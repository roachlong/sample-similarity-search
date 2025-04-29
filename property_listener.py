import threading
import signal
import time
import json
import logging
from collections import defaultdict, deque
from confluent_kafka import Consumer, KafkaException, TopicPartition
from property_preprocessor import process_batch

# Configure logging
logging.basicConfig(level=logging.INFO)

# Application Environment Configuration
CONFIG = {
    'BootstrapServers': "localhost:9092",
    'GroupIdPrefix': "Property-Data-Loader",
    'AutoOffsetReset': 'earliest',
    'EnableAutoCommit': True,
    'EnableAutoOffsetStore': False,
    'topic': "property-updates",
    'numPartitions': 10,
    'batchSize': 10000,  # Number of messages to process in a batch
    'batchWindow': 600,  # seconds
    'maxRetries': 5,
}

running = True

def graceful_shutdown(signum, frame):
    global running
    logging.info("Shutdown signal received")
    running = False

signal.signal(signal.SIGINT, graceful_shutdown)

class PartitionConsumerThread(threading.Thread):
    def __init__(self, partition):
        super().__init__()
        self.partition = partition
        self.buffer = defaultdict(deque)
        self.last_offset = None
        self.failed_attempts = 0
        self.start_time = time.time()
    
    def flush(self):
        threads = []
        for municipality, messages in self.buffer.items():
            if not messages:
                continue
            logging.info(f"Flushing {len(messages)} messages for {municipality} from partition {self.partition}")

            # Clone the deque to list
            data = list(messages)
            thread = threading.Thread(target=process_batch, args=(municipality, data))
            thread.start()
            threads.append(thread)

        # Wait for all processing threads to finish
        for t in threads:
            t.join()

        # Offset commit after successful processing
        if self.last_offset is not None:
            self.consumer.commit(offsets=[TopicPartition(CONFIG['topic'], self.partition, self.last_offset + 1)], asynchronous=False)

        self.buffer.clear()
        self.start_time = time.time()
        self.failed_attempts = 0


    def run(self):
        while running and self.failed_attempts < CONFIG['maxRetries']:
            try:
                while running:
                    consumer_conf = {
                        'bootstrap.servers': CONFIG['BootstrapServers'],
                        'group.id': f"{CONFIG['GroupIdPrefix']}{self.partition}",
                        'auto.offset.reset': CONFIG['AutoOffsetReset'],
                        'enable.auto.commit': CONFIG['EnableAutoCommit'],
                        'enable.auto.offset.store': CONFIG['EnableAutoOffsetStore'],
                    }
                    self.consumer = Consumer(consumer_conf)
                    time.sleep(1)  # Delay before subscription
                    self.consumer.assign([TopicPartition(CONFIG['topic'], self.partition)])
                    
                    while running:
                        msg = self.consumer.poll(timeout=1.0)
                        
                        if msg is None:
                            if time.time() - self.start_time > CONFIG['batchWindow']:
                                self.flush()
                            continue

                        if msg.error():
                            logging.error(f"Consumer error: {msg.error()}")
                            continue

                        try:
                            message = json.loads(msg.value().decode('utf-8'))
                            municipality = message.get('municipality')
                            if municipality:
                                sale_price = message.get(
                                    "property_data", {}
                                ).get(
                                    "countyData", {}
                                ).get("Sale_Price")
                                if sale_price not in (None, "", 0):
                                    self.buffer[municipality].append(message)
                                    self.last_offset = msg.offset()
                            else:
                                logging.warning("Invalid message received: " + str(message))
                                # Publish to error queue
                        except Exception as e:
                            logging.exception("Error processing message: " + str(msg))

                        if sum(len(q) for q in self.buffer.values()) >= CONFIG['batchSize'] or time.time() - self.start_time > CONFIG['batchWindow']:
                            self.flush()

                    if not running:
                        self.flush()
            except Exception as e:
                logging.exception("Exception in partition thread: " + e)
                self.failed_attempts += 1
                time.sleep(2 ** self.failed_attempts)
            finally:
                try:
                    self.consumer.unsubscribe()
                    self.consumer.close()
                except:
                    pass

# Start consumer threads
threads = [PartitionConsumerThread(i) for i in range(CONFIG['numPartitions'])]
for t in threads:
    t.start()
for t in threads:
    t.join()

logging.info("All threads have exited")
