import json
import logging
import os
import psycopg
import threading
from typing import List

class Property_fileloader:
    def __init__(self, args: dict):
        self.batch_size: int = int(args.get("batch_size", 128))
        self.data_folder: str = str(args.get("data_folder", "./data"))
        self.tracker_folder = os.path.join(self.data_folder, ".file_tracker")
        os.makedirs(self.tracker_folder, exist_ok=True)

        self.all_files = sorted([f for f in os.listdir(self.data_folder) if f.endswith(".json")])
        self.lock = threading.Lock()
        self.thread_id = None

        self.current_file = None
        self.current_file_handle = None
        self.current_file_data = []
        self.current_batch = []
        self.file_position = 0

    def _tracker_path(self, filename):
        return os.path.join(self.tracker_folder, f"{filename}")

    def _load_tracker(self, filename):
        path = self._tracker_path(filename)
        if os.path.exists(path):
            with open(path, 'r') as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    logging.error(f"there's an issue with the file at {f}")
                    raise
                except:
                    logging.error(f"there's an issue with the file at {f}")
                    raise
        return None

    def _save_tracker(self, filename, data):
        path = self._tracker_path(filename)
        with open(path, 'w') as f:
            json.dump(data, f)

    def setup(self, conn: psycopg.Connection, id: int, total_thread_count: int):
        self.thread_id = id
        with conn.cursor() as cur:
            print(
                f"My thread ID is {id}. The total count of threads is {total_thread_count}"
            )
            print(cur.execute(f"select version()").fetchone()[0])

    def loop(self):
        return [self.parse, self.load]

    def _assign_file(self):
        with self.lock:
            for filename in self.all_files:
                tracker = self._load_tracker(filename)

                if tracker:
                    if tracker.get("completed"):
                        continue  # skip completed
                    if tracker.get("thread_id") != self.thread_id:
                        continue  # assigned to another thread
                else:
                    tracker = {"processed": 0, "completed": False, "thread_id": self.thread_id}
                    self._save_tracker(filename, tracker)

                return filename
            return None

    def _open_next_file(self):
        self.current_file = self._assign_file()
        if not self.current_file:
            return False

        full_path = os.path.join(self.data_folder, self.current_file)
        with open(full_path, 'r') as f:
            try:
                self.current_file_data = json.load(f)
                tracker = self._load_tracker(self.current_file)
                self.file_position = tracker.get("processed", 0)
                return True
            except json.JSONDecodeError:
                logging.error(f"there's an issue with the file at {full_path}")
                self.current_file_data = []
                return False
            except:
                logging.error(f"there's an issue with the file at {full_path}")
                return False

    def parse(self, conn: psycopg.Connection):
        self.current_batch = []

        if not self.current_file_data and not self._open_next_file():
            return  # No more files

        while len(self.current_batch) < self.batch_size and self.file_position < len(self.current_file_data):
            record = self.current_file_data[self.file_position]
            self.current_batch.append(record)
            self.file_position += 1

    def load(self, conn: psycopg.Connection):
        if not self.current_batch:
            return

        with conn.cursor() as cur:
            placeholders = ', '.join(['(%s::jsonb)'] * len(self.current_batch))
            sql = f"""
            INSERT INTO raw_property_data (property_data) VALUES {placeholders}
            ON CONFLICT (property_id)
            DO UPDATE SET property_data = excluded.property_data;
            """
            values = [json.dumps(obj) for obj in self.current_batch]
            cur.execute(sql, values)

        tracker = self._load_tracker(self.current_file)
        tracker["processed"] = self.file_position
        if self.file_position == len(self.current_file_data):
            tracker["completed"] = True
            self.current_file_data = []
        self._save_tracker(self.current_file, tracker)

        self.current_batch = []
