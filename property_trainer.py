from concurrent.futures import ProcessPoolExecutor, as_completed
from fasteners import InterProcessLock
from filelock import FileLock
import joblib
import logging
from multiprocessing import Manager, freeze_support
import os
import pickle
import shutil
import signal
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tempfile
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential, load_model, save_model
import threading
import time

ARCHIVED_DIR = "data/archived"
POLL_INTERVAL = 5  # seconds between scans
PROCESSED_DIR = "data/processed"
MODEL_DIR = "data/models"
SCALER_DIR = "data/scaler"

def handle_shutdown(running):
    def inner(sig, frame):
        logging.warning("Shutdown signal received.")
        running.value = False
    return inner


# Global lock map by model path
model_thread_locks = {}

def get_thread_lock(model_path):
    if model_path not in model_thread_locks:
        model_thread_locks[model_path] = threading.Lock()
    return model_thread_locks[model_path]


def safe_save_model(model, model_path):
    # Write to a temp file first
    dir_name = os.path.dirname(model_path)
    with tempfile.NamedTemporaryFile(dir=dir_name, delete=False, suffix=".keras") as tmp_file:
        tmp_path = tmp_file.name

    save_model(model, tmp_path)
    logging.info(f"Moving model file from {tmp_path} to {model_path}")
    os.replace(tmp_path, model_path)  # Atomic move


def scale_and_fit(municipality, model_path, X_train, X_test, y_train, y_test):
    scaler_path = os.path.join(SCALER_DIR, municipality + ".features")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        X_train = scaler.transform(X_train)
    else:
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        joblib.dump(scaler, scaler_path)

    X_test = scaler.transform(X_test)
    
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        model = Sequential()
        model.add(Dense(19, activation='relu')) # Inputs

        model.add(Dense(304, activation='relu')) # First hidden layer
        model.add(Dropout(0.2))

        model.add(Dense(304, activation='relu')) # Second hidden layer
        model.add(Dropout(0.2))

        model.add(Dense(304, activation='relu')) # Third hidden layer
        model.add(Dropout(0.2))

        model.add(Dense(304, activation='relu')) # Forth hidden layer
        model.add(Dropout(0.2))

        model.add(Dense(1, activation='relu')) # Output layer (regression)
        model.compile(optimizer='adam', loss='mse')
    
    early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100)
    model.fit(x=X_train, y=y_train, epochs=1000, batch_size=32,
              validation_data=(X_test, y_test), verbose=0, callbacks=[early_stop])
    safe_save_model(model, model_path)


def train_model_for_municipality(municipality, model_path, df):
    X = df.drop('Sale_Price', axis=1).values
    y = df['Sale_Price'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    scale_and_fit(municipality, model_path, X_train, X_test, y_train, y_test)


def process_pickle_file(running, pickle_path):
    if not running.value:
        return
    
    municipality = os.path.basename(pickle_path).replace(".pkl", "")
    lock_path = f"{pickle_path}.lock"

    with FileLock(lock_path):
        if not os.path.exists(pickle_path):
            return

        with open(pickle_path, "rb") as f:
            df = pickle.load(f)

        # if there aren't enough records to train on we'll defer until later
        if df is None or len(df) <= 500:
            return

        # ✅ move the pickle file after loading safely
        destination_path = os.path.join(ARCHIVED_DIR, os.path.basename(pickle_path))
        shutil.move(pickle_path, destination_path)

    model_path = os.path.join(MODEL_DIR, f"{municipality}.keras")
    model_lock_path = f"{model_path}.lock"

    thread_lock = get_thread_lock(model_path)
    with thread_lock:  # Prevent concurrent threads
        with InterProcessLock(model_lock_path): # Also use a write lock for training
            logging.info(f"Training model for {municipality}")
            train_model_for_municipality(municipality, model_path, df)


def run_loader(running, max_processes=10):
    logging.info("Model trainer running...")
    with ProcessPoolExecutor(max_workers=max_processes) as executor:
        futures = {}

        while running.value:
            # Get list of all unprocessed .pkl files
            files = [
                os.path.join(PROCESSED_DIR, f)
                for f in os.listdir(PROCESSED_DIR)
                if f.endswith(".pkl")
            ]

            # Submit new files to the pool
            for file_path in files:
                if not running.value:
                    break
                future = executor.submit(process_pickle_file, running, file_path)
                futures[future] = file_path

            # Clean up completed tasks
            done_futures = [fut for fut in futures if fut.done()]
            for fut in done_futures:
                try:
                    fut.result()
                except Exception as e:
                    logging.exception(f"Error processing {futures[fut]}", exc_info=e)
                del futures[fut]

            time.sleep(POLL_INTERVAL)

        logging.info("Stopping: waiting for in-progress tasks to finish...")
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logging.exception("Error during shutdown task", exc_info=e)


def main():
    logging.basicConfig(level=logging.INFO)
    freeze_support()  # good to keep for macOS/frozen apps

    # This will be shared between processes
    manager = Manager()
    running = manager.Value("b", True)  # boolean flag for graceful shutdown

    # Install signal handlers
    signal.signal(signal.SIGINT, handle_shutdown(running))
    signal.signal(signal.SIGTERM, handle_shutdown(running))

    run_loader(running, max_processes=4)
    logging.info("All threads have exited")


if __name__ == "__main__":
    main()
