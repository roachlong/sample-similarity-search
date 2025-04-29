import os
import psycopg
from property_fileloader import Property_fileloader  # make sure this matches the file/module name

conn_string = os.getenv("DATABASE_URL")

args = {
    "batch_size": 128,
    "data_folder": "./data/tmp"
}

loader = Property_fileloader(args)

with psycopg.connect(conn_string) as conn:
    loader.setup(conn, id=0, total_thread_count=1)

    for step in loader.loop():
        step(conn)
