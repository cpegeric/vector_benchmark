"""
Database Setup and Data Loading

This script prepares the database environment for benchmarking.
It performs the following tasks:
1. **Schema Creation**: Creates the target table and vector index based on the configuration.
2. **Environment Configuration**: Sets session variables (e.g., HNSW parameters) from `cfg.json`.
3. **Data Loading**:
    - **CSV Mode**: Uses `LOAD DATA INFILE` for high-performance bulk loading from a CSV file.
    - **Stream Mode**: Generates data on-the-fly and inserts it in batches.
4. **Index Creation**: Supports both synchronous (Insert -> Index) and asynchronous (Index -> Insert) creation flows.

Modes:
- **Default (Sync)**: Inserts data first, then creates the index. This is generally faster for initial bulk loads.
- **Async (-a)**: Creates the index first, then inserts data. This tests the system's ability to index data as it arrives.

Usage:
    # Sync mode (recommended for initial load)
    python3 create.py -f cfg.json -i dataset.csv

    # Async mode
    python3 create.py -f cfg.json -a -i dataset.csv
"""
import sys
import json
import argparse
import pymysql
import time
import csv
import os
import numpy as np
from gen import Generator
from db import get_db_connection, set_env

optype2distfn = {
    'vector_l2_ops': 'l2_distance',
    'vector_cosine_ops': 'cosine_distance',
    'vector_ip_ops': 'inner_product',
    'vector_l2sq_ops': 'l2_distance_sq'
}

def create_table(cursor, config):
    tbl = config['table']
    dim = config['dimension']
    
    sql = f"""
    CREATE TABLE {tbl} (
        id BIGINT PRIMARY KEY,
        embed VECF32({dim}),
        i32v INT,
        f32v FLOAT,
        strv VARCHAR(255)
    )
    """
    print(f"Executing: {sql}")
    cursor.execute(sql)

def drop_table(cursor, config):
    tbl = config['table']
    sql = f"DROP TABLE IF EXISTS {tbl}"
    print(f"Executing: {sql}")
    cursor.execute(sql)

def create_index(cursor, config, async_mode=False):
    tbl = config['table']
    index_cfg = config.get('index', {})
    
    # Support both old string format and new dict format for index
    if isinstance(index_cfg, str):
        idx_name = index_cfg
        idx_type = config.get('index_type', 'hnsw')
        dist = config.get('distance', 'vector_l2_ops')
        # Default HNSW params
        m, ef_c, ef_s = 100, 500, 200
        lists = None
    else:
        idx_name = index_cfg.get('name', 'myidx')
        idx_type = index_cfg.get('type', 'hnsw')
        dist = index_cfg.get('op_type', config.get('distance', 'vector_l2_ops'))
        m = index_cfg.get('m', 100)
        ef_c = index_cfg.get('ef_construction', 400)
        ef_s = index_cfg.get('ef_search', 200)
        lists = index_cfg.get('lists')
    
    # Check if index type is valid
    if idx_type not in ['hnsw', 'ivfflat']:
        print(f"Unknown index type: {idx_type}, defaulting to hnsw")
        idx_type = 'hnsw'
        
    async_str = "ASYNC" if async_mode else ""
    
    if idx_type == 'hnsw':
        sql = f"""
        CREATE INDEX {idx_name} USING hnsw ON {tbl}(embed) 
        m={m} ef_construction={ef_c} ef_search={ef_s} op_type \"{dist}\" {async_str}
        """
    else:
        # ivfflat
        if lists is None:
            # Calculate lists based on dataset size logic from indextest.py
            nitem = config['dataset_size']
            lists = int(nitem / 1000) if nitem < 1000000 else int(np.sqrt(nitem))
            if lists < 10: lists = 10
        
        sql = f"""
        CREATE INDEX {idx_name} USING ivfflat ON {tbl}(embed) 
        lists={lists} op_type \"{dist}\" {async_str}
        """

    print(f"Executing: {sql}")
    start = time.time()
    cursor.execute(sql)
    end = time.time()
    print(f"Create index finished in {end - start:.4f} seconds")

def insert_data(cursor, config, csv_files=None, seed=8888):
    tbl = config['table']
    batch_size = config.get('batch_size', 1000)
    total_size = config['dataset_size']
    
    print(f"Inserting data into {tbl} (seed: {seed})...")
    start_time = time.time()
    
    if csv_files:
        for csv_file in csv_files:
            abs_path = os.path.abspath(csv_file)
            print(f"Loading data from CSV file: {abs_path}...")
            
            # Use LOAD DATA INFILE for faster loading
            # The CSV format from gen.py corresponds to the table structure
            # We assume standard CSV format with \r\n line terminators (Python default)
            sql = f"""
            LOAD DATA INFILE '{abs_path}' 
            INTO TABLE {tbl} 
            FIELDS TERMINATED BY ',' 
            ENCLOSED BY '"' 
            LINES TERMINATED BY '\\r\\n' 
            IGNORE 1 LINES
            """
            
            print(f"Executing: {sql}")
            cursor.execute(sql)
            # We don't have exact row count from LOAD DATA result easily in pymysql without parsing
            # but the operation is done. 
        
    else:
        # Stream generation
        sql = f"INSERT INTO {tbl} (id, embed, i32v, f32v, strv) VALUES (%s, %s, %s, %s, %s)"
        gen = Generator(config, seed=seed)
        total_inserted = 0
        while total_inserted < total_size:
            current_batch_size = min(batch_size, total_size - total_inserted)
            data = gen.gen_batch(current_batch_size, total_inserted)
            # Convert dict to tuple for executemany
            batch = [(d['id'], d['vector'], d['i32v'], d['f32v'], d['str']) for d in data]
            
            cursor.executemany(sql, batch)
            total_inserted += len(batch)
            if total_inserted % 10000 == 0:
                print(f"Inserted {total_inserted} rows...")

    end_time = time.time()
    print(f"Data load finished in {end_time - start_time:.4f} seconds")

def main():
    parser = argparse.ArgumentParser(description="Create table and index")
    parser.add_argument("-f", "--config", required=True, help="Path to config file")
    parser.add_argument("-i", "--input", action="append", help="Input CSV file(s). Can be specified multiple times.")
    parser.add_argument("-s", "--seed", type=int, default=8888, help="Random seed for stream generation")
    parser.add_argument("-a", "--async_mode", action="store_true", help="Asynchronous mode")
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Connect without database first
    conn = get_db_connection(config, use_db=False)
    
    try:
        with conn.cursor() as cursor:
            # 0. Ensure database exists and use it
            dbname = config['database']
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {dbname}")
            cursor.execute(f"USE {dbname}")
            print(f"Using database '{dbname}'.")

            # 1. Set environment
            set_env(cursor, config)

            # 2. Drop Table
            drop_table(cursor, config)
            
            # 3. Create Table
            create_table(cursor, config)
            
            # 4. Mode handling
            if args.async_mode:
                # Async: Create Index -> Insert
                print("Running in ASYNC mode")
                create_index(cursor, config, async_mode=True)
                insert_data(cursor, config, csv_files=args.input, seed=args.seed)
            else:
                # Sync (Default): Insert -> Create Index
                print("Running in SYNC mode (default)")
                insert_data(cursor, config, csv_files=args.input, seed=args.seed)
                create_index(cursor, config, async_mode=False)
                
    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    main()