"""
Data Manipulation Benchmark (DML)

This script consolidates Insert, Update, Delete, and Mixed workload benchmarks.
It allows performing specific DML operations or a weighted mixture of them.

Modes:
- **Insert (-i)**: Appends new vectors to the table.
- **Update (-u)**: Updates random existing vectors.
- **Delete (-d)**: Deletes random existing vectors.
- **Mix (-m)**: Runs a mixed workload of Insert, Update, and Delete based on provided ratios.

Common Arguments:
- `-n`: Number of operations (default: 1000).
- `-b`: Batch size (default: 1000).
- `-s`: Random seed (default: 8888).

Usage:
    # Insert 1000 rows
    python3 dml.py -f cfg.json -i -n 1000

    # Update 500 rows
    python3 dml.py -f cfg.json -u -n 500

    # Delete 100 rows
    python3 dml.py -f cfg.json -d -n 100

    # Mixed workload (10% insert, 80% update, 10% delete)
    python3 dml.py -f cfg.json -m -n 5000 -r 1,8,1
"""
import sys
import json
import argparse
import pymysql
import time
import random
import numpy as np
from gen import Generator
from db import get_db_connection, set_env

def get_max_id(cursor, table):
    cursor.execute(f"SELECT MAX(id) FROM {table}")
    res = cursor.fetchone()
    return res[0] if res and res[0] is not None else 0

def run_insert(config, count, batch_size=1000, seed=8888):
    conn = get_db_connection(config)
    tbl = config['table']
    
    try:
        with conn.cursor() as cursor:
            set_env(cursor, config)
            start_id = get_max_id(cursor, tbl)
            if start_id is None:
                start_id = 0
            else:
                start_id += 1
            
            print(f"Starting insertion from ID {start_id} for {count} rows (seed: {seed})...")
            
            gen = Generator(config, seed=seed)
            total_inserted = 0
            start_time = time.time()
            
            sql = f"INSERT INTO {tbl} (id, embed, i32v, f32v, strv) VALUES (%s, %s, %s, %s, %s)"
            
            while total_inserted < count:
                current_batch_size = min(batch_size, count - total_inserted)
                data = gen.gen_batch(current_batch_size, start_id + total_inserted)
                
                batch = [(d['id'], d['vector'], d['i32v'], d['f32v'], d['str']) for d in data]
                cursor.executemany(sql, batch)
                
                total_inserted += len(batch)
                if total_inserted % 10000 == 0 or total_inserted == count:
                    print(f"Inserted {total_inserted} rows...")
                    
            end_time = time.time()
            print(f"Inserted {total_inserted} rows in {end_time - start_time:.4f} seconds")
            
    finally:
        conn.close()

def run_update(config, count, batch_size=1000, seed=8888):
    conn = get_db_connection(config)
    tbl = config['table']
    
    try:
        with conn.cursor() as cursor:
            set_env(cursor, config)
            
            # For update, we usually pick random rows. 
            # If count is large, doing it in one SELECT might be heavy, but let's stick to simple logic for now
            # or use batching if count > batch_size to be safe.
            
            total_updated = 0
            start_time = time.time()
            
            # Using a loop to handle large counts safely
            while total_updated < count:
                current_batch_size = min(batch_size, count - total_updated)
                
                print(f"Selecting {current_batch_size} random rows to update...")
                sql = f"SELECT id FROM {tbl} ORDER BY RAND() LIMIT {current_batch_size}"
                cursor.execute(sql)
                ids = [row[0] for row in cursor.fetchall()]
                
                if not ids:
                    print("No rows found to update. Stopping.")
                    break

                gen = Generator(config, seed=seed + total_updated) # Shift seed to vary data
                
                data = gen.gen_batch(len(ids), 0)
                update_batch = []
                for i, target_id in enumerate(ids):
                    row = data[i]
                    update_batch.append((row['vector'], row['i32v'], target_id))
                
                update_sql = f"UPDATE {tbl} SET embed = %s, i32v = %s WHERE id = %s"
                cursor.executemany(update_sql, update_batch)
                
                total_updated += len(ids)
                print(f"Updated batch of {len(ids)} rows...")
                
                if len(ids) < current_batch_size:
                    print("Table has fewer rows than requested.")
                    break
            
            end_time = time.time()
            print(f"Total updated: {total_updated} rows in {end_time - start_time:.4f} seconds")
            
    finally:
        conn.close()

def run_delete(config, count, batch_size=1000):
    conn = get_db_connection(config)
    tbl = config['table']
    
    try:
        with conn.cursor() as cursor:
            set_env(cursor, config)
            
            total_deleted = 0
            start_time = time.time()
            
            # Loop for safety on large counts
            while total_deleted < count:
                current_batch_size = min(batch_size, count - total_deleted)
                
                print(f"Selecting {current_batch_size} random rows to delete...")
                # Optimization: LIMIT is applied here
                sql = f"SELECT id FROM {tbl} ORDER BY RAND() LIMIT {current_batch_size}"
                cursor.execute(sql)
                ids = [row[0] for row in cursor.fetchall()]
                
                if not ids:
                    print("No rows found to delete. Stopping.")
                    break

                id_str = ','.join(str(id) for id in ids)
                delete_sql = f"DELETE FROM {tbl} WHERE id IN ({id_str})"
                cursor.execute(delete_sql)
                
                total_deleted += len(ids)
                print(f"Deleted batch of {len(ids)} rows...")
                
                if len(ids) < current_batch_size:
                    break

            end_time = time.time()
            print(f"Total deleted: {total_deleted} rows in {end_time - start_time:.4f} seconds")
            
    finally:
        conn.close()

def run_mix(config, total_ops, ratios, batch_size=1000, seed=8888):
    conn = get_db_connection(config)
    tbl = config['table']
    
    # Parse ratios (Insert, Update, Delete)
    try:
        r_parts = [float(x) for x in ratios.split(',')]
        if len(r_parts) != 3:
            raise ValueError
    except:
        print("Error: ratios must be 3 comma-separated numbers (e.g. 0.1,0.8,0.1)")
        sys.exit(1)
        
    total_weight = sum(r_parts)
    r_insert = r_parts[0] / total_weight
    r_update = r_parts[1] / total_weight
    r_delete = r_parts[2] / total_weight
    
    print(f"Workload Ratios: Insert={r_insert:.2f}, Update={r_update:.2f}, Delete={r_delete:.2f}")
    
    try:
        with conn.cursor() as cursor:
            set_env(cursor, config)
            
            # Initialize insert ID counter
            current_max_id = get_max_id(cursor, tbl)
            next_insert_id = current_max_id + 1 if current_max_id is not None else 0
            
            gen = Generator(config, seed=seed)
            
            total_executed = 0
            start_time = time.time()
            
            while total_executed < total_ops:
                current_batch = min(batch_size, total_ops - total_executed)
                
                # Calculate counts for this batch
                n_insert = int(current_batch * r_insert)
                n_update = int(current_batch * r_update)
                n_delete = current_batch - n_insert - n_update
                
                # 1. Insert
                if n_insert > 0:
                    data = gen.gen_batch(n_insert, next_insert_id)
                    insert_sql = f"INSERT INTO {tbl} (id, embed, i32v, f32v, strv) VALUES (%s, %s, %s, %s, %s)"
                    batch_args = [(d['id'], d['vector'], d['i32v'], d['f32v'], d['str']) for d in data]
                    cursor.executemany(insert_sql, batch_args)
                    next_insert_id += n_insert
                
                # 2. Update
                if n_update > 0:
                    cursor.execute(f"SELECT id FROM {tbl} ORDER BY RAND() LIMIT {n_update}")
                    ids = [row[0] for row in cursor.fetchall()]
                    
                    if ids:
                        # Generate random data
                        data = gen.gen_batch(len(ids), 0)
                        update_batch = []
                        for i, target_id in enumerate(ids):
                            row = data[i]
                            update_batch.append((row['vector'], row['i32v'], target_id))
                        
                        update_sql = f"UPDATE {tbl} SET embed = %s, i32v = %s WHERE id = %s"
                        cursor.executemany(update_sql, update_batch)

                # 3. Delete
                if n_delete > 0:
                    cursor.execute(f"SELECT id FROM {tbl} ORDER BY RAND() LIMIT {n_delete}")
                    ids = [row[0] for row in cursor.fetchall()]
                    
                    if ids:
                        id_str = ','.join(str(id) for id in ids)
                        cursor.execute(f"DELETE FROM {tbl} WHERE id IN ({id_str})")

                total_executed += current_batch
                if total_executed % 1000 == 0 or total_executed == total_ops:
                    elapsed = time.time() - start_time
                    print(f"Executed {total_executed}/{total_ops} ops (Time: {elapsed:.2f}s)...")

            end_time = time.time()
            duration = end_time - start_time
            print("-" * 40)
            print(f"Mixed workload completed.")
            print(f"Total Operations: {total_executed}")
            print(f"Time: {duration:.4f} s")
            print(f"QPS: {total_executed / duration:.2f}")
            print("-" * 40)

    finally:
        conn.close()

def run_append_csv(config, csv_file):
    conn = get_db_connection(config)
    tbl = config['table']
    
    try:
        with conn.cursor() as cursor:
            set_env(cursor, config)
            
            abs_path = os.path.abspath(csv_file)
            print(f"Appending data from CSV file: {abs_path}...")
            
            # Use LOAD DATA INFILE for faster loading
            sql = f"""
            LOAD DATA INFILE '{abs_path}' 
            INTO TABLE {tbl} 
            FIELDS TERMINATED BY ',' 
            ENCLOSED BY '"' 
            LINES TERMINATED BY '\\r\\n' 
            IGNORE 1 LINES
            """
            
            start_time = time.time()
            print(f"Executing: {sql}")
            cursor.execute(sql)
            end_time = time.time()
            
            print(f"Data append finished in {end_time - start_time:.4f} seconds")
            
    finally:
        conn.close()

def main():
    parser = argparse.ArgumentParser(description="Run DML workloads (Insert, Update, Delete, Mix, Append)")
    parser.add_argument("-f", "--config", required=True, help="Path to config file")
    parser.add_argument("-n", "--number", type=int, default=1000, help="Number of operations")
    parser.add_argument("-b", "--batch_size", type=int, default=1000, help="Batch size")
    parser.add_argument("-s", "--seed", type=int, default=8888, help="Random seed")
    
    # Modes
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-i", "--insert", action="store_true", help="Insert mode (generate data)")
    group.add_argument("-d", "--delete", action="store_true", help="Delete mode")
    group.add_argument("-u", "--update", action="store_true", help="Update mode")
    group.add_argument("-m", "--mix", action="store_true", help="Mixed workload mode")
    group.add_argument("-a", "--append", help="Append data from CSV file")
    
    # Mix specific
    parser.add_argument("-r", "--ratios", default="1,8,1", help="Mix ratios (Insert,Update,Delete)")
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    if args.insert:
        run_insert(config, args.number, args.batch_size, args.seed)
    elif args.delete:
        run_delete(config, args.number, args.batch_size)
    elif args.update:
        run_update(config, args.number, args.batch_size, args.seed)
    elif args.mix:
        run_mix(config, args.number, args.ratios, args.batch_size, args.seed)
    elif args.append:
        run_append_csv(config, args.append)

if __name__ == "__main__":
    main()
