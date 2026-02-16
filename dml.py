"""
Data Manipulation Benchmark (DML)

This script consolidates Insert, Update, Delete, and Mixed workload benchmarks.
It allows performing specific DML operations or a weighted mixture of them using subcommands.

Subcommands:
- **insert**: Appends new vectors to the table.
- **update**: Updates random existing vectors.
- **delete**: Deletes random existing vectors.
- **append**: Bulk loads data from a CSV file using LOAD DATA INFILE.
- **mix**: Runs a mixed workload of Insert, Update, and Delete based on provided ratios.

Usage:
    python3 dml.py insert -f cfg.json -n 1000
    python3 dml.py update -f cfg.json -n 500
    python3 dml.py delete -f cfg.json -n 100
    python3 dml.py append -f cfg.json -i dataset.csv
    python3 dml.py mix -f cfg.json -n 5000 -r 1,8,1
"""
import sys
import json
import argparse
import pymysql
import time
import random
import os
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
            
            total_updated = 0
            start_time = time.time()
            
            while total_updated < count:
                current_batch_size = min(batch_size, count - total_updated)
                
                print(f"Selecting {current_batch_size} random rows to update...")
                sql = f"SELECT id FROM {tbl} ORDER BY RAND() LIMIT {current_batch_size}"
                cursor.execute(sql)
                ids = [row[0] for row in cursor.fetchall()]
                
                if not ids:
                    print("No rows found to update. Stopping.")
                    break

                gen = Generator(config, seed=seed + total_updated)
                
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
            
            while total_deleted < count:
                current_batch_size = min(batch_size, count - total_deleted)
                
                print(f"Selecting {current_batch_size} random rows to delete...")
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

def run_append_csv(config, csv_file):
    conn = get_db_connection(config)
    tbl = config['table']
    
    try:
        with conn.cursor() as cursor:
            set_env(cursor, config)
            
            abs_path = os.path.abspath(csv_file)
            print(f"Appending data from CSV file: {abs_path}...")
            
            sql = f"""
            LOAD DATA INFILE '{abs_path}' 
            INTO TABLE {tbl} 
            FIELDS TERMINATED BY ',' 
            ENCLOSED BY '"' 
            LINES TERMINATED BY '\r\n' 
            IGNORE 1 LINES
            """
            
            start_time = time.time()
            print(f"Executing: {sql}")
            cursor.execute(sql)
            end_time = time.time()
            
            print(f"Data append finished in {end_time - start_time:.4f} seconds")
            
    finally:
        conn.close()

def run_mix(config, total_ops, ratios, batch_size=1000, seed=8888):
    conn = get_db_connection(config)
    tbl = config['table']
    
    try:
        r_parts = [float(x) for x in ratios.split(',')]
        if len(r_parts) != 3:
            raise ValueError
    except:
        print("Error: ratios must be 3 comma-separated numbers (e.g. 1,8,1)")
        sys.exit(1)
        
    total_weight = sum(r_parts)
    r_insert = r_parts[0] / total_weight
    r_update = r_parts[1] / total_weight
    r_delete = r_parts[2] / total_weight
    
    print(f"Workload Ratios: Insert={r_insert:.2f}, Update={r_update:.2f}, Delete={r_delete:.2f}")
    
    try:
        with conn.cursor() as cursor:
            set_env(cursor, config)
            
            current_max_id = get_max_id(cursor, tbl)
            next_insert_id = current_max_id + 1 if current_max_id is not None else 0
            
            gen = Generator(config, seed=seed)
            
            total_executed = 0
            start_time = time.time()
            
            while total_executed < total_ops:
                current_batch = min(batch_size, total_ops - total_executed)
                n_insert = int(current_batch * r_insert)
                n_update = int(current_batch * r_update)
                n_delete = current_batch - n_insert - n_update
                
                if n_insert > 0:
                    data = gen.gen_batch(n_insert, next_insert_id)
                    insert_sql = f"INSERT INTO {tbl} (id, embed, i32v, f32v, strv) VALUES (%s, %s, %s, %s, %s)"
                    batch_args = [(d['id'], d['vector'], d['i32v'], d['f32v'], d['str']) for d in data]
                    cursor.executemany(insert_sql, batch_args)
                    next_insert_id += n_insert
                
                if n_update > 0:
                    cursor.execute(f"SELECT id FROM {tbl} ORDER BY RAND() LIMIT {n_update}")
                    ids = [row[0] for row in cursor.fetchall()]
                    if ids:
                        data = gen.gen_batch(len(ids), 0)
                        update_batch = []
                        for i, target_id in enumerate(ids):
                            row = data[i]
                            update_batch.append((row['vector'], row['i32v'], target_id))
                        update_sql = f"UPDATE {tbl} SET embed = %s, i32v = %s WHERE id = %s"
                        cursor.executemany(update_sql, update_batch)

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

            end_time = time.time() - start_time
            print("-" * 40)
            print(f"Mixed workload completed. QPS: {total_executed / end_time:.2f}")
            print("-" * 40)
    finally:
        conn.close()

def main():
    parser = argparse.ArgumentParser(description="Run DML workloads")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Common arguments helper
    def add_common(p):
        p.add_argument("-f", "--config", required=True, help="Path to config file")
        p.add_argument("-n", "--number", type=int, default=1000, help="Number of operations")
        p.add_argument("-b", "--batch_size", type=int, default=1000, help="Batch size")
        p.add_argument("-s", "--seed", type=int, default=8888, help="Random seed")

    # Insert
    p_insert = subparsers.add_parser("insert")
    add_common(p_insert)

    # Update
    p_update = subparsers.add_parser("update")
    add_common(p_update)

    # Delete
    p_delete = subparsers.add_parser("delete")
    add_common(p_delete)

    # Append
    p_append = subparsers.add_parser("append")
    p_append.add_argument("-f", "--config", required=True, help="Path to config file")
    p_append.add_argument("-i", "--input", required=True, help="Input CSV file")

    # Mix
    p_mix = subparsers.add_parser("mix")
    add_common(p_mix)
    p_mix.add_argument("-r", "--ratios", default="1,8,1", help="Mix ratios (Insert,Update,Delete)")

    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    if args.command == "insert":
        run_insert(config, args.number, args.batch_size, args.seed)
    elif args.command == "update":
        run_update(config, args.number, args.batch_size, args.seed)
    elif args.command == "delete":
        run_delete(config, args.number, args.batch_size)
    elif args.command == "append":
        run_append_csv(config, args.input)
    elif args.command == "mix":
        run_mix(config, args.number, args.ratios, args.batch_size, args.seed)

if __name__ == "__main__":
    main()
