"""
Vector Search Benchmark (Recall & QPS)

This script measures the performance of vector search operations, including:
1. **Recall**: How many true nearest neighbors are retrieved?
2. **QPS**: Queries Per Second.
3. **Latency**: Average query execution time.

Features:
- **Parallel Execution**: Uses `ThreadPoolExecutor` to issue concurrent search queries.
- **Data Generation**: Uses the `Generator` class to reproduce the target vectors, avoiding storage overhead.
- **Search Modes**:
    - **Normal**: Standard k-NN search.
    - **Pre-filtering**: Combines vector similarity with metadata filters (e.g., `WHERE i32v < 10`).
- **Flexible Filters**: Supports CLI options for filtering by integer (`--i32v`), float (`--f32v`), and string (`--str`) metadata.
- **Performance**:
    - **Chunked Processing**: Limits memory usage by processing large datasets in blocks.
    - **Query Timing**: Measures search wall time separately from data generation time.

Usage:
    # Normal Search
    python3 recall.py -f cfg.json -m normal -t 8

    # Pre-filtering
    python3 recall.py -f cfg.json -m pre --i32v 100 --str "abc"
"""
import sys
import json
import argparse
import pymysql
import time
import numpy as np
import csv
import ast
import gzip
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from gen import Generator, AsyncGenerator
from db import get_db_connection, set_env

optype2distfn = {
    'vector_l2_ops': 'l2_distance',
    'vector_cosine_ops': 'cosine_distance',
    'vector_ip_ops': 'inner_product',
    'vector_l2sq_ops': 'l2_distance_sq'
}

def search_worker(config, mode, dataset, start, end, filters=None):
    conn = get_db_connection(config)
    tbl = config['table']
    
    index_cfg = config.get('index', {})
    if isinstance(index_cfg, str):
        dist = config.get('distance', 'vector_l2_ops')
    else:
        dist = index_cfg.get('op_type', config.get('distance', 'vector_l2_ops'))
    
    op_type = optype2distfn.get(dist, 'l2_distance')
    
    correct = 0
    eligible = 0
    total = 0
    start_time = time.time()
    
    # Construct WHERE clause from filters
    where_clause = ""
    if filters:
        conditions = []
        if filters.get('i32v') is not None:
            conditions.append(f"i32v < {filters['i32v']}")
        if filters.get('f32v') is not None:
            conditions.append(f"f32v < {filters['f32v']}")
        if filters.get('str') is not None:
            conditions.append(f"strv = '{filters['str']}'")
            
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)
    
    try:
        with conn.cursor() as cursor:
            # Set environment from config
            set_env(cursor, config)
            
            for i in range(start, end):
                row = dataset[i]
                query_vec = row['vector']
                target_id = row['id']
                
                # Check if target row is eligible (satisfies the filter)
                is_eligible = True
                if filters:
                    if filters.get('i32v') is not None and not (row['i32v'] < filters['i32v']): is_eligible = False
                    if filters.get('f32v') is not None and not (row['f32v'] < filters['f32v']): is_eligible = False
                    if filters.get('str') is not None and not (row['str'] == filters['str']): is_eligible = False
                
                if is_eligible:
                    eligible += 1
                total += 1 # Total queries executed, including non-eligible ones if they pass filters
                # Dynamic SQL generation
                current_where = where_clause
                
                if mode == 'normal':
                    sql = f"SELECT id FROM {tbl} {current_where} ORDER BY {op_type}(embed, '{query_vec}') LIMIT 1"
                elif mode == 'pre':
                    sql = f"SELECT id FROM {tbl} {current_where} ORDER BY {op_type}(embed, '{query_vec}') LIMIT 1 BY RANK WITH OPTION 'mode=pre'"
                elif mode == 'post':
                    sql = f"SELECT id FROM {tbl} {current_where} ORDER BY {op_type}(embed, '{query_vec}') LIMIT 1 BY RANK WITH OPTION 'mode=post'"
                elif mode == 'force':
                    sql = f"SELECT id FROM {tbl} {current_where} ORDER BY {op_type}(embed, '{query_vec}') LIMIT 1 BY RANK WITH OPTION 'mode=force'"
                
                cursor.execute(sql)
                res = cursor.fetchone()
                
                if res and res[0] == target_id and is_eligible:
                    correct += 1
                
                
    finally:
        conn.close()
        
    end_time = time.time()
    return correct, eligible, total, end_time - start_time

def run_recall_test(config, mode, threads, number=None, seed=8888, filters=None, csv_files=None, start_id=0):
    total_size = number if number is not None else config['dataset_size']
    # Use a large chunk size to minimize connection overhead per batch
    # But ensuring it's not too huge for memory
    chunk_size = 10000 
    
    print(f"Starting recall test: queries={total_size}, mode={mode}, threads={threads}, seed={seed}, start_id={start_id}")
    if filters:
        print(f"Filters: {filters}")
    if csv_files:
        print(f"Reading from CSV files: {csv_files}")
        
    print(f"Processing in chunks of {chunk_size}...")

    gen = None
    agen = None
    
    model_load_time_s = 0

    # --- Warm-up Query Acquisition ---
    # Create a temporary Generator to get a random warm-up query without affecting the main data stream
    temp_gen = Generator(config, seed=seed)
    # Generate a random vector for warm-up. Dimension comes from config.
    dim = config['dimension']
    warmup_vector = np.random.RandomState(seed).rand(1, dim).astype(np.float32)[0].tolist()
    warmup_query_data = {'id': -1, 'vector': warmup_vector, 'i32v': 0, 'f32v': 0.0, 'str': ''}

    print("Executing warm-up query...")
    warmup_start_time = time.time()
    # search_worker needs dataset, start, end. So we pass a list with one item. No filters for warm-up.
    _, _, _, _ = search_worker(config, mode, [warmup_query_data], 0, 1, None) 
    warmup_end_time = time.time()
    model_load_time_s = warmup_end_time - warmup_start_time
    print(f"Model warm-up query took {model_load_time_s:.4f} s")
    
    # --- End Warm-up Query Acquisition ---

    if not csv_files: # Only use AsyncGenerator if no CSV files are provided
        gen = Generator(config, seed=seed)
        if start_id > 0:
            print(f"Fast-forwarding generator to ID {start_id}...")
            # Advance the generator state
            left = start_id
            while left > 0:
                step = min(10000, left)
                gen.gen_batch(step, 0)
                left -= step
        
        agen = AsyncGenerator(gen, start_id=start_id, chunk_size=chunk_size)
    
    total_correct = 0
    total_eligible = 0
    total_queries = 0
    total_search_wall_time = 0
    total_worker_cpu_time = 0
    
    processed_count = 0
    
    # --- CSV reading setup for multiple files ---
    csv_files_iterator = iter(csv_files) if csv_files else None
    current_csv_f = None
    current_csv_reader = None

    def get_next_csv_row():
        nonlocal current_csv_f, current_csv_reader, csv_files_iterator
        while True:
            if current_csv_reader:
                try:
                    return next(current_csv_reader)
                except StopIteration:
                    # Current file exhausted, close it and try next
                    current_csv_f.close()
                    current_csv_f = None
                    current_csv_reader = None
            
            if csv_files_iterator:
                try:
                    # Open next file
                    next_file_path = next(csv_files_iterator)
                    if next_file_path.endswith('.gz'):
                        current_csv_f = gzip.open(next_file_path, 'rt', newline='')
                    else:
                        current_csv_f = open(next_file_path, 'r', newline='')
                    current_csv_reader = csv.DictReader(current_csv_f)
                except StopIteration:
                    # No more files to open
                    csv_files_iterator = None
                    raise # Propagate StopIteration
            else:
                # No files at all, or all files processed
                raise StopIteration # Signal no more data

    try:
        while processed_count < total_size:
            current_batch_size = min(chunk_size, total_size - processed_count)
            
            # 1. Get Data
            dataset = []
            if csv_files: # Check if any CSV files were provided
                while len(dataset) < current_batch_size:
                    try:
                        row = get_next_csv_row()
                        # Skip if ID is less than start_id
                        if int(row['id']) < start_id:
                            continue
                            
                        # Parse vector
                        try:
                            # Attempt JSON parse first (faster)
                            row['vector'] = json.loads(row['vector'])
                        except:
                            # Fallback to ast.literal_eval
                            row['vector'] = ast.literal_eval(row['vector'])
                            
                        # Convert types
                        row['id'] = int(row['id'])
                        if 'i32v' in row: row['i32v'] = int(row['i32v'])
                        if 'f32v' in row: row['f32v'] = float(row['f32v'])
                        
                        dataset.append(row)
                    except StopIteration:
                        break # No more rows from any CSV file
                if not dataset: # If no data could be read after trying all files
                    break
            else:
                # Use AsyncGenerator to get batch
                dataset = agen.get_batch(current_batch_size)
            
            if not dataset:
                break

            # 2. Prepare Search (Time ignored)
            items_per_thread = len(dataset) // threads
            remainder = len(dataset) % threads
            
            start_idx = 0
            search_args = []
            for i in range(threads):
                count = items_per_thread + (1 if i < remainder else 0)
                if count > 0:
                    sub_data = dataset[start_idx : start_idx + count]
                    # Pass filters to worker
                    search_args.append((config, mode, sub_data, 0, len(sub_data), filters))
                    start_idx += count
            
            # 3. Execute Search (Time Measured)
            batch_start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=threads) as executor:
                futures = [executor.submit(search_worker, *args) for args in search_args]
                
                for f in as_completed(futures):
                    c, e, t, elapsed = f.result()
                    total_correct += c
                    total_eligible += e
                    total_queries += t
                    total_worker_cpu_time += elapsed
                    
            batch_end_time = time.time()
            
            # Accumulate the wall time spent in this batch's search phase
            total_search_wall_time += (batch_end_time - batch_start_time)
            
            batch_len = len(dataset)
            processed_count += batch_len
            
            if processed_count % 10000 == 0 or processed_count == total_size:
                print(f"Processed {processed_count}/{total_size} queries...")
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Shutting down gracefully...")
    finally:
        if current_csv_f:
            current_csv_f.close()
        if agen:
            agen.close()

    # Calculate Stats
    qps = total_queries / total_search_wall_time if total_search_wall_time > 0 else 0
    recall_rate = total_correct / total_eligible if total_eligible > 0 else 0
    
    # Avg Latency based on total worker execution time (pure search time) / queries
    avg_latency = (total_worker_cpu_time / total_queries) * 1000 if total_queries > 0 else 0
    
    return {
        "mode": mode,
        "seed": seed,
        "total_queries": total_queries,
        "eligible_queries": total_eligible,
        "correct_hits": total_correct,
        "recall_rate": recall_rate,
        "qps": qps,
        "avg_latency_ms": avg_latency,
        "total_search_wall_time_s": total_search_wall_time,
        "model_load_time_s": model_load_time_s
    }

def main():
    parser = argparse.ArgumentParser(description="Run recall test")
    parser.add_argument("-f", "--config", required=True, help="Path to config file")
    parser.add_argument("-m", "--mode", choices=['normal', 'pre', 'post', 'force'], default='normal', help="Search mode")
    parser.add_argument("-t", "--threads", type=int, default=4, help="Number of threads")
    parser.add_argument("-s", "--seed", type=int, default=8888, help="Random seed")
    parser.add_argument("-n", "--number", type=int, help="Number of vectors to search")
    parser.add_argument("-i", "--input", action="append", help="Input CSV file(s). Can be specified multiple times.")
    parser.add_argument("--prefix", help="Input file prefix. Files matching this prefix will be added to the input list.")
    parser.add_argument("--start-id", type=int, default=0, help="Start ID for testing")
    
    # Filter options
    parser.add_argument("--i32v", type=int, help="Filter by i32v value")
    parser.add_argument("--f32v", type=float, help="Filter by f32v value")
    parser.add_argument("--str", type=str, help="Filter by strv value")
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    csv_files = args.input if args.input else []
    if args.prefix:
        directory = os.path.dirname(args.prefix)
        if not directory:
            directory = '.'
        prefix_base = os.path.basename(args.prefix)
        
        try:
            files_in_dir = os.listdir(directory)
            matched_files = [os.path.join(directory, f) for f in files_in_dir if f.startswith(prefix_base)]
            matched_files.sort()
            if matched_files:
                print(f"Found {len(matched_files)} files with prefix '{args.prefix}':")
                for f in matched_files:
                    print(f"  - {f}")
                csv_files.extend(matched_files)
            else:
                print(f"No files found with prefix '{args.prefix}'")
        except FileNotFoundError:
             print(f"Directory not found for prefix: {directory}")

    final_csv_files = csv_files if csv_files else None

    # Collect filters
    filters = {}
    if args.i32v is not None: filters['i32v'] = args.i32v
    if args.f32v is not None: filters['f32v'] = args.f32v
    if args.str is not None: filters['str'] = args.str
        
    stats = run_recall_test(config, args.mode, args.threads, number=args.number, seed=args.seed, filters=filters, csv_files=final_csv_files, start_id=args.start_id)

    print("-" * 40)
    print(f"Mode: {stats['mode']}")
    print(f"Seed: {stats['seed']}")
    print(f"Total Queries: {stats['total_queries']}")
    print(f"Eligible Queries: {stats['eligible_queries']}")
    print(f"Correct Hits: {stats['correct_hits']}")
    print(f"Recall Rate: {stats['recall_rate']:.4f} (Correct / Eligible)")
    print(f"QPS: {stats['qps']:.2f} (Search Time Only)")
    print(f"Avg Latency: {stats['avg_latency_ms']:.4f} ms")
    print(f"Total Search Wall Time: {stats['total_search_wall_time_s']:.4f} s")
    print("-" * 40)

if __name__ == "__main__":
    main()
