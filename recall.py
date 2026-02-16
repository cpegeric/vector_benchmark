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
    python3 recall.py -f cfg.json -m prefilter --i32v 100 --str "abc"
"""
import sys
import json
import argparse
import pymysql
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from gen import Generator
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
                
                # If no filters provided for prefilter mode, fallback to old behavior (filter by current row's i32v)
                # or strictly follow "replace hardcoded" instruction? 
                # To be safe and useful for recall test (identity match), if NO CLI filters are given 
                # but mode is prefilter, we might want to default to something.
                # However, the instruction says "specify the filter value in option".
                # So we assume if options are used, we use them.
                # If mode is prefilter but no options, we'll just have an empty WHERE (which might be valid or not depending on DB).
                # But let's support the specific request: flexible filter with options.
                
                # Dynamic SQL generation
                current_where = where_clause
                
                # Compatibility: If mode is prefilter but no static filters provided, 
                # previously we used row['i32v']. 
                # If we want to strictly follow "specify value in option", we rely on 'current_where'.
                # But if current_where is empty in prefilter mode, it's just a normal search with 'mode=pre' hint.
                
                if mode == 'normal':
                    sql = f"SELECT id FROM {tbl} {current_where} ORDER BY {op_type}(embed, '{query_vec}') LIMIT 1"
                elif mode == 'prefilter':
                    sql = f"SELECT id FROM {tbl} {current_where} ORDER BY {op_type}(embed, '{query_vec}') LIMIT 1 BY RANK WITH OPTION 'mode=pre'"
                
                cursor.execute(sql)
                res = cursor.fetchone()
                
                if res and res[0] == target_id:
                    correct += 1
                total += 1
                
    finally:
        conn.close()
        
    end_time = time.time()
    return correct, total, end_time - start_time

def run_recall_test(config, mode, threads, number=None, seed=8888, filters=None):
    total_size = number if number is not None else config['dataset_size']
    # Use a large chunk size to minimize connection overhead per batch
    # But ensuring it's not too huge for memory
    chunk_size = 10000 
    
    print(f"Starting recall test: {total_size} queries, mode={mode}, threads={threads}, seed={seed}")
    if filters:
        print(f"Filters: {filters}")
    print(f"Processing in chunks of {chunk_size}...")

    gen = Generator(config, seed=seed)
    
    total_correct = 0
    total_queries = 0
    total_search_wall_time = 0
    total_worker_cpu_time = 0
    
    processed = 0
    while processed < total_size:
        current_batch_size = min(chunk_size, total_size - processed)
        
        # 1. Generate Data (Time ignored)
        dataset = gen.gen_batch(current_batch_size, processed)
        
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
                c, t, elapsed = f.result()
                total_correct += c
                total_queries += t
                total_worker_cpu_time += elapsed
                
        batch_end_time = time.time()
        
        # Accumulate the wall time spent in this batch's search phase
        total_search_wall_time += (batch_end_time - batch_start_time)
        
        processed += current_batch_size
        if processed % 10000 == 0 or processed == total_size:
            print(f"Processed {processed}/{total_size} queries...")

    # Calculate Stats
    qps = total_queries / total_search_wall_time if total_search_wall_time > 0 else 0
    recall = total_correct / total_queries if total_queries > 0 else 0
    
    # Avg Latency based on total worker execution time (pure search time) / queries
    avg_latency = (total_worker_cpu_time / total_queries) * 1000 if total_queries > 0 else 0
    
    print("-" * 40)
    print(f"Mode: {mode}")
    print(f"Seed: {seed}")
    print(f"Total Queries: {total_queries}")
    print(f"Recall: {recall:.4f}")
    print(f"QPS: {qps:.2f} (Search Time Only)")
    print(f"Avg Latency: {avg_latency:.4f} ms")
    print(f"Total Search Wall Time: {total_search_wall_time:.4f} s")
    print("-" * 40)

def main():
    parser = argparse.ArgumentParser(description="Run recall test")
    parser.add_argument("-f", "--config", required=True, help="Path to config file")
    parser.add_argument("-m", "--mode", choices=['normal', 'prefilter'], default='normal', help="Search mode")
    parser.add_argument("-t", "--threads", type=int, default=4, help="Number of threads")
    parser.add_argument("-s", "--seed", type=int, default=8888, help="Random seed")
    parser.add_argument("-n", "--number", type=int, help="Number of vectors to search")
    
    # Filter options
    parser.add_argument("--i32v", type=int, help="Filter by i32v value")
    parser.add_argument("--f32v", type=float, help="Filter by f32v value")
    parser.add_argument("--str", type=str, help="Filter by strv value")
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Collect filters
    filters = {}
    if args.i32v is not None: filters['i32v'] = args.i32v
    if args.f32v is not None: filters['f32v'] = args.f32v
    if args.str is not None: filters['str'] = args.str
        
    run_recall_test(config, args.mode, args.threads, number=args.number, seed=args.seed, filters=filters)

if __name__ == "__main__":
    main()
