"""
Vector Search Benchmark (Ground Truth Recall & QPS)

This script measures the performance of vector search operations against pre-computed ground truth:
1. **GT Recall**: Intersection between retrieved results and true nearest neighbors.
2. **QPS**: Queries Per Second.
3. **Latency**: Average query execution time.

Features:
- **Parallel Execution**: Uses `ThreadPoolExecutor` for concurrent queries.
- **Binary Data Support**: Reads query vectors (.fbin) and ground truth (.ibin) files.
- **Recall@k**: Calculates recall based on top-k intersection.

Usage:
    python3 gtrecall.py -f cfg/hnsw.json -q wiki_all_1M/queries.fbin -g wiki_all_1M/groundtruth.1M.neighbors.ibin -k 1 -t 8
"""
import sys
import json
import argparse
import time
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from db import get_db_connection, set_env

optype2distfn = {
    'vector_l2_ops': 'l2_distance',
    'vector_cosine_ops': 'cosine_distance',
    'vector_ip_ops': 'inner_product',
    'vector_l2sq_ops': 'l2_distance_sq'
}

def read_fbin(filename):
    with open(filename, 'rb') as f:
        n = np.fromfile(f, dtype=np.int32, count=1)[0]
        d = np.fromfile(f, dtype=np.int32, count=1)[0]
        data = np.fromfile(f, dtype=np.float32)
        return data.reshape(n, d)

def read_ibin(filename):
    with open(filename, 'rb') as f:
        n = np.fromfile(f, dtype=np.int32, count=1)[0]
        d = np.fromfile(f, dtype=np.int32, count=1)[0]
        data = np.fromfile(f, dtype=np.int32)
        return data.reshape(n, d)

def construct_query(config, mode, query_vec, k, filters):
    tbl = config['table']
    
    index_cfg = config.get('index', {})
    if isinstance(index_cfg, str):
        dist = config.get('distance', 'vector_l2_ops')
    else:
        dist = index_cfg.get('op_type', config.get('distance', 'vector_l2_ops'))
    
    op_type = optype2distfn.get(dist, 'l2_distance')

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

    # MatrixOne vector format: [1.2, 3.4, ...]
    query_vec_str = '[' + ','.join(map(str, query_vec)) + ']'

    if mode == 'normal':
        sql = f"SELECT id FROM {tbl} {where_clause} ORDER BY {op_type}(embed, '{query_vec_str}') LIMIT {k}"
    else:
        sql = f"SELECT id FROM {tbl} {where_clause} ORDER BY {op_type}(embed, '{query_vec_str}') LIMIT {k} BY RANK WITH OPTION 'mode={mode}'"
    
    return sql, op_type

def search_worker(config, mode, query_vectors, gt_neighbors, k, start, end, filters=None):
    conn = get_db_connection(config)
    
    correct = 0
    total_queries = 0
    worker_search_time = 0
    
    try:
        with conn.cursor() as cursor:
            set_env(cursor, config)
            
            for i in range(start, end):
                query_vec = query_vectors[i]
                # Ground truth for this query (top neighbors)
                gt = gt_neighbors[i]
                
                sql, _ = construct_query(config, mode, query_vec, k, filters)
                
                q_start = time.monotonic()
                cursor.execute(sql)
                res = cursor.fetchall()
                q_end = time.monotonic()
                
                worker_search_time += (q_end - q_start)
                retrieved_ids = [row[0] for row in res]
                
                # Compare with top-k ground truth
                # Use standard Recall@k: overlap between top-k retrieved and top-k GT
                actual_k = min(k, len(gt))
                target_gt = gt[:actual_k]
                
                if k == 1:
                    if retrieved_ids and target_gt.size > 0 and retrieved_ids[0] == target_gt[0]:
                        correct += 1
                else:
                    intersection = set(retrieved_ids) & set(target_gt)
                    correct += len(intersection)
                
                total_queries += 1
                
    finally:
        conn.close()
        
    return correct, total_queries, worker_search_time

def run_gt_recall_test(config, mode, threads, query_file, gt_file, k=1, number=None, filters=None):
    print(f"Loading queries from {query_file}...")
    query_vectors = read_fbin(query_file)
    print(f"Loading ground truth from {gt_file}...")
    gt_neighbors = read_ibin(gt_file)
    
    if number is not None:
        query_vectors = query_vectors[:number]
        gt_neighbors = gt_neighbors[:number]
        
    total_size = len(query_vectors)
    print(f"Starting GT recall test: queries={total_size}, mode={mode}, threads={threads}, k={k}")

    # Warm-up (serial)
    search_worker(config, mode, query_vectors[:1], gt_neighbors[:1], k, 0, 1, filters)

    items_per_thread = total_size // threads
    remainder = total_size % threads
    
    start_idx = 0
    search_args = []
    for i in range(threads):
        count = items_per_thread + (1 if i < remainder else 0)
        if count > 0:
            search_args.append((config, mode, query_vectors, gt_neighbors, k, start_idx, start_idx + count, filters))
            start_idx += count
            
    total_correct = 0
    total_queries_executed = 0
    total_worker_search_time = 0
    
    batch_start_time = time.monotonic()
    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = [executor.submit(search_worker, *args) for args in search_args]
        for f in as_completed(futures):
            c, t, elapsed = f.result()
            total_correct += c
            total_queries_executed += t
            total_worker_search_time += elapsed
    batch_end_time = time.monotonic()
    
    total_search_wall_time = batch_end_time - batch_start_time
    
    qps = total_queries_executed / total_search_wall_time if total_search_wall_time > 0 else 0
    # Recall = sum of intersections / (total_queries * k)
    recall_rate = total_correct / (total_queries_executed * k) if total_queries_executed > 0 else 0
    avg_latency = (total_worker_search_time / total_queries_executed) * 1000 if total_queries_executed > 0 else 0
    
    return {
        "mode": mode,
        "total_queries": total_queries_executed,
        "correct_hits": total_correct,
        "recall_rate": recall_rate,
        "qps": qps,
        "avg_latency_ms": avg_latency,
        "total_search_wall_time_s": total_search_wall_time,
        "k": k
    }

def main():
    parser = argparse.ArgumentParser(description="Run Ground Truth recall test using binary files")
    parser.add_argument("-f", "--config", required=True, help="Path to config file")
    parser.add_argument("-m", "--mode", choices=['normal', 'pre', 'post', 'force'], default='normal', help="Search mode")
    parser.add_argument("-t", "--threads", type=int, default=4, help="Number of threads")
    parser.add_argument("-q", "--queries", required=True, help="Path to queries.fbin")
    parser.add_argument("-g", "--gt", required=True, help="Path to groundtruth.ibin")
    parser.add_argument("-k", type=int, default=1, help="Recall@k (checks intersection of top-k)")
    parser.add_argument("-n", "--number", type=int, help="Number of queries to run")
    
    # Filter options (if metadata exists in DB)
    parser.add_argument("--i32v", type=int, help="Filter by i32v value")
    parser.add_argument("--f32v", type=float, help="Filter by f32v value")
    parser.add_argument("--str", type=str, help="Filter by strv value")

    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
        
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    filters = {}
    if args.i32v is not None: filters['i32v'] = args.i32v
    if args.f32v is not None: filters['f32v'] = args.f32v
    if args.str is not None: filters['str'] = args.str

    try:
        stats = run_gt_recall_test(
            config, 
            args.mode, 
            args.threads, 
            args.queries, 
            args.gt, 
            k=args.k, 
            number=args.number, 
            filters=filters
        )

        print("-" * 40)
        print(f"Mode: {stats['mode']}")
        print(f"Recall@{stats['k']}: {stats['recall_rate']:.4f}")
        print(f"Total Queries: {stats['total_queries']}")
        print(f"Total Overlap (Correct Hits): {stats['correct_hits']}")
        print(f"QPS: {stats['qps']:.2f} (Wall time)")
        print(f"Avg Latency: {stats['avg_latency_ms']:.4f} ms (Worker time)")
        print(f"Total Search Wall Time: {stats['total_search_wall_time_s']:.4f} s")
        print("-" * 40)
    except Exception as e:
        print(f"Error running recall test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
