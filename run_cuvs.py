import argparse
import json
import os
import sys
import time
import numpy as np
import pandas as pd
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add path to cuvs python api
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'matrixone', 'cgo', 'cuvs', 'python')))
try:
    import cuvs
except ImportError as e:
    print(f"Error: Could not import cuvs. Make sure it's in matrixone/cgo/cuvs/python/ and the library is built. {e}")
    sys.exit(1)

from gen import Generator

# Mapping matrixone op_type to cuvs DistanceType
dist_mapping = {
    'vector_l2_ops': cuvs.DistanceType.L2Expanded,
    'vector_l2sq_ops': cuvs.DistanceType.L2Unexpanded, 
    'vector_cosine_ops': cuvs.DistanceType.CosineExpanded,
    'vector_ip_ops': cuvs.DistanceType.InnerProduct,
}

# Mapping string quantization types to cuvs.Quantization
qtype_mapping = {
    'fp32': cuvs.Quantization.F32,
    'float32': cuvs.Quantization.F32,
    'fp16': cuvs.Quantization.F16,
    'float16': cuvs.Quantization.F16,
    'half': cuvs.Quantization.F16,
    'int8': cuvs.Quantization.INT8,
    'uint8': cuvs.Quantization.UINT8,
}

def parse_vector(v_str):
    if isinstance(v_str, str):
        try:
            return np.array(ast.literal_eval(v_str), dtype=np.float32)
        except (ValueError, SyntaxError):
            v_str = v_str.strip('[]')
            return np.fromstring(v_str, sep=',', dtype=np.float32)
    return v_str

def cuvs_search_worker(index, query_vectors, query_ids, k, search_params):
    """
    Worker function to run search queries in a thread.
    Following the pattern of recall.py's search_worker.
    """
    start_time = time.monotonic()
    neighbors, distances = index.search(query_vectors, k, search_params=search_params)
    end_time = time.monotonic()
    
    correct = 0
    total = len(query_vectors)
    for i in range(total):
        # Recall@1: Check if the correct ID is in the first result
        # Note: neighbors return might be uint32 or int64 depending on index type
        if query_ids[i] in neighbors[i][:1]: 
            correct += 1
            
    return correct, total, end_time - start_time

def run_cuvs_benchmark(args):
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Override config with CLI args
    dataset_size = args.number if args.number else config.get('dataset_size', 10000)
    dim = config.get('dimension', 1024)
    chunk_size = args.chunk_size
    
    # 1. Setup Index Parameters
    idx_cfg = config.get('index', {})
    
    # Priority: 1. CLI Argument, 2. Config File, 3. Default (cagra)
    if args.index_type:
        idx_type = args.index_type
    elif isinstance(idx_cfg, str):
        idx_type = idx_cfg
    else:
        idx_type = idx_cfg.get('type', 'cagra')
    
    # Priority for qtype: 1. CLI Argument, 2. Config File, 3. Default (fp32)
    qtype_str = args.qtype or config.get('qtype', 'fp32')
    qtype = qtype_mapping.get(qtype_str.lower(), cuvs.Quantization.F32)
    
    # Clean up idx_cfg if it was just a string in config
    if isinstance(idx_cfg, str):
        idx_cfg = {}
    
    dist_type_str = config.get('distance', 'vector_l2_ops')
    if isinstance(config.get('index'), dict):
        dist_type_str = config['index'].get('op_type', dist_type_str)
    
    metric = dist_mapping.get(dist_type_str, cuvs.DistanceType.L2Expanded)
    print(f"Index type: {idx_type}, Metric: {metric.name}, Quantization: {qtype.name}, Dataset size: {dataset_size}")

    # 2. Create Empty Index
    index = None
    if idx_type == 'cagra':
        build_params = cuvs.CagraBuildParams.default()
        if 'graph_degree' in idx_cfg: build_params.graph_degree = idx_cfg['graph_degree']
        if 'intermediate_graph_degree' in idx_cfg: build_params.intermediate_graph_degree = idx_cfg['intermediate_graph_degree']
        index = cuvs.CagraIndex.create_empty(dataset_size, dim, metric=metric, build_params=build_params, qtype=qtype)
    elif idx_type == 'ivfflat':
        build_params = cuvs.IvfFlatBuildParams.default()
        if 'n_lists' in idx_cfg: build_params.n_lists = idx_cfg['n_lists']
        index = cuvs.IvfFlatIndex.create_empty(dataset_size, dim, metric=metric, build_params=build_params, qtype=qtype)
    elif idx_type == 'ivfpq':
        build_params = cuvs.IvfPqBuildParams.default()
        if 'n_lists' in idx_cfg: build_params.n_lists = idx_cfg['n_lists']
        if 'm' in idx_cfg: build_params.m = idx_cfg['m']
        index = cuvs.IvfPqIndex.create_empty(dataset_size, dim, metric=metric, build_params=build_params, qtype=qtype)
    else:
        print(f"Unsupported index type for chunked addition: {idx_type}. Defaulting to CAGRA.")
        index = cuvs.CagraIndex.create_empty(dataset_size, dim, metric=metric, qtype=qtype)

    index.start()

    # 3. Load or Build Index
    if args.input_dir:
        print(f"Loading index from {args.input_dir}...")
        index.load_dir(args.input_dir)
        added_count = dataset_size # Assume it's fully loaded for reporting
    else:
        # Load/Generate Data in Chunks and Add to Index
        csv_files = args.input_csv if args.input_csv else []
        if args.prefix:
            directory = os.path.dirname(args.prefix) or '.'
            prefix_base = os.path.basename(args.prefix)
            try:
                matched_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.startswith(prefix_base)]
                matched_files.sort()
                csv_files.extend(matched_files)
            except FileNotFoundError:
                pass
        
        added_count = 0
        query_vectors = []
        query_ids = []
        max_queries = args.number_queries

        if csv_files:
            print(f"Adding data from CSV files in chunks of {chunk_size}...")
            for f in csv_files:
                if added_count >= dataset_size:
                    break
                for df_chunk in pd.read_csv(f, chunksize=chunk_size):
                    if added_count >= dataset_size:
                        break
                    needed = dataset_size - added_count
                    if len(df_chunk) > needed:
                        df_chunk = df_chunk.head(needed)
                    
                    vecs = np.array([parse_vector(v) for v in df_chunk['vector'].values], dtype=np.float32)
                    
                    # Save some queries for recall test
                    if len(query_vectors) < max_queries:
                        take = min(max_queries - len(query_vectors), len(vecs))
                        query_vectors.append(vecs[:take])
                        query_ids.append(df_chunk['id'].values[:take].astype(np.uint32))

                    index.add_chunk(vecs)
                    added_count += len(vecs)
                    if added_count % 50000 == 0 or added_count == dataset_size:
                        print(f"Added {added_count}/{dataset_size} vectors...")
        else:
            print(f"Generating and adding synthetic data in chunks of {chunk_size}...")
            gen = Generator(config, seed=args.seed)
            while added_count < dataset_size:
                current_batch_size = min(chunk_size, dataset_size - added_count)
                batch = gen.gen_batch(current_batch_size, added_count)
                vecs = np.array([parse_vector(row['vector']) for row in batch], dtype=np.float32)
                
                # Save some queries for recall test
                if len(query_vectors) < max_queries:
                    take = min(max_queries - len(query_vectors), len(vecs))
                    query_vectors.append(vecs[:take])
                    query_ids.append(np.array([row['id'] for row in batch[:take]], dtype=np.uint32))

                index.add_chunk(vecs)
                added_count += len(vecs)
                if added_count % 50000 == 0 or added_count == dataset_size:
                    print(f"Added {added_count}/{dataset_size} vectors...")

        # 4. Build Index
        print("Building index...")
        build_start = time.monotonic()
        index.build()
        build_end = time.monotonic()
        print(f"Index build took {build_end - build_start:.4f} s")
        
        # Save index if requested
        if args.output_dir:
            print(f"Saving index to {args.output_dir}...")
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
            index.save_dir(args.output_dir)

    # 5. Parallel Recall Test using ThreadPoolExecutor
    if args.input_dir:
        # If we loaded the index, we need to generate some queries for recall test
        # as we didn't collect them during addition.
        print(f"Generating {args.number_queries} queries for recall test...")
        gen = Generator(config, seed=args.seed)
        batch = gen.gen_batch(args.number_queries, 0)
        query_vectors = np.array([parse_vector(row['vector']) for row in batch], dtype=np.float32)
        query_ids = np.array([row['id'] for row in batch], dtype=np.uint32)
    else:
        query_vectors = np.vstack(query_vectors)
        query_ids = np.concatenate(query_ids)
    
    n_queries = len(query_vectors)
    k = args.k
    threads = args.threads
    
    search_params = None
    if idx_type == 'cagra':
        search_params = cuvs.CagraSearchParams.default()
        if 'itopk_size' in idx_cfg: search_params.itopk_size = idx_cfg['itopk_size']
        if 'search_width' in idx_cfg: search_params.search_width = idx_cfg['search_width']
    elif idx_type == 'ivfflat':
        search_params = cuvs.IvfFlatSearchParams.default()
        if 'n_probes' in idx_cfg: search_params.n_probes = idx_cfg['n_probes']
    elif idx_type == 'ivfpq':
        search_params = cuvs.IvfPqSearchParams.default()
        if 'n_probes' in idx_cfg: search_params.n_probes = idx_cfg['n_probes']

    print(f"Running parallel search for {n_queries} queries with {threads} threads...")

    # Split queries among threads
    queries_per_thread = n_queries // threads
    remainder = n_queries % threads
    
    search_args = []
    start_idx = 0
    for i in range(threads):
        count = queries_per_thread + (1 if i < remainder else 0)
        if count > 0:
            sub_vectors = query_vectors[start_idx : start_idx + count]
            sub_ids = query_ids[start_idx : start_idx + count]
            search_args.append((index, sub_vectors, sub_ids, k, search_params))
            start_idx += count
    
    # Warm-up (serial)
    index.search(query_vectors[:min(10, n_queries)], k, search_params=search_params)

    total_correct = 0
    total_queries = 0
    total_worker_time = 0
    
    batch_start = time.monotonic()
    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = [executor.submit(cuvs_search_worker, *args) for args in search_args]
        for f in as_completed(futures):
            c, t, elapsed = f.result()
            total_correct += c
            total_queries += t
            total_worker_time += elapsed
    batch_end = time.monotonic()
    
    total_search_wall_time = batch_end - batch_start
    qps = total_queries / total_search_wall_time if total_search_wall_time > 0 else 0
    avg_latency = (total_worker_time / total_queries) * 1000 if total_queries > 0 else 0
    recall_at_1 = total_correct / total_queries if total_queries > 0 else 0

    print("-" * 40)
    print(f"Index Type: {idx_type}")
    print(f"Dataset Size: {added_count}")
    print(f"Threads: {threads}")
    print(f"Queries: {total_queries}")
    print(f"Recall@1: {recall_at_1:.4f}")
    print(f"QPS (Wall time): {qps:.2f}")
    print(f"Avg Latency (Worker time): {avg_latency:.4f} ms")
    print("-" * 40)

def main():
    parser = argparse.ArgumentParser(description="Run cuVS benchmark with chunked addition and parallel search")
    parser.add_argument("-f", "--config", required=True, help="Path to config file")
    parser.add_argument("--index-type", choices=['cagra', 'ivfflat', 'ivfpq'], help="Index type (overrides config)")
    parser.add_argument("--qtype", choices=['fp32', 'fp16', 'int8', 'uint8'], help="Quantization type (overrides config)")
    parser.add_argument("--input-dir", help="Directory to load index from")
    parser.add_argument("--output-dir", help="Directory to save index to")
    parser.add_argument("-n", "--number", type=int, help="Total number of vectors to add")
    parser.add_argument("-nq", "--number-queries", type=int, default=100, help="Number of queries for recall test")
    parser.add_argument("-k", type=int, default=1, help="K for search")
    parser.add_argument("-t", "--threads", type=int, default=1, help="Number of threads for parallel search")
    parser.add_argument("-s", "--seed", type=int, default=8888, help="Random seed")
    parser.add_argument("--input-csv", action="append", help="Input CSV file(s)")
    parser.add_argument("--prefix", help="Input file prefix")
    parser.add_argument("--chunk-size", type=int, default=10000, help="Chunk size for adding data")
    
    args = parser.parse_args()
    
    run_cuvs_benchmark(args)

if __name__ == "__main__":
    main()
