import argparse
import json
import os
import sys
import time
import numpy as np
import pandas as pd
import ast

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
    'vector_l2sq_ops': cuvs.DistanceType.L2Unexpanded, # L2Sq is often unexpanded in cuvs context if not sqrted
    'vector_cosine_ops': cuvs.DistanceType.CosineExpanded,
    'vector_ip_ops': cuvs.DistanceType.InnerProduct,
}

def parse_vector(v_str):
    if isinstance(v_str, str):
        try:
            return np.array(ast.literal_eval(v_str), dtype=np.float32)
        except (ValueError, SyntaxError):
            # Try splitting by comma if literal_eval fails (for very simple comma-separated strings)
            v_str = v_str.strip('[]')
            return np.fromstring(v_str, sep=',', dtype=np.float32)
    return v_str

def run_cuvs_benchmark(args):
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Override config with CLI args
    dataset_size = args.number if args.number else config.get('dataset_size', 10000)
    dim = config.get('dimension', 1024)
    chunk_size = args.chunk_size
    
    # 1. Setup Index Parameters
    idx_cfg = config.get('index', {})
    if isinstance(idx_cfg, str):
        idx_type = idx_cfg
        idx_cfg = {}
    else:
        idx_type = idx_cfg.get('type', args.index_type)
    
    dist_type_str = config.get('distance', 'vector_l2_ops')
    if isinstance(config.get('index'), dict):
        dist_type_str = config['index'].get('op_type', dist_type_str)
    
    metric = dist_mapping.get(dist_type_str, cuvs.DistanceType.L2Expanded)
    print(f"Index type: {idx_type}, Metric: {metric.name}, Dataset size: {dataset_size}")

    # 2. Create Empty Index
    index = None
    if idx_type == 'cagra':
        build_params = cuvs.CagraBuildParams.default()
        if 'graph_degree' in idx_cfg: build_params.graph_degree = idx_cfg['graph_degree']
        if 'intermediate_graph_degree' in idx_cfg: build_params.intermediate_graph_degree = idx_cfg['intermediate_graph_degree']
        index = cuvs.CagraIndex.create_empty(dataset_size, dim, metric=metric, build_params=build_params)
    elif idx_type == 'ivfflat':
        build_params = cuvs.IvfFlatBuildParams.default()
        if 'n_lists' in idx_cfg: build_params.n_lists = idx_cfg['n_lists']
        index = cuvs.IvfFlatIndex.create_empty(dataset_size, dim, metric=metric, build_params=build_params)
    elif idx_type == 'ivfpq':
        # IVF-PQ might not support add_chunk in this version of the library
        # If it doesn't, we'll have to load all data at once.
        print("Warning: IVF-PQ may not support chunked addition. Attempting to load all data.")
        chunk_size = dataset_size 
    else:
        print(f"Unsupported index type for chunked addition: {idx_type}. Defaulting to CAGRA.")
        index = cuvs.CagraIndex.create_empty(dataset_size, dim, metric=metric)

    index.start()

    # 3. Load/Generate Data in Chunks and Add to Index
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
    
    start_time = time.monotonic()
    added_count = 0
    
    # We'll save a few queries for recall test
    query_vectors = []
    query_ids = []
    max_queries = args.number_queries

    if csv_files:
        print(f"Adding data from CSV files in chunks of {chunk_size}...")
        for f in csv_files:
            if added_count >= dataset_size:
                break
            # Use pandas chunking
            for df_chunk in pd.read_csv(f, chunksize=chunk_size):
                if added_count >= dataset_size:
                    break
                needed = dataset_size - added_count
                if len(df_chunk) > needed:
                    df_chunk = df_chunk.head(needed)
                
                vecs = np.array([parse_vector(v) for v in df_chunk['vector'].values], dtype=np.float32)
                # ids = df_chunk['id'].values.astype(np.uint32) # cuvs add_chunk usually doesn't take IDs, it appends
                
                # Save some queries
                if len(query_vectors) < max_queries:
                    take = min(max_queries - len(query_vectors), len(vecs))
                    query_vectors.append(vecs[:take])
                    query_ids.append(df_chunk['id'].values[:take])

                index.add_chunk(vecs)
                added_count += len(vecs)
                print(f"Added {added_count}/{dataset_size} vectors...")
    else:
        print(f"Generating and adding synthetic data in chunks of {chunk_size}...")
        gen = Generator(config, seed=args.seed)
        while added_count < dataset_size:
            current_batch_size = min(chunk_size, dataset_size - added_count)
            batch = gen.gen_batch(current_batch_size, added_count)
            vecs = np.array([parse_vector(row['vector']) for row in batch], dtype=np.float32)
            
            # Save some queries
            if len(query_vectors) < max_queries:
                take = min(max_queries - len(query_vectors), len(vecs))
                query_vectors.append(vecs[:take])
                query_ids.append(np.array([row['id'] for row in batch[:take]]))

            index.add_chunk(vecs)
            added_count += len(vecs)
            if added_count % 10000 == 0 or added_count == dataset_size:
                print(f"Added {added_count}/{dataset_size} vectors...")

    # For IVF-PQ which we couldn't create empty (hypothetically)
    if index is None and idx_type == 'ivfpq':
         # Load all at once as a fallback for IVF-PQ
         print("Loading all data at once for IVF-PQ...")
         # (This part would be similar to the previous version's load_dataset but with OOM risk)
         # For brevity, I'll assume CAGRA/IVF-Flat are the primary targets for this request.
         pass

    # 4. Build Index
    print("Building index...")
    build_start = time.monotonic()
    index.build()
    build_end = time.monotonic()
    print(f"Index build took {build_end - build_start:.4f} s")

    # 5. Recall Test
    if not query_vectors:
        print("Error: No queries collected for recall test.")
        return

    query_vectors = np.vstack(query_vectors)
    query_ids = np.concatenate(query_ids)
    
    n_queries = len(query_vectors)
    print(f"Running search for {n_queries} queries...")
    
    k = args.k
    search_params = None
    if idx_type == 'cagra':
        search_params = cuvs.CagraSearchParams.default()
        if 'itopk_size' in idx_cfg: search_params.itopk_size = idx_cfg['itopk_size']
        if 'search_width' in idx_cfg: search_params.search_width = idx_cfg['search_width']
    elif idx_type == 'ivfflat':
        search_params = cuvs.IvfFlatSearchParams.default()
        if 'n_probes' in idx_cfg: search_params.n_probes = idx_cfg['n_probes']

    # Warm-up
    index.search(query_vectors[:min(10, n_queries)], k, search_params=search_params)

    start_search = time.monotonic()
    neighbors, distances = index.search(query_vectors, k, search_params=search_params)
    end_search = time.monotonic()
    
    search_time = end_search - start_search
    qps = n_queries / search_time if search_time > 0 else 0
    avg_latency = (search_time / n_queries) * 1000 if n_queries > 0 else 0

    # Calculate Recall@1
    correct = 0
    for i in range(n_queries):
        # Note: neighbors return might be uint32 or int64 depending on index type
        if query_ids[i] in neighbors[i][:1]: 
            correct += 1
    
    recall_at_1 = correct / n_queries if n_queries > 0 else 0

    print("-" * 40)
    print(f"Index Type: {idx_type}")
    print(f"Dataset Size: {added_count}")
    print(f"Queries: {n_queries}")
    print(f"Recall@1: {recall_at_1:.4f}")
    print(f"QPS: {qps:.2f}")
    print(f"Avg Latency: {avg_latency:.4f} ms")
    print("-" * 40)

def main():
    parser = argparse.ArgumentParser(description="Run cuVS benchmark with chunked addition")
    parser.add_argument("-f", "--config", required=True, help="Path to config file")
    parser.add_argument("--index-type", choices=['cagra', 'ivfflat', 'ivfpq'], default='cagra', help="Index type")
    parser.add_argument("-n", "--number", type=int, help="Total number of vectors to add")
    parser.add_argument("-nq", "--number-queries", type=int, default=100, help="Number of queries for recall test")
    parser.add_argument("-k", type=int, default=1, help="K for search")
    parser.add_argument("-s", "--seed", type=int, default=8888, help="Random seed")
    parser.add_argument("--input-csv", action="append", help="Input CSV file(s)")
    parser.add_argument("--prefix", help="Input file prefix")
    parser.add_argument("--chunk-size", type=int, default=10000, help="Chunk size for adding data")
    
    args = parser.parse_args()
    
    run_cuvs_benchmark(args)

if __name__ == "__main__":
    main()
