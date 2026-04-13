"""
Vector Dataset Generator

This script generates synthetic vector datasets for benchmarking purposes.
It supports generating data with the following schema:
- id: Unique integer identifier
- vector: High-dimensional float vector (normalized)
- i32v: Random integer metadata
- f32v: Random float metadata
- str: Random string metadata

Features:
- **Reproducibility**: Uses a seed to ensure deterministic data generation.
- **Independence**: Each column has its own random state, allowing for consistent generation even if column logic changes.
- **Modes**:
    - **Library**: `Generator` class can be imported to generate data batches on-the-fly.
    - **CLI**: Can generate a CSV file for offline loading.

Usage:
    python3 gen.py -f cfg.json -o dataset.csv -s 8888
"""
import sys
import json
import argparse
import numpy as np
import math
import csv
import string
import random
import threading
import queue
import gzip
import multiprocessing
import os

# Seed for reproducibility
DEFAULT_SEED = 8888

def normalize(array):
    norm = np.linalg.norm(array)
    if norm == 0:
        return array
    return array / norm

class Generator:
    def __init__(self, config, seed=None):
        self.dim = config['dimension']
        self.nitem = config['dataset_size']
        
        actual_seed = DEFAULT_SEED if seed is None else seed
        
        # Initialize a list of RandomState for each column to ensure independence
        # 0: vector, 1: i32v, 2: f32v, 3: str
        self.rss = [
            np.random.RandomState(actual_seed),     # vector
            np.random.RandomState(actual_seed + 1), # i32v
            np.random.RandomState(actual_seed + 2), # f32v
            np.random.RandomState(actual_seed + 3)  # str
        ]
        
        self.str_choices = list(string.ascii_letters + string.digits)

    def gen_batch(self, size, start_id):
        # Use specific random state for each column
        vectors = self.rss[0].rand(size, self.dim).astype(np.float32)
        # Normalize vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / norms
        
        i32vs = self.rss[1].randint(0, 1000, size, dtype=np.int32)
        f32vs = self.rss[2].rand(size).astype(np.float32)
        
        batch_data = []
        for i in range(size):
            vec_str = '[' + ','.join(str(x) for x in vectors[i]) + ']'
            # Generate random string
            s_len = self.rss[3].randint(5, 10)
            s_val = ''.join(self.rss[3].choice(self.str_choices, s_len))
            
            row = {
                'id': start_id + i,
                'vector': vec_str,
                'i32v': int(i32vs[i]),
                'f32v': float(f32vs[i]),
                'str': s_val
            }
            batch_data.append(row)
        return batch_data

class AsyncGenerator:
    def __init__(self, generator, start_id=0, chunk_size=10000, buffer_size=4):
        self.generator = generator
        self.chunk_size = chunk_size
        self.current_id = start_id
        self.queue = queue.Queue(maxsize=buffer_size)
        self.stop_event = threading.Event()
        
        self.buffer = []
        self.buffer_pos = 0
        
        self.thread = threading.Thread(target=self._producer_loop, daemon=True)
        self.thread.start()
        
    def _producer_loop(self):
        while not self.stop_event.is_set():
            try:
                data = self.generator.gen_batch(self.chunk_size, self.current_id)
                self.current_id += self.chunk_size
                
                while not self.stop_event.is_set():
                    try:
                        self.queue.put(data, timeout=1)
                        break
                    except queue.Full:
                        continue
            except Exception as e:
                print(f"Async generation error: {e}")
                break

    def get_batch(self, size):
        result = []
        needed = size
        
        while needed > 0:
            if self.buffer_pos >= len(self.buffer):
                self.buffer = self.queue.get()
                self.buffer_pos = 0
            
            available = len(self.buffer) - self.buffer_pos
            take = min(needed, available)
            
            result.extend(self.buffer[self.buffer_pos : self.buffer_pos + take])
            self.buffer_pos += take
            needed -= take
            
        return result

    def close(self):
        self.stop_event.set()
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break

def _generate_csv_chunk(args_tuple):
    config, output_file, seed, start_id, num_items_in_chunk, is_first_chunk, compress_level = args_tuple
    gen = Generator(config, seed=seed)
    batch_size = config.get('batch_size', 1000)
    
    # Always open as a plain CSV file for temporary chunks
    csv_fp = open(output_file, 'w', newline='')
    
    with csv_fp as csvfile:
        fieldnames = ['id', 'vector', 'i32v', 'f32v', 'str']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, restval='')
        
        if is_first_chunk: # Only write header for the first chunk
            writer.writeheader()
        
        count = 0
        while count < num_items_in_chunk:
            current_batch = min(batch_size, num_items_in_chunk - count)
            data = gen.gen_batch(current_batch, start_id + count)
            writer.writerows(data)
            count += current_batch
            
    return output_file

def generate_csv(config, output_file=None, output_prefix=None, seed=DEFAULT_SEED, start_id=0, num_items=None, num_processes=1, compress_level=6):
    batch_size = config.get('batch_size', 1000)
    
    if num_items is not None:
        total_size = num_items
    elif 'dataset_size' in config:
        total_size = config['dataset_size']
    else:
        print("Error: 'dataset_size' not found in config and no --number specified.", file=sys.stderr)
        raise ValueError("Missing 'dataset_size' in config and no --number specified.") # Print and then re-raise


    # Determine the base directory for output and create it if it doesn't exist
    target_dir = None
    if output_file:
        target_dir = os.path.dirname(output_file)
    elif output_prefix:
        target_dir = os.path.dirname(output_prefix)
    
    if target_dir and not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
        print(f"Created output directory: {target_dir}")

    op_target = output_prefix if output_prefix else output_file
    print(f"Generating {total_size} rows to '{op_target}' with seed {seed} starting from ID {start_id} using {num_processes} processes...")

    multi_file_no_merge = num_processes > 1 and output_prefix is not None

    if num_processes > 1:
        if multi_file_no_merge:
            output_filenames = []
        else:
            temp_dir = os.path.dirname(output_file)
            if not temp_dir:
                temp_dir = '.'
            temp_prefix = os.path.join(temp_dir, f"temp_gen_{os.getpid()}_")
            temp_csv_files = []

        items_per_process = total_size // num_processes
        remainder = total_size % num_processes

        pool_args = []
        current_global_id = start_id
        for i in range(num_processes):
            chunk_size = items_per_process + (1 if i < remainder else 0)
            if chunk_size == 0:
                continue
            
            if multi_file_no_merge:
                chunk_output_file = f"{output_prefix}{i}.csv"
                output_filenames.append(chunk_output_file)
                process_output_file = chunk_output_file
            else:
                # Always generate plain CSV temp files
                temp_csv_file = f"{temp_prefix}{i}.csv" 
                temp_csv_files.append(temp_csv_file)
                process_output_file = temp_csv_file
            
            pool_args.append((config, process_output_file, seed + i * 100, current_global_id, chunk_size, (i == 0) or multi_file_no_merge, compress_level))
            current_global_id += chunk_size
        
        with multiprocessing.Pool(processes=num_processes) as pool:
            generated_files = pool.map(_generate_csv_chunk, pool_args)

        if multi_file_no_merge:
            print("Finished generating chunk files. Merging is skipped as per --prefix option.")
            for fname in generated_files:
                print(f"  - Generated: {fname}")
            print(f"Finished generating {total_size} rows.")
            return generated_files
        else:
            # Concatenate generated temp files into the final output file
            if output_file.endswith('.gz'):
                with gzip.open(output_file, 'wt', newline='', compresslevel=compress_level) as outfile:
                    for idx, fname in enumerate(generated_files):
                        with open(fname, 'rt', newline='') as infile:
                            for line_idx, line in enumerate(infile):
                                if idx > 0 and line_idx == 0:
                                    continue
                                outfile.write(line)
            else:
                with open(output_file, 'w', newline='') as outfile:
                    for idx, fname in enumerate(generated_files):
                        with open(fname, 'r', newline='') as infile:
                            for line_idx, line in enumerate(infile):
                                if idx > 0 and line_idx == 0:
                                    continue
                                outfile.write(line)
            
            for fname in temp_csv_files:
                os.remove(fname)
                print(f"Cleaned up temporary file: {fname}")
            print(f"Finished generating {total_size} rows.")
            return [output_file]

    else: # Sequential generation (existing logic)
        # When using prefix with single process, just append .csv
        if output_prefix:
            output_file = f"{output_prefix}.csv"

        gen = Generator(config, seed=seed)
        
        if output_file.endswith('.gz'):
            csv_fp = gzip.open(output_file, 'wt', newline='', compresslevel=compress_level)
        else:
            csv_fp = open(output_file, 'w', newline='')

        with csv_fp as csvfile:
            fieldnames = ['id', 'vector', 'i32v', 'f32v', 'str']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, restval='')
            writer.writeheader()
            
            count = 0
            while count < total_size:
                current_batch = min(batch_size, total_size - count)
                data = gen.gen_batch(current_batch, start_id + count)
                writer.writerows(data)
                count += current_batch
                if count % 10000 == 0:
                    print(f"{count} rows written...")
                    
        print(f"Finished generating {total_size} rows.")
        return [output_file]

def fbin_read(filename):
    """Read a binary vector file in the cuvs benchmark format.

    Supported suffixes and their dtypes:
      .fbin   -> float32
      .f16bin -> float16
      .ibin   -> int32
      .u8bin  -> uint8
      .i8bin  -> int8

    Format: first 8 bytes are num_vectors (uint32) and num_dimensions (uint32),
    followed by num_vectors * num_dimensions * sizeof(type) bytes in row-major order.
    """
    suffix_to_dtype = {
        '.fbin':   np.float32,
        '.f16bin': np.float16,
        '.ibin':   np.int32,
        '.u8bin':  np.uint8,
        '.i8bin':  np.int8,
    }
    # Match the suffix
    fname_lower = filename.lower()
    dtype = None
    for suffix, dt in suffix_to_dtype.items():
        if fname_lower.endswith(suffix):
            dtype = dt
            break
    if dtype is None:
        raise ValueError(f"Unrecognized fbin file suffix for '{filename}'. "
                         f"Expected one of: {list(suffix_to_dtype.keys())}")

    with open(filename, 'rb') as f:
        header = np.frombuffer(f.read(8), dtype=np.uint32)
        num_vectors, num_dimensions = int(header[0]), int(header[1])
        data = np.frombuffer(f.read(), dtype=dtype)

    assert data.size == num_vectors * num_dimensions, (
        f"Expected {num_vectors * num_dimensions} elements, got {data.size}"
    )
    return data.reshape(num_vectors, num_dimensions)


def convert_fbin_to_csv(fbin_path, output_file, seed=DEFAULT_SEED, start_id=0, batch_size=1000, compress_level=6):
    print(f"Reading vectors from {fbin_path}...")
    vectors = fbin_read(fbin_path)
    # Cast to float32 for consistent CSV output
    vectors = vectors.astype(np.float32)
    total_size = len(vectors)
    print(f"Found {total_size} vectors with dimension {vectors.shape[1]}. Converting to CSV at {output_file}...")

    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")

    rss = [
        None,
        np.random.RandomState(seed + 1),  # i32v
        np.random.RandomState(seed + 2),  # f32v
        np.random.RandomState(seed + 3),  # str
    ]
    str_choices = list(string.ascii_letters + string.digits)

    if output_file.endswith('.gz'):
        csv_fp = gzip.open(output_file, 'wt', newline='', compresslevel=compress_level)
    else:
        csv_fp = open(output_file, 'w', newline='')

    with csv_fp as csvfile:
        fieldnames = ['id', 'vector', 'i32v', 'f32v', 'str']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, restval='')
        writer.writeheader()

        for i in range(0, total_size, batch_size):
            batch_end = min(i + batch_size, total_size)
            batch_vectors = vectors[i:batch_end]
            current_batch_size = len(batch_vectors)

            i32vs = rss[1].randint(0, 1000, current_batch_size, dtype=np.int32)
            f32vs = rss[2].rand(current_batch_size).astype(np.float32)

            rows = []
            for j in range(current_batch_size):
                vec_str = '[' + ','.join(map(str, batch_vectors[j])) + ']'
                s_len = rss[3].randint(5, 10)
                s_val = ''.join(rss[3].choice(str_choices, s_len))
                rows.append({
                    'id': start_id + i + j,
                    'vector': vec_str,
                    'i32v': int(i32vs[j]),
                    'f32v': float(f32vs[j]),
                    'str': s_val,
                })

            writer.writerows(rows)
            if (i + current_batch_size) % 10000 == 0 or (i + current_batch_size) == total_size:
                print(f"{i + current_batch_size} rows written...")

    print(f"Finished converting {total_size} vectors to CSV.")


def fvecs_read(filename, c_contiguous=True):
    fv = np.fromfile(filename, dtype=np.float32)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    assert dim > 0, "Dimension must be greater than 0"
    fv = fv.reshape(-1, 1 + dim)
    if not c_contiguous:
        fv = fv.copy()
    fv = fv[:, 1:]
    return fv

def convert_fvecs_to_csv(fvecs_path, output_file, seed=DEFAULT_SEED, start_id=0, batch_size=1000, compress_level=6):
    print(f"Reading vectors from {fvecs_path}...")
    vectors = fvecs_read(fvecs_path)
    total_size = len(vectors)
    print(f"Found {total_size} vectors. Converting to CSV at {output_file}...")

    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")

    # Initialize random states for other columns
    rss = [
        None,  # vector (already loaded)
        np.random.RandomState(seed + 1),  # i32v
        np.random.RandomState(seed + 2),  # f32v
        np.random.RandomState(seed + 3)   # str
    ]
    str_choices = list(string.ascii_letters + string.digits)

    # Open the file, gzipped if the extension is .gz
    if output_file.endswith('.gz'):
        csv_fp = gzip.open(output_file, 'wt', newline='', compresslevel=compress_level)
    else:
        csv_fp = open(output_file, 'w', newline='')

    with csv_fp as csvfile:
        fieldnames = ['id', 'vector', 'i32v', 'f32v', 'str']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, restval='')
        writer.writeheader()

        for i in range(0, total_size, batch_size):
            batch_end = min(i + batch_size, total_size)
            batch_vectors = vectors[i:batch_end]
            current_batch_size = len(batch_vectors)
            
            # Generate other columns for the batch
            i32vs = rss[1].randint(0, 1000, current_batch_size, dtype=np.int32)
            f32vs = rss[2].rand(current_batch_size).astype(np.float32)
            
            rows = []
            for j in range(current_batch_size):
                vec_str = '[' + ','.join(map(str, batch_vectors[j])) + ']'
                s_len = rss[3].randint(5, 10)
                s_val = ''.join(rss[3].choice(str_choices, s_len))
                
                rows.append({
                    'id': start_id + i + j,
                    'vector': vec_str,
                    'i32v': int(i32vs[j]),
                    'f32v': float(f32vs[j]),
                    'str': s_val
                })
            
            writer.writerows(rows)
            if (i + current_batch_size) % 10000 == 0 or (i + current_batch_size) == total_size:
                print(f"{i + current_batch_size} rows written...")
                
    print(f"Finished converting {total_size} vectors to CSV.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate vector dataset")
    parser.add_argument("-f", "--config", help="Path to configuration file")
    parser.add_argument("-o", "--output", help="Output CSV file path")
    parser.add_argument("--prefix", help="Output file prefix. When used with -p > 1, files are not merged.")
    parser.add_argument("-n", "--number", type=int, help="Number of items to generate (overrides dataset_size in config for generate_csv)")
    parser.add_argument("-p", "--processes", type=int, default=1, help="Number of parallel processes for CSV generation")
    parser.add_argument("-cl", "--compress-level", type=int, default=6, choices=range(1, 10), metavar="[1-9]", help="Gzip compression level (1-9, 1=fastest, 9=best compression)")
    parser.add_argument("-s", "--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument("--start-id", type=int, default=0, help="Starting ID for the dataset")
    parser.add_argument("--fvecs", help="Path to .fvecs file to convert to CSV")
    parser.add_argument("--fbin", help="Path to binary vector file (.fbin/.f16bin/.ibin/.u8bin/.i8bin) to convert to CSV")
    
    args = parser.parse_args()

    if args.output and args.prefix:
        print("Error: --output and --prefix are mutually exclusive.", file=sys.stderr)
        sys.exit(1)
    
    if args.fbin:
        if args.prefix:
            print("Error: --prefix is not supported with --fbin conversion.", file=sys.stderr)
            raise ValueError("--prefix is not supported with --fbin conversion.")
        if not args.output:
            print("Error: --output is required when using --fbin", file=sys.stderr)
            raise ValueError("--output is required when using --fbin")
        convert_fbin_to_csv(args.fbin, args.output, args.seed, args.start_id, compress_level=args.compress_level)
        sys.exit(0)

    if args.fvecs:
        if args.prefix:
            print("Error: --prefix is not supported with --fvecs conversion.", file=sys.stderr)
            raise ValueError("--prefix is not supported with --fvecs conversion.")
        if not args.output:
            print("Error: --output is required when using --fvecs", file=sys.stderr)
            raise ValueError("--output is required when using --fvecs")
        convert_fvecs_to_csv(args.fvecs, args.output, args.seed, args.start_id, compress_level=args.compress_level)
        sys.exit(0)

    # The rest of the script requires config
    if not args.config:
        print("Error: --config is required for data generation", file=sys.stderr)
        raise ValueError("--config is required for data generation")

    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except Exception as e:
        print(f"Error reading config file: {e}", file=sys.stderr)
        raise # Re-raise the exception

    if args.output or args.prefix:
        generate_csv(config, output_file=args.output, output_prefix=args.prefix, seed=args.seed, start_id=args.start_id, num_items=args.number, num_processes=args.processes, compress_level=args.compress_level)
    else:
        # If no output file, just print a sample batch
        gen = Generator(config, seed=args.seed)
        batch = gen.gen_batch(5, args.start_id)
        print(f"Sample batch (5 rows) with seed {args.seed} and start-id {args.start_id}:")
        for row in batch:
            print(row)
