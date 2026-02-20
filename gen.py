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
                # Don't produce if queue is full (put blocks, but we check stop_event occasionally)
                # Actually queue.put blocks, so that's fine, but if we want to exit cleanly
                # we might want a timeout.
                data = self.generator.gen_batch(self.chunk_size, self.current_id)
                self.current_id += self.chunk_size
                
                # Put with timeout to allow checking stop_event
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
            # Refill local buffer from queue if empty
            if self.buffer_pos >= len(self.buffer):
                self.buffer = self.queue.get()
                self.buffer_pos = 0
            
            # Take from local buffer
            available = len(self.buffer) - self.buffer_pos
            take = min(needed, available)
            
            result.extend(self.buffer[self.buffer_pos : self.buffer_pos + take])
            self.buffer_pos += take
            needed -= take
            
        return result

    def close(self):
        self.stop_event.set()
        # Drain queue
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break


def generate_csv(config, output_file, seed=DEFAULT_SEED, start_id=0, num_items=None):
    gen = Generator(config, seed=seed)
    batch_size = config.get('batch_size', 1000)
    
    # Determine total_size: prioritize num_items, then config['dataset_size']
    if num_items is not None:
        total_size = num_items
    elif 'dataset_size' in config:
        total_size = config['dataset_size']
    else:
        print("Error: 'dataset_size' not found in config and no --number specified.", file=sys.stderr)
        sys.exit(1)
    
    print(f"Generating {total_size} rows to {output_file} with seed {seed} starting from ID {start_id}...")
    
    # Open the file, gzipped if the extension is .gz
    if output_file.endswith('.gz'):
        csv_fp = gzip.open(output_file, 'wt', newline='')
    else:
        csv_fp = open(output_file, 'w', newline='')

    with csv_fp as csvfile:
        fieldnames = ['id', 'vector', 'i32v', 'f32v', 'str']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
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

def convert_fvecs_to_csv(fvecs_path, output_file, seed=DEFAULT_SEED, start_id=0, batch_size=1000):
    print(f"Reading vectors from {fvecs_path}...")
    vectors = fvecs_read(fvecs_path)
    total_size = len(vectors)
    print(f"Found {total_size} vectors. Converting to CSV at {output_file}...")

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
        csv_fp = gzip.open(output_file, 'wt', newline='')
    else:
        csv_fp = open(output_file, 'w', newline='')

    with csv_fp as csvfile:
        fieldnames = ['id', 'vector', 'i32v', 'f32v', 'str']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
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
    parser.add_argument("-n", "--number", type=int, help="Number of items to generate (overrides dataset_size in config for generate_csv)")
    parser.add_argument("-s", "--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument("--start-id", type=int, default=0, help="Starting ID for the dataset")
    parser.add_argument("--fvecs", help="Path to .fvecs file to convert to CSV")
    
    args = parser.parse_args()
    
    if args.fvecs:
        if not args.output:
            print("Error: --output is required when using --fvecs")
            sys.exit(1)
        convert_fvecs_to_csv(args.fvecs, args.output, args.seed, args.start_id)
        sys.exit(0)

    # The rest of the script requires config
    if not args.config:
        print("Error: --config is required for data generation")
        sys.exit(1)

    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except Exception as e:
        print(f"Error reading config file: {e}")
        sys.exit(1)

    if args.output:
        generate_csv(config, args.output, args.seed, args.start_id, num_items=args.number)
    else:
        # If no output file, just print a sample batch
        gen = Generator(config, seed=args.seed)
        batch = gen.gen_batch(5, args.start_id)
        print(f"Sample batch (5 rows) with seed {args.seed} and start-id {args.start_id}:")
        for row in batch:
            print(row)
