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

def generate_csv(config, output_file, seed=DEFAULT_SEED, start_id=0):
    gen = Generator(config, seed=seed)
    batch_size = config.get('batch_size', 1000)
    total_size = config['dataset_size']
    
    print(f"Generating {total_size} rows to {output_file} with seed {seed} starting from ID {start_id}...")
    
    with open(output_file, 'w', newline='') as csvfile:
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate vector dataset")
    parser.add_argument("-f", "--config", required=True, help="Path to configuration file")
    parser.add_argument("-o", "--output", help="Output CSV file path")
    parser.add_argument("-s", "--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument("--start-id", type=int, default=0, help="Starting ID for the dataset")
    
    args = parser.parse_args()
    
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except Exception as e:
        print(f"Error reading config file: {e}")
        sys.exit(1)

    if args.output:
        generate_csv(config, args.output, args.seed, args.start_id)
    else:
        # If no output file, just print a sample batch
        gen = Generator(config, seed=args.seed)
        batch = gen.gen_batch(5, args.start_id)
        print(f"Sample batch (5 rows) with seed {args.seed} and start-id {args.start_id}:")
        for row in batch:
            print(row)
