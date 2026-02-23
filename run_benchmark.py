import argparse
import json
import os
import csv
import sys
from gen import generate_csv
from create import setup_database
from recall import run_recall_test
from dml import run_insert, run_update, run_delete, run_append_csv, run_mix

def run_suite(config_path, output_format='human'):
    """
    Runs a full benchmark suite based on a config file.
    This function will orchestrate the steps from the .sh scripts.
    """
    all_stats = []

    print(f"=== Running Benchmark Suite for {config_path} ===")
    
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Define file paths based on config for clarity
    base_csv = f"data/temp_{config.get('table', 'default')}_base.csv"
    extra_csv = f"data/temp_{config.get('table', 'default')}_extra.csv"
    
    # === Step 1: Data Generation ===
    print("\n--- Step 1: Data Generation ---")
    generate_csv(config, base_csv, seed=1234)
    # Generate extra data for DML append tests
    dataset_size = config.get('dataset_size')
    if dataset_size is None:
        print("Warning: 'dataset_size' not in config, using 10000 for extra data start_id.", file=sys.stderr)
        dataset_size = 10000
    extra_data_start_id = dataset_size + 1
    generate_csv(config, extra_csv, seed=5678, start_id=extra_data_start_id, num_items=1000)
    print("Data generation complete.")

    # === Step 2: Setup Table and Index ===
    print("\n--- Step 2: Setup Table and Index ---")
    setup_database(config, csv_files=[base_csv])
    
    # === Step 3: Recall Tests ===
    print("\n--- Step 3: Recall Tests ---")
    recall_modes = ['normal', 'pre', 'post', 'force']
    for mode in recall_modes:
        # Using a smaller number for testing to speed up the suite
        stats = run_recall_test(config, mode=mode, threads=4, number=100, seed=1234, filters={'i32v': 500})
        stats['test_name'] = f"recall_{mode}"
        all_stats.append(stats)

    # === Step 4: DML Operations ===
    print("\n--- Step 4: DML Operations ---")
    
    # Append
    append_stats = run_append_csv(config, extra_csv)
    append_stats['test_name'] = 'dml_append'
    all_stats.append(append_stats)
    
    # Insert
    insert_stats = run_insert(config, count=500)
    insert_stats['test_name'] = 'dml_insert'
    all_stats.append(insert_stats)

    # Update
    update_stats = run_update(config, count=200)
    update_stats['test_name'] = 'dml_update'
    all_stats.append(update_stats)

    # Delete
    delete_stats = run_delete(config, count=100)
    delete_stats['test_name'] = 'dml_delete'
    all_stats.append(delete_stats)

    # Mix
    mix_stats = run_mix(config, total_ops=1000, ratios="1,8,1")
    mix_stats['test_name'] = 'dml_mix'
    all_stats.append(mix_stats)

    # === Final Results ===
    print("\n\n--- Benchmark Results ---")
    if output_format == 'csv':
        if all_stats:
            # Get a comprehensive list of all possible keys
            fieldnames = sorted(list(set(key for stats in all_stats for key in stats.keys())))
            writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_stats)
    else: # human-readable
        for stats in all_stats:
            print(f"\n--- Test: {stats.get('test_name', 'N/A')} ---")
            for key, value in stats.items():
                if key == 'test_name': continue
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")

def main():
    parser = argparse.ArgumentParser(description="Run benchmark suites in Python.")
    parser.add_argument("-c", "--config", required=True, help="Path to config file for the suite (e.g., cfg/hnsw.json)")
    parser.add_argument("-o", "--output", choices=['human', 'csv'], default='human', help="Output format")
    args = parser.parse_args()

    run_suite(args.config, args.output)

if __name__ == "__main__":
    main()
