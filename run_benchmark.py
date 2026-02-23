import argparse
import json
import os
import csv
import sys
from gen import generate_csv
from create import setup_database
from recall import run_recall_test
from dml import run_insert, run_update, run_delete, run_append_csv, run_mix

def run_suite(config_path, output_format='human', input_csv=None, extra_csv_in=None, skip_create=False, skip_append=False, threads=4, seed=8888, number=None, filter_i32v=None, filter_f32v=None, filter_str=None):
    """
    Runs a full benchmark suite based on a config file.
    This function will orchestrate the steps from the .sh scripts.
    """
    all_stats = []

    print(f"=== Running Benchmark Suite for {config_path} ===")
    
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Determine which base and extra CSV files to *potentially* use/generate
    base_csv_path = input_csv
    extra_csv_path = extra_csv_in # This will hold the path to the extra CSV, if used
    
    temp_files = [] # For cleanup

    # Flag to control DML append test: set to True unless explicitly skipped
    run_dml_append_test = not skip_append 

    # --- Conditional Skip Create Logic ---
    if skip_create:
        print("\n--- Skipping Data Generation and Table Setup ---")
        if not base_csv_path:
            print("Error: --skip-create requires --input-csv to be specified.", file=sys.stderr)
            sys.exit(1)
        if not os.path.exists(base_csv_path):
            print(f"Error: --skip-create specified, but base CSV file '{base_csv_path}' is missing.", file=sys.stderr)
            sys.exit(1)
        
        # If DML append test is enabled AND skip_create is true, then extra_csv_in MUST be provided and exist
        if run_dml_append_test:
            if not extra_csv_path: # This means extra_csv_in was not provided
                 print("Error: --skip-create and DML append test requires --extra-csv to be specified.", file=sys.stderr)
                 sys.exit(1)
            if not os.path.exists(extra_csv_path):
                print(f"Error: --skip-create specified, but extra CSV file '{extra_csv_path}' is missing.", file=sys.stderr)
                sys.exit(1)
        
        print(f"Using pre-existing base CSV: {base_csv_path}")
        if run_dml_append_test:
            print(f"Using pre-existing extra CSV: {extra_csv_path}")

    else: # Not skipping creation (generate and setup)
        # === Step 1: Data Generation ===
        print("\n--- Step 1: Data Generation ---")
        # Generate base CSV if not provided
        if not base_csv_path:
            temp_base_csv_file = f"data/temp_{config.get('table', 'default')}_base.csv"
            generate_csv(config, temp_base_csv_file, seed=seed) # Use the suite's seed here
            base_csv_path = temp_base_csv_file
            temp_files.append(base_csv_path)
            print(f"Base data generated: {base_csv_path}")
        else:
            print(f"Using provided base CSV: {base_csv_path}")

        # Generate extra CSV only if DML append test is enabled
        if run_dml_append_test:
            if not extra_csv_path: # If extra_csv_in was not provided, generate it
                temp_extra_csv_file = f"data/temp_{config.get('table', 'default')}_extra.csv"
                dataset_size = config.get('dataset_size')
                if dataset_size is None:
                    print("Warning: 'dataset_size' not in config, using 10000 for extra data start_id.", file=sys.stderr)
                    dataset_size = 10000
                extra_data_start_id = dataset_size + 1
                generate_csv(config, temp_extra_csv_file, seed=seed + 5678, start_id=extra_data_start_id, num_items=1000) # seed + 5678
                extra_csv_path = temp_extra_csv_file
                temp_files.append(extra_csv_path)
                print(f"Extra data generated: {extra_csv_path}")
            else: # extra_csv_in was provided, use it
                print(f"Using provided extra CSV: {extra_csv_path}")
        else: # DML append test is skipped, so no need for extra_csv
            print("Extra CSV generation skipped as DML append test will be skipped (--skip-append was used).")
            extra_csv_path = None # Ensure it's explicitly None if not used
        
        # === Step 2: Setup Table and Index ===
        print("\n--- Step 2: Setup Table and Index ---")
        setup_database(config, csv_files=[base_csv_path])
    
    # === Step 3: Recall Tests ===
    print("\n--- Step 3: Recall Tests ---")
    
    # Collect filters for recall tests
    recall_filters = {}
    if filter_i32v is not None: recall_filters['i32v'] = filter_i32v
    if filter_f32v is not None: recall_filters['f32v'] = filter_f32v
    if filter_str is not None: recall_filters['str'] = filter_str

    recall_modes = ['normal', 'pre', 'post', 'force']
    for mode in recall_modes:
        stats = run_recall_test(config, mode=mode, threads=threads, number=number, seed=seed, filters=recall_filters)
        stats['test_name'] = f"recall_{mode}" # Corrected f-string here
        all_stats.append(stats)

    # === Step 4: DML Operations ===
    print("\n--- Step 4: DML Operations ---")
    
    # Append
    if run_dml_append_test:
        append_stats = run_append_csv(config, extra_csv_path) 
        append_stats['test_name'] = 'dml_append'
        all_stats.append(append_stats)
    else:
        print("Skipping DML append test (--skip-append was used).")
    
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
    
    # --- Cleanup ---
    for f_path in temp_files:
        if os.path.exists(f_path):
            os.remove(f_path)
            print(f"Cleaned up temporary file: {f_path}")

def main():
    parser = argparse.ArgumentParser(description="Run benchmark suites in Python.")
    parser.add_argument("-c", "--config", required=True, help="Path to config file for the suite (e.g., cfg/hnsw.json)")
    parser.add_argument("-o", "--output", choices=['human', 'csv'], default='human', help="Output format")
    parser.add_argument("--input-csv", help="Path to an existing base CSV file to use (skips generation if --skip-create is used).")
    parser.add_argument("--extra-csv", help="Path to an existing extra CSV file for DML tests (skips generation if --skip-create is used and --skip-append is False).")
    parser.add_argument("--skip-create", action="store_true", help="Skip data generation and table creation. Requires --input-csv and --extra-csv if DML append is run.")
    parser.add_argument("--skip-append", action="store_true", help="Skip the DML append test.")
    parser.add_argument("-t", "--threads", type=int, default=4, help="Number of threads for recall tests")
    parser.add_argument("-s", "--seed", type=int, default=8888, help="Random seed for recall tests")
    parser.add_argument("-n", "--number", type=int, help="Number of vectors for recall tests (overrides config's dataset_size for recall)")
    parser.add_argument("--i32v", type=int, help="Filter by i32v value for recall tests")
    parser.add_argument("--f32v", type=float, help="Filter by f32v value for recall tests")
    parser.add_argument("--str", type=str, help="Filter by strv value for recall tests")
    args = parser.parse_args()

    run_suite(args.config, args.output, input_csv=args.input_csv, extra_csv_in=args.extra_csv, skip_create=args.skip_create, skip_append=args.skip_append, threads=args.threads, seed=args.seed, number=args.number, filter_i32v=args.i32v, filter_f32v=args.f32v, filter_str=args.str)

if __name__ == "__main__":
    main()