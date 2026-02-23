import argparse
import json
import os
import csv
import sys
from gen import generate_csv
from create import setup_database
from recall import run_recall_test
from dml import run_insert, run_update, run_delete, run_append_csv, run_mix

def _extra_csv_step(config, extra_csv_in, skip_create, run_dml_append_test, seed, temp_files):
    """
    Handles generation/validation of extra data for DML append tests.
    """
    extra_csv_path = extra_csv_in

    if skip_create:
        if run_dml_append_test:
            if not extra_csv_path:
                 print("Error: --skip-create and DML append test requires --extra-csv to be specified.", file=sys.stderr)
                 sys.exit(1)
            if not os.path.exists(extra_csv_path):
                print(f"Error: --skip-create specified, but extra CSV file '{extra_csv_path}' is missing.", file=sys.stderr)
                sys.exit(1)
            print(f"Using pre-existing extra CSV: {extra_csv_path}")
    else: # Not skipping creation
        if run_dml_append_test:
            if not extra_csv_path:
                temp_extra_csv_file = f"data/temp_{config.get('table', 'default')}_extra.csv"
                dataset_size = config.get('dataset_size')
                if dataset_size is None:
                    print("Warning: 'dataset_size' not in config, using 10000 for extra data start_id.", file=sys.stderr)
                    dataset_size = 10000
                extra_data_start_id = dataset_size + 1
                generate_csv(config, temp_extra_csv_file, seed=seed + 5678, start_id=extra_data_start_id, num_items=1000)
                extra_csv_path = temp_extra_csv_file
                temp_files.append(extra_csv_path)
                print(f"Extra data generated: {extra_csv_path}")
            else:
                print(f"Using provided extra CSV: {extra_csv_path}")
        else:
            print("Extra CSV generation skipped as DML append test will be skipped (--skip-append was used).")
            extra_csv_path = None
    
    return extra_csv_path

def _create_step(config, input_csvs, skip_create, seed, temp_files, gen_prefix=None, gen_processes=1):
    """
    Handles data generation for base dataset and database setup.
    Returns base_csv_paths used for subsequent steps.
    """
    base_csv_paths = input_csvs
    
    if skip_create:
        print("\n--- Skipping Data Generation and Table Setup ---")
        if not base_csv_paths and not gen_prefix:
            print("Error: --skip-create requires at least one --input-csv or --prefix to be specified.", file=sys.stderr)
            sys.exit(1)
        
        if base_csv_paths:
            for csv_file in base_csv_paths:
                if not os.path.exists(csv_file):
                    print(f"Error: --skip-create specified, but base CSV file '{csv_file}' is missing.", file=sys.stderr)
                    sys.exit(1)
            print(f"Using pre-existing base CSV files: {base_csv_paths}")

    else: # Not skipping creation (generate and setup)
        # === Step 1: Data Generation ===
        print("\n--- Step 1: Data Generation ---")
        if gen_prefix:
            print(f"Generating data with prefix '{gen_prefix}' using {gen_processes} processes.")
            base_csv_paths = generate_csv(config, output_prefix=gen_prefix, seed=seed, num_processes=gen_processes)
            temp_files.extend(base_csv_paths)
            print(f"Base data generated in {len(base_csv_paths)} files with prefix '{gen_prefix}'")
        elif not base_csv_paths:
            temp_base_csv_file = f"data/temp_{config.get('table', 'default')}_base.csv"
            base_csv_paths = generate_csv(config, temp_base_csv_file, seed=seed)
            temp_files.extend(base_csv_paths)
            print(f"Base data generated: {base_csv_paths[0]}")
        else:
            print(f"Using provided base CSV files: {base_csv_paths}")
        
        # === Step 2: Setup Table and Index ===
        print("\n--- Step 2: Setup Table and Index ---")
        setup_database(config, csv_files=base_csv_paths)
    
    return base_csv_paths

def _recall_step(config, threads, seed, number, recall_filters, all_stats, enable_force_recall=False, query_csv_files=None):
    """
    Handles recall tests.
    """
    print("\n--- Step 3: Recall Tests ---")
    recall_modes = ['normal', 'pre', 'post', 'force']
    for mode in recall_modes:
        if mode == 'force' and not enable_force_recall:
            print(f"Skipping recall test for mode '{mode}' (not enabled).")
            continue
        stats = run_recall_test(config, mode=mode, threads=threads, number=number, seed=seed, filters=recall_filters, csv_files=query_csv_files)
        stats['test_name'] = f"recall_{mode}"
        all_stats.append(stats)

def _dml_step(config, extra_csv_path, run_dml_append_test, dml_count, dml_batch_size, dml_ratios, all_stats):
    """
    Handles DML operations.
    """
    print("\n--- Step 4: DML Operations ---")
    
    # Append
    if run_dml_append_test:
        append_stats = run_append_csv(config, extra_csv_path) 
        append_stats['test_name'] = 'dml_append'
        all_stats.append(append_stats)
    else:
        print("Skipping DML append test (--skip-append was used).")
    
    # Insert
    insert_stats = run_insert(config, count=dml_count, batch_size=dml_batch_size)
    insert_stats['test_name'] = 'dml_insert'
    all_stats.append(insert_stats)

    # Update
    update_stats = run_update(config, count=dml_count // 2, batch_size=dml_batch_size) # Using half dml_count for update
    update_stats['test_name'] = 'dml_update'
    all_stats.append(update_stats)

    # Delete
    delete_stats = run_delete(config, count=dml_count // 4, batch_size=dml_batch_size) # Using quarter dml_count for delete
    delete_stats['test_name'] = 'dml_delete'
    all_stats.append(delete_stats)

    # Mix
    mix_stats = run_mix(config, total_ops=dml_count, ratios=dml_ratios, batch_size=dml_batch_size)
    mix_stats['test_name'] = 'dml_mix'
    all_stats.append(mix_stats)

def run_suite(config_path, output_format='human', input_csvs=None, extra_csv_in=None, skip_create=False, skip_append=False, skip_recall=False, skip_dml=False, gen_prefix=None, gen_processes=1, enable_force_recall=False, threads=4, seed=8888, number=100, filter_i32v=None, filter_f32v=None, filter_str=None, dml_count=1000, dml_batch_size=1000, dml_ratios="1,8,1"):
    """
    Orchestrates the full benchmark suite execution.
    """
    all_stats = []

    print(f"=== Running Benchmark Suite for {config_path} ===")
    
    with open(config_path, 'r') as f:
        config = json.load(f)

    # --- Setup and Data Generation ---
    base_csv_paths = []
    extra_csv_path = None
    temp_files = []

    skip_dml_append = skip_append or skip_dml
    
    base_csv_paths = _create_step(config, input_csvs, skip_create, seed, temp_files, gen_prefix, gen_processes)
    extra_csv_path = _extra_csv_step(config, extra_csv_in, skip_create, not skip_dml_append, seed, temp_files)
    
    # --- Recall Tests ---
    if not skip_recall:
        recall_filters = {}
        if filter_i32v is not None: recall_filters['i32v'] = filter_i32v
        if filter_f32v is not None: recall_filters['f32v'] = filter_f32v
        if filter_str is not None: recall_filters['str'] = filter_str
        _recall_step(config, threads, seed, number, recall_filters, all_stats, enable_force_recall=enable_force_recall, query_csv_files=base_csv_paths)
    else:
        print("\n--- Skipping Recall Tests ---")

    # --- DML Operations ---
    if not skip_dml:
        _dml_step(config, extra_csv_path, not skip_append, dml_count, dml_batch_size, dml_ratios, all_stats)
    else:
        print("\n--- Skipping DML Operations ---")

    # === Final Results ===
    print("\n\n--- Benchmark Results ---")
    if output_format == 'csv':
        if all_stats:
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
    parser.add_argument("--input-csv", action="append", help="Path to existing base CSV file(s) to use (skips generation if --skip-create is used). Can be specified multiple times.")
    parser.add_argument("--prefix", help="Input file prefix. Files matching this prefix will be added to the input list.")
    parser.add_argument("--extra-csv", help="Path to an existing extra CSV file for DML tests (skips generation if --skip-create is used and --skip-append is False).")
    parser.add_argument("--skip-create", action="store_true", help="Skip data generation and table creation. Requires --input-csv and --extra-csv if DML append is run.")
    parser.add_argument("--skip-append", action="store_true", help="Skip the DML append test.")
    parser.add_argument("--skip-recall", action="store_true", help="Skip all recall tests.")
    parser.add_argument("--skip-dml", action="store_true", help="Skip all DML tests.")
    parser.add_argument("--gen-prefix", help="Prefix for generating chunked data files.")
    parser.add_argument("--gen-processes", type=int, default=1, help="Number of processes for data generation.")
    parser.add_argument("-t", "--threads", type=int, default=4, help="Number of threads for recall tests")
    parser.add_argument("-s", "--seed", type=int, default=8888, help="Random seed for recall tests and generated data.")
    parser.add_argument("-n", "--number", type=int, default=100, help="Number of vectors for recall tests (overrides config's dataset_size for recall).")
    parser.add_argument("--i32v", type=int, help="Filter by i32v value for recall tests.")
    parser.add_argument("--f32v", type=float, help="Filter by f32v value for recall tests.")
    parser.add_argument("--str", type=str, help="Filter by strv value for recall tests.")
    parser.add_argument("--dml-count", type=int, default=1000, help="Total number of DML operations (insert, update, delete, mix).")
    parser.add_argument("--dml-batch-size", type=int, default=1000, help="Batch size for DML operations.")
    parser.add_argument("--dml-ratios", type=str, default="1,8,1", help="Mix ratios for DML (Insert,Update,Delete, e.g., '1,8,1').")
    parser.add_argument("--skip-force-recall", action="store_true", help="Skip recall test with 'force' mode.")
    parser.add_argument("--enable-force-recall", action="store_true", help="Enable recall test with 'force' mode (default is false).")
    args = parser.parse_args()

    csv_files = args.input_csv if args.input_csv else []
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

    run_suite(config_path=args.config, 
              output_format=args.output, 
              input_csvs=final_csv_files, 
              extra_csv_in=args.extra_csv, 
              skip_create=args.skip_create, 
              skip_append=args.skip_append, 
              skip_recall=args.skip_recall,
              skip_dml=args.skip_dml,
              gen_prefix=args.gen_prefix,
              gen_processes=args.gen_processes,
              enable_force_recall=args.enable_force_recall,
              threads=args.threads, 
              seed=args.seed, 
              number=args.number, 
              filter_i32v=args.i32v, 
              filter_f32v=args.f32v, 
              filter_str=args.str,
              dml_count=args.dml_count,
              dml_batch_size=args.dml_batch_size,
              dml_ratios=args.dml_ratios)

if __name__ == "__main__":
    main()