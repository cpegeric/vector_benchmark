# Vector Benchmark Suite

A comprehensive benchmarking toolset for vector databases (optimized for MatrixOne). This suite allows you to measure index creation time, search performance (QPS, Recall, Latency), and DML efficiency (Insert, Update, Delete, Mixed workloads).

## Features

-   **Reproducible Data**: Deterministic data generation using a base seed with independent random states for each column (`vector`, `i32v`, `f32v`, `str`).
-   **Parallel CSV Generation**: Accelerate dataset creation with the `-p` option for `gen.py`.
-   **Chunked Data Handling**: New `--prefix` option in `gen.py`, `create.py`, `recall.py`, and `run_benchmark.py` for generating and consuming data in multiple chunked files.
-   **Automatic Directory Creation**: `gen.py` now automatically creates output directories if they don't exist.
-   **Improved Recall Accuracy**: Fixed floating-point precision issue in `recall.py` when reading vectors from CSV, ensuring accurate recall rate calculation.
-   **`.fvecs` Support**: Convert vector datasets from the standard `.fvecs` format into the benchmark's CSV format.
-   **Flexible Indexing**: Supports HNSW and IVFflat with configurable parameters via JSON.
-   **High-Performance Loading**: Uses `LOAD DATA INFILE` for rapid CSV ingestion, including support for gzipped CSV files.
-   **Advanced Search**: Benchmarks parallel search with support for pre-filtering (dynamic metadata conditions) and model warm-up time reporting.
-   **Mixed Workloads**: Simulates real-world traffic with configurable ratios for Insert, Update, and Delete operations.
-   **Orchestrated Benchmarking**: A Python-based runner (`run_benchmark.py`) for comprehensive and customizable test suites.

## Prerequisites

- Python 3.x
- `pymysql`
- `numpy`

```bash
pip install pymysql numpy
```

## Configuration (`cfg.json`)

The `cfg.json` file centralizes all database and benchmark settings.

```json
{
  "host": "localhost",
  "port": 6001,
  "user": "root",
  "password": "111",
  "database": "mydb",
  "table": "mytbl",
  "index": {
    "name": "myidx",
    "type": "hnsw",
    "m": 100,
    "ef_construction": 400,
    "ef_search": 200,
    "op_type": "vector_l2_ops"
  },
  "dimension": 1024,
  "dataset_size": 10000,
  "batch_size": 1000,
  "env": {
    "experimental_hnsw_index": 1,
    "probe_limit": 3
  }
}
```

- **env**: Session-level SQL variables executed upon connection (`SET key = value`).
- **index**: Supports `hnsw` and `ivfflat`. For IVF, use `"lists": N` instead of HNSW parameters.

---

## Usage Guide

### 1. Data Generation (`gen.py`)
Generate a reproducible dataset into a CSV file for offline loading.
```bash
# Generate 10k rows (based on cfg.json) to dataset.csv with seed 8888
python3 gen.py -f cfg.json -o dataset.csv -s 8888

# Generate 500 rows, overriding dataset_size from cfg.json
python3 gen.py -f cfg.json -o dataset_small.csv -n 500

# Generate 10k rows in parallel using 4 processes, outputting to a single merged file
python3 gen.py -f cfg.json -o dataset_parallel.csv -s 8888 -p 4

# Generate 10k rows in parallel using 4 processes, outputting to multiple chunked files
# (e.g., my_data0.csv, my_data1.csv, etc. without merging)
python3 gen.py -f cfg.json --prefix my_data -s 8888 -p 4

# Generate extra data starting from a specific ID to avoid duplicates
python3 gen.py -f cfg.json -o extra_data.csv --start-id 10001

# Convert a .fvecs file to CSV, generating id and metadata columns
python3 gen.py --fvecs /path/to/sift1m/sift_base.fvecs -o sift_base.csv
```

### 2. Setup Database and Index (`create.py`)
Initialize the database, table, and index. 
```bash
# Synchronous Mode (Insert -> Create Index) - Default
python3 create.py -f cfg.json -i dataset.csv

# Load multiple CSV files at once
python3 create.py -f cfg.json -i dataset_part1.csv -i dataset_part2.csv

# Load multiple CSV files by specifying a common prefix (e.g., my_data0.csv, my_data1.csv)
python3 create.py -f cfg.json --prefix my_data

# Asynchronous Mode (Create Index -> Insert)
python3 create.py -f cfg.json -a -i dataset.csv

# Stream Mode (Generate data on-the-fly instead of using CSV)
python3 create.py -f cfg.json
```

### 3. Search & Recall Benchmark (`recall.py`)
Measure search performance. Data can be generated in blocks on-the-fly or read from one or more CSV files. Data generation is done asynchronously in the background to minimize blocking and improve performance in multi-threaded tests. Reports `model_load_time_s`.
```bash
# Standard k-NN Search with 8 threads (data generated on-the-fly)
python3 recall.py -f cfg.json -m normal -t 8

# Test recall on the first 1000 vectors from the dataset (generated on-the-fly)
python3 recall.py -f cfg.json -m normal -t 8 -n 1000

# Standard k-NN Search using data from multiple CSV files specified by prefix
python3 recall.py -f cfg.json -m normal -t 8 --prefix my_data

# Pre-filtering Search (WHERE i32v < 100 AND strv = 'abc')
python3 recall.py -f cfg.json -m pre -t 4 --i32v 100 --str "abc"

# Run EXPLAIN ANALYZE on a single query and print the execution plan
python3 recall.py -f cfg.json -m normal --explain
# With filters:
python3 recall.py -f cfg.json -m pre --i32v 100 --str "abc" --explain
```

### 4. DML Benchmarks (`dml.py`)
Consolidated tool for Data Manipulation Language operations.

```bash
# Insert 1000 new vectors
python3 dml.py insert -f cfg.json -n 1000

# Update 500 random vectors
python3 dml.py update -f cfg.json -n 500

# Delete 200 random vectors
python3 dml.py delete -f cfg.json -n 200

# Bulk load data from an existing CSV file
python3 dml.py append -f cfg.json -i extra_data.csv

# Run a mixed workload of operations
# 10% Insert, 80% Update, 10% Delete (Total 5000 ops, batch size 500)
python3 dml.py mix -f cfg.json -n 5000 -r 1,8,1 -b 500
```

### 5. Orchestrated Benchmark Suite (`run_benchmark.py`)

This is the primary script to run comprehensive benchmark suites. It orchestrates data generation, database setup, recall, and DML tests, collecting and presenting all results.

**Note**: `run_benchmark.py` currently only supports **synchronous** index creation. This means that while it can create various index types (like HNSW or IVFflat), they will all be built synchronously. This might not be optimal or desired for certain index types (e.g., HNSW) that often benefit from or are intended for asynchronous building.

```bash
# Example: Run a full suite with IVFflat config, human-readable output
python3 run_benchmark.py -c cfg/ivfflat.json -o human

# Example: Run a suite, skipping create and DML, outputting CSV
python3 run_benchmark.py -c cfg/ivfflat.json \
                         --input-csv data/my_base.csv \
                         --extra-csv data/my_extra.csv \
                         --skip-create --skip-dml -o csv

# Example: Run a suite, skipping create and DML, using pre-existing chunked files by prefix
# (e.g., my_preexisting_chunks0.csv, my_preexisting_chunks1.csv)
python3 run_benchmark.py -c cfg/ivfflat.json \
                         --prefix data/my_preexisting_chunks --extra-csv data/my_extra.csv \
                         --skip-create --skip-dml -o csv

# Example: Generate base data with a prefix and 2 processes, then run the suite
python3 run_benchmark.py -c cfg/hnsw.json \
                         --gen-prefix data/my_gen_chunks --gen-processes 2

# Example: Customize recall tests and enable force mode
python3 run_benchmark.py -c cfg/hnsw.json \
                         -t 8 -s 9999 -n 500 \
                         --i32v 750 --enable-force-recall
```

**Options:**

-   `-c CONFIG`, `--config CONFIG`: Path to configuration file (required).
-   `-o {human,csv}`, `--output {human,csv}`: Output format (default: `human`).
-   `--input-csv INPUT_CSV`: Path to an existing base CSV file(s) to use. Skips generation if `--skip-create` is active. Can be specified multiple times.
-   `--prefix PREFIX`: Input file prefix. Files matching this prefix will be added to the input list (for `create.py`, `recall.py`).
-   `--extra-csv EXTRA_CSV`: Path to an existing extra CSV file for DML tests. Skips generation if `--skip-create` is active and `--skip-append` is false.
-   `--skip-create`: Skip data generation and table creation. Requires `--input-csv` or `--prefix` (and optionally `--extra-csv`) if DML append is run.
-   `--skip-append`: Skip the DML append test.
-   `--skip-recall`: Skip all recall tests.
-   `--skip-dml`: Skip all DML tests.
-   `--gen-prefix GEN_PREFIX`: Prefix for generating chunked data files using `gen.py`. When specified, `gen.py` will create multiple files (e.g., `prefix0.csv`, `prefix1.csv`) if `gen-processes > 1`.
-   `--gen-processes GEN_PROCESSES`: Number of parallel processes for data generation when using `--gen-prefix`.
-   `--enable-force-recall`: Enable recall test with 'force' mode (default is false).
-   `-t THREADS`, `--threads THREADS`: Number of threads for recall tests (default: `4`).
-   `-s SEED`, `--seed SEED`: Random seed for recall tests and generated data (default: `8888`).
-   `-n NUMBER`, `--number NUMBER`: Number of vectors for recall tests (default: `100`). Overrides `config['dataset_size']` for recall.
-   `--i32v I32V`: Filter by `i32v` value for recall tests.
-   `--f32v F32V`: Filter by `f32v` value for recall tests.
-   `--str STR`: Filter by `strv` value for recall tests.
-   `--dml-count DML_COUNT`: Total number of DML operations (insert, update, delete, mix) (default: `1000`).
-   `--dml-batch-size DML_BATCH_SIZE`: Batch size for DML operations (default: `1000`).
-   `--dml-ratios DML_RATIOS`: Mix ratios for DML (Insert,Update,Delete, e.g., `'1,8,1'`) (default: `'1,8,1'`).

---

## File Structure

- `gen.py`: Core data generation logic (Generator class + CSV CLI).
- `create.py`: Table/Index lifecycle management and CSV loading.
- `recall.py`: Search benchmark engine (parallel execution + recall math).
- `dml.py`: Combined Insert/Update/Delete/Mix/Append benchmark tool.
- `db.py`: Shared database connection and session environment logic.
- `cfg.json`: Centralized configuration.
- `run_benchmark.py`: Python-based orchestrator for running full benchmark suites.

## Testing

The primary way to run comprehensive test suites is now via `run_benchmark.py`. The shell scripts (`test_hnsw.sh`, `test_ivfflat.sh`, `test_ivfflat_sift128.sh`) can still be used for quick direct runs of specific configurations, but for full suite testing and detailed reporting, `run_benchmark.py` is recommended.

```bash
# Run a full IVFflat suite using the new Python orchestrator
python3 run_benchmark.py -c cfg/ivfflat.json

# Run an IVFflat suite, skipping creation and DML ops (assumes data is already in DB)
python3 run_benchmark.py -c cfg/ivfflat.json --skip-create --skip-dml \
                         --input-csv data/test_data.csv --extra-csv data/extra_data.csv
```
