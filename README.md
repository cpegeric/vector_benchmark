# Vector Benchmark Suite

A comprehensive benchmarking toolset for vector databases (optimized for MatrixOne). This suite allows you to measure index creation time, search performance (QPS, Recall, Latency), and DML efficiency (Insert, Update, Delete, Mixed workloads).

## Features

- **Reproducible Data**: Deterministic data generation using a base seed with independent random states for each column (`vector`, `i32v`, `f32v`, `str`).
- **Flexible Indexing**: Supports HNSW and IVFflat with configurable parameters via JSON.
- **High-Performance Loading**: Uses `LOAD DATA INFILE` for rapid CSV ingestion.
- **Advanced Search**: Benchmarks parallel search with support for pre-filtering (dynamic metadata conditions).
- **Mixed Workloads**: Simulates real-world traffic with configurable ratios for Insert, Update, and Delete operations.

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

# Generate extra data starting from a specific ID to avoid duplicates
python3 gen.py -f cfg.json -o extra_data.csv --start-id 10001
```

### 2. Setup Database and Index (`create.py`)
Initialize the database, table, and index. 
```bash
# Synchronous Mode (Insert -> Create Index) - Default
python3 create.py -f cfg.json -i dataset.csv

# Asynchronous Mode (Create Index -> Insert)
python3 create.py -f cfg.json -a -i dataset.csv

# Stream Mode (Generate data on-the-fly instead of using CSV)
python3 create.py -f cfg.json
```

### 3. Search & Recall Benchmark (`recall.py`)
Measure search performance. Data is generated in blocks on-the-fly to minimize memory usage, and generation time is excluded from performance metrics.
```bash
# Standard k-NN Search with 8 threads
python3 recall.py -f cfg.json -m normal -t 8

# Pre-filtering Search (WHERE i32v < 100 AND strv = 'abc')
python3 recall.py -f cfg.json -m prefilter -t 4 --i32v 100 --str "abc"

# Specific number of queries
python3 recall.py -f cfg.json -n 500
```

### 4. DML Benchmarks (`dml.py`)
Consolidated tool for Data Manipulation Language operations.

#### Insert
Generate and append new vectors starting from the current `MAX(id)`.
```bash
python3 dml.py insert -f cfg.json -n 1000
```

#### Update
Randomly select existing rows and update their vector and metadata.
```bash
python3 dml.py update -f cfg.json -n 500
```

#### Delete
Randomly select and remove rows.
```bash
python3 dml.py delete -f cfg.json -n 200
```

#### Append (CSV)
Bulk load data from an existing CSV file.
```bash
python3 dml.py append -f cfg.json -i extra_data.csv
```

#### Mixed Workload
Run a mixture of operations in blocks.
```bash
# 10% Insert, 80% Update, 10% Delete (Total 5000 ops, batch size 500)
python3 dml.py mix -f cfg.json -n 5000 -r 1,8,1 -b 500
```

---

## File Structure

- `gen.py`: Core data generation logic (Generator class + CSV CLI).
- `create.py`: Table/Index lifecycle management and CSV loading.
- `recall.py`: Search benchmark engine (parallel execution + recall math).
- `dml.py`: Combined Insert/Update/Delete/Mix/Append benchmark tool.
- `db.py`: Shared database connection and session environment logic.
- `cfg.json`: Centralized configuration.