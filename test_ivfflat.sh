#!/bin/bash
set -e

CONFIG="cfg/ivfflat.json"
CSV_FILE="data/test_data.csv"
EXTRA_CSV="data/extra_data.csv"

# --- Parse Options ---
SKIP_CREATE=false
for arg in "$@"; do
  if [ "$arg" == "--skip-create" ]; then
    SKIP_CREATE=true
    echo "Option -skip-create enabled: Skipping data generation and table creation."
    break
  fi
done

# --- Conditional Steps ---
if [ "$SKIP_CREATE" = true ]; then
    echo "Checking for existing data files..."
    if [ ! -f "$CSV_FILE" ] || [ ! -f "$EXTRA_CSV" ]; then
        echo "Error: -skip-create specified, but required data files are missing."
        echo "Please ensure '$CSV_FILE' and '$EXTRA_CSV' exist or run without -skip-create first."
        exit 1
    fi
    echo "Data files found."
else
    echo "=== Step 1: Data Generation ==="
    python3 gen.py -f $CONFIG -o $CSV_FILE -s 1234
    python3 gen.py -f $CONFIG -o $EXTRA_CSV -s 5678 --start-id 1000001 -n 10000

    echo "=== Step 2: Setup Table and IVFflat Index ==="
    python3 create.py -f $CONFIG -i $CSV_FILE
fi

echo "=== Step 3: Recall Tests ==="
python3 recall.py -f $CONFIG -m normal -n 1000 -t 8 -s 1234
echo "--- Pre Mode ---"
python3 recall.py -f $CONFIG -m pre -n 1000 -t 8 --i32v 50 -s 1234
echo "--- Post-filtering Mode ---"
python3 recall.py -f $CONFIG -m post -n 1000 -t 8 --i32v 50 -s 1234
echo "--- Force Mode ---"
python3 recall.py -f $CONFIG -m force -n 1000 -t 8 --i32v 50 -s 1234
echo "--- CSV Input Mode ---"
python3 recall.py -f $CONFIG -m normal -n 1000 -t 8 -i $CSV_FILE --start-id 0

echo "=== Step 4: DML Operations ==="
echo "--- Append ---"
python3 dml.py append -f $CONFIG -i $EXTRA_CSV
echo "--- Extra CSV Input Mode ---"
python3 recall.py -f $CONFIG -m normal -n 100 -t 4 -i $EXTRA_CSV
echo "--- Insert ---"
python3 dml.py insert -f $CONFIG -n 500
echo "--- Update ---"
python3 dml.py update -f $CONFIG -n 200
echo "--- Delete ---"
python3 dml.py delete -f $CONFIG -n 100
echo "--- Mix ---"
python3 dml.py mix -f $CONFIG -n 1000 -r 1,8,1 -b 200

echo "=== All Tests Completed Successfully ==="
