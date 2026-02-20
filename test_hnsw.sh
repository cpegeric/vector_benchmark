#!/bin/bash
set -e

CONFIG="cfg/hnsw.json"
CSV_FILE="data/test_data_hnsw.csv"
EXTRA_CSV="data/extra_data_hnsw.csv"

echo "=== Step 1: Data Generation ==="
python3 gen.py -f $CONFIG -o $CSV_FILE -s 1234
python3 gen.py -f $CONFIG -o $EXTRA_CSV -s 5678 --start-id 10001

echo "=== Step 2: Setup Table and HNSW Index ==="
python3 create.py -f $CONFIG -i $CSV_FILE -i $EXTRA_CSV

# HNSW is async so sleep to make sure index updated before recall
#echo "--- Append ---"
#python3 dml.py append -f $CONFIG -i $EXTRA_CSV
#sleep 30

echo "=== Step 3: Recall Tests ==="
echo "--- Normal Mode ---"
python3 recall.py -f $CONFIG -m normal -n 100 -t 4 -s 1234
echo "--- Pre-filtering Mode ---"
python3 recall.py -f $CONFIG -m prefilter -n 100 -t 4 --i32v 500 -s 1234
echo "--- Post-filtering Mode ---"
python3 recall.py -f $CONFIG -m post -n 100 -t 4 --i32v 500 -s 1234
echo "--- CSV Input Mode ---"
python3 recall.py -f $CONFIG -m normal -n 100 -t 4 -i $CSV_FILE --start-id 0
echo "--- Extra CSV Input Mode ---"
python3 recall.py -f $CONFIG -m normal -n 100 -t 4 -i $EXTRA_CSV

echo "=== Step 4: DML Operations ==="
echo "--- Insert ---"
python3 dml.py insert -f $CONFIG -n 500
echo "--- Update ---"
python3 dml.py update -f $CONFIG -n 200
echo "--- Delete ---"
python3 dml.py delete -f $CONFIG -n 100
echo "--- Mix ---"
python3 dml.py mix -f $CONFIG -n 1000 -r 1,8,1 -b 200

echo "=== HNSW Tests Completed Successfully ==="
