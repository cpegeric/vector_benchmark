#!/bin/bash
set -e

CONFIG="cfg/ivfflat-sift128.json"
VECS_FILE="data/sift_base.fvecs"
CSV_FILE="data/sift_base.csv.gz"

echo "=== Step 1: Data Generation ==="
rm -rf $CSV_FILE
python3 gen.py --fvecs $VECS_FILE -o $CSV_FILE

echo "=== Step 2: Setup Table and IVFflat Index ==="
python3 create.py -f $CONFIG -i $CSV_FILE

echo "=== Step 3: Recall Tests ==="
echo "--- Normal Mode ---"
python3 recall.py -f $CONFIG -m normal -n 1000000 -t 4 -i $CSV_FILE
echo "--- Pre-filtering Mode ---"
python3 recall.py -f $CONFIG -m pre -n 1000000 -t 4 --i32v 500 -i $CSV_FILE
echo "--- Post-filtering Mode ---"
python3 recall.py -f $CONFIG -m post -n 1000000 -t 4 --i32v 500 -i $CSV_FILE
echo "--- Force Mode ---"
python3 recall.py -f $CONFIG -m force -n 1000000 -t 4 --i32v 500 -i $CSV_FILE

echo "=== Step 4: DML Operations ==="
echo "--- Insert ---"
python3 dml.py insert -f $CONFIG -n 500
echo "--- Update ---"
python3 dml.py update -f $CONFIG -n 200
echo "--- Delete ---"
python3 dml.py delete -f $CONFIG -n 100
echo "--- Mix ---"
python3 dml.py mix -f $CONFIG -n 1000 -r 1,8,1 -b 200

echo "=== All Tests Completed Successfully ==="
