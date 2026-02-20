#!/bin/bash
set -e

# --- Script Configuration ---
CONFIG="cfg/ivfflat-sift128.json"
# The final CSV file that will be used for the benchmark.
# It can be the one passed via -i, or the result of a conversion.
TARGET_CSV_FILE="" 

# --- Parse Command-line Arguments ---
usage() {
    echo "Usage: $0 -i <input_file>"
    echo "  <input_file> can be a .fvecs, .csv, or .csv.gz file."
    exit 1
}

while getopts ":i:" opt; do
  case ${opt} in
    i)
      INPUT_FILE=$OPTARG
      ;;
    \?)
      echo "Invalid Option: -$OPTARG" 1>&2
      usage
      ;;
    :)
      echo "Invalid Option: -$OPTARG requires an argument" 1>&2
      usage
      ;;
  esac
done

if [ -z "$INPUT_FILE" ]; then
    echo "Error: Input file must be specified with -i."
    usage
fi

# --- Step 1: Data Preparation ---
echo "=== Step 1: Data Preparation ==="
if [[ "$INPUT_FILE" == *.fvecs ]]; then
    GENERATED_CSV="data/sift_generated_from_fvecs.csv"
    echo "Input is a .fvecs file. Converting '$INPUT_FILE' to '$GENERATED_CSV'..."
    python3 gen.py --fvecs "$INPUT_FILE" -o "$GENERATED_CSV"
    TARGET_CSV_FILE="$GENERATED_CSV"
elif [[ "$INPUT_FILE" == *.csv || "$INPUT_FILE" == *.csv.gz ]]; then
    echo "Input is a CSV file. Using '$INPUT_FILE' directly."
    TARGET_CSV_FILE="$INPUT_FILE"
else
    echo "Error: Unsupported file type '$INPUT_FILE'. Must be .fvecs, .csv, or .csv.gz."
    exit 1
fi

# --- Step 2: Setup Table and IVFflat Index ---
echo "=== Step 2: Setup Table and IVFflat Index ==="
python3 create.py -f $CONFIG -i "$TARGET_CSV_FILE"

# --- Step 3: Recall Tests ---
echo "=== Step 3: Recall Tests ==="
echo "--- Normal Mode ---"
python3 recall.py -f $CONFIG -m normal -n 1000000 -t 8 -i "$TARGET_CSV_FILE"
echo "--- Pre Mode ---"
python3 recall.py -f $CONFIG -m pre -n 1000 -t 8 --i32v 50 -i "$TARGET_CSV_FILE"
echo "--- Post-filtering Mode ---"
python3 recall.py -f $CONFIG -m post -n 1000 -t 8 --i32v 50 -i "$TARGET_CSV_FILE"
echo "--- Force Mode ---"
python3 recall.py -f $CONFIG -m force -n 1000 -t 8 --i32v 50 -i "$TARGET_CSV_FILE"

# --- Step 4: DML Operations (Example) ---
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
