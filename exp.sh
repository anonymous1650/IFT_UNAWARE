#!/bin/bash

# =================================================================
# IFT_UNAWARE Replication Script
# This script trains classifiers, generates test pairs, and summarizes results.
# =================================================================

# Exit immediately if a command exits with a non-zero status
set -e

# Python command (use python3 or python depending on your environment)
PYTHON="python3"

# --- Configuration ---

# 1. Classifiers to use
# The paper focuses on DNN (Section IV), so we default to "dnn".
# To test other models, uncomment the line below:
# MODELS=("dnn" "rf" "svm" "lr")
MODELS=("dnn")

# 2. Dataset and Sensitive Attribute pairs
# Based on Table II in the paper (Adult, Bank, German)
DATA_PAIRS=(
    "adult gender"
    "adult race"
    "adult age"
    "bank age"
    "german gender"
    "german age"
)

# 3. Testing parameters
NUM_TEST=50000
THRESHOLD=0.5

# --- Setup Directories ---
mkdir -p models/main
mkdir -p models/proxy
mkdir -p results/proxy
mkdir -p exp_individual_results_e1

echo "================================================================="
echo "STEP 1: Training Classifiers (CuT and Proxy)"
echo "================================================================="

for pair in "${DATA_PAIRS[@]}"; do
    # Split the pair string into dataset and attribute
    read -r DATASET ATTRIBUTE <<< "$pair"

    for MODEL in "${MODELS[@]}"; do
        # --- Train Main Classifier ---
        MAIN_PKL="models/main/main_${MODEL}_${DATASET}_${ATTRIBUTE}.pkl"
        if [ -f "$MAIN_PKL" ]; then
            echo "[Skip] Main Model ($MODEL) for $DATASET-$ATTRIBUTE already exists."
        else
            echo "[Train] Main Model ($MODEL) for $DATASET-$ATTRIBUTE..."
            $PYTHON train_classifier.py -cm "$MODEL" -d "$DATASET" -s "$ATTRIBUTE"
        fi

        # --- Train Proxy Classifier ---
        PROXY_PKL="models/proxy/proxy_${MODEL}_${DATASET}_${ATTRIBUTE}.pkl"
        if [ -f "$PROXY_PKL" ]; then
            echo "[Skip] Proxy Model ($MODEL) for $DATASET-$ATTRIBUTE already exists."
        else
            echo "[Train] Proxy Model ($MODEL) for $DATASET-$ATTRIBUTE..."
            $PYTHON train_classifier.py -cp "$MODEL" -d "$DATASET" -s "$ATTRIBUTE" --train-proxy
        fi
    done
done

echo ""
echo "================================================================="
echo "STEP 2: Running Fairness Testing"
echo "================================================================="

COUNT=1
TOTAL_STEPS=$((${#MODELS[@]} * ${#MODELS[@]} * ${#DATA_PAIRS[@]}))

for cm in "${MODELS[@]}"; do
    for cp in "${MODELS[@]}"; do
        for pair in "${DATA_PAIRS[@]}"; do
            read -r DATASET ATTRIBUTE <<< "$pair"

            echo "-------------------------------------------"
            echo "Running (${COUNT}/${TOTAL_STEPS}): Main=${cm}, Proxy=${cp}, Dataset=${DATASET}, Attr=${ATTRIBUTE}"
            echo "-------------------------------------------"

            $PYTHON exp_individual.py \
                -cm "$cm" \
                -cp "$cp" \
                -d "$DATASET" \
                -s "$ATTRIBUTE" \
                --num-test $NUM_TEST \
                --threshold $THRESHOLD

            COUNT=$((COUNT + 1))
        done
    done
done

echo ""
echo "================================================================="
echo "STEP 3: Summarizing Results"
echo "================================================================="

# This script calculates IFr and Confidence Intervals from the CSVs generated in Step 2
$PYTHON summarize_IFr_growth.py

echo ""
echo "All experiments finished successfully."