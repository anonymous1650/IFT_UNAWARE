# IFT_UNAWARE Replication Package

This package contains the source code and experimental scripts for the paper:
**"Individual Fairness Testing in Fairness through Unawareness"**.

It implements the framework **$IFT_{UNAWARE}$** to evaluate individual fairness in the FTU setting (Fairness Through Unawareness), where protected attributes are not explicitly used by the classifier but are inferred by a proxy model. This framework corresponds to **Algorithm 2** proposed in the paper.

## Directory Structure

    IFT_UNAWARE/
    ├── README.md               # This file
    ├── requirements.txt        # Python dependencies
    ├── exp.sh                  # Shell script to run the experimental pipeline
    ├── config.py               # Configuration for datasets, models, and hyperparameters
    ├── common.py               # Utility functions (data loading, args parsing)
    ├── train_classifier.py     # Script to train CuT (Classifier under Test) and Proxy models
    ├── exp_individual.py       # Script to generate fairness testing data (Batch/Offline mode)
    ├── test_fairness.py        # Script for online fairness testing (Algorithm 2 loop)
    ├── summarize_IFr_growth.py # Script to calculate IFr and Confidence Intervals from generated data
    ├── datasets/               # Place CSV datasets here
    │   ├── adult.csv
    │   ├── bank.csv
    │   └── german.csv
    ├── models/                 # Trained models will be saved here
    │   ├── main/
    │   └── proxy/
    └── results/                # Experiment results will be saved here
        └── proxy/              # Proxy model performance reports

## Prerequisites

The experiments were conducted in the following environment:

* **OS:** macOS 15.7.2
* **Python:** 3.10.3

### Python Dependencies

To ensure reproducibility, please use the specific versions listed below. You can install them using the `requirements.txt` file.

    pip install -r requirements.txt

**Content of `requirements.txt`:**

    joblib==1.5.1
    matplotlib==3.10.3
    numpy==2.2.6
    pandas==2.3.0
    scikit-learn==1.7.0

## Usage

Follow these steps to reproduce the experiments.

### 1. Prepare Datasets
Ensure the following dataset files are placed in the `datasets/` directory.
* `adult.csv`
* `bank.csv`
* `german.csv`

*(Note: These datasets should be pre-processed or compatible with the loading logic defined in `common.py`.)*

### 2. Train Models (CuT and Proxy)
Before testing, you need to train the Main Classifier (CuT) and the Proxy Classifier. The trained models will be saved in the `models/` directory.

**Example: Training models for the ADULT dataset**

    # 1. Train the Main Classifier (CuT)
    # -cm: classifier model (dnn), -d: dataset, -s: sensitive attribute
    python3 train_classifier.py -cm dnn -d adult -s gender

    # 2. Train the Proxy Classifier
    # --train-proxy: Flag to indicate proxy training
    python3 train_classifier.py -cp dnn -d adult -s gender --train-proxy

**Arguments:**
* `-d` / `--dataset`: `adult`, `bank`, `german`
* `-s` / `--sensitive`: `gender`, `race`, `age` (availability depends on dataset)
* `-cm`/`-cp`: Classifier type (default: `dnn`)

### 3. Run Fairness Testing (Test Generation)
Generate test cases (pairs of individuals) and record predictions. This script simulates the sampling and testing process (Algorithm 2).

    # Generate 50,000 test pairs with Hamming distance epsilon=1
    python3 exp_individual.py --dataset adult --sensitive gender --num-test 50000 --epsilon 1

* **Output:** The script generates raw experimental data (pairs, proxy confidence, predictions) and saves it to a CSV file in the `exp_individual_results_e1/` directory.

### 4. Analyze Results (Calculate IFr)
To obtain the **Individual Fairness Ratio (IFr)** and High-Confidence IFr values (as shown in Table V of the paper), run the summary script on the generated CSV files.

    python3 summarize_IFr_growth.py

* **Configuration:** To change the Hamming distance threshold ($\epsilon$) for analysis or the number of samples processed, modify the `EPSILON` and `NUM_SAMPLES` variables inside `summarize_IFr_growth.py`.

## Automating the Experiments

To reproduce the full experiments described in **Section IV** of the paper (iterating over all datasets and attributes), you can use the provided shell script `exp.sh`.

    bash exp.sh

## Configuration

Experimental settings are defined in `config.py`. You can adjust:
* **Model Hyperparameters:** Defined in `dnn_classifier` (Corresponds to **Table III** in the paper).
* **Binarization Rules:** Logic for handling sensitive attributes (Corresponds to **Table IV** in the paper).
* **Dataset Columns:** Feature definitions for each dataset.

## Notes on Validity and Reproducibility

* **Random Seed:** A fixed random seed (`42`) is used in `config.py` and training scripts to ensure deterministic model behavior.
* **Invalid Estimates:** As discussed in **RQ1** of the paper, for some dataset configurations (e.g., German Credit), the proxy classifier may fail to find enough high-confidence test cases. In such cases, the summary script will report "NA" or flags indicating invalid confidence intervals. This is an expected result demonstrating the difficulty of fairness testing in certain FTU scenarios.