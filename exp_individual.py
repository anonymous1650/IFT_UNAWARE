import csv
import os
import random
import sys
import joblib
import numpy as np
import pandas as pd
import common
import config

def get_dataset_config(dataset_name):
    if dataset_name == "adult":
        return config.adult_dataset_summary, config.adult_columns
    elif dataset_name == "german":
        return config.german_dataset_summary, config.german_columns
    elif dataset_name == "bank":
        return config.bank_dataset_summary, config.bank_columns
    else:
        raise ValueError(f"Dataset '{dataset_name}' is not configured in get_dataset_config.")


def generate_pair(dataset_summary, feature_names, epsilon):
    individual_A = {}
    for feature in feature_names:
        min_val, max_val = dataset_summary[feature]
        individual_A[feature] = random.randint(int(min_val), int(max_val))
    
    individual_B = individual_A.copy()

    for i in range(epsilon):
        feature_to_change = random.choice(feature_names)
        min_val, max_val = dataset_summary[feature_to_change]
        individual_B[feature_to_change] = random.randint(int(min_val), int(max_val))
            
    return individual_A, individual_B


def main(argv=None):
    args = common.getArgs(argv)

    main_classifier = args.main_classifier
    proxy_classifier = args.proxy_classifier
    dataset_name = args.dataset
    sensitive_attr = args.sensitive
    num_total_test = int(args.num_test)
    threshold = float(args.threshold)
    epsilon = int(args.epsilon)
    
    print("--- Test Configuration ---")
    print(f"# Main Classifier: {main_classifier}")
    print(f"# Proxy Classifier: {proxy_classifier}")
    print(f"# Dataset: {dataset_name}")
    print(f"# Sensitive Attribute: {sensitive_attr}")
    print(f"# Number of Test Pairs: {num_total_test}")
    print(f"# Confidence Threshold: {threshold}")
    print(f"# epsilon: {epsilon}")
    print("--------------------------\n")

    main_model_dir = "models/main"
    proxy_model_dir = "models/proxy"
    
    main_model_path = os.path.join(main_model_dir, f"main_{main_classifier}_{dataset_name}_{sensitive_attr}.pkl")
    proxy_model_path = os.path.join(proxy_model_dir, f"proxy_{proxy_classifier}_{dataset_name}_{sensitive_attr}.pkl")
    
    print(f"Loading main model from: {main_model_path}")
    main_model = joblib.load(main_model_path)
    print(f"Loading proxy model from: {proxy_model_path}")
    proxy_model = joblib.load(proxy_model_path)
    
    dataset_summary, columns_info = get_dataset_config(dataset_name)

    feature_names = [col for col in main_model.feature_names_in_]

    results = []

    print("Starting...\n")

    all_individuals_A = []
    all_individuals_B = []

    print(f"Generating {num_total_test} pairs...")
    for _ in range(num_total_test):
        individual_A, individual_B = generate_pair(dataset_summary, feature_names, epsilon) 
        all_individuals_A.append(individual_A)
        all_individuals_B.append(individual_B)

    df_A = pd.DataFrame(all_individuals_A, columns=feature_names) 
    df_B = pd.DataFrame(all_individuals_B, columns=feature_names)

    all_pairs_df = pd.concat([df_A, df_B], ignore_index=True)

    print("Predicting with proxy model (batch)...")
    all_probabilities = proxy_model.predict_proba(all_pairs_df)

    print("Predicting with main model (batch)...")
    all_main_predictions = main_model.predict(all_pairs_df)

    print("Formatting results...")
    results = []
    for i in range(num_total_test):
        idx_A = i
        idx_B = i + num_total_test

        data_A_str = ",".join(df_A.iloc[idx_A].astype(str).tolist())
        data_B_str = ",".join(df_B.iloc[idx_A].astype(str).tolist()) 

        row = [
            data_A_str,
            round(all_probabilities[idx_A][0], 3), # probability of A (Class 0)
            int(all_main_predictions[idx_A]),      # prediction of A
            data_B_str,
            round(all_probabilities[idx_B][0], 3), # probability of B (Class 0)
            int(all_main_predictions[idx_B])       # prediction of B
        ]
        results.append(row)

        i += 1

    dir_name = f'exp_individual_results_e{epsilon}'

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    base_filename = f"exp_{dataset_name}_{sensitive_attr}_{main_classifier}_{proxy_classifier}.csv"
    filename = os.path.join(dir_name, base_filename)

    file_exists = os.path.exists(filename)

    with open(filename, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        if not file_exists:
            header = ['ind_0', 'proxy_conf_0', 'pred_prob_0', 'ind1', 'proxy_conf_1', 'pred_prob_1']
            writer.writerow(header)
        
        writer.writerows(results)

    print(f"Data saved to '{filename}'.")

if __name__ == "__main__":
    main(sys.argv[1:])