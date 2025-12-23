import pandas as pd
import numpy as np
import random
import joblib
import os
import sys
import config
import common
import time


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
    epsilon = 1
    
    print("--- Test Configuration ---")
    print(f"# Main Classifier: {main_classifier}")
    print(f"# Proxy Classifier: {proxy_classifier}")
    print(f"# Dataset: {dataset_name}")
    print(f"# Sensitive Attribute: {sensitive_attr}")
    print(f"# Number of Test Pairs: {num_total_test}")
    print(f"# Confidence Threshold: {threshold}")
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
    
    discriminatory_pairs_count = 0
    tested_pairs_count = 0

    timeout_count = 0

    print("Starting fairness test...\n")
    start_time = time.time()

    while tested_pairs_count < num_total_test:
        
        pair_dict_A, pair_dict_B = generate_pair(dataset_summary, feature_names, epsilon)
        pair_df = pd.DataFrame([pair_dict_A, pair_dict_B])
        
        predicted_attrs = proxy_model.predict(pair_df)

        if predicted_attrs[0] == predicted_attrs[1]:
            continue

        probabilities = proxy_model.predict_proba(pair_df)
        confidences = probabilities.max(axis=1)

        if confidences[0] < threshold or confidences[1] < threshold:
            timeout_count += 1
            if timeout_count > 100000:
                print("[TIMEOUT] Terminated due to consecutive failures.")
                sys.exit()
            continue 

        tested_pairs_count += 1
        main_predictions = main_model.predict(pair_df)

        if main_predictions[0] != main_predictions[1]:
            discriminatory_pairs_count += 1
        

        if tested_pairs_count > 0 and tested_pairs_count % (num_total_test // 10) == 0:
            progress_percentage = (tested_pairs_count / num_total_test) * 100
            current_time = time.time()
            mid_time = current_time - start_time
            print(f"Progress: {progress_percentage:.0f}% ({tested_pairs_count}/{num_total_test} completed.) " + str(mid_time) + " sec")

    end_time = time.time()
    elapsed_time = end_time - start_time
    discrimination_rate = (discriminatory_pairs_count / tested_pairs_count) * 100

    print("\n--- Test Results ---")

    print(f"Total Pairs Tested: {tested_pairs_count}")
    print(f"Discriminatory Pairs Found: {discriminatory_pairs_count}")
    print(f"Discrimination Rate: {discrimination_rate:.2f}%")
    print(f"Time Elapsed: {elapsed_time:.2f} seconds")
    
    result_dir = "results"
    os.makedirs(result_dir, exist_ok=True)
    filename = f"results_{main_classifier}_{dataset_name}_{sensitive_attr}.txt"
    filepath = os.path.join(result_dir, filename)
    
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(f"Proxy Classifier: {proxy_classifier}, ")
        f.write(f"Threshold: {threshold}, ")
        f.write(f"Tested Pairs: {tested_pairs_count}, ")
        f.write(f"Discriminatory Pairs: {discriminatory_pairs_count}, ")
        f.write(f"Rate: {discrimination_rate:.2f}%\n")
        
    print(f"Results appended to: {filepath}")

if __name__ == "__main__":
    main(sys.argv[1:])
