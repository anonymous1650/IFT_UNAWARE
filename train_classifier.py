import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_score, classification_report, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import config
import common
import test_fairness

def get_dataset_config(dataset_name):
    if dataset_name == "adult":
        return config.adult_dataset_summary, config.adult_columns
    elif dataset_name == "german":
        return config.german_dataset_summary, config.german_columns
    elif dataset_name == "bank":
        return config.bank_dataset_summary, config.bank_columns
    else:
        raise ValueError(f"Dataset '{dataset_name}' is not configured in get_dataset_config.")


def load_dataset(dataset_name, sensitive):
    dataset_file = config.dataset_path.get(dataset_name)
    print("# Dataset:\t"+ dataset_file)

    dataset = pd.read_csv(dataset_file)
    dataset.head()
    all_columns = common.set_all_columns(dataset_name)
    dataset = dataset[all_columns]

    return dataset


def binarize_specified_attribute(df, dataset_name, attribute_to_binarize):

    rules_for_dataset = config.binarization_rules[dataset_name]

    print(f"\nApplying binarization for attribute: '{attribute_to_binarize}'...")
    
    df_copy = df.copy()
    rule_function = rules_for_dataset[attribute_to_binarize]
    df_copy[attribute_to_binarize] = df_copy[attribute_to_binarize].apply(rule_function)
    
    return df_copy

def plot_testdata_histogram(pipeline, X_test, classifier_name, class_label=0, bin_width=0.02):

    y_proba = pipeline.predict_proba(X_test)
    probabilities_for_class = y_proba[:, class_label]
    bins = np.arange(0, 1 + bin_width, bin_width)

    plt.figure(figsize=(10, 6))
    plt.hist(probabilities_for_class, bins=bins, edgecolor='black', alpha=0.7)
    
    plt.title(f'[{classifier_name.upper()}] Prediction Probability for Class {class_label}', fontsize=16)
    plt.xlabel(f'Predicted Probability of being Class {class_label}', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.xticks(bins, rotation=90)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.show()


def generate_random_data(dataset_summary, num_samples=10000):

    all_feature_names = [key for key in dataset_summary.keys() if key != 'Class']
    
    random_individuals = []
    for _ in range(num_samples):
        instance = {
            feature: random.randint(int(min_val), int(max_val))
            for feature, (min_val, max_val) in dataset_summary.items()
            if feature in all_feature_names
        }
        random_individuals.append(instance)
    
    return pd.DataFrame(random_individuals)

def plot_randomdata_histogram(pipeline, classifier_name, dataset_name, sensitive_attr, feature_names_for_model, class_label=0, bin_width=0.02, num_samples=5000):
    dataset_summary, _ = test_fairness.get_dataset_config(dataset_name)

    print(f"Plotting histogram for RANDOMLY generated data (Class {class_label})...")
    
    X_random_full = generate_random_data(dataset_summary, num_samples)
    X_random_for_prediction = X_random_full[feature_names_for_model]
    
    y_proba = pipeline.predict_proba(X_random_for_prediction)
    probabilities_for_class = y_proba[:, class_label]

    bins = np.arange(0, 1 + bin_width, bin_width)

    plt.figure(figsize=(10, 6))
    plt.hist(probabilities_for_class, bins=bins, edgecolor='black', alpha=0.7, color='skyblue')
    
    title_text = (f'[{classifier_name.upper()}] Probability on Random Data\n'
                  f'Dataset: {dataset_name}, Sensitive Attr: {sensitive_attr}, Target Class: {class_label}')
    plt.title(title_text, fontsize=14)
    plt.xlabel(f'Predicted Probability of being Class {class_label}', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def main(argv=None):
    args = common.getArgs(argv)

    dataset_name = args.dataset
    sensitive_attr = args.sensitive
    threshold = float(args.threshold)

    dataset = load_dataset(dataset_name, sensitive_attr)
    dataset = binarize_specified_attribute(dataset, dataset_name, sensitive_attr)

    dataset_summary, columns_info = get_dataset_config(dataset_name)

    output_column = columns_info["output"][0]

    X = dataset.drop(columns=[sensitive_attr, output_column])

    if args.train_proxy:
        y = dataset[sensitive_attr]
        classifier = args.proxy_classifier
    else:
        y = dataset[output_column]
        classifier = args.main_classifier

    param_grid = {}

    if classifier == "dnn":
        base_model = config.dnn_classifier
        param_grid = {
            'classifier__hidden_layer_sizes': [(32, 16), (64, 32), (100,)], 
            'classifier__learning_rate_init': [0.001, 0.01]
        }
        print("# Classifier: Deep Neural Network (DNN) with HPO")
        
    else:
        print("# Classifier specified wrong. Exit.")
        exit()

    base_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', base_model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    grid = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1
    )
    
    print(f"\nStarting HPO for {classifier}...")
    grid.fit(X_train, y_train)
    print("HPO finished.")
    print(f"Best params found: {grid.best_params_}")

    best_model_pipeline = grid.best_estimator_

    print("\nStarting Calibration...")
    
    final_model = CalibratedClassifierCV(
        estimator=best_model_pipeline,
        method='sigmoid',
        cv=3,
        n_jobs=-1
    )
    
    final_model.fit(X_train, y_train)
    print("Calibration finished.")

    if args.train_proxy:
        y_proba = final_model.predict_proba(X_test)
        y_test_array = np.array(y_test)

        max_proba = np.max(y_proba, axis=1)
        pred_label = np.argmax(y_proba, axis=1)
        mask = max_proba >= threshold
        y_test_filtered = y_test_array[mask]
        y_pred_filtered = pred_label[mask]
        print(f"Number of evaluation samples: {len(y_test_filtered)} / {len(y_test)}")
        if len(y_test_filtered) > 0:
            macro_precision = precision_score(y_test_filtered, y_pred_filtered, average='macro')
            print("Macro Precision: " + str(macro_precision))
        else:
            print("No samples found with prediction confidence above the threshold")
        
        prec_filename = classifier + "_" + args.dataset + "_" + args.sensitive + ".txt"    
        output_path = os.path.join('results/proxy/', prec_filename)
        with open(output_path, 'a') as f:
            f.write(f"Dataset: {dataset_name}, Sensitive Attr: {sensitive_attr}, Threshold: {threshold}\n")
            f.write(classification_report(y_test_filtered, y_pred_filtered, digits=4))
            f.write("-" * 50 + "\n")
    else:
        # If not train_proxy, display only Accuracy.
        y_pred = final_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print("Accuracy: " + str(acc))


    # Model saving
    if args.train_proxy:
        save_dir = "models/proxy/"
        proxy_or_main = "proxy"
    else:
        save_dir = "models/main/"
        proxy_or_main = "main"

    pkl_filename = save_dir + proxy_or_main + "_" + classifier + "_" + args.dataset + "_" + args.sensitive + ".pkl"
    
    joblib.dump(final_model, pkl_filename)
    print(f"Model saved to '{pkl_filename}'.")


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])