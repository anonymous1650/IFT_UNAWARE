import argparse
import random
import numpy as np
import pandas as pd
import config


def getArgs(argv):
    # Arguments handler
    parser = argparse.ArgumentParser(description='This is an evaluator for machine learning models.')
    parser.add_argument('-cm', '--main-classifier', default='dnn',
                        help='Type of main classifier to use (default: dnn)')
    parser.add_argument('-cp', '--proxy-classifier', default='dnn',
                        help='Type of proxy classifier to use (default: dnn)')
    parser.add_argument('-d', '--dataset', default='adult',
                        help='Name of the dataset to use (default: adult)')
    parser.add_argument('-s', '--sensitive', default='gender',
                        help='Name of the sensitive attribute (default: gender)')
    parser.add_argument('-th', '--threshold', default='0.5',
                        help='Classification threshold (default: 0.5)')
    parser.add_argument('-ep', '--epsilon', default='1',
                        help='Value of epsilon (default: 1)')
    parser.add_argument('--plot-save', action='store_true',
                        help='Set this flag to save plots to a file.')
    parser.add_argument('--num-test', default='10000',
                        help='Number of tested pairs')
    parser.add_argument('--train-proxy', action='store_true',
                        help='Set this flag to train proxy models.') # set '--train-proxy'

    args = parser.parse_args(argv)

    print('# main_classifier =\t' + args.main_classifier)
    print('# proxy_classifier =\t' + args.proxy_classifier)
    print('# dataset =\t' + args.dataset)
    print('# sensitive =\t' + args.sensitive)
    print('# threshold =\t' + args.threshold)
    print('# epsilon =\t' + args.epsilon)
    print('# plot_save =\t' + str(args.plot_save))
    print('# train_proxy =\t' + str(args.train_proxy))
    # print('# save_model =\t' + str(args.save_model))
    
    return args

def set_all_columns(dataset_name):
    if dataset_name == "adult":
        categorical_columns = config.adult_columns.get("sensitive") + config.adult_columns.get("categorical")
        numerical_columns = config.adult_columns.get("numerical")
        output_columns = config.adult_columns.get("output")
    elif dataset_name == "bank":
        categorical_columns = config.bank_columns.get("sensitive") + config.bank_columns.get("categorical")
        numerical_columns = config.bank_columns.get("numerical")
        output_columns = config.bank_columns.get("output")
    elif dataset_name == "german":
        categorical_columns = config.german_columns.get("sensitive") + config.german_columns.get("categorical")
        numerical_columns = config.german_columns.get("numerical")
        output_columns = config.german_columns.get("output")
    elif dataset_name == "compas":
        categorical_columns = config.compas_columns.get("sensitive") + config.compas_columns.get("categorical")
        numerical_columns = config.compas_columns.get("numerical")
        output_columns = config.compas_columns.get("output")
    elif dataset_name == "lsa":
        categorical_columns = config.lsa_columns.get("sensitive") + config.lsa_columns.get("categorical")
        numerical_columns = config.lsa_columns.get("numerical")
        output_columns = config.lsa_columns.get("output")
    else:
        exit("ERROR: Dataset specified wrong")

    all_columns = categorical_columns + numerical_columns + output_columns
    return all_columns

def generate_data(dataset_name, feature_order, num_samples, sensitive_attr):
    if (dataset_name == "adult"):
        dataset_summary = config.adult_dataset_summary
    elif (dataset_name == "german"):
        dataset_summary = config.german_dataset_summary
    else:
        exit("Error. config error.")

    random_data = generate_data_from_summary(dataset_summary, num_samples, sensitive_attr)
    return random_data[feature_order]

def generate_data_from_summary(summary_dict, k, exclude_columns=None):
    if exclude_columns is None:
        exclude_columns = []

    data = []

    for _ in range(k):
        instance = {}
        for key, val in summary_dict.items():
            if key in exclude_columns:
                continue

            if isinstance(val, list) and len(val) == 2 and all(isinstance(v, (int, float)) for v in val):
                generated_val = round(np.random.uniform(val[0], val[1]))
            else:
                generated_val = random.choice(val)

            instance[key] = generated_val
        data.append(instance)

    pd.set_option('display.max_columns', None)
    return pd.DataFrame(data)