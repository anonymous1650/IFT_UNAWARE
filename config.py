from sklearn.neural_network import MLPClassifier

dnn_classifier = MLPClassifier(
    hidden_layer_sizes=(32, 16),
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    max_iter=300,  
    early_stopping=True,
    random_state=42
)

adult_dataset_summary = {
    "age": [0, 9],
    "workclass": [0, 7],
    "fnlwgt":  [0, 74],
    "education": [0, 15],
    "marital_status": [0,6],
    "occupation": [0, 13],
    "relationship": [0, 5],
    "race": [0, 4],
    "gender": [0, 1],
    "capital_gain": [0, 41],
    "capital_loss": [0, 43],
    "hours_per_week": [1, 99],
    "native_country": [0, 41],
    "Class": [0, 1]
}

german_dataset_summary = {
    'account_status': [1, 4],
    'duration_in_month': [4, 72],
    'credit_history': [1, 5],
    'purpose': [1, 10],
    'credit_amount': [250, 18424],
    'savings_status': [1, 5],
    'employment_since': [1, 5],
    'installment_commitment': [1, 4],
    'gender': [0, 1],
    'other_parties': [1, 3],
    'residence_since': [1, 4],
    'property': [1, 4],
    'age': [19, 75],
    'other_installment_plans': [1, 3],
    'housing': [1, 3],
    'num_credits': [1, 4],
    'job': [1, 4],
    'num_dependent': [1, 2],
    'telephone': [1, 2],
    'foreign_worker': [1, 2],
    'Class': [0, 1]
}

bank_dataset_summary = {
    "age": [1, 9], 
    "job": [0, 11],
    "marital": [0, 2],
    "education": [0, 3],
    "default": [0, 1],
    "balance": [-20, 179],
    "housing": [0, 1],
    "loan": [0, 1],
    "contact": [0, 2],
    "day": [1, 31],
    "month": [0, 11],
    "duration": [0, 99],
    "campaign": [1, 63],
    "pdays": [0, 1],
    "previous": [0, 1],
    "poutcome": [0, 3],
    "Class": [0, 1]
}

# Configurations for datasets
adult_columns = {
    "sensitive": [ "age", "gender", "race"],
    "numerical": [ "fnlwgt", "education", "capital_gain", "capital_loss", "hours_per_week"],
    "categorical": [ "workclass", "marital_status", "occupation",
                     "relationship", "native_country"],
    "output": ["Class"]
}
german_columns = {
    "sensitive": ["gender", "age"],
    "numerical": ["duration_in_month", "credit_amount"],
    "categorical": ["account_status","job", "credit_history", "purpose", "savings_status", "employment_since", "installment_commitment", "other_parties", "residence_since", "property", "other_installment_plans", \
    "housing", "num_credits", "num_dependent", "telephone", "foreign_worker"],
    "output": ["Class"]
}
bank_columns = {
    "sensitive": ["age"],
    "numerical": ["job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome"],
    "categorical": [],
    "output": ["Class"]
}


binarization_rules = {
    "adult": {
        "gender": lambda x: x,
        "race": lambda x: 0 if x == 0 else 1,
        "age": lambda x: 0 if 2 <= x <= 5 else 1,
    },
    "german": {
        "gender": lambda x: x,
        "age": lambda x: 0 if 2 <= x <= 5 else 1,
    },
    "bank": {
        "age": lambda x: 0 if 2 <= x <= 5 else 1,
    }
}

dataset_path = {
    "adult": "datasets/adult.csv",
    "bank": "datasets/bank.csv",
    "german": "datasets/german.csv",
}

# source(Adult): https://archive.ics.uci.edu/dataset/2/adult
# source(Bank): https://archive.ics.uci.edu/dataset/222/bank+marketing
# source(German): https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data