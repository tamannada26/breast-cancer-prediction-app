# data_preparation.py

import pandas as pd
from sklearn.datasets import load_breast_cancer
from imblearn.over_sampling import SMOTE

# Load the dataset
def load_data():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df

# Apply SMOTE to handle class imbalance
def apply_smote(X, y):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled
