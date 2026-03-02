# src/features/feature_engineering.py

import pandas as pd

def preprocess_data(df):
    """
    Cleans and prepares dataset for modeling.
    """

    # Encode target
    df['deposit'] = df['deposit'].map({'yes': 1, 'no': 0})

    # Separate features and target
    X = df.drop('deposit', axis=1)
    y = df['deposit']

    # One-hot encode categorical variables
    X = pd.get_dummies(X, drop_first=True)

    return X, y