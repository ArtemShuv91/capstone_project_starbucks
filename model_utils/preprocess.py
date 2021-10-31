import os
import pandas as pd
import argparse

os.system('pip install joblib')
os.system('pip install imblearn')
os.system('pip install --upgrade scikit-learn')

import joblib
from imblearn.under_sampling import RandomUnderSampler

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# Variables
variables = ['gender', 'weekday', 'age', 'income', 'day','became_member_from', 
             'number_of_transactions', 'avg_number_of_transctions',
             'number_of_offers_completed', 'number_of_offers_viewed', 
             'avg_reward', 'receival_to_view_avg', 'view_to_completion_avg']

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, required=True)
    parser.add_argument('--val_split_ratio', type=float, default=0.2)
    parser.add_argument('--test_split_ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=1123)
    args = parser.parse_args()

    # Random seed to reproduce the results
    seed = args.seed
    
    # Check the target
    target_name = args.target
    
    if target_name not in ['bogo', 'discount', 'info']:
        raise ValueError(
            ('Target name must be "bogo", "discount" or "info" - '
             f'{target_name} passed'))

    # Read the data
    input_path = os.path.join('/opt/ml/processing/input', f'{target_name}.csv')
    df = pd.read_csv(input_path, header=None, names=[target_name] + variables)
    
    rus = RandomUnderSampler(random_state=seed)
    X_tot, y_tot = rus.fit_resample(df.drop(target_name, 1), df[target_name])

    # Split the data
    X, X_val, y, y_val = train_test_split(
        X_tot, y_tot, test_size=args.val_split_ratio,
        stratify=y_tot, random_state=seed)
    
    X, X_test, y, y_test = train_test_split(
        X, y, test_size=args.test_split_ratio / (1 - args.val_split_ratio),
        stratify=y, random_state=seed)

    # Preprocessing for gender. Replace with "O", then One-Hot Encoding
    gender_pipe = Pipeline(
        [
            ('imputer', SimpleImputer(strategy='constant', fill_value='O')),
            ('ohe', OneHotEncoder())
        ])
    
    # Preprocessing for numerical features. Replace with median and standardize
    num_pipe = Pipeline(
        [
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

    # Join via ColumnTransformer
    preprocessing = ColumnTransformer(transformers=[
        ('gender_pipeline', gender_pipe, ['gender']),
        ('numeric_pipeline', num_pipe, variables[1:])
    ])

    # Fit Transformer
    X = preprocessing.fit_transform(X)
    X_val = preprocessing.transform(X_val)
    X_test = preprocessing.transform(X_test)

    # Save the data
    pd.concat([y.reset_index(drop=True), pd.DataFrame(X)], axis=1).to_csv(
        os.path.join(f'/opt/ml/processing/output/{target_name}_train.csv'),
        header=False, index=False
    )
    pd.concat([y_val.reset_index(drop=True), pd.DataFrame(X_val)], axis=1).to_csv(
        os.path.join(f'/opt/ml/processing/output/{target_name}_val.csv'),
        header=False, index=False
    )
    pd.Series(y_test.reset_index(drop=True)).to_csv(
        os.path.join(f'/opt/ml/processing/output/{target_name}_test_target.csv'),
        header=False, index=False
    )
    pd.DataFrame(X_test).to_csv(
        os.path.join(f'/opt/ml/processing/output/{target_name}_test.csv'),
        header=False, index=False
    )

    # Save transformer
    joblib.dump(
        preprocessing, f'/opt/ml/processing/output/{target_name}_transformer.joblib'
    )