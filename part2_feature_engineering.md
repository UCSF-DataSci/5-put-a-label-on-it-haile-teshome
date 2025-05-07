# Install necessary packages

```python
%pip install -r requirements.txt
```
# Part 2: Time Series Features & Tree-Based Models

**Objective:** Extract basic time-series features from heart rate data, train Random Forest and XGBoost models, and compare their performance.

## 1. Setup

Import necessary libraries.

```python
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
```

## 2. Data Loading

Load the dataset.

```python
def load_data(file_path):
    """
    Load the synthetic health data from a CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame containing the data with timestamp parsed as datetime
    """
    
    return pd.read_csv(file_path, parse_dates=['timestamp'])
```

## 3. Feature Engineering

Implement `extract_rolling_features` to calculate rolling mean and standard deviation for the `heart_rate`.

```python
def extract_rolling_features(df, window_size_seconds):
    """
    Calculate rolling mean and standard deviation for heart rate.
    
    Args:
        df: DataFrame with timestamp and heart_rate columns
        window_size_seconds: Size of the rolling window in seconds
        
    Returns:
        DataFrame with added hr_rolling_mean and hr_rolling_std columns
    """
    df = df.sort_values('timestamp')
    df = df.set_index('timestamp')

    rolling = df['heart_rate'].rolling(f'{window_size_seconds}s')
    df['hr_rolling_mean'] = rolling.mean()
    df['hr_rolling_std'] = rolling.std()

    df = df.reset_index()
    df = df.fillna(method='bfill')  

    return df
```

## 4. Data Preparation

Implement `prepare_data_part2` using the newly engineered features.

```python
def prepare_data_part2(df_with_features, test_size=0.2, random_state=42):
    """
    Prepare data for modeling with time-series features.
    
    Args:
        df_with_features: DataFrame with original and rolling features
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    features = ['age', 'systolic_bp', 'diastolic_bp', 'glucose_level', 'bmi',
                'hr_rolling_mean', 'hr_rolling_std']
    target = 'disease_outcome'
    
    X = df_with_features[features]
    y = df_with_features[target]
    
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    return train_test_split(X_imputed, y, test_size=test_size, random_state=random_state)
```

## 5. Random Forest Model

Implement `train_random_forest`.

```python
def train_random_forest(X_train, y_train, n_estimators=100, max_depth=10, random_state=42):
    """
    Train a Random Forest classifier.
    
    Args:
        X_train: Training features
        y_train: Training target
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of the trees
        random_state: Random seed for reproducibility
        
    Returns:
        Trained Random Forest model
    """
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)
    return model
```

## 6. XGBoost Model

Implement `train_xgboost`.

```python
def train_xgboost(X_train, y_train, n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42):
    """
    Train an XGBoost classifier.
    
    Args:
        X_train: Training features
        y_train: Training target
        n_estimators: Number of boosting rounds
        learning_rate: Boosting learning rate
        max_depth: Maximum depth of a tree
        random_state: Random seed for reproducibility
        
    Returns:
        Trained XGBoost model
    """
    model = xgb.XGBClassifier(n_estimators=n_estimators,
                              learning_rate=learning_rate,
                              max_depth=max_depth,
                              use_label_encoder=False,
                              eval_metric='logloss',
                              random_state=random_state)
    model.fit(X_train, y_train)
    return model
```

## 7. Model Comparison

Calculate and compare AUC scores for both models.

```python

rf_probs = rf_model.predict_proba(X_test)[:, 1]
xgb_probs = xgb_model.predict_proba(X_test)[:, 1]

rf_auc = roc_auc_score(y_test, rf_probs)
xgb_auc = roc_auc_score(y_test, xgb_probs)

print(f"Random Forest AUC: {rf_auc:.4f}")
print(f"XGBoost AUC: {xgb_auc:.4f}")

```

## 8. Save Results

Save the AUC scores to a text file.

```python
def save_auc_results(rf_auc, xgb_auc, filename='results/results_part2.txt'):
    """
    Save the AUC scores to a results text file.
    """
    os.makedirs('results', exist_ok=True)
    with open(filename, 'w') as f:
        f.write(f"Random Forest AUC: {rf_auc:.4f}\n")
        f.write(f"XGBoost AUC: {xgb_auc:.4f}\n")

```

## 9. Main Execution

Run the complete workflow.

```python
# Main execution
if __name__ == "__main__":
    # 1. Load data
    data_file = 'data/synthetic_health_data.csv'
    df = load_data(data_file)
    
    # 2. Extract rolling features
    window_size = 300  # 5 minutes in seconds
    df_with_features = extract_rolling_features(df, window_size)
    
    # 3. Prepare data
    X_train, X_test, y_train, y_test = prepare_data_part2(df_with_features)
    
    # 4. Train models
    rf_model = train_random_forest(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)
    
    # 5. Calculate AUC scores
    rf_probs = rf_model.predict_proba(X_test)[:, 1]
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    
    rf_auc = roc_auc_score(y_test, rf_probs)
    xgb_auc = roc_auc_score(y_test, xgb_probs)
    
    print(f"Random Forest AUC: {rf_auc:.4f}")
    print(f"XGBoost AUC: {xgb_auc:.4f}")
    
    # 6. Save results
    save_auc_results(rf_auc, xgb_auc)
