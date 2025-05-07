```python
%pip install -r requirements.txt
```
# Part 1: Introduction to Classification & Evaluation

**Objective:** Load the synthetic health data, train a Logistic Regression model, and evaluate its performance.

## 1. Setup

Import necessary libraries.

```python
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.impute import SimpleImputer
```

## 2. Data Loading

Implement the `load_data` function to read the dataset.

```python
def load_data(file_path):
    """
    Load the synthetic health data from a CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame containing the data
    """
    
    return pd.read_csv(file_path)
```

## 3. Data Preparation

Implement `prepare_data_part1` to select features, split data, and handle missing values.

```python
def prepare_data_part1(df, test_size=0.2, random_state=42):
    """
    Prepare data for modeling: select features, split into train/test sets, handle missing values.
    
    Args:
        df: Input DataFrame
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    features = ['age', 'systolic_bp', 'diastolic_bp', 'glucose_level', 'bmi']
    target = 'disease_outcome'
    
    X = df[features]
    y = df[target]
    
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    return train_test_split(X_imputed, y, test_size=test_size, random_state=random_state)
```

## 4. Model Training

Implement `train_logistic_regression`.

```python
def train_logistic_regression(X_train, y_train):
    """
    Train a logistic regression model.
    
    Args:
        X_train: Training features
        y_train: Training target
        
    Returns:
        Trained logistic regression model
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model
```

## 5. Model Evaluation

Implement `calculate_evaluation_metrics` to assess the model's performance.

```python
def calculate_evaluation_metrics(model, X_test, y_test):
    """
    Calculate classification evaluation metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        
    Returns:
        Dictionary containing accuracy, precision, recall, f1, auc, and confusion_matrix
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_prob),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }
```

## 6. Save Results

Save the calculated metrics to a text file.

```python
def save_results(metrics, filename='results/results_part1.txt'):
    """
    Save metrics dictionary to a text file in JSON format.
    """
    os.makedirs('results', exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(metrics, f, indent=4)
```

## 7. Main Execution

Run the complete workflow.

```python
# Main execution
if __name__ == "__main__":
    # 1. Load data
    data_file = 'data/synthetic_health_data.csv'
    df = load_data(data_file)
    
    # 2. Prepare data
    X_train, X_test, y_train, y_test = prepare_data_part1(df)
    
    # 3. Train model
    model = train_logistic_regression(X_train, y_train)
    
    # 4. Evaluate model
    metrics = calculate_evaluation_metrics(model, X_test, y_test)
    
    # 5. Print metrics
    for metric, value in metrics.items():
        if metric != 'confusion_matrix':
            print(f"{metric}: {value:.4f}")
    
    # 6. Save results
    save_results(metrics)
    
    # 7. Interpret results
    interpretation = interpret_results(metrics)
    print("\nResults Interpretation:")
    for key, value in interpretation.items():
        print(f"{key}: {value}")
```

## 8. Interpret Results

Implement a function to analyze the model performance on imbalanced data.

```python
def interpret_results(metrics):
    """
    Analyze model performance on imbalanced data.
    
    Args:
        metrics: Dictionary containing evaluation metrics
        
    Returns:
        Dictionary with keys:
        - 'best_metric': Name of the metric that performed best
        - 'worst_metric': Name of the metric that performed worst
        - 'imbalance_impact_score': A score from 0-1 indicating how much
          the class imbalance affected results (0=no impact, 1=severe impact)
    """
    metric_values = {k: v for k, v in metrics.items() if k != 'confusion_matrix'}
    best_metric = max(metric_values, key=metric_values.get)
    worst_metric = min(metric_values, key=metric_values.get)
    
    imbalance_impact_score = float(
        abs(metrics['accuracy'] - metrics['recall'] + metrics['accuracy'] - metrics['f1']) / 2
    )
    
    imbalance_impact_score = min(1.0, round(imbalance_impact_score, 3))  

    return {
        'best_metric': best_metric,
        'worst_metric': worst_metric,
        'imbalance_impact_score': imbalance_impact_score
    }
