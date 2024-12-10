# Import required libraries
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

def train_and_evaluate_model(data):
    """Train a Logistic Regression model and evaluate its performance."""
    # One-hot encoding for categorical columns
    categorical_columns = ['cp', 'restecg', 'slope']
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

    # Normalize continuous variables
    scaler = StandardScaler()
    continuous_columns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    data[continuous_columns] = scaler.fit_transform(data[continuous_columns])

    # Split data into train, validation, and test sets
    X = data.drop('num', axis=1)  # Features
    y = data['num']  # Target
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Train Logistic Regression model
    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000)  # Increase max_iter to handle convergence warning
    model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(model, 'logistic_regression_model.pkl')
    print("Model saved as 'logistic_regression_model.pkl'.")

    # Evaluate on validation set
    y_val_pred = model.predict(X_val)
    val_metrics = {
        "Validation Accuracy": accuracy_score(y_val, y_val_pred),
        "Validation Precision": precision_score(y_val, y_val_pred, average='weighted'),  # Use weighted for multiclass
        "Validation Recall": recall_score(y_val, y_val_pred, average='weighted')         # Use weighted for multiclass
    }
    print("Validation Metrics:", val_metrics)

    # Save validation metrics to a file
    with open('validation_metrics.txt', 'w') as f:
        for key, value in val_metrics.items():
            f.write(f"{key}: {value}\n")

    # Evaluate on test set
    y_test_pred = model.predict(X_test)
    test_metrics = {
        "Test Accuracy": accuracy_score(y_test, y_test_pred),
        "Test Precision": precision_score(y_test, y_test_pred, average='weighted'),  # Use weighted for multiclass
        "Test Recall": recall_score(y_test, y_test_pred, average='weighted')         # Use weighted for multiclass
    }
    print("Test Metrics:", test_metrics)

    # Save test metrics to a file
    with open('test_metrics.txt', 'w') as f:
        for key, value in test_metrics.items():
            f.write(f"{key}: {value}\n")

    print("Model training and evaluation completed.")
