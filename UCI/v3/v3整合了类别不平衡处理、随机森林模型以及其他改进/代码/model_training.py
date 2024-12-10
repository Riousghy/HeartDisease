# Import required libraries
import pandas as pd
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

def train_and_evaluate_model(data):
    """Train a Random Forest model and evaluate its performance."""
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

    # Apply SMOTE for balancing classes
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print("Resampled class distribution:\n", pd.Series(y_train_resampled).value_counts())

    # Train Random Forest model
    print("Training Random Forest model...")
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_resampled, y_train_resampled)

    # Save the trained model
    joblib.dump(model, 'random_forest_model.pkl')
    print("Model saved as 'random_forest_model.pkl'.")

    # Evaluate on validation set
    y_val_pred = model.predict(X_val)
    val_metrics = {
        "Validation Accuracy": accuracy_score(y_val, y_val_pred),
        "Validation Precision": precision_score(y_val, y_val_pred, average='weighted', zero_division=1),
        "Validation Recall": recall_score(y_val, y_val_pred, average='weighted', zero_division=1)
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
        "Test Precision": precision_score(y_test, y_test_pred, average='weighted', zero_division=1),
        "Test Recall": recall_score(y_test, y_test_pred, average='weighted', zero_division=1)
    }
    print("Test Metrics:", test_metrics)

    # Save test metrics to a file
    with open('test_metrics.txt', 'w') as f:
        for key, value in test_metrics.items():
            f.write(f"{key}: {value}\n")

    print("Model training and evaluation completed.")
