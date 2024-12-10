# Import required libraries
import pandas as pd
import numpy as np
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    RocCurveDisplay,
)
import matplotlib.pyplot as plt

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

    # Apply SMOTE for balancing classes
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print("Resampled class distribution:\n", pd.Series(y_train_resampled).value_counts())

    # Hyperparameter tuning using GridSearchCV
    print("Starting hyperparameter tuning...")
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],  # Regularization strength
        'penalty': ['l2'],             # Regularization type (L2 for stability)
        'solver': ['liblinear', 'saga']  # Solvers compatible with L2
    }
    grid_search = GridSearchCV(
        LogisticRegression(random_state=42, max_iter=5000),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search.fit(X_train_resampled, y_train_resampled)

    # Best parameters and model
    best_params = grid_search.best_params_
    print("Best Parameters from GridSearchCV:", best_params)

    best_model = grid_search.best_estimator_

    # Save the best model
    joblib.dump(best_model, 'logistic_regression_optimized_model.pkl')
    print("Optimized model saved as 'logistic_regression_optimized_model.pkl'.")

    # Evaluate on validation set
    y_val_pred = best_model.predict(X_val)
    y_val_prob = best_model.predict_proba(X_val)
    if y_val_prob.ndim == 1:  # If single probability, adjust for multi-class compatibility
        y_val_prob = np.vstack([1 - y_val_prob, y_val_prob]).T

    val_metrics = {
        "Validation Accuracy": accuracy_score(y_val, y_val_pred),
        "Validation Precision": precision_score(y_val, y_val_pred, average='weighted', zero_division=1),
        "Validation Recall": recall_score(y_val, y_val_pred, average='weighted', zero_division=1),
        "Validation F1 Score": f1_score(y_val, y_val_pred, average='weighted', zero_division=1),
        "Validation ROC AUC Score": roc_auc_score(y_val, y_val_prob, multi_class='ovr')
    }
    print("Validation Metrics:", val_metrics)

    # Confusion matrix for validation set
    cm_val = confusion_matrix(y_val, y_val_pred)
    print("Validation Confusion Matrix:\n", cm_val)
    ConfusionMatrixDisplay(confusion_matrix=cm_val, display_labels=best_model.classes_).plot()
    plt.title("Confusion Matrix (Validation Set)")
    plt.show()

    # Evaluate on test set
    y_test_pred = best_model.predict(X_test)
    y_test_prob = best_model.predict_proba(X_test)
    if y_test_prob.ndim == 1:
        y_test_prob = np.vstack([1 - y_test_prob, y_test_prob]).T

    test_metrics = {
        "Test Accuracy": accuracy_score(y_test, y_test_pred),
        "Test Precision": precision_score(y_test, y_test_pred, average='weighted', zero_division=1),
        "Test Recall": recall_score(y_test, y_test_pred, average='weighted', zero_division=1),
        "Test F1 Score": f1_score(y_test, y_test_pred, average='weighted', zero_division=1),
        "Test ROC AUC Score": roc_auc_score(y_test, y_test_prob, multi_class='ovr')
    }
    print("Test Metrics:", test_metrics)

    # Confusion matrix for test set
    cm_test = confusion_matrix(y_test, y_test_pred)
    print("Test Confusion Matrix:\n", cm_test)
    ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=best_model.classes_).plot()
    plt.title("Confusion Matrix (Test Set)")
    plt.show()

    # ROC Curve for test set
    fpr, tpr, _ = roc_curve(y_test, y_test_prob[:, 1], pos_label=1)
    RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
    plt.title("ROC Curve (Test Set)")
    plt.show()

    print("Model training and evaluation completed.")
