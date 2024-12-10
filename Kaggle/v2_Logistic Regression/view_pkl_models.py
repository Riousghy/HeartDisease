import joblib
import json
import pandas as pd
import os

# Process Logistic Regression model
def process_model():
    print("\nProcessing Logistic Regression model...")
    
    # Dynamically determine the path to the model file
    current_dir = os.path.dirname(__file__)
    model_path = os.path.join(current_dir, 'logistic_regression_model.pkl')
    
    # Load the model
    model = joblib.load(model_path)
    print("Model Information:")
    print(model)

    # Export model parameters to a JSON file
    model_params = model.get_params()
    params_path = os.path.join(current_dir, 'logistic_regression_model_params.json')
    with open(params_path, 'w') as f:
        json.dump(model_params, f, indent=4)
    print(f"Model parameters saved to {params_path}")

    # Export logistic regression coefficients to a CSV file
    if hasattr(model, 'coef_'):
        coef_df = pd.DataFrame(model.coef_)
        coef_path = os.path.join(current_dir, 'logistic_regression_coefficients.csv')
        coef_df.to_csv(coef_path, index=False)
        print(f"Logistic regression coefficients saved to {coef_path}")

if __name__ == "__main__":
    process_model()
