import os
import joblib
import json
import pandas as pd

def process_model():
    print("\nProcessing model...")
    
    # Dynamically determine the path to the model file
    current_dir = os.path.dirname(__file__)
    model_path = os.path.join(current_dir, 'svm_model.pkl')
    
    # Load the model
    model = joblib.load(model_path)
    print("Model Information:")
    print(model)

    # Export model parameters to a JSON file
    model_params = model.get_params()
    with open(os.path.join(current_dir, 'svm_model_params.json'), 'w') as f:
        json.dump(model_params, f, indent=4)
    print("Model parameters saved to svm_model_params.json")

    # Export additional attributes if available
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'Feature': [f'Feature_{i+1}' for i in range(len(model.feature_importances_))],
            'Importance': model.feature_importances_
        })
        feature_importance.to_csv(os.path.join(current_dir, 'svm_model_feature_importance.csv'), index=False)
        print("Model feature importance saved to svm_model_feature_importance.csv")

    if hasattr(model, 'support_vectors_'):
        support_vectors_df = pd.DataFrame(model.support_vectors_)
        support_vectors_df.to_csv(os.path.join(current_dir, 'svm_model_support_vectors.csv'), index=False)
        print("Model support vectors saved to svm_model_support_vectors.csv")

if __name__ == "__main__":
    process_model()
