import os
import joblib
import json
import pandas as pd

# Get the directory of the current script
current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, 'random_forest_model.pkl')

# Load the random forest model
random_forest_model = joblib.load(model_path)
print("\nRandom Forest Model Information:")
print(random_forest_model)

# Export random forest model parameters to a JSON file
random_forest_params = random_forest_model.get_params()
with open(os.path.join(current_dir, 'random_forest_params.json'), 'w') as f:
    json.dump(random_forest_params, f, indent=4)
print("Random forest model parameters saved to random_forest_params.json")

# Export random forest feature importance to a CSV file
if hasattr(random_forest_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'Feature': [f'Feature_{i+1}' for i in range(len(random_forest_model.feature_importances_))],
        'Importance': random_forest_model.feature_importances_
    })
    feature_importance.to_csv(os.path.join(current_dir, 'random_forest_feature_importance.csv'), index=False)
    print("Random forest model feature importance saved to random_forest_feature_importance.csv")
