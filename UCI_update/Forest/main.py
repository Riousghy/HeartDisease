import os
import pandas as pd
from model_training import train_and_evaluate_model

def run_data_cleaning():
    """Run the data_cleaning.py script to clean the dataset."""
    current_dir = os.path.dirname(__file__)
    data_cleaning_script = os.path.join(current_dir, 'data_cleaning.py')

    print("Starting data cleaning process...")
    os.system(f'python {data_cleaning_script}')
    print("Data cleaning completed.")

def load_cleaned_data():
    """Load the cleaned dataset."""
    current_dir = os.path.dirname(__file__)
    cleaned_file = os.path.join(current_dir, 'cleaned_heart_disease.csv')

    if not os.path.exists(cleaned_file):
        print(f"Cleaned dataset not found at: {cleaned_file}")
        exit(1)

    print(f"Loading cleaned data from: {cleaned_file}")
    data = pd.read_csv(cleaned_file)
    return data

def main():
    """Main function to orchestrate the workflow."""
    run_data_cleaning()
    data = load_cleaned_data()
    train_and_evaluate_model(data)

if __name__ == "__main__":
    main()
