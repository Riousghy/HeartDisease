import os
import subprocess
import pandas as pd
from model_training import train_and_evaluate_svm

def run_data_cleaning():
    """Run the data_cleaning.py script to clean the dataset."""
    current_dir = os.path.dirname(__file__)
    data_cleaning_script = os.path.join(current_dir, 'data_cleaning.py')

    print("Starting data cleaning process...")
    try:
        subprocess.run(['python', data_cleaning_script], check=True)
        print("Data cleaning completed.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while cleaning data: {e}")
        exit(1)

def load_cleaned_data():
    """Load the cleaned dataset."""
    current_dir = os.path.dirname(__file__)
    cleaned_file = os.path.join(current_dir, 'cleaned_heart_disease.csv')

    if not os.path.exists(cleaned_file):
        print(f"Cleaned dataset not found at: {cleaned_file}")
        print("Please ensure that data_cleaning.py has completed successfully.")
        exit(1)

    print(f"Loading cleaned data from: {cleaned_file}")
    data = pd.read_csv(cleaned_file)
    print("Cleaned Data Info:")
    print(data.info())
    return data

def main():
    """Main function to orchestrate the workflow."""
    # Step 1: Run data cleaning
    run_data_cleaning()

    # Step 2: Load cleaned data
    data = load_cleaned_data()

    # Step 3: Train and evaluate SVM model
    train_and_evaluate_svm(data)

if __name__ == "__main__":
    main()
