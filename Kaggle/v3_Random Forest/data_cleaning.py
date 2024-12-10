# Import required libraries
import pandas as pd
import numpy as np

# Load dataset from UCI repository
file_path = "Dataset/kaggle/heart.csv"
data = pd.read_csv(file_path)  

# Data Cleaning
# 1. Handle Missing Values
# Replace missing values in categorical columns with mode, and numerical columns with median
missing_columns = ['ca', 'thal']
for col in missing_columns:
    data[col] = data[col].fillna(data[col].mode()[0])

# 2. Remove Duplicate Rows
data = data.drop_duplicates()

# 3. Check for Outliers using IQR (Interquartile Range) Method
def remove_outliers(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

# Continuous numeric columns to check for outliers
continuous_columns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
data = remove_outliers(data, continuous_columns)

# Feature Engineering
# Transform categorical columns to numeric using one-hot encoding
categorical_columns = data.select_dtypes(include=['object', 'category']).columns
data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# Save cleaned data to a single CSV file
data.to_csv('cleaned_heart_disease.csv', index=False)

print("Data cleaning completed and saved as 'cleaned_heart_disease.csv'.")
