import pandas as pd

# Dataset path
file_path = "../data/processed.cleveland.data"

# Column names
columns = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target"
]

# Load data
df = pd.read_csv(
    file_path,
    names=columns,
    na_values="?"
)

print("Original dataset shape:", df.shape)

# 1. Remove duplicate rows
df = df.drop_duplicates()

# 2. Handle missing values
df["ca"] = df["ca"].fillna(df["ca"].median())
df["thal"] = df["thal"].fillna(df["thal"].mode()[0])

# 3. Convert target to binary classification
df["target"] = (df["target"] > 0).astype(int)

# 4. Convert selected columns to integers
integer_columns = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "slope",
    "ca",
    "thal",
    "target"
]

for column in integer_columns:
    df[column] = df[column].astype(int)

print("\nCleaned dataset shape:", df.shape)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nTarget distribution:")
print(df["target"].value_counts())

# Save cleaned dataset
df.to_csv("../data/heart_cleaned.csv", index=False)

print("\nCleaned dataset saved successfully!")