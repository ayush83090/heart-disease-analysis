import pandas as pd
import matplotlib.pyplot as plt

# Load cleaned dataset
df = pd.read_csv("../data/heart_cleaned.csv")

print("========== DATASET OVERVIEW ==========")

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nBasic Statistics:")
print(df.describe())


# ------------------------------------------------
# 1. Heart Disease Distribution
# ------------------------------------------------

print("\n========== HEART DISEASE DISTRIBUTION ==========")

disease_counts = df["target"].value_counts()

print(disease_counts)

print("\nPercentage distribution:")
print(df["target"].value_counts(normalize=True) * 100)


# ------------------------------------------------
# 2. Average Age
# ------------------------------------------------

print("\n========== AGE ANALYSIS ==========")

print("Average age:", round(df["age"].mean(), 2))

print("\nAverage age by heart disease:")
print(
    df.groupby("target")["age"]
    .mean()
    .round(2)
)


# ------------------------------------------------
# 3. Cholesterol Analysis
# ------------------------------------------------

print("\n========== CHOLESTEROL ANALYSIS ==========")

print("Average cholesterol:", round(df["chol"].mean(), 2))

print("\nAverage cholesterol by target:")
print(
    df.groupby("target")["chol"]
    .mean()
    .round(2)
)


# ------------------------------------------------
# 4. Maximum Heart Rate
# ------------------------------------------------

print("\n========== MAX HEART RATE ANALYSIS ==========")

print(
    df.groupby("target")["thalach"]
    .mean()
    .round(2)
)


# ------------------------------------------------
# 5. Chest Pain Analysis
# ------------------------------------------------

print("\n========== CHEST PAIN ANALYSIS ==========")

chest_pain_analysis = pd.crosstab(
    df["cp"],
    df["target"],
    normalize="index"
) * 100

print(chest_pain_analysis.round(2))


# ------------------------------------------------
# 6. Correlation with Target
# ------------------------------------------------

print("\n========== CORRELATION WITH TARGET ==========")

correlation = (
    df.corr(numeric_only=True)["target"]
    .sort_values(ascending=False)
)

print(correlation)


# ------------------------------------------------
# 7. Age Distribution Plot
# ------------------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(df["age"], bins=15)

plt.xlabel("Age")
plt.ylabel("Number of Patients")
plt.title("Age Distribution")

plt.tight_layout()

plt.savefig("../data/age_distribution.png")

plt.show()


print("\nEDA completed successfully!")