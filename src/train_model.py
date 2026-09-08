import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


# ----------------------------------------
# 1. Load cleaned dataset
# ----------------------------------------

df = pd.read_csv("../data/heart_cleaned.csv")

print("Dataset shape:", df.shape)


# ----------------------------------------
# 2. Separate features and target
# ----------------------------------------

X = df.drop("target", axis=1)
y = df["target"]


# ----------------------------------------
# 3. Train-test split
# ----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ----------------------------------------
# 4. Feature scaling
# ----------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ----------------------------------------
# 5. Train Logistic Regression
# ----------------------------------------

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train_scaled, y_train)


# ----------------------------------------
# 6. Predictions
# ----------------------------------------

y_pred = model.predict(X_test_scaled)

y_probability = model.predict_proba(X_test_scaled)[:, 1]


# ----------------------------------------
# 7. Evaluation
# ----------------------------------------

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

roc_auc = roc_auc_score(y_test, y_probability)


print("\n========== MODEL PERFORMANCE ==========")

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")


# ----------------------------------------
# 8. Classification Report
# ----------------------------------------

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "No Heart Disease",
            "Heart Disease"
        ]
    )
)


# ----------------------------------------
# 9. Confusion Matrix
# ----------------------------------------

print("\n========== CONFUSION MATRIX ==========")

print(confusion_matrix(y_test, y_pred))


# ----------------------------------------
# 10. Save model + scaler
# ----------------------------------------

joblib.dump(
    model,
    "../models/heart_disease_model.pkl"
)

joblib.dump(
    scaler,
    "../models/scaler.pkl"
)

print("\nModel saved successfully!")

print("Scaler saved successfully!")