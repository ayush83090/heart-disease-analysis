import pandas as pd
import sqlite3

csv_path = "../data/heart_cleaned.csv"
db_path = "../data/heart_disease.db"

# Load cleaned dataset
df = pd.read_csv(csv_path)

# Create unique patient ID
df.insert(0, "patient_id", range(1, len(df) + 1))

# Connect to database
connection = sqlite3.connect(db_path)

# Store patients table
df.to_sql(
    "patients",
    connection,
    if_exists="replace",
    index=False
)

print("Patients table created successfully!")

# Check records
query = """
SELECT COUNT(*) AS total_patients
FROM patients;
"""

result = pd.read_sql_query(query, connection)

print("\nTotal patients:")
print(result)

# Create patient_groups table
groups = pd.DataFrame({
    "patient_id": df["patient_id"],
    "age_group": pd.cut(
        df["age"],
        bins=[0, 39, 49, 59, 69, 100],
        labels=["Under 40", "40-49", "50-59", "60-69", "70+"]
    )
})

groups.to_sql(
    "patient_groups",
    connection,
    if_exists="replace",
    index=False
)

print("\nPatient groups table created successfully!")

connection.close()