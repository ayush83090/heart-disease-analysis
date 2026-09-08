import sqlite3
import pandas as pd

db_path = "../data/heart_disease.db"

connection = sqlite3.connect(db_path)


# ----------------------------------------
# Query 1: Total patients
# ----------------------------------------

query1 = """
SELECT COUNT(*) AS total_patients
FROM patients;
"""

result1 = pd.read_sql_query(query1, connection)

print("\n1. Total Patients")
print(result1)


# ----------------------------------------
# Query 2: Heart disease distribution
# ----------------------------------------

query2 = """
SELECT
    target,
    COUNT(*) AS patient_count
FROM patients
GROUP BY target;
"""

result2 = pd.read_sql_query(query2, connection)

print("\n2. Heart Disease Distribution")
print(result2)


# ----------------------------------------
# Query 3: Average age by outcome
# ----------------------------------------

query3 = """
SELECT
    target,
    ROUND(AVG(age), 2) AS average_age
FROM patients
GROUP BY target;
"""

result3 = pd.read_sql_query(query3, connection)

print("\n3. Average Age by Outcome")
print(result3)


# ----------------------------------------
# Query 4: Average cholesterol by outcome
# ----------------------------------------

query4 = """
SELECT
    target,
    ROUND(AVG(chol), 2) AS average_cholesterol
FROM patients
GROUP BY target;
"""

result4 = pd.read_sql_query(query4, connection)

print("\n4. Average Cholesterol by Outcome")
print(result4)


# ----------------------------------------
# Query 5: Chest pain analysis
# ----------------------------------------

query5 = """
SELECT
    cp,
    COUNT(*) AS total_patients,
    SUM(target) AS heart_disease_cases
FROM patients
GROUP BY cp
ORDER BY heart_disease_cases DESC;
"""

result5 = pd.read_sql_query(query5, connection)

print("\n5. Chest Pain Analysis")
print(result5)


# ----------------------------------------
# Query 6: Heart disease by age group
# ----------------------------------------

query6 = """
SELECT
    pg.age_group,
    COUNT(p.patient_id) AS total_patients,
    SUM(p.target) AS heart_disease_cases,
    ROUND(
        100.0 * SUM(p.target) / COUNT(p.patient_id),
        2
    ) AS disease_rate
FROM patients p
JOIN patient_groups pg
    ON p.patient_id = pg.patient_id
GROUP BY pg.age_group
ORDER BY disease_rate DESC;
"""

result6 = pd.read_sql_query(query6, connection)

print("\n6. Heart Disease Rate by Age Group")
print(result6)


# ----------------------------------------
# Query 7: Rank age groups by disease rate
# ----------------------------------------

query7 = """
WITH age_group_stats AS (
    SELECT
        pg.age_group,
        COUNT(p.patient_id) AS total_patients,
        SUM(p.target) AS heart_disease_cases,
        ROUND(
            100.0 * SUM(p.target) / COUNT(p.patient_id),
            2
        ) AS disease_rate
    FROM patients p
    JOIN patient_groups pg
        ON p.patient_id = pg.patient_id
    GROUP BY pg.age_group
)

SELECT
    age_group,
    total_patients,
    heart_disease_cases,
    disease_rate,
    RANK() OVER (
        ORDER BY disease_rate DESC
    ) AS disease_rate_rank
FROM age_group_stats
ORDER BY disease_rate_rank;
"""

result7 = pd.read_sql_query(query7, connection)

print("\n7. Age Group Ranking by Disease Rate")
print(result7)

connection.close()