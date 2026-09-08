import sqlite3
import time


DB_PATH = "../data/heart_disease.db"

connection = sqlite3.connect(DB_PATH)


# ----------------------------------------
# Test query
# ----------------------------------------

query = """
SELECT
    target,
    COUNT(*) AS patient_count,
    AVG(chol) AS average_cholesterol
FROM patients
WHERE age >= 50
GROUP BY target;
"""


# ----------------------------------------
# Function to measure query time
# ----------------------------------------

def measure_query(query, runs=1000):

    start_time = time.perf_counter()

    for _ in range(runs):
        connection.execute(query).fetchall()

    end_time = time.perf_counter()

    total_time = end_time - start_time

    return total_time


# ----------------------------------------
# BEFORE INDEX
# ----------------------------------------

print("\n========== BEFORE INDEX ==========")

before_time = measure_query(query)

print(
    f"Execution time for 1000 runs: "
    f"{before_time:.6f} seconds"
)


# ----------------------------------------
# Query Plan BEFORE INDEX
# ----------------------------------------

print("\nQuery Plan BEFORE INDEX:")

plan = connection.execute(
    "EXPLAIN QUERY PLAN " + query
).fetchall()

for row in plan:
    print(row)


# ----------------------------------------
# CREATE INDEX
# ----------------------------------------

print("\nCreating index on age...")

connection.execute(
    "CREATE INDEX IF NOT EXISTS idx_patients_age "
    "ON patients(age)"
)

connection.commit()


# ----------------------------------------
# AFTER INDEX
# ----------------------------------------

print("\n========== AFTER INDEX ==========")

after_time = measure_query(query)

print(
    f"Execution time for 1000 runs: "
    f"{after_time:.6f} seconds"
)


# ----------------------------------------
# Query Plan AFTER INDEX
# ----------------------------------------

print("\nQuery Plan AFTER INDEX:")

plan = connection.execute(
    "EXPLAIN QUERY PLAN " + query
).fetchall()

for row in plan:
    print(row)


# ----------------------------------------
# Performance comparison
# ----------------------------------------

if after_time > 0:

    improvement = (
        (before_time - after_time)
        / before_time
    ) * 100

    print(
        f"\nPerformance change: "
        f"{improvement:.2f}%"
    )


connection.close()