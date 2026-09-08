import sqlite3
import pandas as pd
import time

DB_PATH = "../data/heart_disease.db"

connection = sqlite3.connect(DB_PATH)

# Load original cleaned data
df = pd.read_csv("../data/heart_cleaned.csv")

# Create a large dataset
repeat_count = 1000

large_df = pd.concat(
    [df] * repeat_count,
    ignore_index=True
)

# Create unique patient IDs
large_df.insert(
    0,
    "patient_id",
    range(1, len(large_df) + 1)
)

print("Large dataset size:", len(large_df))

# Store large dataset
large_df.to_sql(
    "patients_large",
    connection,
    if_exists="replace",
    index=False
)

print("Large table created successfully!")


# ----------------------------------------
# Query
# ----------------------------------------

query = """
SELECT
    patient_id,
    age,
    target,
    chol,
    thalach
FROM patients_large
WHERE age = 77
AND target = 1;
"""


# ----------------------------------------
# Benchmark function
# ----------------------------------------

def benchmark(query, runs=20):

    start = time.perf_counter()

    for _ in range(runs):
        connection.execute(query).fetchall()

    end = time.perf_counter()

    return end - start


# ----------------------------------------
# WITHOUT INDEX
# ----------------------------------------

print("\n========== WITHOUT INDEX ==========")

connection.execute(
    "DROP INDEX IF EXISTS idx_age_target"
)

connection.commit()

without_index = benchmark(query)

print(
    f"Execution time for 20 runs: "
    f"{without_index:.6f} seconds"
)

print("\nQuery Plan:")

plan = connection.execute(
    "EXPLAIN QUERY PLAN " + query
).fetchall()

for row in plan:
    print(row)


# ----------------------------------------
# CREATE INDEX
# ----------------------------------------

print("\nCreating index...")

connection.execute(
    "CREATE INDEX idx_age_target "
    "ON patients_large(age, target)"
)

connection.commit()


# ----------------------------------------
# WITH INDEX
# ----------------------------------------

print("\n========== WITH INDEX ==========")

with_index = benchmark(query)

print(
    f"Execution time for 20 runs: "
    f"{with_index:.6f} seconds"
)

print("\nQuery Plan:")

plan = connection.execute(
    "EXPLAIN QUERY PLAN " + query
).fetchall()

for row in plan:
    print(row)


# ----------------------------------------
# Comparison
# ----------------------------------------

print("\n========== COMPARISON ==========")

print(f"Without index: {without_index:.6f}s")
print(f"With index:    {with_index:.6f}s")

if with_index < without_index:

    improvement = (
        (without_index - with_index)
        / without_index
    ) * 100

    print(
        f"Index improvement: "
        f"{improvement:.2f}%"
    )

else:

    slowdown = (
        (with_index - without_index)
        / without_index
    ) * 100

    print(
        f"Index slowdown: "
        f"{slowdown:.2f}%"
    )


connection.close()