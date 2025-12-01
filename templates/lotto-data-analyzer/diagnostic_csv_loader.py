# diagnostic_csv_loader.py

import pandas as pd

csv_path = "data/powerball_complete_dataset.csv"

print("=== First 3 lines of the CSV (raw text) ===")
with open(csv_path, "r", encoding="utf-8") as f:
    for i in range(3):
        print(f.readline().strip())

print("\n=== Attempt 1: Default pd.read_csv ===")
try:
    df1 = pd.read_csv(csv_path)
    print(f"Rows: {len(df1)}, Columns: {df1.columns.tolist()}")
except Exception as e:
    print(f"Failed: {e}")

print("\n=== Attempt 2: engine='python' ===")
try:
    df2 = pd.read_csv(csv_path, engine="python")
    print(f"Rows: {len(df2)}, Columns: {df2.columns.tolist()}")
except Exception as e:
    print(f"Failed: {e}")

print("\n=== Attempt 3: sep=';' ===")
try:
    df3 = pd.read_csv(csv_path, sep=";")
    print(f"Rows: {len(df3)}, Columns: {df3.columns.tolist()}")
except Exception as e:
    print(f"Failed: {e}")

print("\n=== Attempt 4: encoding='utf-16' ===")
try:
    df4 = pd.read_csv(csv_path, encoding="utf-16")
    print(f"Rows: {len(df4)}, Columns: {df4.columns.tolist()}")
except Exception as e:
    print(f"Failed: {e}")

print("\n=== Attempt 5: engine='python', quoting=3 ===")
try:
    import csv
    df5 = pd.read_csv(csv_path, engine="python", quoting=csv.QUOTE_NONE)
    print(f"Rows: {len(df5)}, Columns: {df5.columns.tolist()}")
except Exception as e:
    print(f"Failed: {e}")