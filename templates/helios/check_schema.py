#!/usr/bin/env python3
import sqlite3

conn = sqlite3.connect('helios_memory.db')
cursor = conn.cursor()

print("=== DATABASE SCHEMA ANALYSIS ===")

# List all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
print(f"\nTables found: {len(tables)}")
for table in tables:
    print(f"  - {table[0]}")

# Check training_sessions structure
print("\n=== TRAINING_SESSIONS TABLE ===")
cursor.execute("PRAGMA table_info(training_sessions)")
columns = cursor.fetchall()
print("Columns:")
for col in columns:
    print(f"  {col[1]} ({col[2]})")

# Check if there are other training-related tables
training_tables = [t[0] for t in tables if 'training' in t[0]]
print(f"\nTraining-related tables: {training_tables}")

for table in training_tables:
    if table != 'training_sessions':
        print(f"\n=== {table.upper()} TABLE ===")
        cursor.execute(f"PRAGMA table_info({table})")
        cols = cursor.fetchall()
        for col in cols:
            print(f"  {col[1]} ({col[2]})")

conn.close()
