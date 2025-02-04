import sqlite3
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

def format_json(json_str):
    """Format JSON string for better readability"""
    if not json_str:
        return None
    try:
        return json.loads(json_str)
    except:
        return json_str

def inspect_registry(db_path: str = "experiments/registry.db"):
    """Display contents of the model registry database"""
    print("\n" + "="*80)
    print("Model Registry Inspector")
    print("="*80)

    conn = sqlite3.connect(db_path)

    # Get all tables
    tables = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table'",
        conn
    )

    for table_name in tables['name']:
        print(f"\n{'='*40}")
        print(f"Table: {table_name}")
        print(f"{'='*40}")

        # Get table data
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)

        # Format JSON columns
        for col in df.columns:
            if df[col].dtype == object:  # Usually string columns
                # Try to format any JSON strings
                df[col] = df[col].apply(format_json)

        if len(df) > 0:
            print(df.to_string())
        else:
            print("No records found")

    conn.close()

def main():
    try:
        inspect_registry()
        return 0
    except Exception as e:
        print(f"Error inspecting registry: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 