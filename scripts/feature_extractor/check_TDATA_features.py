import pandas as pd
import numpy as np

# Load the data
data = pd.read_csv('../../dataset/training/Data.csv')

print("="*70)
print("ANALYZING TRAINING DATA FEATURES")
print("="*70)

# Column indices used during training
FEATURE_INDICES = [2, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 26]

print(f"\nTotal columns in CSV: {len(data.columns)}")
print(f"Features used: {len(FEATURE_INDICES)}\n")

print("Column Index | Column Name              | Sample Value      | Min          | Max          | Mean")
print("-" * 110)

for idx in FEATURE_INDICES:
    col_name = data.columns[idx]
    sample_val = data.iloc[0, idx]
    min_val = data.iloc[:, idx].min()
    max_val = data.iloc[:, idx].max()
    mean_val = data.iloc[:, idx].mean()
    
    print(f"{idx:12d} | {col_name:24s} | {sample_val:16.6f} | {min_val:12.6f} | {max_val:12.6f} | {mean_val:12.6f}")

print("\n" + "="*70)
print("\nALL COLUMNS IN THE DATASET:")
print("-" * 70)
for i, col in enumerate(data.columns):
    print(f"{i:3d}. {col}")

print("\n" + "="*70)
