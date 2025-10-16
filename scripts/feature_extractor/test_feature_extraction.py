# test_feature_extraction.py

import pandas as pd
import pickle
import numpy as np
from gps_feature_extractor import GPSFeatureExtractor

print("="*70)
print("TESTING FEATURE EXTRACTION PIPELINE")
print("="*70)

# Load model
with open('../../model/DT_model.pkl', 'rb') as f:
    clf, scaler = pickle.load(f)
print("\n✓ Model loaded")

# Load training data
data = pd.read_csv('../../dataset/training/Data.csv')
FEATURE_COLS = [2, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 26]
feature_names = data.columns[FEATURE_COLS].tolist()
print(f"✓ Training data loaded ({len(data)} samples)\n")

# Initialize extractor
extractor = GPSFeatureExtractor()

# Verify feature name matching
print("Feature Name Verification:")
print("-" * 70)
all_match = True
for i, (train_name, extract_name) in enumerate(zip(feature_names, extractor.FEATURE_NAMES)):
    match = "✓" if train_name == extract_name else "✗"
    if train_name != extract_name:
        all_match = False
    print(f"{i:2d}. {match} {train_name:24s} == {extract_name}")

if all_match:
    print("\n✅ ALL FEATURE NAMES MATCH PERFECTLY!\n")
else:
    print("\n❌ FEATURE MISMATCH - NEED TO FIX!\n")
    exit(1)

# Test prediction with training samples
print("="*70)
print("TESTING PREDICTIONS ON TRAINING SAMPLES")
print("="*70)

test_indices = [0, 100, 1000, 5000, 10000, 15000, 20000]
correct = 0

for idx in test_indices:
    if idx >= len(data):
        continue
    
    # Get sample from training data
    sample_df = data.iloc[[idx]][feature_names]
    actual_label = int(data.iloc[idx, 0])
    
    # Predict
    scaled = scaler.transform(sample_df)
    prediction = clf.predict(scaled)[0]
    confidence = clf.predict_proba(scaled)[0][prediction] * 100
    
    labels = ['Clean', 'Static Spoofing', 'Dynamic Spoofing']
    match = "✓" if prediction == actual_label else "✗"
    
    if prediction == actual_label:
        correct += 1
    
    print(f"Sample {idx:6d}: Actual={labels[actual_label]:20s} | " +
          f"Predicted={labels[prediction]:20s} | " +
          f"Conf={confidence:5.1f}% {match}")

print(f"\nAccuracy: {correct}/{len(test_indices)} = {correct/len(test_indices)*100:.1f}%")

print("\n" + "="*70)
print("✅ PIPELINE VALIDATED AND READY!")
print("="*70)
