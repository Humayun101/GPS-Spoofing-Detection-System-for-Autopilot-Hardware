import pickle
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Load the trained model
print("Loading model...")
with open('../../model/DT_model.pkl', 'rb') as f:
    clf, scaler = pickle.load(f)
print("✓ Model loaded successfully\n")

# Define the exact feature columns used during training
FEATURE_COLUMNS = [2, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 26]

def predict_spoofing(gps_features_df, show_details=True):
    """
    Predict GPS spoofing from features
    
    Parameters:
    - gps_features_df: DataFrame with proper column names
    - show_details: show prediction probabilities
    
    Returns: prediction label
    """
    features_scaled = scaler.transform(gps_features_df)
    prediction = clf.predict(features_scaled)
    
    labels = {0: 'Clean', 1: 'Static Spoofing', 2: 'Dynamic Spoofing'}
    result = labels[prediction[0]]
    
    if show_details and hasattr(clf, 'predict_proba'):
        probas = clf.predict_proba(features_scaled)[0]
        print(f"Prediction: {result}")
        print(f"Confidence:")
        for i, label in labels.items():
            print(f"  {label}: {probas[i]*100:.2f}%")
    
    return result, prediction[0]

# Test with multiple samples
print("="*60)
print("TESTING GPS SPOOFING DETECTION")
print("="*60)

# Load data with proper column names preserved
data = pd.read_csv('../../dataset/training/Data.csv')
label_col = data.columns[0]  # First column is the label

# Get column names for features
feature_names = data.columns[FEATURE_COLUMNS].tolist()
print(f"\nFeatures used: {feature_names}\n")

# Test samples from each class
print("📍 Testing Sample 1 (First row):")
sample1 = data.iloc[[0]][feature_names]  # Keep as DataFrame
actual1 = int(data.iloc[0, 0])
pred1, pred_num1 = predict_spoofing(sample1)
labels = {0: 'Clean', 1: 'Static Spoofing', 2: 'Dynamic Spoofing'}
print(f"Actual: {labels[actual1]}")
print(f"Match: {'✓ CORRECT' if actual1 == pred_num1 else '✗ WRONG'}\n")

# Find and test a Static spoofing sample
if len(data[data.iloc[:, 0] == 1]) > 0:
    print("📍 Testing Sample from Static Spoofing Class:")
    static_idx = data[data.iloc[:, 0] == 1].index[0]
    sample2 = data.iloc[[static_idx]][feature_names]
    actual2 = int(data.iloc[static_idx, 0])
    pred2, pred_num2 = predict_spoofing(sample2)
    print(f"Actual: {labels[actual2]}")
    print(f"Match: {'✓ CORRECT' if actual2 == pred_num2 else '✗ WRONG'}\n")

# Find and test a Dynamic spoofing sample
if len(data[data.iloc[:, 0] == 2]) > 0:
    print("📍 Testing Sample from Dynamic Spoofing Class:")
    dynamic_idx = data[data.iloc[:, 0] == 2].index[0]
    sample3 = data.iloc[[dynamic_idx]][feature_names]
    actual3 = int(data.iloc[dynamic_idx, 0])
    pred3, pred_num3 = predict_spoofing(sample3)
    print(f"Actual: {labels[actual3]}")
    print(f"Match: {'✓ CORRECT' if actual3 == pred_num3 else '✗ WRONG'}\n")

# Test random samples
print("="*60)
print("RANDOM SAMPLE TESTING (10 samples)")
print("="*60)

np.random.seed(42)
random_indices = np.random.choice(len(data), 10, replace=False)
correct = 0

for i, idx in enumerate(random_indices, 1):
    sample = data.iloc[[idx]][feature_names]
    actual = int(data.iloc[idx, 0])
    prediction, pred_num = predict_spoofing(sample, show_details=False)
    
    match = "✓" if pred_num == actual else "✗"
    if pred_num == actual:
        correct += 1
    
    print(f"{i:2d}. Actual: {labels[actual]:20s} | Predicted: {prediction:20s} {match}")

print(f"\nAccuracy: {correct}/10 = {correct*10}%")

print("\n" + "="*60)
print("MODEL READY FOR DEPLOYMENT")
print("="*60)
