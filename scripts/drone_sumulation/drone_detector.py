# drone_detector.py - Production GPS Spoofing Detector

import pickle
import pandas as pd
import time
from datetime import datetime
from gps_feature_extractor import GPSFeatureExtractor

# Configuration
GPS_TYPE = 'simulation'  # Options: 'simulation', 'mavlink', 'ublox', 'nmea'
GPS_PORT = '/dev/ttyACM0'
GPS_BAUD = 57600

# Load model
print("Loading GPS spoofing detection model...")
with open('../../model/DT_model.pkl', 'rb') as f:
    clf, scaler = pickle.load(f)
print("✓ Model loaded\n")

# Initialize extractor
extractor = GPSFeatureExtractor()

def detect_spoofing(features_dict):
    """Run detection on GPS features"""
    features_df = extractor.to_dataframe(features_dict)
    
    # Validate features
    warnings = extractor.validate_features(features_dict)
    if warnings:
        print("⚠️  Feature warnings:")
        for w in warnings[:3]:  # Show first 3
            print(f"   {w}")
    
    # Predict
    features_scaled = scaler.transform(features_df)
    prediction = clf.predict(features_scaled)[0]
    confidence = clf.predict_proba(features_scaled)[0][prediction] * 100
    
    return prediction, confidence

# SIMULATION MODE (for testing)
if GPS_TYPE == 'simulation':
    print("🧪 SIMULATION MODE - Using training data\n")
    print("="*70)
    print("🛰️  GPS SPOOFING DETECTOR ACTIVE")
    print("="*70)
    print("Press Ctrl+C to stop\n")
    
    data = pd.read_csv('../../dataset/testing/test_data.csv')
    FEATURE_COLS = [2, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 26]
    feature_names = data.columns[FEATURE_COLS]
    
    stats = {'clean': 0, 'static': 0, 'dynamic': 0}
    
    try:
        for i in range(1000):
            timestamp = datetime.now().strftime('%H:%M:%S')
            
            # Get random sample
            idx = i % len(data)
            features_dict = {name: data.iloc[idx][name] for name in feature_names}
            
            # Detect
            prediction, confidence = detect_spoofing(features_dict)
            
            labels = {0: 'CLEAN', 1: 'STATIC SPOOFING', 2: 'DYNAMIC SPOOFING'}
            status = labels[prediction]
            
            # Update stats
            if prediction == 0:
                stats['clean'] += 1
                color = '\033[92m'  # Green
                symbol = '✓'
            elif prediction == 1:
                stats['static'] += 1
                color = '\033[93m'  # Yellow
                symbol = '⚠'
            else:
                stats['dynamic'] += 1
                color = '\033[91m'  # Red
                symbol = '🚨'
            
            reset = '\033[0m'
            
            # Display
            print(f"{color}[{timestamp}] {symbol} {status:20s} | " +
                  f"Sats: {int(features_dict['satellites_used']):2d} | " +
                  f"HDOP: {features_dict['hdop']:4.1f} | " +
                  f"Vel: {features_dict['vel_m_s']:5.2f} m/s | " +
                  f"Conf: {confidence:5.1f}%{reset}")
            
            if prediction != 0:
                print(f"{color}    → ALERT: Take protective action!{reset}")
            
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\n\n" + "="*70)
        print("DETECTION STOPPED - STATISTICS")
        print("="*70)
        total = sum(stats.values())
        print(f"Total samples: {total}")
        print(f"Clean: {stats['clean']} ({stats['clean']/total*100:.1f}%)")
        print(f"Static spoofing: {stats['static']} ({stats['static']/total*100:.1f}%)")
        print(f"Dynamic spoofing: {stats['dynamic']} ({stats['dynamic']/total*100:.1f}%)")

# Add MAVLink/uBlox/NMEA modes as needed
elif GPS_TYPE == 'mavlink':
    print("🚁 MAVLINK MODE - Connect to Pixhawk")
    # Implement MAVLink connection
    
elif GPS_TYPE == 'ublox':
    print("📡 UBLOX MODE - Connect to uBlox GPS")
    # Implement uBlox connection
    
elif GPS_TYPE == 'nmea':
    print("📍 NMEA MODE - Standard GPS")
    # Implement NMEA connection
