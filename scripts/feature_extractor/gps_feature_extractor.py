# gps_feature_extractor.py - UPDATED FOR YOUR EXACT FEATURES

import numpy as np
import pandas as pd
import pynmea2
from collections import deque
from datetime import datetime

class GPSFeatureExtractor:
    """
    Extract GPS features matching your training data EXACTLY
    
    Training feature order (columns 2,7,8,10,11,12,13,14,15,16,17,18,19,26):
    0.  time_utc_usec      (column 2)
    1.  s_variance_m_s     (column 7)  - Speed variance
    2.  c_variance_rad     (column 8)  - Course variance
    3.  epv                (column 10) - Vertical position error
    4.  hdop               (column 11) - Horizontal dilution
    5.  vdop               (column 12) - Vertical dilution
    6.  noise_per_ms       (column 13) - RF noise level
    7.  jamming_indicator  (column 14) - Jamming detection
    8.  vel_m_s            (column 15) - Ground speed
    9.  vel_n_m_s          (column 16) - North velocity
    10. vel_e_m_s          (column 17) - East velocity
    11. vel_d_m_s          (column 18) - Down velocity
    12. cog_rad            (column 19) - Course over ground
    13. satellites_used    (column 26) - Number of satellites
    """
    
    FEATURE_NAMES = [
        'time_utc_usec',
        's_variance_m_s',
        'c_variance_rad',
        'epv',
        'hdop',
        'vdop',
        'noise_per_ms',
        'jamming_indicator',
        'vel_m_s',
        'vel_n_m_s',
        'vel_e_m_s',
        'vel_d_m_s',
        'cog_rad',
        'satellites_used'
    ]
    
    # Expected ranges from your training data
    FEATURE_RANGES = {
        'time_utc_usec': (0, 1.66e15),
        's_variance_m_s': (0.053, 1.483),
        'c_variance_rad': (0.010832, 3.141593),
        'epv': (1.261, 34.469),
        'hdop': (0.840, 1.990),
        'vdop': (1.060, 3.110),
        'noise_per_ms': (78, 243),
        'jamming_indicator': (2, 97),
        'vel_m_s': (0, 5.576),
        'vel_n_m_s': (-5.516, 5.431),
        'vel_e_m_s': (-5.373, 5.549),
        'vel_d_m_s': (-0.912, 1.555),
        'cog_rad': (0.000004, 6.283182),
        'satellites_used': (5, 11)
    }
    
    def __init__(self, history_size=20):
        """Initialize with history tracking for variance calculation"""
        self.speed_history = deque(maxlen=history_size)
        self.course_history = deque(maxlen=history_size)
        self.altitude_history = deque(maxlen=history_size)
        self.time_history = deque(maxlen=history_size)
        
    def extract_from_nmea_basic(self, gga_msg=None, gsa_msg=None, rmc_msg=None, vtg_msg=None):
        """
        Extract features from STANDARD NMEA sentences
        ⚠️ WARNING: Some features will be ESTIMATED (not ideal for production)
        
        Args:
            gga_msg: pynmea2.GGA message
            gsa_msg: pynmea2.GSA message
            rmc_msg: pynmea2.RMC message
            vtg_msg: pynmea2.VTG message (optional)
            
        Returns:
            dict: Features (some estimated)
        """
        features = {name: 0.0 for name in self.FEATURE_NAMES}
        
        # TIME (column 2)
        if rmc_msg and rmc_msg.timestamp:
            time_seconds = (rmc_msg.timestamp.hour * 3600 + 
                          rmc_msg.timestamp.minute * 60 + 
                          rmc_msg.timestamp.second)
            # Add microseconds if available
            if hasattr(rmc_msg.timestamp, 'microsecond'):
                time_seconds += rmc_msg.timestamp.microsecond / 1_000_000
            features['time_utc_usec'] = float(time_seconds * 1_000_000)
        elif gga_msg and gga_msg.timestamp:
            time_seconds = (gga_msg.timestamp.hour * 3600 + 
                          gga_msg.timestamp.minute * 60 + 
                          gga_msg.timestamp.second)
            features['time_utc_usec'] = float(time_seconds * 1_000_000)
        
        # HDOP (column 11)
        if gsa_msg and gsa_msg.hdop:
            features['hdop'] = float(gsa_msg.hdop)
        elif gga_msg and gga_msg.horizontal_dil:
            features['hdop'] = float(gga_msg.horizontal_dil)
        
        # VDOP (column 12)
        if gsa_msg and gsa_msg.vdop:
            features['vdop'] = float(gsa_msg.vdop)
        
        # SATELLITES (column 26)
        if gga_msg and gga_msg.num_sats:
            features['satellites_used'] = float(gga_msg.num_sats)
        
        # VELOCITY (column 15)
        if rmc_msg and rmc_msg.spd_over_grnd:
            speed_knots = float(rmc_msg.spd_over_grnd)
            features['vel_m_s'] = speed_knots * 0.514444  # knots to m/s
        elif vtg_msg and vtg_msg.spd_over_grnd_kmph:
            speed_kmh = float(vtg_msg.spd_over_grnd_kmph)
            features['vel_m_s'] = speed_kmh / 3.6  # km/h to m/s
        
        # COURSE (column 19)
        if rmc_msg and rmc_msg.true_course:
            course_deg = float(rmc_msg.true_course)
            features['cog_rad'] = course_deg * 0.0174533  # degrees to radians
        elif vtg_msg and vtg_msg.true_track:
            course_deg = float(vtg_msg.true_track)
            features['cog_rad'] = course_deg * 0.0174533
        
        # VELOCITY COMPONENTS (columns 16, 17)
        if features['vel_m_s'] > 0:
            features['vel_n_m_s'] = features['vel_m_s'] * np.cos(features['cog_rad'])
            features['vel_e_m_s'] = features['vel_m_s'] * np.sin(features['cog_rad'])
        
        # ALTITUDE (for vel_d_m_s calculation)
        if gga_msg and gga_msg.altitude:
            current_alt = float(gga_msg.altitude)
            current_time = features['time_utc_usec']
            
            if len(self.altitude_history) > 0 and len(self.time_history) > 0:
                dt = (current_time - self.time_history[-1]) / 1_000_000  # to seconds
                if dt > 0:
                    dalt = current_alt - self.altitude_history[-1]
                    features['vel_d_m_s'] = -dalt / dt  # Negative because down is positive
            
            self.altitude_history.append(current_alt)
            self.time_history.append(current_time)
        
        # VARIANCES (columns 7, 8) - ESTIMATED from history
        self.speed_history.append(features['vel_m_s'])
        self.course_history.append(features['cog_rad'])
        
        if len(self.speed_history) >= 10:
            features['s_variance_m_s'] = float(np.std(self.speed_history))
            features['c_variance_rad'] = float(np.std(self.course_history))
        else:
            # Use training mean as default
            features['s_variance_m_s'] = 0.226  # Mean from training
            features['c_variance_rad'] = 0.602  # Mean from training
        
        # ⚠️ NOT AVAILABLE in standard NMEA - use training means
        features['epv'] = 2.940  # Mean from training
        features['noise_per_ms'] = 133.222  # Mean from training
        features['jamming_indicator'] = 34.831  # Mean from training
        
        return features
    
    def extract_from_ublox(self, ubx_nav_pvt=None, ubx_nav_dop=None, ubx_mon_hw=None):
        """
        Extract features from uBlox UBX protocol messages
        ✅ This gives ALL features accurately!
        
        Args:
            ubx_nav_pvt: UBX-NAV-PVT message (Position Velocity Time)
            ubx_nav_dop: UBX-NAV-DOP message (Dilution of Precision)
            ubx_mon_hw: UBX-MON-HW message (Hardware status)
            
        Returns:
            dict: All features from UBX messages
        """
        features = {name: 0.0 for name in self.FEATURE_NAMES}
        
        if ubx_nav_pvt:
            # Time
            features['time_utc_usec'] = float(ubx_nav_pvt['iTOW']) * 1000
            
            # Velocity
            features['vel_m_s'] = float(ubx_nav_pvt['gSpeed']) / 1000.0  # mm/s to m/s
            features['vel_n_m_s'] = float(ubx_nav_pvt['velN']) / 1000.0
            features['vel_e_m_s'] = float(ubx_nav_pvt['velE']) / 1000.0
            features['vel_d_m_s'] = float(ubx_nav_pvt['velD']) / 1000.0
            
            # Course
            features['cog_rad'] = float(ubx_nav_pvt['headMot']) * 1e-5 * 0.0174533
            
            # Accuracy
            features['s_variance_m_s'] = float(ubx_nav_pvt['sAcc']) / 1000.0
            features['c_variance_rad'] = float(ubx_nav_pvt['headAcc']) * 1e-5 * 0.0174533
            features['epv'] = float(ubx_nav_pvt['vAcc']) / 1000.0
            
            # Satellites
            features['satellites_used'] = float(ubx_nav_pvt['numSV'])
        
        if ubx_nav_dop:
            features['hdop'] = float(ubx_nav_dop['hDOP']) / 100.0
            features['vdop'] = float(ubx_nav_dop['vDOP']) / 100.0
        
        if ubx_mon_hw:
            features['noise_per_ms'] = float(ubx_mon_hw['noisePerMS'])
            features['jamming_indicator'] = float(ubx_mon_hw['jamInd'])
        
        return features
    
    def extract_from_mavlink(self, gps_raw_int=None, global_position_int=None):
        """
        Extract features from MAVLink messages (Pixhawk/ArduPilot)
        ✅ Most features available!
        
        Args:
            gps_raw_int: GPS_RAW_INT message
            global_position_int: GLOBAL_POSITION_INT message
            
        Returns:
            dict: Features from MAVLink
        """
        features = {name: 0.0 for name in self.FEATURE_NAMES}
        
        if gps_raw_int:
            features['time_utc_usec'] = float(gps_raw_int.time_usec)
            features['satellites_used'] = float(gps_raw_int.satellites_visible)
            features['hdop'] = float(gps_raw_int.eph) / 100.0  # cm to m
            features['vdop'] = float(gps_raw_int.epv) / 100.0
            features['epv'] = float(gps_raw_int.epv) / 100.0
            features['vel_m_s'] = float(gps_raw_int.vel) / 100.0  # cm/s to m/s
            features['cog_rad'] = float(gps_raw_int.cog) / 100.0 * 0.0174533
        
        if global_position_int:
            features['vel_n_m_s'] = float(global_position_int.vx) / 100.0
            features['vel_e_m_s'] = float(global_position_int.vy) / 100.0
            features['vel_d_m_s'] = float(global_position_int.vz) / 100.0
        else:
            # Calculate from speed and course
            features['vel_n_m_s'] = features['vel_m_s'] * np.cos(features['cog_rad'])
            features['vel_e_m_s'] = features['vel_m_s'] * np.sin(features['cog_rad'])
        
        # Calculate variances from history
        self.speed_history.append(features['vel_m_s'])
        self.course_history.append(features['cog_rad'])
        
        if len(self.speed_history) >= 10:
            features['s_variance_m_s'] = float(np.std(self.speed_history))
            features['c_variance_rad'] = float(np.std(self.course_history))
        else:
            features['s_variance_m_s'] = 0.226
            features['c_variance_rad'] = 0.602
        
        # Not available in basic MAVLink - use defaults
        features['noise_per_ms'] = 133.222
        features['jamming_indicator'] = 34.831
        
        return features
    
    def to_dataframe(self, features):
        """Convert features dict to DataFrame with correct column names"""
        return pd.DataFrame([[features[name] for name in self.FEATURE_NAMES]], 
                          columns=self.FEATURE_NAMES)
    
    def validate_features(self, features):
        """Check if features are within expected ranges"""
        warnings = []
        for name, value in features.items():
            if name in self.FEATURE_RANGES:
                min_val, max_val = self.FEATURE_RANGES[name]
                if not (min_val <= value <= max_val):
                    warnings.append(f"{name}: {value:.3f} outside range [{min_val:.3f}, {max_val:.3f}]")
        return warnings
