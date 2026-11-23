import os
import json
import numpy as np
import pandas as pd
import joblib

class FloodPredictor:
    def __init__(self, model_regime='split_chronological_70_20_10', model_type='xgboost'):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.root_dir = os.path.abspath(os.path.join(self.base_dir, '..'))
        
        # Paths
        meta_path = os.path.join(self.base_dir, 'station_metadata.json')
        models_dir = os.path.join(self.root_dir, 'models', model_regime)
        
        model_filename = 'xgboost_model.pkl' if model_type == 'xgboost' else 'stacking_ensemble_model.pkl'
        
        # Check direct local models folder first (standalone deployment repo)
        local_models = os.path.join(self.base_dir, 'models')
        if os.path.exists(os.path.join(local_models, model_filename)):
            model_path = os.path.join(local_models, model_filename)
            feature_cols_path = os.path.join(local_models, 'feature_columns.pkl')
        elif os.path.exists(os.path.join(local_models, model_regime, model_filename)):
            model_path = os.path.join(local_models, model_regime, model_filename)
            feature_cols_path = os.path.join(local_models, model_regime, 'feature_columns.pkl')
        else:
            models_dir = os.path.join(self.root_dir, 'models', model_regime)
            model_path = os.path.join(models_dir, model_filename)
            feature_cols_path = os.path.join(models_dir, 'feature_columns.pkl')
            if not os.path.exists(model_path):
                models_dir = os.path.join(self.root_dir, 'models', 'split_shuffled_70_15_15')
                model_path = os.path.join(models_dir, model_filename)
                feature_cols_path = os.path.join(models_dir, 'feature_columns.pkl')

        # Load artifacts
        with open(meta_path, 'r', encoding='utf-8') as f:
            self.station_meta = json.load(f)
            
        self.feature_columns = joblib.load(feature_cols_path)
        self.model = joblib.load(model_path)
        self.model_type = model_type
        self.model_regime = model_regime

    def get_stations(self):
        return sorted(list(self.station_meta.keys()))

    def get_station_defaults(self, station_name, month=7):
        meta = self.station_meta.get(station_name, {})
        if not meta:
            # Fallback to Dhaka if not found
            meta = self.station_meta.get('Dhaka', {})
            station_name = 'Dhaka'
            
        m_str = str(int(month))
        return {
            'station_name': station_name,
            'latitude': meta.get('latitude', 23.7),
            'longitude': meta.get('longitude', 90.38),
            'alt': meta.get('alt', 8.0),
            'x_cor': meta.get('x_cor', 539000.0),
            'y_cor': meta.get('y_cor', 625000.0),
            'rainfall': meta.get('monthly_avg_rainfall', {}).get(m_str, 380.0),
            'max_temp': meta.get('monthly_avg_max_temp', {}).get(m_str, 32.0),
            'min_temp': meta.get('monthly_avg_min_temp', {}).get(m_str, 26.0),
            'relative_humidity': meta.get('monthly_avg_humidity', {}).get(m_str, 84.0),
            'wind_speed': meta.get('monthly_avg_wind', {}).get(m_str, 1.8),
            'cloud_coverage': meta.get('monthly_avg_cloud', {}).get(m_str, 6.2),
            'bright_sunshine': meta.get('monthly_avg_sun', {}).get(m_str, 4.5)
        }

    def prepare_features(self, data):
        st = data.get('station_name', 'Dhaka')
        meta = self.station_meta.get(st, self.station_meta.get('Dhaka', {}))
        
        month = int(data.get('month', 7))
        rainfall = float(data.get('rainfall', 350.0))
        max_temp = float(data.get('max_temp', 32.0))
        min_temp = float(data.get('min_temp', 25.0))
        humidity = float(data.get('relative_humidity', 82.0))
        wind = float(data.get('wind_speed', 1.8))
        cloud = float(data.get('cloud_coverage', 6.0))
        sun = float(data.get('bright_sunshine', 4.5))
        
        lat = float(data.get('latitude', meta.get('latitude', 23.7)))
        lon = float(data.get('longitude', meta.get('longitude', 90.38)))
        alt = float(data.get('alt', meta.get('alt', 8.0)))
        x_cor = float(data.get('x_cor', meta.get('x_cor', 539000.0)))
        y_cor = float(data.get('y_cor', meta.get('y_cor', 625000.0)))
        
        # Antecedent rainfall lags: if not provided, estimate from station monthly normals
        prev_m1 = 12 if month == 1 else month - 1
        prev_m2 = 12 if prev_m1 == 1 else prev_m1 - 1
        
        default_lag1 = meta.get('monthly_avg_rainfall', {}).get(str(prev_m1), rainfall * 0.8)
        default_lag2 = meta.get('monthly_avg_rainfall', {}).get(str(prev_m2), rainfall * 0.6)
        
        lag1 = float(data.get('rainfall_lag1', default_lag1))
        lag2 = float(data.get('rainfall_lag2', default_lag2))
        rolling3m = rainfall + lag1 + lag2
        
        # Runoff & Hydrological ratios
        runoff_ratio = rainfall / (alt + 1.0)
        moisture_index = humidity * cloud
        
        # Fourier Cyclical Month Signals
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        is_monsoon = 1 if month in [6, 7, 8, 9] else 0
        
        # Log features
        rain_log = np.log1p(max(0.0, rainfall))
        wind_log = np.log1p(max(0.0, wind))
        alt_log = np.log1p(max(0.0, alt))
        lag1_log = np.log1p(max(0.0, lag1))
        lag2_log = np.log1p(max(0.0, lag2))
        rolling3m_log = np.log1p(max(0.0, rolling3m))
        runoff_ratio_log = np.log1p(max(0.0, runoff_ratio))
        
        # Dictionary of base features
        feature_dict = {
            'Max_Temp': max_temp,
            'Min_Temp': min_temp,
            'Rainfall': rainfall,
            'Relative_Humidity': humidity,
            'Wind_Speed': wind,
            'Cloud_Coverage': cloud,
            'Bright_Sunshine': sun,
            'X_COR': x_cor,
            'Y_COR': y_cor,
            'LATITUDE': lat,
            'LONGITUDE': lon,
            'ALT': alt,
            'Rainfall log': rain_log,
            'Wind_Speed log': wind_log,
            'ALT log': alt_log,
            'Rainfall_Lag1': lag1,
            'Rainfall_Lag2': lag2,
            'Rainfall_Rolling3M': rolling3m,
            'Rainfall_Lag1_log': lag1_log,
            'Rainfall_Lag2_log': lag2_log,
            'Rainfall_Rolling3M_log': rolling3m_log,
            'Runoff_Ratio': runoff_ratio,
            'Runoff_Ratio_log': runoff_ratio_log,
            'Moisture_Index': moisture_index,
            'Month_Sin': month_sin,
            'Month_Cos': month_cos,
            'Is_Monsoon': is_monsoon
        }
        
        # One-hot station columns
        for col in self.feature_columns:
            if col.startswith('Station_'):
                target_station = col.replace('Station_', '')
                feature_dict[col] = 1 if st == target_station else 0
                
        # Build DataFrame with exact column order
        df_row = pd.DataFrame([feature_dict])
        df_row = df_row[self.feature_columns]
        return df_row, {
            'rainfall': rainfall,
            'lag1': lag1,
            'lag2': lag2,
            'rolling3m': rolling3m,
            'runoff_ratio': runoff_ratio,
            'alt': alt,
            'is_monsoon': bool(is_monsoon)
        }

    def predict(self, data, threshold_mode='balanced'):
        # Operational threshold setpoints
        # Balanced: 0.50 (Max F1 balance)
        # Alert / Life-Safety: 0.35 (Maximizes detection recall >= 98%)
        threshold = 0.35 if threshold_mode == 'alert' else 0.50
        
        df_features, hydro_stats = self.prepare_features(data)
        
        # Inference
        probs = self.model.predict_proba(df_features)[0]
        prob_flood = float(probs[1])
        is_flood = bool(prob_flood >= threshold)
        
        # Risk Tier Classification
        if prob_flood >= 0.70:
            risk_tier = 'severe'
            badge_en = 'SEVERE FLOOD EMERGENCY'
            badge_bn = 'চরম বন্যা পরিস্থিতি ও জরুরি বিপদসংকেত'
            color = '#ef4444' # red
        elif prob_flood >= 0.45:
            risk_tier = 'high'
            badge_en = 'HIGH FLOOD ALERT'
            badge_bn = 'উচ্চ বন্যা সতর্কতা ও বিপদসীমা পূর্বাভাস'
            color = '#f97316' # orange
        elif prob_flood >= 0.25:
            risk_tier = 'moderate'
            badge_en = 'GUARDED / WATCH ADVISORY'
            badge_bn = 'নজরদারি ও মধ্যম সতর্কতা'
            color = '#eab308' # yellow
        else:
            risk_tier = 'low'
            badge_en = 'NORMAL SEASONAL CONDITIONS'
            badge_bn = 'স্বাভাবিক পরিস্থিতি (ঝুঁকিমুক্ত)'
            color = '#10b981' # emerald green
            
        # Hydrological drivers breakdown
        drivers = []
        if hydro_stats['rainfall'] > 450:
            drivers.append({
                'en': f"Extreme precipitation volume ({hydro_stats['rainfall']:.0f} mm) exceeds direct catchment runoff thresholds.",
                'bn': f"অতিরিক্ত বৃষ্টিপাত ({hydro_stats['rainfall']:.0f} মিমি) সরাসরি নদীর পানি ধারণক্ষমতা অতিক্রম করেছে।"
            })
        if hydro_stats['rolling3m'] > 900:
            drivers.append({
                'en': f"Prolonged 3-month monsoonal saturation ({hydro_stats['rolling3m']:.0f} mm) leaves ground permeability saturated.",
                'bn': f"পূর্ববর্তী ৩ মাসের তীব্র বৃষ্টিপাতে ({hydro_stats['rolling3m']:.0f} মিমি) মাটি ও জলাশয় সম্পূর্ণ সম্পৃক্ত।"
            })
        if hydro_stats['alt'] <= 8:
            drivers.append({
                'en': f"Low station elevation ({hydro_stats['alt']}m) elevates gravity-driven runoff pooling risk.",
                'bn': f"স্টেশনের নিচু ভূপ্রকৃতি ও উচ্চতা ({hydro_stats['alt']} মিটার) পানি নিষ্কাশনে ধীরগতি তৈরি করছে।"
            })
            
        if not drivers:
            drivers.append({
                'en': "Meteorological and topographic parameters remain within safe historical baseline thresholds.",
                'bn': "আবহাওয়া ও ভূপ্রকৃতির সকল সূচক ঐতিহাসিক নিরাপদ সীমার মধ্যে রয়েছে।"
            })

        # Operational Advisory
        if risk_tier in ['severe', 'high']:
            advisory_en = "Immediate flood precautions advised. Residents in low-lying char and haor areas should move livestock and essentials to high-ground flood shelters. Monitor BWDB water level bulletins."
            advisory_bn = "অবিলম্বে বন্যা সতর্কতামূলক ব্যবস্থা গ্রহণ করুন। নিম্নাঞ্চল, চর ও হাওর এলাকার বাসিন্দাদের গবাদিপশু ও জরুরি সামগ্রীসহ নিকটস্থ আশ্রয়কেন্দ্রে প্রস্তুত থাকতে অনুরোধ করা হচ্ছে। পানি উন্নয়ন বোর্ডের সার্বক্ষণিক বুলেটিন অনুসরণ করুন।"
        elif risk_tier == 'moderate':
            advisory_en = "Elevated soil moisture and catchment build-up observed. Agricultural drainage and embankment status should be actively inspected."
            advisory_bn = "মাটির আর্দ্রতা ও নদীর পানির স্তর বৃদ্ধি পাচ্ছে। ফসলি জমির নিষ্কাশন ব্যবস্থা এবং বেড়িবাঁধের নিরাপত্তা জোরদারভাবে পর্যবেক্ষণ করুন।"
        else:
            advisory_en = "Routine seasonal activity can proceed safely. No emergency interventions currently warranted."
            advisory_bn = "পরিস্থিতি স্বাভাবিক রয়েছে। স্বাভাবিক জীবনযাত্রা ও কৃষি কার্যক্রম নিরাপদে পরিচালনা করা যাবে।"

        return {
            'status': 'success',
            'station_name': data.get('station_name', 'Dhaka'),
            'month': data.get('month', 7),
            'probability': round(prob_flood * 100, 2),
            'probability_decimal': round(prob_flood, 4),
            'threshold_used': threshold,
            'is_flood_predicted': is_flood,
            'risk_tier': risk_tier,
            'badge_en': badge_en,
            'badge_bn': badge_bn,
            'badge_color': color,
            'drivers': drivers,
            'advisory_en': advisory_en,
            'advisory_bn': advisory_bn,
            'hydro_stats': hydro_stats,
            'model_name': self.model_type.upper(),
            'model_regime': self.model_regime
        }
