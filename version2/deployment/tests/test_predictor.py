import os
import sys
import unittest

# Ensure deployment directory is on sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
deployment_dir = os.path.abspath(os.path.join(current_dir, '..'))
if deployment_dir not in sys.path:
    sys.path.insert(0, deployment_dir)

from predictor import FloodPredictor
from app import app

class TestFloodDeployment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = FloodPredictor()
        cls.client = app.test_client()

    def test_station_count(self):
        stations = self.predictor.get_stations()
        self.assertEqual(len(stations), 33, f"Expected 33 stations, got {len(stations)}")

    def test_zero_dimension_mismatch(self):
        # Test across all 33 stations that prepared features match exact column length
        for st in self.predictor.get_stations():
            df_feat, _ = self.predictor.prepare_features({'station_name': st, 'rainfall': 300})
            self.assertEqual(df_feat.shape[1], len(self.predictor.feature_columns),
                             f"Dimension mismatch for station {st}: {df_feat.shape[1]} vs {len(self.predictor.feature_columns)}")

    def test_dry_winter_low_risk(self):
        # January dry winter in Dhaka
        res = self.predictor.predict({
            'station_name': 'Dhaka',
            'month': 1,
            'rainfall': 2.0,
            'rainfall_lag1': 0.0,
            'rainfall_lag2': 5.0,
            'relative_humidity': 62.0,
            'cloud_coverage': 1.0
        })
        self.assertFalse(res['is_flood_predicted'])
        self.assertEqual(res['risk_tier'], 'low')
        self.assertLess(res['probability'], 25.0)

    def test_extreme_monsoon_high_risk(self):
        # July peak deluge in Sylhet haor basin
        res = self.predictor.predict({
            'station_name': 'Sylhet',
            'month': 7,
            'rainfall': 850.0,
            'rainfall_lag1': 700.0,
            'rainfall_lag2': 600.0,
            'relative_humidity': 94.0,
            'cloud_coverage': 7.8
        })
        self.assertTrue(res['is_flood_predicted'])
        self.assertIn(res['risk_tier'], ['high', 'severe'])
        self.assertGreater(res['probability'], 75.0)

    def test_threshold_modes(self):
        sample = {
            'station_name': 'Comilla',
            'month': 8,
            'rainfall': 380.0,
            'rainfall_lag1': 350.0,
            'rainfall_lag2': 320.0
        }
        res_balanced = self.predictor.predict(sample, threshold_mode='balanced')
        res_alert = self.predictor.predict(sample, threshold_mode='alert')
        
        self.assertEqual(res_balanced['threshold_used'], 0.50)
        self.assertEqual(res_alert['threshold_used'], 0.35)

    def test_flask_endpoints(self):
        # 1. Root Dashboard
        resp_root = self.client.get('/')
        self.assertEqual(resp_root.status_code, 200)

        # 2. Health Endpoint
        resp_health = self.client.get('/api/health')
        self.assertEqual(resp_health.status_code, 200)
        data_health = resp_health.get_json()
        self.assertEqual(data_health['status'], 'healthy')

        # 3. Stations Metadata API
        resp_stations = self.client.get('/api/stations')
        self.assertEqual(resp_stations.status_code, 200)
        data_stations = resp_stations.get_json()
        self.assertEqual(len(data_stations), 33)

        # 4. Predict API
        resp_predict = self.client.post('/api/predict', json={
            'station_name': 'Cox\'s Bazar',
            'month': 6,
            'rainfall': 550.0
        })
        self.assertEqual(resp_predict.status_code, 200)
        data_predict = resp_predict.get_json()
        self.assertEqual(data_predict['status'], 'success')
        self.assertIn('probability', data_predict)

if __name__ == '__main__':
    unittest.main()
