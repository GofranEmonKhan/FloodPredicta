import os
import json
from flask import Flask, render_template, request, jsonify
from predictor import FloodPredictor

app = Flask(__name__)

# Initialize Predictor
# Using XGBoost Champion from Chronological 70-20-10 regime (high recall & fast sub-millisecond inference)
predictor = FloodPredictor(model_regime='split_chronological_70_20_10', model_type='xgboost')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/stations', methods=['GET'])
def get_stations():
    return jsonify(predictor.station_meta)

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        threshold_mode = data.get('threshold_mode', 'balanced')
        result = predictor.predict(data, threshold_mode=threshold_mode)
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'service': 'FloodPredicta Early Warning API',
        'version': '2.0.0',
        'stations_loaded': len(predictor.get_stations()),
        'model': predictor.model_type,
        'regime': predictor.model_regime
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🌊 FloodPredicta Web Application running at: http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
