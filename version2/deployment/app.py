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

@app.after_request
def add_cache_headers(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

@app.route('/favicon.ico')
def favicon():
    return ('', 204)

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

def serve_port(p):
    try:
        from werkzeug.serving import run_simple
        run_simple('0.0.0.0', p, app, threaded=True, use_reloader=False)
    except Exception as e:
        print(f"Server on port {p} error: {e}")

if __name__ == '__main__':
    import threading
    default_port = int(os.environ.get('PORT', 5000))
    secondary_port = 8080 if default_port == 5000 else 5000
    
    print(f"🌊 FloodPredicta Web Application running at:")
    print(f"   • Primary:   http://localhost:{default_port}  (or http://127.0.0.1:{default_port})")
    print(f"   • Secondary: http://localhost:{secondary_port}  (or http://127.0.0.1:{secondary_port})")
    
    # Run secondary port on background thread
    t = threading.Thread(target=serve_port, args=(secondary_port,), daemon=True)
    t.start()
    
    # Run primary port on main thread
    serve_port(default_port)
