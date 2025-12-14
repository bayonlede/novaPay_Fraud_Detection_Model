"""
NovaPay Fraud Detection Web Application
Flask backend for serving the ML model and handling predictions
"""

import os
import pickle
import warnings
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

# Suppress sklearn version warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Get model path
def get_model_path():
    """Get the model path, checking multiple locations"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        os.path.join(base_dir, 'model', 'rf_model_with_thresholds.pkl'),
        os.path.join(base_dir, '..', 'best_model', 'rf_model_with_thresholds.pkl'),
        'model/rf_model_with_thresholds.pkl',
    ]
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Found model at: {path}")
            return path
    print(f"Model not found. Checked: {possible_paths}")
    return possible_paths[0]

MODEL_PATH = get_model_path()

def load_model():
    """Load the trained Random Forest model with thresholds"""
    try:
        print(f"Loading model from: {MODEL_PATH}")
        with open(MODEL_PATH, 'rb') as f:
            model_package = pickle.load(f)
        print("Model loaded successfully!")
        return model_package
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

model_package = load_model()

# Feature mappings
CATEGORICAL_MAPPINGS = {
    'home_country': {'CA': 0, 'UK': 1, 'US': 2},
    'source_currency': {'CAD': 0, 'GBP': 1, 'USD': 2},
    'dest_currency': {'CAD': 0, 'CNY': 1, 'EUR': 2, 'GBP': 3, 'INR': 4, 'MXN': 5, 'NGN': 6, 'PHP': 7, 'USD': 8},
    'channel': {'ATM': 0, 'MOBILE': 1, 'WEB': 2},
    'ip_country': {'CA': 0, 'UK': 1, 'US': 2},
    'kyc_tier': {'ENHANCED': 0, 'LOW': 1, 'STANDARD': 2},
    'new_device': {False: 0, True: 1},
    'days_only': {'Friday': 0, 'Monday': 1, 'Saturday': 2, 'Sunday': 3, 'Thursday': 4, 'Tuesday': 5, 'Wednesday': 6},
    'period_of_the_day': {'Day': 0, 'Evening': 1, 'Late Night': 2, 'Night': 3},
    'fee_bracket': {'high risk': 0, 'no risk': 1},
    'ip_risk_score_bracket': {'high risk': 0, 'no risk': 1},
    'device_trust_bucket': {'high risk': 0, 'no risk': 1}
}

FEATURE_ORDER = [
    'home_country', 'source_currency', 'dest_currency', 'channel',
    'amount_src', 'amount_usd', 'fee', 'exchange_rate_src_to_dest',
    'new_device', 'ip_country', 'location_mismatch', 'ip_risk_score',
    'kyc_tier', 'account_age_days', 'device_trust_score',
    'chargeback_history_count', 'risk_score_internal', 'txn_velocity_1h',
    'txn_velocity_24h', 'corridor_risk', 'days_only',
    'period_of_the_day', 'fee_bracket', 'ip_risk_score_bracket',
    'device_trust_bucket'
]

ROBUST_SCALE_PARAMS = {
    'amount_src': {'median': 200.0, 'iqr': 300.0},
    'amount_usd': {'median': 180.0, 'iqr': 280.0},
    'fee': {'median': 4.0, 'iqr': 5.0},
    'exchange_rate_src_to_dest': {'median': 1.0, 'iqr': 10.0},
    'chargeback_history_count': {'median': 0.0, 'iqr': 1.0}
}

STANDARD_SCALE_PARAMS = {
    'ip_risk_score': {'mean': 0.5, 'std': 0.25},
    'account_age_days': {'mean': 500.0, 'std': 300.0},
    'device_trust_score': {'mean': 0.65, 'std': 0.25}
}


def preprocess_input(data):
    """Preprocess input data to match the model's expected format"""
    processed = {}
    
    for feature, mapping in CATEGORICAL_MAPPINGS.items():
        if feature in data:
            value = data[feature]
            if feature == 'new_device':
                value = value == 'true' or value == True
            processed[feature] = mapping.get(value, 0)
    
    processed['location_mismatch'] = 1 if data.get('location_mismatch') in ['true', True, 1, '1'] else 0
    
    numerical_features = ['amount_src', 'amount_usd', 'fee', 'exchange_rate_src_to_dest',
                         'ip_risk_score', 'account_age_days', 'device_trust_score',
                         'chargeback_history_count', 'risk_score_internal',
                         'txn_velocity_1h', 'txn_velocity_24h', 'corridor_risk']
    
    for feature in numerical_features:
        processed[feature] = float(data.get(feature, 0))
    
    for feature, params in ROBUST_SCALE_PARAMS.items():
        if params['iqr'] != 0:
            processed[feature] = (processed[feature] - params['median']) / params['iqr']
    
    for feature, params in STANDARD_SCALE_PARAMS.items():
        if params['std'] != 0:
            processed[feature] = (processed[feature] - params['mean']) / params['std']
    
    feature_array = [processed.get(f, 0) for f in FEATURE_ORDER]
    return np.array(feature_array).reshape(1, -1)


@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_package is not None
    })


@app.route('/predict', methods=['POST'])
def predict():
    """Handle fraud prediction requests"""
    try:
        data = request.json
        
        if data is None:
            return jsonify({'success': False, 'error': 'No JSON data received'}), 400
        
        if model_package is None:
            return jsonify({'success': False, 'error': 'Model not loaded'}), 500
        
        features = preprocess_input(data)
        model = model_package['model']
        best_threshold = model_package.get('best_threshold', 0.5)
        default_threshold = model_package.get('default_threshold', 0.5)
        
        fraud_probability = float(model.predict_proba(features)[0][1])
        is_fraud_best = fraud_probability >= best_threshold
        
        if fraud_probability >= 0.7:
            risk_level, risk_color = 'CRITICAL', '#dc2626'
        elif fraud_probability >= 0.5:
            risk_level, risk_color = 'HIGH', '#ea580c'
        elif fraud_probability >= 0.3:
            risk_level, risk_color = 'MEDIUM', '#f59e0b'
        elif fraud_probability >= 0.1:
            risk_level, risk_color = 'LOW', '#84cc16'
        else:
            risk_level, risk_color = 'MINIMAL', '#22c55e'
        
        recommendations = {
            'CRITICAL': {'action': 'BLOCK TRANSACTION', 'details': 'Extremely high fraud indicators. Block immediately.', 'icon': '🚫'},
            'HIGH': {'action': 'HOLD FOR REVIEW', 'details': 'High fraud probability. Require additional verification.', 'icon': '⚠️'},
            'MEDIUM': {'action': 'ENHANCED MONITORING', 'details': 'Moderate risk signals. Proceed with monitoring.', 'icon': '👁️'},
            'LOW': {'action': 'PROCEED WITH CAUTION', 'details': 'Low fraud indicators. Include in routine monitoring.', 'icon': '✅'},
            'MINIMAL': {'action': 'APPROVE', 'details': 'Transaction appears legitimate. Safe to process.', 'icon': '✨'}
        }
        
        return jsonify({
            'success': True,
            'fraud_probability': round(fraud_probability * 100, 2),
            'is_fraud_prediction': bool(is_fraud_best),
            'risk_level': risk_level,
            'risk_color': risk_color,
            'thresholds': {
                'best': round(best_threshold * 100, 2),
                'default': round(default_threshold * 100, 2)
            },
            'recommendation': recommendations.get(risk_level)
        })
        
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 400


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
