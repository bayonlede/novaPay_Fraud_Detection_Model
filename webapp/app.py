"""
NovaPay Fraud Detection Web Application
Flask backend for serving the ML model and handling predictions
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler

# Suppress sklearn version warnings
warnings.filterwarnings('ignore', category=UserWarning)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model
import os

def get_model_path():
    """Get the model path, checking multiple locations"""
    possible_paths = [
        os.path.join(os.path.dirname(__file__), 'model', 'rf_model_with_thresholds.pkl'),
        os.path.join(os.path.dirname(__file__), '..', 'best_model', 'rf_model_with_thresholds.pkl'),
        'model/rf_model_with_thresholds.pkl',
        '../best_model/rf_model_with_thresholds.pkl',
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return possible_paths[0]  # Default to first option

MODEL_PATH = get_model_path()

def load_model():
    """Load the trained Random Forest model with thresholds"""
    try:
        print(f"Loading model from: {MODEL_PATH}")
        with open(MODEL_PATH, 'rb') as f:
            model_package = pickle.load(f)
        print("Model loaded successfully!")
        return model_package
    except FileNotFoundError:
        print(f"Model file not found at {MODEL_PATH}")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

model_package = load_model()

# Define feature categories for encoding
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

# Feature order as expected by the model
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

# Scaling parameters (approximate from training data statistics)
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
    """
    Preprocess input data to match the model's expected format
    """
    processed = {}
    
    # Encode categorical features
    for feature, mapping in CATEGORICAL_MAPPINGS.items():
        if feature in data:
            value = data[feature]
            if feature == 'new_device':
                value = value == 'true' or value == True
            processed[feature] = mapping.get(value, 0)
    
    # Handle boolean features
    processed['location_mismatch'] = 1 if data.get('location_mismatch') in ['true', True, 1, '1'] else 0
    
    # Process numerical features
    numerical_features = ['amount_src', 'amount_usd', 'fee', 'exchange_rate_src_to_dest',
                         'ip_risk_score', 'account_age_days', 'device_trust_score',
                         'chargeback_history_count', 'risk_score_internal',
                         'txn_velocity_1h', 'txn_velocity_24h', 'corridor_risk']
    
    for feature in numerical_features:
        processed[feature] = float(data.get(feature, 0))
    
    # Apply Robust Scaling
    for feature, params in ROBUST_SCALE_PARAMS.items():
        if params['iqr'] != 0:
            processed[feature] = (processed[feature] - params['median']) / params['iqr']
        else:
            processed[feature] = 0
    
    # Apply Standard Scaling
    for feature, params in STANDARD_SCALE_PARAMS.items():
        if params['std'] != 0:
            processed[feature] = (processed[feature] - params['mean']) / params['std']
        else:
            processed[feature] = 0
    
    # Create feature array in correct order
    feature_array = [processed.get(f, 0) for f in FEATURE_ORDER]
    
    return np.array(feature_array).reshape(1, -1)


@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle fraud prediction requests"""
    try:
        data = request.json
        
        if data is None:
            return jsonify({
                'success': False,
                'error': 'No JSON data received'
            }), 400
        
        if model_package is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded. Please check if the model file exists.'
            }), 500
        
        # Preprocess input
        features = preprocess_input(data)
        
        # Get model and thresholds
        model = model_package['model']
        best_threshold = model_package.get('best_threshold', 0.5)
        default_threshold = model_package.get('default_threshold', 0.5)
        
        # Get prediction probability
        fraud_probability = float(model.predict_proba(features)[0][1])
        
        # Apply best threshold for prediction
        is_fraud_best = fraud_probability >= best_threshold
        is_fraud_default = fraud_probability >= default_threshold
        
        # Determine risk level
        if fraud_probability >= 0.7:
            risk_level = 'CRITICAL'
            risk_color = '#dc2626'
        elif fraud_probability >= 0.5:
            risk_level = 'HIGH'
            risk_color = '#ea580c'
        elif fraud_probability >= 0.3:
            risk_level = 'MEDIUM'
            risk_color = '#f59e0b'
        elif fraud_probability >= 0.1:
            risk_level = 'LOW'
            risk_color = '#84cc16'
        else:
            risk_level = 'MINIMAL'
            risk_color = '#22c55e'
        
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
            'recommendation': get_recommendation(risk_level, fraud_probability)
        })
        
    except Exception as e:
        import traceback
        print(f"Error in predict: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


def get_recommendation(risk_level, probability):
    """Generate recommendation based on risk level"""
    recommendations = {
        'CRITICAL': {
            'action': 'BLOCK TRANSACTION',
            'details': 'This transaction shows extremely high fraud indicators. Immediate blocking is recommended. Flag for manual review and contact customer for verification.',
            'icon': '🚫'
        },
        'HIGH': {
            'action': 'HOLD FOR REVIEW',
            'details': 'Transaction flagged with high fraud probability. Place on hold and require additional verification before processing.',
            'icon': '⚠️'
        },
        'MEDIUM': {
            'action': 'ENHANCED MONITORING',
            'details': 'Transaction shows moderate risk signals. Proceed with enhanced monitoring and log for pattern analysis.',
            'icon': '👁️'
        },
        'LOW': {
            'action': 'PROCEED WITH CAUTION',
            'details': 'Low fraud indicators detected. Transaction can proceed but include in routine monitoring reports.',
            'icon': '✅'
        },
        'MINIMAL': {
            'action': 'APPROVE',
            'details': 'Transaction appears legitimate with minimal risk indicators. Safe to process normally.',
            'icon': '✨'
        }
    }
    return recommendations.get(risk_level, recommendations['MEDIUM'])


@app.route('/api/options', methods=['GET'])
def get_options():
    """Return dropdown options for the frontend"""
    return jsonify({
        'home_countries': ['US', 'CA', 'UK'],
        'source_currencies': ['USD', 'CAD', 'GBP'],
        'dest_currencies': ['USD', 'CAD', 'GBP', 'EUR', 'MXN', 'CNY', 'INR', 'PHP', 'NGN'],
        'channels': ['ATM', 'WEB', 'MOBILE'],
        'ip_countries': ['US', 'CA', 'UK'],
        'kyc_tiers': ['STANDARD', 'ENHANCED', 'LOW'],
        'days': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
        'periods': ['Day', 'Evening', 'Night', 'Late Night']
    })


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(debug=debug, host='0.0.0.0', port=port)


