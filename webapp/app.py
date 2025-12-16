"""
NovaPay Fraud Detection Web Application
Flask backend for serving the LightGBM model with SHAP explanations
"""

import os
import pickle
import warnings
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

# Suppress sklearn version warnings
warnings.filterwarnings('ignore')

# Try to import shap, but make it optional
try:
    import shap
    SHAP_AVAILABLE = True
    print("SHAP library loaded successfully")
except ImportError as e:
    SHAP_AVAILABLE = False
    print(f"SHAP library not available: {e}")
    shap = None

app = Flask(__name__)
CORS(app)

# Get model path
def get_model_path():
    """Get the model path, checking multiple locations for LightGBM first, then Random Forest as fallback"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Try LightGBM model first
    lgb_paths = [
        os.path.join(base_dir, 'model', 'lgb_model_with_thresholds.pkl'),
        os.path.join(base_dir, '..', 'best_model', 'lgb_model_with_thresholds.pkl'),
        'model/lgb_model_with_thresholds.pkl',
    ]
    
    for path in lgb_paths:
        if os.path.exists(path):
            print(f"Found LightGBM model at: {path}")
            return path, 'LightGBM'
    
    # Fallback to Random Forest model
    rf_paths = [
        os.path.join(base_dir, 'model', 'rf_model_with_thresholds.pkl'),
        os.path.join(base_dir, '..', 'best_model', 'rf_model_with_thresholds.pkl'),
        'model/rf_model_with_thresholds.pkl',
    ]
    
    for path in rf_paths:
        if os.path.exists(path):
            print(f"Found Random Forest model at: {path}")
            return path, 'RandomForest'
    
    print(f"No model found. Checked LightGBM paths: {lgb_paths}")
    print(f"Checked Random Forest paths: {rf_paths}")
    return lgb_paths[0], 'LightGBM'

MODEL_PATH, MODEL_TYPE = get_model_path()

# Global variables for model and explainer
model_package = None
shap_explainer = None

def load_model():
    """Load the trained LightGBM model with thresholds"""
    global shap_explainer
    try:
        print(f"Loading model from: {MODEL_PATH}")
        print(f"Model file exists: {os.path.exists(MODEL_PATH)}")
        
        if not os.path.exists(MODEL_PATH):
            print(f"ERROR: Model file not found at {MODEL_PATH}")
            # List available files in model directory
            model_dir = os.path.dirname(MODEL_PATH)
            if os.path.exists(model_dir):
                print(f"Files in {model_dir}: {os.listdir(model_dir)}")
            return None
        
        with open(MODEL_PATH, 'rb') as f:
            loaded_package = pickle.load(f)
        print("Model loaded successfully!")
        print(f"Model package keys: {loaded_package.keys() if isinstance(loaded_package, dict) else 'Not a dict'}")
        
        # Initialize SHAP explainer for LightGBM
        model = loaded_package['model']
        print(f"Model type: {type(model)}")
        
        if SHAP_AVAILABLE:
            try:
                shap_explainer = shap.TreeExplainer(model)
                print("SHAP explainer initialized successfully!")
            except Exception as shap_error:
                print(f"Warning: Could not initialize SHAP explainer: {shap_error}")
                shap_explainer = None
        else:
            print("SHAP not available, skipping explainer initialization")
            shap_explainer = None
        
        return loaded_package
    except Exception as e:
        import traceback
        print(f"Error loading model: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return None

# Print startup diagnostics
print("=" * 50)
print("NovaPay Fraud Detection - Starting Up")
print("=" * 50)
print(f"Working directory: {os.getcwd()}")
print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
print(f"Model path: {MODEL_PATH}")
print(f"Model type: {MODEL_TYPE}")
print(f"SHAP available: {SHAP_AVAILABLE}")

# List model directory contents
model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')
if os.path.exists(model_dir):
    print(f"Model directory contents: {os.listdir(model_dir)}")
else:
    print(f"Model directory does not exist: {model_dir}")

model_package = load_model()
print(f"Model loaded: {model_package is not None}")
print("=" * 50)

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
    'period_of_the_day': {'Day': 0, 'Evening': 1, 'Late Night': 2, 'Night': 3}
}

FEATURE_ORDER = [
    'home_country', 'source_currency', 'dest_currency', 'channel',
    'amount_src', 'amount_usd', 'fee', 'exchange_rate_src_to_dest',
    'new_device', 'ip_country', 'location_mismatch', 'ip_risk_score',
    'kyc_tier', 'account_age_days', 'device_trust_score',
    'chargeback_history_count', 'risk_score_internal', 'txn_velocity_1h',
    'txn_velocity_24h', 'corridor_risk', 'days_only',
    'period_of_the_day'
]

# Human-readable feature names for SHAP explanations
FEATURE_DISPLAY_NAMES = {
    'home_country': 'Home Country',
    'source_currency': 'Source Currency',
    'dest_currency': 'Destination Currency',
    'channel': 'Transaction Channel',
    'amount_src': 'Source Amount',
    'amount_usd': 'Amount (USD)',
    'fee': 'Transaction Fee',
    'exchange_rate_src_to_dest': 'Exchange Rate',
    'new_device': 'New Device',
    'ip_country': 'IP Country',
    'location_mismatch': 'Location Mismatch',
    'ip_risk_score': 'IP Risk Score',
    'kyc_tier': 'KYC Tier',
    'account_age_days': 'Account Age',
    'device_trust_score': 'Device Trust Score',
    'chargeback_history_count': 'Chargeback History',
    'risk_score_internal': 'Internal Risk Score',
    'txn_velocity_1h': 'Txns (1 Hour)',
    'txn_velocity_24h': 'Txns (24 Hours)',
    'corridor_risk': 'Corridor Risk',
    'days_only': 'Day of Week',
    'period_of_the_day': 'Time Period'
}

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


def get_shap_explanations(features, top_n=5):
    """Get SHAP explanations for a prediction"""
    global shap_explainer
    
    if not SHAP_AVAILABLE or shap_explainer is None:
        return []
    
    try:
        # Get SHAP values for the prediction
        shap_values = shap_explainer.shap_values(features)
        
        # Handle the case where SHAP returns a list (for binary classification)
        if isinstance(shap_values, list):
            # Use the positive class (fraud) SHAP values
            shap_vals = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        else:
            shap_vals = shap_values[0]
        
        # Create feature-impact pairs
        feature_impacts = []
        for i, feature_name in enumerate(FEATURE_ORDER):
            impact = float(shap_vals[i])
            display_name = FEATURE_DISPLAY_NAMES.get(feature_name, feature_name)
            feature_impacts.append({
                'feature': display_name,
                'feature_key': feature_name,
                'impact': impact,
                'abs_impact': abs(impact),
                'direction': 'increases' if impact > 0 else 'decreases'
            })
        
        # Sort by absolute impact and get top N
        feature_impacts.sort(key=lambda x: x['abs_impact'], reverse=True)
        top_explanations = feature_impacts[:top_n]
        
        return top_explanations
    except Exception as e:
        print(f"Error getting SHAP explanations: {e}")
        return []


@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_package is not None,
        'shap_available': SHAP_AVAILABLE,
        'shap_enabled': shap_explainer is not None,
        'model_type': MODEL_TYPE,
        'model_path': MODEL_PATH
    })


@app.route('/predict', methods=['POST'])
def predict():
    """Handle fraud prediction requests with SHAP explanations"""
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
        
        # Get SHAP explanations
        shap_explanations = get_shap_explanations(features, top_n=5)
        
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
            'recommendation': recommendations.get(risk_level),
            'shap_explanations': shap_explanations
        })
        
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 400


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
