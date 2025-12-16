# NovaPay Fraud Detection System

A professional machine learning-powered web application for detecting fraudulent transactions in real-time.

![Fraud Detection](https://img.shields.io/badge/ML-LightGBM-brightgreen) ![Accuracy](https://img.shields.io/badge/Accuracy-98%25-blue) ![Python](https://img.shields.io/badge/Python-3.8%2B-yellow) ![SHAP](https://img.shields.io/badge/Explainability-SHAP-orange)

## 🎯 Project Overview

This project provides an end-to-end fraud detection solution for NovaPay, a cross-border digital money transfer platform. The system uses a trained LightGBM classifier to analyze transaction features and predict the likelihood of fraudulent activity with SHAP-based explainability.

### Key Features
- **Real-time Fraud Prediction**: Analyze transactions instantly with 22 key features
- **SHAP Explainability**: Understand why the model made its prediction
- **Risk Level Assessment**: Get detailed risk levels (MINIMAL, LOW, MEDIUM, HIGH, CRITICAL)
- **Actionable Recommendations**: Receive specific actions for each risk level
- **Beautiful UI**: Modern, dark-themed interface with animated visualizations
- **98% Accuracy**: Trained model with high precision and F1-Score of 0.88

## 📁 Project Structure

```
novaPay_fraud_detection_model/
├── best_model/
│   └── lgb_model_with_thresholds.pkl   # Trained LightGBM model
├── cleaned data/
│   └── nova_master_df.csv              # Preprocessed dataset
├── raw data/
│   └── nova_pay_merged.csv             # Raw merged data
├── notebooks/
│   ├── NovaPay_Data_Cleaning.ipynb
│   ├── NovaPay_EDA.ipynb
│   └── NovaPay_Model_Training.ipynb
├── webapp/
│   ├── app.py                          # Flask backend with SHAP
│   ├── requirements.txt                # Python dependencies
│   ├── model/
│   │   └── lgb_model_with_thresholds.pkl  # Deployed model
│   ├── templates/
│   │   └── index.html                  # Main HTML template
│   └── static/
│       ├── styles.css                  # Styling
│       └── script.js                   # Frontend JavaScript
├── initial_nova_pay_merged.csv
└── README.md
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Navigate to the webapp directory:**
   ```bash
   cd webapp
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python app.py
   ```

4. **Open in browser:**
   Navigate to `http://localhost:5000`

## 🔍 Features Analyzed

The model analyzes 22 transaction features:

### Geographic & Currency
- Home Country (US, CA, UK)
- IP Country
- Source Currency (USD, CAD, GBP)
- Destination Currency (EUR, MXN, CNY, INR, GBP, PHP, NGN, etc.)

### Transaction Details
- Source Amount
- Amount in USD
- Transaction Fee
- Exchange Rate
- Transaction Channel (Mobile, Web, ATM)
- Corridor Risk Score

### Device & Security
- Device Trust Score
- IP Risk Score
- New Device Flag
- Location Mismatch

### Customer Profile
- KYC Tier (Standard, Enhanced, Low)
- Account Age (Days)
- Chargeback History Count
- Internal Risk Score

### Velocity & Timing
- Transaction Velocity (1 Hour)
- Transaction Velocity (24 Hours)
- Day of Week
- Time Period (Day, Evening, Night, Late Night)

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 98% |
| Precision | High |
| Recall | High |
| F1-Score | 0.88 |
| ROC AUC | High |

## 🎨 User Interface

The web application features:
- **Dark Neon Theme**: Professional dark mode with vibrant accent colors
- **Animated Background**: Floating orbs and gradient effects
- **Interactive Gauge**: Real-time fraud probability visualization
- **Risk Assessment Cards**: Color-coded risk levels with recommendations
- **Responsive Design**: Works on desktop and mobile devices

## 🛡️ Risk Levels

| Level | Probability | Action |
|-------|-------------|--------|
| CRITICAL | ≥70% | Block Transaction |
| HIGH | 50-70% | Hold for Review |
| MEDIUM | 30-50% | Enhanced Monitoring |
| LOW | 10-30% | Proceed with Caution |
| MINIMAL | <10% | Approve |

## 🔧 Technical Details

### Model: LightGBM Classifier
- **Training Method**: SMOTE for class imbalance handling
- **Best Threshold**: ~86% (optimized via ROC curve)
- **Features**: 22 engineered features
- **Preprocessing**: Label encoding + Robust/Standard scaling
- **Explainability**: SHAP (SHapley Additive exPlanations) integration

### Backend: Flask
- RESTful API endpoints
- JSON request/response
- Model inference with probability scores
- SHAP explanations for predictions

### Frontend: HTML/CSS/JavaScript
- Vanilla JavaScript (no frameworks)
- CSS animations and transitions
- Real-time form validation
- Interactive SHAP visualization

## 📝 API Endpoints

### `POST /predict`
Analyze a transaction for fraud.

**Request Body:**
```json
{
  "home_country": "US",
  "source_currency": "USD",
  "dest_currency": "EUR",
  "amount_src": 250.00,
  "channel": "MOBILE",
  ...
}
```

**Response:**
```json
{
  "success": true,
  "fraud_probability": 15.5,
  "is_fraud_prediction": false,
  "risk_level": "LOW",
  "risk_color": "#84cc16",
  "recommendation": {
    "action": "PROCEED WITH CAUTION",
    "details": "Low fraud indicators detected...",
    "icon": "✅"
  }
}
```

### `GET /api/options`
Get dropdown options for the form.

## 📈 Future Improvements

- [ ] Add user authentication
- [ ] Implement transaction history logging
- [ ] Add batch prediction capability
- [x] ~~Integrate SHAP for model explainability~~ ✅ Implemented
- [ ] Deploy with Docker container
- [ ] Add real-time model retraining pipeline

## 📄 License

This project is for educational and demonstration purposes.

## 👥 Contributors

NovaPay Data Science Team

---

**Built with ❤️ using Machine Learning and Flask**
