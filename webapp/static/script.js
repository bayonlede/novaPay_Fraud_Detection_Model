/**
 * NovaPay Fraud Detection System
 * Frontend JavaScript
 */

document.addEventListener('DOMContentLoaded', function() {
    initializeForm();
    initializeSliders();
    addFormAnimations();
});

/**
 * Initialize the fraud detection form
 */
function initializeForm() {
    const form = document.getElementById('fraud-form');
    const submitBtn = document.getElementById('analyze-btn');
    
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Set loading state
        submitBtn.classList.add('loading');
        submitBtn.disabled = true;
        
        try {
            const formData = collectFormData();
            const result = await analyzeFraud(formData);
            displayResults(result);
        } catch (error) {
            console.error('Error:', error);
            displayError(error.message);
        } finally {
            submitBtn.classList.remove('loading');
            submitBtn.disabled = false;
        }
    });
}

/**
 * Collect all form data
 */
function collectFormData() {
    const form = document.getElementById('fraud-form');
    const data = {};
    
    // Get all form elements
    const inputs = form.querySelectorAll('input, select');
    
    inputs.forEach(input => {
        if (input.type === 'checkbox') {
            data[input.name] = input.checked;
        } else if (input.type === 'range' || input.type === 'number') {
            data[input.name] = parseFloat(input.value) || 0;
        } else {
            data[input.name] = input.value;
        }
    });
    
    return data;
}

/**
 * Send data to the API for fraud analysis
 */
async function analyzeFraud(data) {
    const response = await fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    });
    
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Failed to analyze transaction');
    }
    
    return await response.json();
}

/**
 * Display results in the results panel
 */
function displayResults(result) {
    const placeholder = document.getElementById('placeholder');
    const resultsContent = document.getElementById('results-content');
    
    // Hide placeholder, show results
    placeholder.style.display = 'none';
    resultsContent.style.display = 'flex';
    
    // Animate the gauge
    animateGauge(result.fraud_probability);
    
    // Update risk label
    const riskLabel = document.getElementById('risk-label');
    riskLabel.textContent = result.risk_level;
    riskLabel.style.color = result.risk_color;
    riskLabel.style.borderColor = result.risk_color;
    
    // Update prediction card
    const predictionCard = document.getElementById('prediction-card');
    const predictionIcon = document.getElementById('prediction-icon');
    const predictionTitle = document.getElementById('prediction-title');
    const predictionAction = document.getElementById('prediction-action');
    const predictionDetails = document.getElementById('prediction-details');
    
    predictionCard.className = 'prediction-card ' + (result.is_fraud_prediction ? 'fraud' : 'safe');
    predictionIcon.textContent = result.recommendation.icon;
    predictionTitle.textContent = result.is_fraud_prediction ? 'Fraud Detected' : 'Transaction Safe';
    predictionAction.textContent = result.recommendation.action;
    predictionDetails.textContent = result.recommendation.details;
    
    // Update metrics
    document.getElementById('fraud-probability').textContent = result.fraud_probability + '%';
    document.getElementById('best-threshold').textContent = result.thresholds.best + '%';
    document.getElementById('default-threshold').textContent = result.thresholds.default + '%';
    document.getElementById('risk-level-value').textContent = result.risk_level;
    document.getElementById('risk-level-value').style.color = result.risk_color;
    
    // Add animation class
    resultsContent.classList.add('slide-up');
    setTimeout(() => resultsContent.classList.remove('slide-up'), 600);
}

/**
 * Animate the risk gauge
 */
function animateGauge(probability) {
    const gaugeValue = document.getElementById('gauge-value');
    const gaugeFill = document.querySelector('.gauge-fill');
    const gaugeNeedle = document.querySelector('.gauge-needle');
    
    // Calculate stroke offset (251.2 is the full arc length)
    const maxOffset = 251.2;
    const offset = maxOffset - (probability / 100) * maxOffset;
    
    // Animate the fill
    setTimeout(() => {
        gaugeFill.style.strokeDashoffset = offset;
    }, 100);
    
    // Animate the value counter
    animateCounter(gaugeValue, 0, probability, 1500);
    
    // Calculate needle position (angle from -90 to 90 degrees)
    const angle = -90 + (probability / 100) * 180;
    const radians = (angle * Math.PI) / 180;
    const needleLength = 70;
    const cx = 100;
    const cy = 100;
    const nx = cx + needleLength * Math.cos(radians);
    const ny = cy + needleLength * Math.sin(radians);
    
    setTimeout(() => {
        gaugeNeedle.setAttribute('cx', nx);
        gaugeNeedle.setAttribute('cy', ny);
    }, 100);
}

/**
 * Animate a counter from start to end value
 */
function animateCounter(element, start, end, duration) {
    const startTime = performance.now();
    const range = end - start;
    
    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        // Ease out cubic
        const easeProgress = 1 - Math.pow(1 - progress, 3);
        const current = start + range * easeProgress;
        
        element.textContent = current.toFixed(1) + '%';
        
        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }
    
    requestAnimationFrame(update);
}

/**
 * Display an error message
 */
function displayError(message) {
    const placeholder = document.getElementById('placeholder');
    const resultsContent = document.getElementById('results-content');
    
    placeholder.style.display = 'flex';
    resultsContent.style.display = 'none';
    
    placeholder.innerHTML = `
        <div class="placeholder-icon" style="color: var(--danger);">
            <svg viewBox="0 0 80 80" fill="none" stroke="currentColor" stroke-width="1.5">
                <circle cx="40" cy="40" r="35" opacity="0.3"/>
                <path d="M28 28L52 52M52 28L28 52" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
        </div>
        <h3 style="color: var(--danger);">Error</h3>
        <p>${message}</p>
    `;
}

/**
 * Initialize slider value displays
 */
function initializeSliders() {
    const sliders = document.querySelectorAll('input[type="range"]');
    
    sliders.forEach(slider => {
        const valueDisplay = slider.parentElement.querySelector('.slider-value');
        
        // Set initial value
        updateSliderValue(slider, valueDisplay);
        
        // Update on change
        slider.addEventListener('input', () => {
            updateSliderValue(slider, valueDisplay);
        });
    });
}

/**
 * Update slider value display
 */
function updateSliderValue(slider, display) {
    const value = parseFloat(slider.value);
    display.textContent = value.toFixed(2);
    
    // Update slider track color
    const percent = ((value - slider.min) / (slider.max - slider.min)) * 100;
    slider.style.background = `linear-gradient(to right, var(--accent-primary) 0%, var(--accent-primary) ${percent}%, var(--bg-input) ${percent}%, var(--bg-input) 100%)`;
}

/**
 * Add form field animations
 */
function addFormAnimations() {
    const formGroups = document.querySelectorAll('.form-group');
    
    formGroups.forEach((group, index) => {
        group.style.opacity = '0';
        group.style.transform = 'translateY(10px)';
        group.style.transition = 'all 0.3s ease';
        
        setTimeout(() => {
            group.style.opacity = '1';
            group.style.transform = 'translateY(0)';
        }, 50 + index * 30);
    });
}

/**
 * Auto-sync amount fields based on currency
 */
document.addEventListener('DOMContentLoaded', function() {
    const amountSrc = document.getElementById('amount_src');
    const amountUsd = document.getElementById('amount_usd');
    const sourceCurrency = document.getElementById('source_currency');
    
    // Simple exchange rates for demo
    const rates = {
        'USD': 1,
        'CAD': 0.74,
        'GBP': 1.27
    };
    
    function updateUsdAmount() {
        const amount = parseFloat(amountSrc.value) || 0;
        const currency = sourceCurrency.value;
        const rate = rates[currency] || 1;
        amountUsd.value = (amount * rate).toFixed(2);
    }
    
    amountSrc.addEventListener('input', updateUsdAmount);
    sourceCurrency.addEventListener('change', updateUsdAmount);
});

/**
 * Auto-detect location mismatch
 */
document.addEventListener('DOMContentLoaded', function() {
    const homeCountry = document.getElementById('home_country');
    const ipCountry = document.getElementById('ip_country');
    const locationMismatch = document.getElementById('location_mismatch');
    
    function checkLocationMismatch() {
        const mismatch = homeCountry.value !== ipCountry.value;
        locationMismatch.checked = mismatch;
    }
    
    homeCountry.addEventListener('change', checkLocationMismatch);
    ipCountry.addEventListener('change', checkLocationMismatch);
});

/**
 * Quick test scenarios
 */
window.loadTestScenario = function(scenario) {
    const scenarios = {
        safe: {
            home_country: 'US',
            ip_country: 'US',
            source_currency: 'USD',
            dest_currency: 'EUR',
            amount_src: 150,
            fee: 3.5,
            exchange_rate_src_to_dest: 0.92,
            channel: 'MOBILE',
            corridor_risk: 0.05,
            device_trust_score: 0.85,
            ip_risk_score: 0.15,
            new_device: false,
            location_mismatch: false,
            kyc_tier: 'ENHANCED',
            account_age_days: 730,
            chargeback_history_count: 0,
            risk_score_internal: 0.1,
            txn_velocity_1h: 0,
            txn_velocity_24h: 1,
            days_only: 'Wednesday',
            period_of_the_day: 'Day'
        },
        suspicious: {
            home_country: 'US',
            ip_country: 'UK',
            source_currency: 'USD',
            dest_currency: 'NGN',
            amount_src: 5000,
            fee: 150,
            exchange_rate_src_to_dest: 1600,
            channel: 'WEB',
            corridor_risk: 0.8,
            device_trust_score: 0.25,
            ip_risk_score: 0.85,
            new_device: true,
            location_mismatch: true,
            kyc_tier: 'LOW',
            account_age_days: 15,
            chargeback_history_count: 2,
            risk_score_internal: 0.75,
            txn_velocity_1h: 5,
            txn_velocity_24h: 12,
            days_only: 'Saturday',
            period_of_the_day: 'Late Night'
        }
    };
    
    const data = scenarios[scenario];
    if (data) {
        Object.keys(data).forEach(key => {
            const element = document.getElementById(key);
            if (element) {
                if (element.type === 'checkbox') {
                    element.checked = data[key];
                } else if (element.type === 'range') {
                    element.value = data[key];
                    const valueDisplay = element.parentElement.querySelector('.slider-value');
                    if (valueDisplay) {
                        updateSliderValue(element, valueDisplay);
                    }
                } else {
                    element.value = data[key];
                }
            }
        });
    }
};


