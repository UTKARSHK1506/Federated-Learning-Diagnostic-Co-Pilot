/**
 * Clinical Diagnostic Co-Pilot - Client Script
 * Dual-Mode Inference Engine (Online FastAPI REST + Offline Local FNN Matrix Engine)
 * Data-Grounded Explainability & Clinical Decision Support
 */

const API_BASE_URL = window.location.origin.includes('8000') 
    ? window.location.origin 
    : 'http://127.0.0.1:8000';

let currentInferenceMode = 'online'; // 'online' or 'offline'
let offlineWeights = null;
let offlineScaler = null;

// ============================================================================
// Core Abstraction: InferenceRepository (Preserved Logic)
// ============================================================================
class InferenceRepository {
    static async predict(payload) {
        if (currentInferenceMode === 'offline') {
            return await InferenceRepository.runOfflineInference(payload);
        } else {
            return await InferenceRepository.runOnlineInference(payload);
        }
    }

    // 1. ONLINE MODE: Fetch from FastAPI backend
    static async runOnlineInference(payload) {
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        if (!response.ok) {
            throw new Error(`Server returned status ${response.status}`);
        }
        const res = await response.json();
        return {
            ...res,
            engine_origin: 'FastAPI Server (Online)'
        };
    }

    // 2. OFFLINE MODE: Local Standalone Inference Engine (0ms network dependency)
    static async runOfflineInference(payload) {
        if (!offlineWeights || !offlineScaler) {
            await InferenceRepository.loadOfflineAssets();
        }

        // Feature order: ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
        const featureNames = offlineScaler.feature_columns;
        const continuousCols = offlineScaler.continuous_columns;
        const mean = offlineScaler.mean;
        const scale = offlineScaler.scale;

        // Local Standardization for continuous features: (x - mean) / scale
        const x_input = new Array(featureNames.length);
        
        for (let i = 0; i < featureNames.length; i++) {
            const fname = featureNames[i];
            const rawVal = payload[fname];
            const contIndex = continuousCols.indexOf(fname);

            if (contIndex !== -1) {
                x_input[i] = (rawVal - mean[contIndex]) / scale[contIndex];
            } else {
                x_input[i] = rawVal;
            }
        }

        // Forward Pass: Layer 1 (11 -> 64 with ReLU)
        const fc1_w = offlineWeights.fc1_w; // shape (64, 11)
        const fc1_b = offlineWeights.fc1_b; // shape (64)
        const h1 = new Array(64);
        for (let i = 0; i < 64; i++) {
            let sum = fc1_b[i];
            for (let j = 0; j < 11; j++) {
                sum += fc1_w[i][j] * x_input[j];
            }
            h1[i] = Math.max(0.0, sum); // ReLU activation
        }

        // Forward Pass: Layer 2 (64 -> 32 with ReLU)
        const fc2_w = offlineWeights.fc2_w; // shape (32, 64)
        const fc2_b = offlineWeights.fc2_b; // shape (32)
        const h2 = new Array(32);
        for (let i = 0; i < 32; i++) {
            let sum = fc2_b[i];
            for (let j = 0; j < 64; j++) {
                sum += fc2_w[i][j] * h1[j];
            }
            h2[i] = Math.max(0.0, sum); // ReLU activation
        }

        // Forward Pass: Layer 3 (32 -> 1 raw logit)
        const fc3_w = offlineWeights.fc3_w; // shape (1, 32)
        const fc3_b = offlineWeights.fc3_b; // shape (1)
        let logit = fc3_b[0];
        for (let j = 0; j < 32; j++) {
            logit += fc3_w[0][j] * h2[j];
        }

        // Sigmoid Output Probability
        const probability = 1.0 / (1.0 + Math.exp(-logit));
        const predicted_class = probability >= 0.5 ? 1 : 0;

        return {
            predicted_class: predicted_class,
            probability: parseFloat(probability.toFixed(4)),
            model_type: "Federated FNN (Offline)",
            engine_origin: "Local On-Device Engine (Offline)"
        };
    }

    static async loadOfflineAssets() {
        const pathCandidates = [
            ['global_model_weights.json', 'scaler_params.json'],
            ['static/global_model_weights.json', 'static/scaler_params.json'],
            ['/static/global_model_weights.json', '/static/scaler_params.json']
        ];

        for (const [wPath, sPath] of pathCandidates) {
            try {
                const [wRes, sRes] = await Promise.all([fetch(wPath), fetch(sPath)]);
                if (wRes.ok && sRes.ok) {
                    offlineWeights = await wRes.json();
                    offlineScaler = await sRes.json();
                    console.log(`InferenceRepository: Offline assets loaded from [${wPath}, ${sPath}]`);
                    return;
                }
            } catch (e) {
                // Continue to next candidate
            }
        }
        console.warn("InferenceRepository: Could not pre-fetch offline assets automatically. Will retry upon offline inference trigger.");
    }
}

// ============================================================================
// Navigation & Mode Management
// ============================================================================
function showScreen(screenId) {
    document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
    const target = document.getElementById(screenId);
    if (target) {
        target.classList.add('active');
    }

    // Update nav tab highlights
    document.querySelectorAll('.nav-tab').forEach(tab => tab.classList.remove('active'));
    if (screenId === 'screen-home') {
        const homeTab = document.getElementById('tab-home');
        if (homeTab) homeTab.classList.add('active');
    } else if (screenId === 'screen-assessment') {
        const assessTab = document.getElementById('tab-assessment');
        if (assessTab) assessTab.classList.add('active');
    } else if (screenId === 'screen-federated') {
        const flTab = document.getElementById('tab-federated');
        if (flTab) flTab.classList.add('active');
    }

    // Clear any previous validation alerts when navigating
    hideValidationAlert();
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

function handleModeChange(mode) {
    currentInferenceMode = mode;
    
    const headerStatus = document.getElementById('header-status');
    const statusDot = document.getElementById('status-dot');
    const statusLabel = document.getElementById('status-label');
    const modeSelect = document.getElementById('home-mode-select');
    const modeDesc = document.getElementById('mode-description-text');
    const bannerTitle = document.getElementById('offline-banner-title');
    const bannerDesc = document.getElementById('offline-banner-desc');

    if (modeSelect && modeSelect.value !== mode) {
        modeSelect.value = mode;
    }

    if (mode === 'offline') {
        headerStatus.className = 'header-status offline-active';
        statusDot.className = 'status-dot offline';
        statusLabel.innerText = 'Offline Mode (Local)';
        
        if (modeDesc) {
            modeDesc.innerText = 'Local engine active. Standalone neural network computes risk predictions on-device with 0ms network latency.';
        }
        if (bannerTitle) {
            bannerTitle.innerText = 'You are in Offline Mode (Local)';
        }
        if (bannerDesc) {
            bannerDesc.innerText = 'Analysis is performed locally on-device. Model updates synchronize when connectivity is available.';
        }
    } else {
        headerStatus.className = 'header-status';
        statusDot.className = 'status-dot';
        statusLabel.innerText = 'Online Mode (FastAPI)';
        
        if (modeDesc) {
            modeDesc.innerText = 'Connected to central FastAPI server for remote inference verification and federated parameter synchronization.';
        }
        if (bannerTitle) {
            bannerTitle.innerText = 'Online Centralized Mode';
        }
        if (bannerDesc) {
            bannerDesc.innerText = 'Connected to FastAPI clinical server. Switch to Offline Mode anytime for autonomous local decision support.';
        }
    }
}

// ============================================================================
// Frontend Physiological Validation
// ============================================================================
function validatePatientForm(payload) {
    const errors = [];

    // Demographics validation
    if (isNaN(payload.age) || payload.age < 1 || payload.age > 120) {
        errors.push('Age must be a valid positive value between 1 and 120 years.');
    }
    if (isNaN(payload.height) || payload.height < 50 || payload.height > 250) {
        errors.push('Height must be between 50 cm and 250 cm.');
    }
    if (isNaN(payload.weight) || payload.weight < 20 || payload.weight > 300) {
        errors.push('Weight must be between 20 kg and 300 kg.');
    }

    // Blood Pressure validation
    if (isNaN(payload.ap_hi) || payload.ap_hi < 60 || payload.ap_hi > 260) {
        errors.push('Systolic BP (ap_hi) must be a realistic numeric reading (60–260 mmHg).');
    }
    if (isNaN(payload.ap_lo) || payload.ap_lo < 40 || payload.ap_lo > 180) {
        errors.push('Diastolic BP (ap_lo) must be a realistic numeric reading (40–180 mmHg).');
    }
    if (payload.ap_hi <= payload.ap_lo) {
        errors.push('Systolic BP (ap_hi) must be strictly greater than Diastolic BP (ap_lo).');
    }

    return errors;
}

function showValidationAlert(errors) {
    const alertBox = document.getElementById('validation-alert');
    const list = document.getElementById('validation-errors-list');
    if (!alertBox || !list) return;

    list.innerHTML = errors.map(err => `<li>${err}</li>`).join('');
    alertBox.classList.add('active');
    alertBox.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

function hideValidationAlert() {
    const alertBox = document.getElementById('validation-alert');
    if (alertBox) alertBox.classList.remove('active');
}

// ============================================================================
// Explainability & Grounded Factor Analysis
// ============================================================================
function evaluateClinicalFactors(payload, bmi) {
    const factors = [];
    const elevatedAspects = [];

    // 1. Blood Pressure Analysis
    if (payload.ap_hi >= 140 || payload.ap_lo >= 90) {
        factors.push({
            type: 'elevated',
            label: `Elevated Blood Pressure: ${payload.ap_hi}/${payload.ap_lo} mmHg (Hypertension threshold)`,
            aspect: 'elevated blood pressure'
        });
        elevatedAspects.push('blood pressure');
    } else if (payload.ap_hi >= 130 || payload.ap_lo >= 80) {
        factors.push({
            type: 'elevated',
            label: `Prehypertension Blood Pressure Reading: ${payload.ap_hi}/${payload.ap_lo} mmHg`,
            aspect: 'borderline blood pressure'
        });
        elevatedAspects.push('blood pressure');
    } else {
        factors.push({
            type: 'optimal',
            label: `Normotensive Blood Pressure: ${payload.ap_hi}/${payload.ap_lo} mmHg`,
            aspect: null
        });
    }

    // 2. Cholesterol Level
    if (payload.cholesterol === 3) {
        factors.push({
            type: 'elevated',
            label: 'Well Above Normal Serum Cholesterol (Level 3, ≥ 240 mg/dL)',
            aspect: 'significantly elevated cholesterol'
        });
        elevatedAspects.push('cholesterol level');
    } else if (payload.cholesterol === 2) {
        factors.push({
            type: 'elevated',
            label: 'Above Normal Serum Cholesterol (Level 2, 200–239 mg/dL)',
            aspect: 'above-normal cholesterol'
        });
        elevatedAspects.push('cholesterol level');
    } else {
        factors.push({
            type: 'optimal',
            label: 'Normal Serum Cholesterol Level (< 200 mg/dL)',
            aspect: null
        });
    }

    // 3. Glucose Level
    if (payload.gluc === 3) {
        factors.push({
            type: 'elevated',
            label: 'Well Above Normal Fasting Glucose (Level 3, ≥ 126 mg/dL)',
            aspect: 'elevated blood glucose'
        });
        elevatedAspects.push('glucose level');
    } else if (payload.gluc === 2) {
        factors.push({
            type: 'elevated',
            label: 'Above Normal Fasting Glucose (Level 2, 100–125 mg/dL)',
            aspect: 'borderline blood glucose'
        });
        elevatedAspects.push('glucose level');
    } else {
        factors.push({
            type: 'optimal',
            label: 'Normal Fasting Blood Glucose (< 100 mg/dL)',
            aspect: null
        });
    }

    // 4. Physical Activity
    if (payload.active === 0) {
        factors.push({
            type: 'elevated',
            label: 'Physical Inactivity Reported (< 30 min daily activity)',
            aspect: 'physical inactivity'
        });
        elevatedAspects.push('physical inactivity');
    } else {
        factors.push({
            type: 'optimal',
            label: 'Physically Active Lifestyle (≥ 30 min daily activity)',
            aspect: null
        });
    }

    // 5. Smoking Status
    if (payload.smoke === 1) {
        factors.push({
            type: 'elevated',
            label: 'Active Tobacco Smoker (Increased vascular risk factor)',
            aspect: 'tobacco smoking'
        });
        elevatedAspects.push('smoking status');
    } else {
        factors.push({
            type: 'optimal',
            label: 'Non-Smoker Status',
            aspect: null
        });
    }

    // 6. Body Mass Index (BMI)
    if (bmi >= 30) {
        factors.push({
            type: 'elevated',
            label: `High Body Mass Index: ${bmi} kg/m² (Obese class)`,
            aspect: 'high BMI'
        });
        elevatedAspects.push('body mass index');
    } else if (bmi >= 25) {
        factors.push({
            type: 'elevated',
            label: `Elevated Body Mass Index: ${bmi} kg/m² (Overweight class)`,
            aspect: 'elevated BMI'
        });
        elevatedAspects.push('body mass index');
    }

    // 7. Age factor
    if (payload.age >= 55) {
        factors.push({
            type: 'elevated',
            label: `Age Demographics: ${payload.age} years (Associated cardiovascular baseline factor)`,
            aspect: 'patient age profile'
        });
        elevatedAspects.push('age factor');
    }

    return { factors, elevatedAspects };
}

function constructReasoningStatement(predictedClass, probability, elevatedAspects) {
    if (predictedClass === 1) {
        if (elevatedAspects.length > 0) {
            const uniqueAspects = [...new Set(elevatedAspects)];
            let aspectList = '';
            if (uniqueAspects.length === 1) {
                aspectList = uniqueAspects[0];
            } else if (uniqueAspects.length === 2) {
                aspectList = `${uniqueAspects[0]} and ${uniqueAspects[1]}`;
            } else {
                aspectList = `${uniqueAspects.slice(0, -1).join(', ')}, and ${uniqueAspects[uniqueAspects.length - 1]}`;
            }
            return `The assessment is influenced primarily by the patient's ${aspectList}. AI decision support indicates statistical correlation with potential cardiovascular risk patterns derived from multicenter federated hospital cohorts.`;
        } else {
            return `The assessment reflects subtle multivariate interaction patterns across clinical baselines with an estimated statistical probability of ${(probability * 100).toFixed(1)}%.`;
        }
    } else {
        return `The assessment indicates optimal alignment across primary vascular metrics (blood pressure, lipid profile, and activity levels), resulting in a lower potential cardiovascular risk estimation.`;
    }
}

// ============================================================================
// Assessment Submission Handler
// ============================================================================
async function handleAssessmentSubmit(event) {
    event.preventDefault();
    hideValidationAlert();

    const btn = document.getElementById('btn-submit');
    const originalText = btn.innerHTML;
    btn.innerHTML = '<span class="status-dot"></span> Analyzing Patient...';
    btn.disabled = true;

    const formData = new FormData(event.target);
    
    const heightCm = parseFloat(formData.get('height'));
    const weightKg = parseFloat(formData.get('weight'));
    const apHi = parseInt(formData.get('ap_hi'), 10);
    const apLo = parseInt(formData.get('ap_lo'), 10);
    const cholVal = parseInt(formData.get('cholesterol'), 10);
    const glucVal = parseInt(formData.get('gluc'), 10);
    const smokeVal = parseInt(formData.get('smoke'), 10);
    const alcoVal = parseInt(formData.get('alco'), 10);
    const activeVal = parseInt(formData.get('active'), 10);
    const ageYears = parseFloat(formData.get('age'));
    const genderVal = parseInt(formData.get('gender'), 10);

    const payload = {
        age: ageYears,
        gender: genderVal,
        height: heightCm,
        weight: weightKg,
        ap_hi: apHi,
        ap_lo: apLo,
        cholesterol: cholVal,
        gluc: glucVal,
        smoke: smokeVal,
        alco: alcoVal,
        active: activeVal
    };

    // Frontend validation check
    const validationErrors = validatePatientForm(payload);
    if (validationErrors.length > 0) {
        showValidationAlert(validationErrors);
        btn.innerHTML = originalText;
        btn.disabled = false;
        return;
    }

    try {
        const data = await InferenceRepository.predict(payload);

        // Calculate derived BMI
        const heightM = heightCm / 100.0;
        const bmi = (heightM > 0) ? parseFloat((weightKg / (heightM * heightM)).toFixed(1)) : 22.0;

        // Update Screen 3 UI Elements
        const boxEl = document.getElementById('res-box');
        const predLabelEl = document.getElementById('res-prediction-label');
        const probEl = document.getElementById('res-probability');
        const engineTagEl = document.getElementById('res-engine-tag');

        if (data.predicted_class === 1) {
            predLabelEl.innerText = 'Elevated Potential Risk';
            boxEl.className = 'risk-primary-card risk-high';
        } else {
            predLabelEl.innerText = 'Lower Potential Risk';
            boxEl.className = 'risk-primary-card risk-low';
        }

        const pct = (data.probability * 100).toFixed(2);
        probEl.innerText = `${pct}%`;
        engineTagEl.innerText = `Inference Engine: ${data.engine_origin || (currentInferenceMode === 'offline' ? 'Local Neural Net' : 'FastAPI REST Server')}`;

        // Populate Contributing Factors
        const { factors, elevatedAspects } = evaluateClinicalFactors(payload, bmi);
        const factorsListEl = document.getElementById('factors-list');
        factorsListEl.innerHTML = factors.map(f => `
            <li class="factor-item ${f.type}">
                <span class="factor-icon">${f.type === 'elevated' ? '▲' : '✓'}</span>
                <span>${f.label}</span>
            </li>
        `).join('');

        // Populate Supporting Clinical Reasoning
        const reasoningText = constructReasoningStatement(data.predicted_class, data.probability, elevatedAspects);
        document.getElementById('reasoning-text').innerText = reasoningText;

        // Derived Patient Metrics Summary Grid
        document.getElementById('summary-bp').innerText = `${apHi}/${apLo} mmHg`;
        document.getElementById('summary-bmi').innerText = `${bmi} kg/m²`;

        const cholMap = { 1: 'Normal', 2: 'Above Normal', 3: 'Well Above Normal' };
        document.getElementById('summary-cholesterol').innerText = cholMap[cholVal] || 'Normal';

        const glucMap = { 1: 'Normal', 2: 'Above Normal', 3: 'Well Above Normal' };
        document.getElementById('summary-gluc').innerText = glucMap[glucVal] || 'Normal';

        document.getElementById('summary-smoking').innerText = smokeVal === 1 ? 'Active Smoker' : 'Non-Smoker';
        document.getElementById('summary-activity').innerText = activeVal === 1 ? 'Active (≥30m)' : 'Inactive';

        // Display results screen
        showScreen('screen-result');
    } catch (err) {
        console.error('Inference error:', err);
        alert(`Inference Request Issue (${currentInferenceMode.toUpperCase()} MODE): ${err.message}\n\nNote: Switch to Offline Mode on the Dashboard to execute local standalone inference.`);
    } finally {
        btn.innerHTML = originalText;
        btn.disabled = false;
    }
}

// ============================================================================
// Fetch Central Model Info
// ============================================================================
async function fetchModelInfo() {
    try {
        const response = await fetch(`${API_BASE_URL}/model-info`);
        if (response.ok) {
            const data = await response.json();
            const statusEl = document.getElementById('model-status-text');
            if (statusEl) {
                statusEl.innerHTML = '<span class="status-dot"></span> Operational';
            }
            const roundsEl = document.getElementById('rounds-text');
            if (roundsEl && data.federated_rounds) {
                roundsEl.innerText = data.federated_rounds;
            }
            const clinicsEl = document.getElementById('clinics-text');
            if (clinicsEl && data.participating_clinics) {
                clinicsEl.innerText = data.participating_clinics.length;
            }
        }
    } catch (err) {
        console.warn('Backend API connection check fallback (Local mode ready):', err);
    }
}

// Initial setup on page load
document.addEventListener('DOMContentLoaded', () => {
    fetchModelInfo();
    InferenceRepository.loadOfflineAssets();
});
