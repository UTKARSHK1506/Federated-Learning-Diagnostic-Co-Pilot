const API_BASE_URL = window.location.origin.includes('8000') 
    ? window.location.origin 
    : 'http://127.0.0.1:8000';

let currentInferenceMode = 'online'; // 'online' or 'offline'
let offlineWeights = null;
let offlineScaler = null;

// Clean Abstraction: InferenceRepository
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
        return await response.json();
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
            model_type: "Federated FNN (Offline)"
        };
    }

    static async loadOfflineAssets() {
        try {
            const [wRes, sRes] = await Promise.all([
                fetch('global_model_weights.json'),
                fetch('scaler_params.json')
            ]);
            offlineWeights = await wRes.json();
            offlineScaler = await sRes.json();
        } catch (err) {
            console.error("Failed to load local offline model assets:", err);
            throw new Error("Offline model assets could not be loaded.");
        }
    }
}

function handleModeChange(mode) {
    currentInferenceMode = mode;
    const headerStatus = document.getElementById('header-status');
    if (mode === 'offline') {
        headerStatus.innerHTML = '<span class="status-dot offline"></span> Offline Mode';
    } else {
        headerStatus.innerHTML = '<span class="status-dot"></span> Online Mode';
    }
}

function showScreen(screenId) {
    document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
    const target = document.getElementById(screenId);
    if (target) {
        target.classList.add('active');
    }
}

async function fetchModelInfo() {
    try {
        const response = await fetch(`${API_BASE_URL}/model-info`);
        if (response.ok) {
            const data = await response.json();
            document.getElementById('model-status-text').innerText = data.model_status.includes('Operational') ? 'Operational' : data.model_status;
            document.getElementById('rounds-text').innerText = data.federated_rounds || 5;
            document.getElementById('clinics-text').innerText = (data.participating_clinics && data.participating_clinics.length) || 3;
        }
    } catch (err) {
        console.warn('Backend API connection check failed:', err);
    }
}

async function handleAssessmentSubmit(event) {
    event.preventDefault();
    const btn = document.getElementById('btn-submit');
    const originalText = btn.innerText;
    btn.innerText = 'Analyzing...';
    btn.disabled = true;

    const formData = new FormData(event.target);
    
    const heightCm = parseFloat(formData.get('height'));
    const weightKg = parseFloat(formData.get('weight'));
    const apHi = parseInt(formData.get('ap_hi'));
    const apLo = parseInt(formData.get('ap_lo'));
    const cholVal = parseInt(formData.get('cholesterol'));
    const smokeVal = parseInt(formData.get('smoke'));
    const activeVal = parseInt(formData.get('active'));

    const payload = {
        age: parseFloat(formData.get('age')),
        gender: parseInt(formData.get('gender')),
        height: heightCm,
        weight: weightKg,
        ap_hi: apHi,
        ap_lo: apLo,
        cholesterol: cholVal,
        gluc: parseInt(formData.get('gluc')),
        smoke: smokeVal,
        alco: parseInt(formData.get('alco')),
        active: activeVal
    };

    try {
        const data = await InferenceRepository.predict(payload);

        // Update Screen 3 UI
        const boxEl = document.getElementById('res-box');
        const predEl = document.getElementById('res-prediction');

        if (data.predicted_class === 1) {
            predEl.innerText = 'Elevated Risk';
            boxEl.className = 'risk-primary-box risk-high';
        } else {
            predEl.innerText = 'Lower Risk';
            boxEl.className = 'risk-primary-box risk-low';
        }

        const pct = (data.probability * 100).toFixed(2);
        document.getElementById('res-probability').innerText = `${pct}%`;

        // Derived Patient Summary
        document.getElementById('summary-bp').innerText = `${apHi}/${apLo} mmHg`;
        
        const heightM = heightCm / 100.0;
        const bmi = (heightM > 0) ? (weightKg / (heightM * heightM)).toFixed(1) : 'N/A';
        document.getElementById('summary-bmi').innerText = `${bmi} kg/m²`;

        const cholMap = { 1: 'Normal', 2: 'Above Normal', 3: 'Well Above Normal' };
        document.getElementById('summary-cholesterol').innerText = cholMap[cholVal] || 'Normal';

        document.getElementById('summary-activity').innerText = activeVal === 1 ? 'Active' : 'Inactive';
        document.getElementById('summary-smoking').innerText = smokeVal === 1 ? 'Smoker' : 'Non-Smoker';

        showScreen('screen-result');
    } catch (err) {
        alert(`Inference Request Failed (${currentInferenceMode.toUpperCase()} MODE): ${err.message}`);
    } finally {
        btn.innerText = originalText;
        btn.disabled = false;
    }
}

// Initial setup on page load
document.addEventListener('DOMContentLoaded', () => {
    fetchModelInfo();
    InferenceRepository.loadOfflineAssets();
});
