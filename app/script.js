const API_BASE_URL = window.location.origin.includes('8000') 
    ? window.location.origin 
    : 'http://127.0.0.1:8000';

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
            document.getElementById('header-status').innerHTML = '<span class="status-dot"></span> System Connected';
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
    
    // Raw inputs
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
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            throw new Error(`Server returned status ${response.status}`);
        }

        const data = await response.json();

        // 1. Primary Risk & Probability Focus
        const boxEl = document.getElementById('res-box');
        const predEl = document.getElementById('res-prediction');

        if (data.predicted_class === 1) {
            predEl.innerText = 'Elevated Risk';
            boxEl.className = 'risk-primary-box risk-high';
        } else {
            predEl.innerText = 'Lower Risk';
            boxEl.className = 'risk-primary-box risk-low';
        }

        // Format probability to 2 decimal places e.g. 73.88%
        const pct = (data.probability * 100).toFixed(2);
        document.getElementById('res-probability').innerText = `${pct}%`;

        // 2. Derive Patient Summary Values (For display reference only)
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
        alert(`Inference Request Failed: ${err.message}`);
    } finally {
        btn.innerText = originalText;
        btn.disabled = false;
    }
}

// Initial fetch on page load
document.addEventListener('DOMContentLoaded', fetchModelInfo);
