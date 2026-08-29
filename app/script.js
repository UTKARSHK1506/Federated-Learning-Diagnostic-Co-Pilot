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
    const payload = {
        age: parseFloat(formData.get('age')),
        gender: parseInt(formData.get('gender')),
        height: parseFloat(formData.get('height')),
        weight: parseFloat(formData.get('weight')),
        ap_hi: parseInt(formData.get('ap_hi')),
        ap_lo: parseInt(formData.get('ap_lo')),
        cholesterol: parseInt(formData.get('cholesterol')),
        gluc: parseInt(formData.get('gluc')),
        smoke: parseInt(formData.get('smoke')),
        alco: parseInt(formData.get('alco')),
        active: parseInt(formData.get('active'))
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

        // Update Screen 3 UI
        const boxEl = document.getElementById('res-box');
        const predEl = document.getElementById('res-prediction');

        if (data.predicted_class === 1) {
            predEl.innerText = 'Cardiovascular Risk Detected';
            boxEl.className = 'prediction-box risk-high';
        } else {
            predEl.innerText = 'No Significant Risk Detected';
            boxEl.className = 'prediction-box risk-low';
        }

        const pct = (data.probability * 100).toFixed(2);
        document.getElementById('res-probability').innerText = `${pct}%`;
        document.getElementById('res-model').innerText = data.model_type || 'Federated FNN';

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
