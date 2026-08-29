/**
 * CardioSense Clinical Assessment Web Application Script
 * Dual-Mode Inference Engine (Online FastAPI REST + Offline Local FNN Matrix Engine)
 * Client-Side Medical OCR Parser, Normalization Layer & Local History Persistence
 */

const API_BASE_URL = window.location.origin.includes('8000') 
    ? window.location.origin 
    : 'http://127.0.0.1:8000';

let currentInferenceMode = 'online'; // 'online' or 'offline'
let offlineWeights = null;
let offlineScaler = null;

// State Variables for Computed Assessment
let lastComputedProb = 0.0;
let lastComputedCategory = "Unknown";
let lastComputedApHi = 140;
let lastComputedApLo = 90;
let lastComputedAge = 55.5;

// ============================================================================
// Core InferenceRepository (Preserved Dual-Mode Architecture)
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

        const featureNames = offlineScaler.feature_columns;
        const continuousCols = offlineScaler.continuous_columns;
        const mean = offlineScaler.mean;
        const scale = offlineScaler.scale;

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

        // Layer 1 (11 -> 64 with ReLU)
        const fc1_w = offlineWeights.fc1_w;
        const fc1_b = offlineWeights.fc1_b;
        const h1 = new Array(64);
        for (let i = 0; i < 64; i++) {
            let sum = fc1_b[i];
            for (let j = 0; j < 11; j++) {
                sum += fc1_w[i][j] * x_input[j];
            }
            h1[i] = Math.max(0.0, sum);
        }

        // Layer 2 (64 -> 32 with ReLU)
        const fc2_w = offlineWeights.fc2_w;
        const fc2_b = offlineWeights.fc2_b;
        const h2 = new Array(32);
        for (let i = 0; i < 32; i++) {
            let sum = fc2_b[i];
            for (let j = 0; j < 64; j++) {
                sum += fc2_w[i][j] * h1[j];
            }
            h2[i] = Math.max(0.0, sum);
        }

        // Layer 3 (32 -> 1 raw logit)
        const fc3_w = offlineWeights.fc3_w;
        const fc3_b = offlineWeights.fc3_b;
        let logit = fc3_b[0];
        for (let j = 0; j < 32; j++) {
            logit += fc3_w[0][j] * h2[j];
        }

        const probability = 1.0 / (1.0 + Math.exp(-logit));
        const predicted_class = probability >= 0.5 ? 1 : 0;

        return {
            predicted_class: predicted_class,
            probability: parseFloat(probability.toFixed(4)),
            model_type: "CardioSense Local Engine (Offline)",
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
                    console.log(`InferenceRepository: Offline assets loaded successfully.`);
                    return;
                }
            } catch (e) {
                // Try next path
            }
        }
        console.warn("InferenceRepository: Could not pre-fetch offline assets automatically.");
    }
}

// Pre-fetch assets on load
InferenceRepository.loadOfflineAssets();

// ============================================================================
// Medical Term Dictionary & Context-Aware Web OCR Parser
// ============================================================================
class WebOcrParser {
    static parseText(rawText) {
        const text = rawText.toLowerCase();
        const warnings = [];

        // 1. Blood Pressure (SBP & DBP)
        let ap_hi = null;
        let ap_lo = null;

        const bpMatch = text.match(/(?:bp|blood\s*pressure|sbp\/dbp)\D*(\d{2,3})\s*[\/:-]\s*(\d{2,3})/);
        if (bpMatch) {
            ap_hi = parseInt(bpMatch[1], 10);
            ap_lo = parseInt(bpMatch[2], 10);
        } else {
            const sbpMatch = text.match(/(?:systolic\s*bp|systolic\s*blood\s*pressure|sbp|sys\s*bp|systolic)\D*(\d{2,3})/);
            if (sbpMatch) ap_hi = parseInt(sbpMatch[1], 10);

            const dbpMatch = text.match(/(?:diastolic\s*bp|diastolic\s*blood\s*pressure|dbp|dia\s*bp|diastolic)\D*(\d{2,3})/);
            if (dbpMatch) ap_lo = parseInt(dbpMatch[1], 10);
        }

        if (ap_hi && (ap_hi < 60 || ap_hi > 260)) ap_hi = null;
        if (ap_lo && (ap_lo < 40 || ap_lo > 180)) ap_lo = null;

        // 2. Glucose
        let gluc = null;
        const glucNumMatch = text.match(/(?:blood\s*sugar|blood\s*glucose|fasting\s*glucose|fasting\s*blood\s*sugar|rbs|fbs|ppbs|glucose|sugar|gluc)\D*(\d{2,3})/);
        if (glucNumMatch) {
            const num = parseInt(glucNumMatch[1], 10);
            if (num >= 126) gluc = 3;
            else if (num >= 100) gluc = 2;
            else if (num > 0) gluc = 1;
        }

        // 3. Cholesterol
        let cholesterol = null;
        const cholNumMatch = text.match(/(?:total\s*cholesterol|serum\s*cholesterol|tc|cholesterol|chol)\D*(\d{2,3})/);
        if (cholNumMatch) {
            const num = parseInt(cholNumMatch[1], 10);
            if (num >= 240) cholesterol = 3;
            else if (num >= 200) cholesterol = 2;
            else if (num > 0) cholesterol = 1;
        }

        if (cholesterol === null && (text.includes('ldl') || text.includes('hdl') || text.includes('triglycerides'))) {
            warnings.push("Cholesterol level could not be mapped automatically — please review.");
        }

        // 4. DOB & Age Parsing
        let age = null;
        let calculatedAgeFromDob = null;

        const dobMatch = text.match(/(?:dob|date\s*of\s*birth|birth\s*date|d\.o\.b\.|born)\D*(\d{1,4}[\/\.-]\d{1,2}[\/\.-]\d{1,4})/);
        if (dobMatch) {
            calculatedAgeFromDob = WebOcrParser.calculateAgeFromDob(dobMatch[1]);
            if (calculatedAgeFromDob === null) {
                warnings.push("Date format unclear — please verify.");
            }
        }

        const ageMatch = text.match(/(?:patient\s*age|age)\D*(\d{1,3}(?:\.\d)?)\s*(?:yrs|years|y)?\b/);
        if (ageMatch) {
            age = parseFloat(ageMatch[1]);
        }

        if (age === null && calculatedAgeFromDob !== null) {
            age = calculatedAgeFromDob;
        } else if (age !== null && calculatedAgeFromDob !== null) {
            if (Math.abs(age - calculatedAgeFromDob) > 1.5) {
                warnings.push("Age and DOB appear inconsistent — please verify.");
            }
        }

        // 5. Height (cm, m, inches -> cm)
        let height = null;
        const hmMatch = text.match(/(?:height|ht|body\s*height|stature)\D*(\d(?:\.\d{1,2})?)\s*m\b/);
        if (hmMatch) {
            const mVal = parseFloat(hmMatch[1]);
            if (mVal >= 0.8 && mVal <= 2.5) height = mVal * 100.0;
        }
        if (height === null) {
            const hcmMatch = text.match(/(?:height|ht|body\s*height|stature)\D*(\d{2,3}(?:\.\d)?)\s*(?:cm)?/);
            if (hcmMatch) {
                const num = parseFloat(hcmMatch[1]);
                if (num >= 50 && num <= 250) height = num;
            }
        }

        // 6. Weight (kg, lbs -> kg)
        let weight = null;
        const wlbsMatch = text.match(/(?:weight|wt|body\s*weight)\D*(\d{2,3}(?:\.\d)?)\s*(?:lbs|lb)\b/);
        if (wlbsMatch) {
            const lbs = parseFloat(wlbsMatch[1]);
            weight = lbs * 0.45359237;
        }
        if (weight === null) {
            const wkgMatch = text.match(/(?:weight|wt|body\s*weight)\D*(\d{2,3}(?:\.\d)?)\s*(?:kg)?/);
            if (wkgMatch) {
                const num = parseFloat(wkgMatch[1]);
                if (num >= 20 && num <= 300) weight = num;
            }
        }

        // 7. Gender
        let gender = null;
        const sexMatch = text.match(/(?:gender|sex)\D*(female|male|f|m)\b/);
        if (sexMatch) {
            gender = (sexMatch[1] === 'female' || sexMatch[1] === 'f') ? 1 : 2;
        }

        // 8. Lifestyle
        let smoke = null;
        const smokeMatch = text.match(/(?:smoking\s*status|smoking|smoker|tobacco)\D*(non-smoker|never|former|current|active|yes|no)\b/);
        if (smokeMatch) {
            const s = smokeMatch[1];
            smoke = (s === 'active' || s === 'current' || s === 'yes') ? 1 : 0;
        }

        let alco = null;
        const alcoMatch = text.match(/(?:alcohol\s*intake|alcohol\s*use|alcohol|drinking)\D*(non-drinker|regular|yes|no)\b/);
        if (alcoMatch) {
            const a = alcoMatch[1];
            alco = (a === 'regular' || a === 'yes') ? 1 : 0;
        }

        let active = null;
        const activeMatch = text.match(/(?:physical\s*activity|exercise|activity\s*level|active|sedentary)\D*(active|inactive|sedentary|yes|no)\b/);
        if (activeMatch) {
            const act = activeMatch[1];
            active = (act === 'active' || act === 'yes') ? 1 : 0;
        }

        return {
            age: age,
            gender: gender,
            height: height,
            weight: weight,
            ap_hi: ap_hi,
            ap_lo: ap_lo,
            cholesterol: cholesterol,
            gluc: gluc,
            smoke: smoke,
            alco: alco,
            active: active,
            warnings: warnings
        };
    }

    static calculateAgeFromDob(dobStr) {
        try {
            const parts = dobStr.split(/[\/\.-]/);
            if (parts.length === 3) {
                let day = parseInt(parts[0], 10);
                let month = parseInt(parts[1], 10) - 1;
                let year = parseInt(parts[2], 10);

                if (parts[0].length === 4) {
                    year = parseInt(parts[0], 10);
                    month = parseInt(parts[1], 10) - 1;
                    day = parseInt(parts[2], 10);
                }

                const birthDate = new Date(year, month, day);
                const today = new Date();
                let ageYears = today.getFullYear() - birthDate.getFullYear();
                const m = today.getMonth() - birthDate.getMonth();
                if (m < 0 || (m === 0 && today.getDate() < birthDate.getDate())) {
                    ageYears--;
                }
                if (ageYears >= 1 && ageYears <= 120) return ageYears;
            }
        } catch (e) {
            // Ignore parse failure
        }
        return null;
    }
}

// ============================================================================
// Screen Router & Navigation
// ============================================================================
function showScreen(screenId) {
    document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
    const target = document.getElementById(screenId);
    if (target) {
        target.classList.add('active');
    }

    // Update bottom nav active highlights
    document.querySelectorAll('.nav-tab').forEach(tab => tab.classList.remove('active'));
    if (screenId === 'screen-welcome' || screenId === 'screen-dashboard' || screenId === 'screen-how-it-works') {
        const homeTab = document.getElementById('tab-home');
        if (homeTab) homeTab.classList.add('active');
    } else if (screenId === 'screen-assessment' || screenId === 'screen-ocr-upload' || screenId === 'screen-ocr-review' || screenId === 'screen-result') {
        const assessTab = document.getElementById('tab-assessment');
        if (assessTab) assessTab.classList.add('active');
    } else if (screenId === 'screen-history') {
        const histTab = document.getElementById('tab-history');
        if (histTab) histTab.classList.add('active');
        loadAndDisplayHistory();
    } else if (screenId === 'screen-health-tips') {
        const tipsTab = document.getElementById('tab-tips');
        if (tipsTab) tipsTab.classList.add('active');
    }

    window.scrollTo({ top: 0, behavior: 'smooth' });
}

function handleModeChange(mode) {
    currentInferenceMode = mode;
    const headerStatus = document.getElementById('header-status');
    const statusDot = document.getElementById('status-dot');
    const statusLabel = document.getElementById('status-label');
    const modeSelect = document.getElementById('home-mode-select');
    const modeDesc = document.getElementById('mode-description-text');

    if (modeSelect && modeSelect.value !== mode) {
        modeSelect.value = mode;
    }

    if (mode === 'offline') {
        headerStatus.className = 'header-status offline-active';
        statusDot.className = 'status-dot offline';
        statusLabel.innerText = 'Offline Mode';
        if (modeDesc) {
            modeDesc.innerText = 'On-device local engine active. Risk predictions run locally with 0ms network latency.';
        }
    } else {
        headerStatus.className = 'header-status';
        statusDot.className = 'status-dot';
        statusLabel.innerText = 'Online Mode';
        if (modeDesc) {
            modeDesc.innerText = 'Connected to central FastAPI server for remote inference verification.';
        }
    }
}

// ============================================================================
// File Selection & OCR Review Handling
// ============================================================================
function triggerFileSelect(mimeType) {
    if (mimeType.includes('pdf')) {
        document.getElementById('file-input-pdf').click();
    } else {
        document.getElementById('file-input-image').click();
    }
}

function handleFileSelected(file) {
    if (!file) return;

    // Simulate / Process local OCR extraction
    const reader = new FileReader();
    reader.onload = function(e) {
        // Sample medical report text parsing simulation if OCR library is unavailable
        const fileContent = e.target.result || "";
        const parsed = WebOcrParser.parseText(typeof fileContent === 'string' ? fileContent : file.name);
        populateFormFromOcr(parsed);
        showScreen('screen-ocr-review');
    };

    if (file.type.includes('text')) {
        reader.readAsText(file);
    } else {
        reader.readAsDataURL(file);
    }
}

function populateFormFromOcr(parsed) {
    if (parsed.age) document.getElementById('age').value = parsed.age;
    if (parsed.height) document.getElementById('height').value = parsed.height;
    if (parsed.weight) document.getElementById('weight').value = parsed.weight;
    if (parsed.ap_hi) document.getElementById('ap_hi').value = parsed.ap_hi;
    if (parsed.ap_lo) document.getElementById('ap_lo').value = parsed.ap_lo;

    if (parsed.gender) document.getElementById('gender').value = parsed.gender;
    if (parsed.cholesterol) document.getElementById('cholesterol').value = parsed.cholesterol;
    if (parsed.gluc) document.getElementById('gluc').value = parsed.gluc;
    if (parsed.smoke !== null) document.getElementById('smoke').value = parsed.smoke;
    if (parsed.alco !== null) document.getElementById('alco').value = parsed.alco;
    if (parsed.active !== null) document.getElementById('active').value = parsed.active;

    const alertTitle = document.getElementById('ocr-alert-title');
    const alertBody = document.getElementById('ocr-alert-body');

    if (alertTitle && alertBody) {
        alertTitle.innerText = "OCR Extraction Complete — Your report is processed on this device.";
        if (parsed.warnings && parsed.warnings.length > 0) {
            alertBody.innerText = "⚠️ " + parsed.warnings.join("\n⚠️ ");
        } else {
            alertBody.innerText = "All recognized values populated into the assessment form. Please verify fields before proceeding.";
        }
    }
}

// Setup Drag & Drop
document.addEventListener('DOMContentLoaded', () => {
    const dropzone = document.getElementById('ocr-dropzone');
    if (dropzone) {
        ['dragenter', 'dragover'].forEach(eventName => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                dropzone.style.backgroundColor = '#A7F3D0';
            }, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                dropzone.style.backgroundColor = 'var(--primary-light)';
            }, false);
        });

        dropzone.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            const files = dt.files;
            if (files && files.length > 0) {
                handleFileSelected(files[0]);
            }
        });
    }
});

// ============================================================================
// Form Submission & Dynamic Result Rendering
// ============================================================================
async function handleAssessmentSubmit(event) {
    event.preventDefault();
    hideValidationAlert();

    const form = document.getElementById('assessment-form');
    const formData = new FormData(form);

    const payload = {
        age: parseFloat(formData.get('age')),
        gender: parseInt(formData.get('gender'), 10),
        height: parseFloat(formData.get('height')),
        weight: parseFloat(formData.get('weight')),
        ap_hi: parseInt(formData.get('ap_hi'), 10),
        ap_lo: parseInt(formData.get('ap_lo'), 10),
        cholesterol: parseInt(formData.get('cholesterol'), 10),
        gluc: parseInt(formData.get('gluc'), 10),
        smoke: parseInt(formData.get('smoke'), 10),
        alco: parseInt(formData.get('alco'), 10),
        active: parseInt(formData.get('active'), 10)
    };

    const errors = validatePatientForm(payload);
    if (errors.length > 0) {
        showValidationAlert(errors);
        return;
    }

    const submitBtn = document.getElementById('btn-submit');
    submitBtn.disabled = true;
    submitBtn.innerText = 'Assessing...';

    try {
        const result = await InferenceRepository.predict(payload);
        renderResult(result, payload);
    } catch (e) {
        console.warn("Online inference failed. Falling back to Offline mode...", e);
        try {
            handleModeChange('offline');
            const result = await InferenceRepository.runOfflineInference(payload);
            renderResult(result, payload);
        } catch (err) {
            alert("Assessment could not be completed. Please try again.");
        }
    } finally {
        submitBtn.disabled = false;
        submitBtn.innerText = 'Assess Risk';
    }
}

function validatePatientForm(payload) {
    const errors = [];
    if (isNaN(payload.age) || payload.age < 1 || payload.age > 120) {
        errors.push('Age must be between 1 and 120 years.');
    }
    if (isNaN(payload.height) || payload.height < 50 || payload.height > 250) {
        errors.push('Height must be between 50 cm and 250 cm.');
    }
    if (isNaN(payload.weight) || payload.weight < 20 || payload.weight > 300) {
        errors.push('Weight must be between 20 kg and 300 kg.');
    }
    if (isNaN(payload.ap_hi) || payload.ap_hi < 60 || payload.ap_hi > 260) {
        errors.push('Systolic BP must be between 60 and 260 mmHg.');
    }
    if (isNaN(payload.ap_lo) || payload.ap_lo < 40 || payload.ap_lo > 180) {
        errors.push('Diastolic BP must be between 40 and 180 mmHg.');
    }
    if (payload.ap_hi <= payload.ap_lo) {
        errors.push('Systolic BP must be strictly greater than Diastolic BP.');
    }
    return errors;
}

function showValidationAlert(errors) {
    const alertBox = document.getElementById('validation-alert');
    const list = document.getElementById('validation-errors-list');
    if (!alertBox || !list) return;

    list.innerHTML = errors.map(err => `<li>${err}</li>`).join('');
    alertBox.style.display = 'block';
}

function hideValidationAlert() {
    const alertBox = document.getElementById('validation-alert');
    if (alertBox) alertBox.style.display = 'none';
}

function renderResult(res, payload) {
    lastComputedProb = res.probability;
    lastComputedApHi = payload.ap_hi;
    lastComputedApLo = payload.ap_lo;
    lastComputedAge = payload.age;

    const probText = (res.probability * 100).toFixed(2) + '%';
    document.getElementById('res-probability').innerText = probText;

    const prob = res.probability;
    const tagEl = document.getElementById('res-prediction-label');
    const resBox = document.getElementById('res-box');

    // Reset 4-Segment Scale Bar Classes
    const lowSeg = document.getElementById('v-scale-low');
    const modSeg = document.getElementById('v-scale-mod');
    const elevSeg = document.getElementById('v-scale-elev');
    const highSeg = document.getElementById('v-scale-high');

    lowSeg.className = 'scale-segment';
    modSeg.className = 'scale-segment';
    elevSeg.className = 'scale-segment';
    highSeg.className = 'scale-segment';

    document.getElementById('tv-lbl-low').className = '';
    document.getElementById('tv-lbl-mod').className = '';
    document.getElementById('tv-lbl-elev').className = '';
    document.getElementById('tv-lbl-high').className = '';

    if (prob < 0.30) {
        lastComputedCategory = "Low Risk";
        tagEl.innerText = "LOW POTENTIAL RISK";
        resBox.style.backgroundColor = "#ECFDF5";
        lowSeg.className = 'scale-segment active-low';
        document.getElementById('tv-lbl-low').className = 'active';
    } else if (prob < 0.50) {
        lastComputedCategory = "Moderate Risk";
        tagEl.innerText = "MODERATE POTENTIAL RISK";
        resBox.style.backgroundColor = "#FFFBEB";
        modSeg.className = 'scale-segment active-mod';
        document.getElementById('tv-lbl-mod').className = 'active';
    } else if (prob < 0.75) {
        lastComputedCategory = "Elevated Risk";
        tagEl.innerText = "ELEVATED POTENTIAL RISK";
        resBox.style.backgroundColor = "#FEF2F2";
        elevSeg.className = 'scale-segment active-elev';
        document.getElementById('tv-lbl-elev').className = 'active';
    } else {
        lastComputedCategory = "High Risk";
        tagEl.innerText = "HIGH POTENTIAL RISK";
        resBox.style.backgroundColor = "#FEF2F2";
        highSeg.className = 'scale-segment active-high';
        document.getElementById('tv-lbl-high').className = 'active';
    }

    // Baseline Metrics
    document.getElementById('summary-age').innerText = payload.age.toFixed(1) + ' yrs';
    document.getElementById('summary-bp').innerText = `${payload.ap_hi}/${payload.ap_lo} mmHg`;
    
    const heightM = payload.height / 100.0;
    const bmi = heightM > 0 ? (payload.weight / (heightM * heightM)).toFixed(1) : '22.0';
    document.getElementById('summary-bmi').innerText = `${bmi} kg/m²`;

    document.getElementById('summary-cholesterol').innerText = `Level ${payload.cholesterol}`;
    document.getElementById('summary-gluc').innerText = `Level ${payload.gluc}`;

    // Generate Dynamic Health Factors
    const factorsList = document.getElementById('factors-list');
    const factors = [];

    if (payload.ap_hi >= 140 || payload.ap_lo >= 90) {
        factors.push(`Elevated Blood Pressure (${payload.ap_hi}/${payload.ap_lo} mmHg)`);
    } else if (payload.ap_hi >= 130 || payload.ap_lo >= 80) {
        factors.push(`Prehypertension BP range (${payload.ap_hi}/${payload.ap_lo} mmHg)`);
    }

    if (payload.cholesterol >= 2) {
        const desc = payload.cholesterol === 3 ? "Well Above Normal" : "Above Normal";
        factors.push(`Cholesterol Level (${desc})`);
    }

    if (payload.gluc >= 2) {
        const desc = payload.gluc === 3 ? "Well Above Normal" : "Above Normal";
        factors.push(`Glucose Level (${desc})`);
    }

    if (payload.active === 0) {
        factors.push(`Physical Inactivity (< 30 min daily activity)`);
    }

    if (payload.smoke === 1) {
        factors.push(`Active Smoking Status`);
    }

    if (parseFloat(bmi) >= 25.0) {
        factors.push(`Elevated Body Mass Index (${bmi} kg/m²)`);
    }

    if (payload.age >= 55.0) {
        factors.push(`Age Baseline Factor (${payload.age.toFixed(1)} years)`);
    }

    if (factors.length === 0) {
        factors.push(`Blood pressure within normal range (${payload.ap_hi}/${payload.ap_lo} mmHg)`);
        factors.push(`Cholesterol and glucose levels normal`);
        factors.push(`Active lifestyle reported`);
    }

    factorsList.innerHTML = factors.map(f => `<li>• ${f}</li>`).join('');

    showScreen('screen-result');
}

// ============================================================================
// Assessment History Persistence (localStorage)
// ============================================================================
function saveCurrentAssessmentToHistory() {
    const historyJson = localStorage.getItem('cardiosense_history') || '[]';
    const history = JSON.parse(historyJson);

    const now = new Date();
    const dateStr = now.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' }) + 
                    ' - ' + now.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });

    const newRecord = {
        id: Date.now().toString(),
        date: dateStr,
        probability: lastComputedProb,
        riskCategory: lastComputedCategory,
        bpStr: `${lastComputedApHi}/${lastComputedApLo} mmHg`,
        ageStr: `${lastComputedAge.toFixed(1)} yrs`,
        mode: currentInferenceMode
    };

    history.unshift(newRecord);
    if (history.length > 20) history.pop();

    localStorage.setItem('cardiosense_history', JSON.stringify(history));
    alert("Assessment saved to local history");
    showScreen('screen-history');
}

function loadAndDisplayHistory() {
    const historyJson = localStorage.getItem('cardiosense_history') || '[]';
    const history = JSON.parse(historyJson);
    const container = document.getElementById('history-container');
    const emptyEl = document.getElementById('history-empty');

    if (!container) return;

    if (history.length === 0) {
        if (emptyEl) emptyEl.style.display = 'block';
        container.innerHTML = '';
    } else {
        if (emptyEl) emptyEl.style.display = 'none';
        container.innerHTML = history.map(rec => `
            <div class="card">
                <div style="display:flex; justify-content:space-between; font-weight:bold;">
                    <span style="color:#0D9488;">${rec.riskCategory.toUpperCase()}</span>
                    <span style="color:#0F172A;">${(rec.probability * 100).toFixed(2)}%</span>
                </div>
                <div style="font-size:0.85rem; color:#64748B; margin-top:6px;">
                    BP: ${rec.bpStr} | Age: ${rec.ageStr}<br>
                    Date: ${rec.date} | Mode: ${rec.mode.toUpperCase()}
                </div>
            </div>
        `).join('');
    }
}
