/**
 * CardioSense Clinical Assessment Web Application Script
 * Dual-Mode Inference Engine (Online FastAPI REST + Offline Local Engine)
 * Client-Side OCR Engine (Tesseract.js + Canvas Preprocessor + PDF.js), 3-Pass Medical Parser & Stale Data Cleansing
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
let lastComputedApHi = 0;
let lastComputedApLo = 0;
let lastComputedAge = 0.0;

// ============================================================================
// Core InferenceRepository (Preserved Architecture)
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

    // 2. OFFLINE MODE: Local Engine
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
                    return;
                }
            } catch (e) {
                // Try next candidate
            }
        }
    }
}

// Pre-fetch assets on load
InferenceRepository.loadOfflineAssets();

// ============================================================================
// 3-Pass Semantic Web OCR Parser Engine
// ============================================================================
class WebOcrParser {
    static parseText(rawText) {
        if (!rawText || typeof rawText !== 'string' || rawText.trim().length === 0) {
            return {
                age: null, gender: null, height: null, weight: null,
                ap_hi: null, ap_lo: null, cholesterol: null, gluc: null,
                smoke: null, alco: null, active: null, warnings: [],
                fieldsFoundCount: 0
            };
        }

        const normalizedText = WebOcrParser.normalizeTypos(rawText.toLowerCase());
        const lines = normalizedText.split('\n').map(l => l.trim()).filter(l => l.length > 0);
        const warnings = [];

        let age = null;
        let calculatedAgeFromDob = null;
        let gender = null;
        let height = null;
        let weight = null;
        let ap_hi = null;
        let ap_lo = null;
        let cholesterol = null;
        let gluc = null;
        let smoke = null;
        let alco = null;
        let active = null;

        // --------------------------------------------------------------------
        // PASS 1: FULL-TEXT MULTI-LINE PROXIMITY SEARCH
        // --------------------------------------------------------------------

        // 1. Blood Pressure
        const fullBpMatch = normalizedText.match(/(?:bp|b\.p\.|blood\s*pressure|sbp\/dbp|pressure|reading)[\s:=\n-]*(\d{2,3})\s*[\/\\:-]\s*(\d{2,3})\b/);
        if (fullBpMatch) {
            const hi = parseInt(fullBpMatch[1], 10);
            const lo = parseInt(fullBpMatch[2], 10);
            if (hi >= 60 && hi <= 260 && lo >= 40 && lo <= 180 && hi > lo) {
                ap_hi = hi;
                ap_lo = lo;
            }
        }
        if (ap_hi === null) {
            const sbpMatch = normalizedText.match(/(?:systolic\s*bp(?:\s*\([^)]*\))?|systolic\s*blood\s*pressure|sbp|sys\s*bp|systolic)[\s:=\n-]*(\d{2,3})\b/);
            if (sbpMatch) {
                const hi = parseInt(sbpMatch[1], 10);
                if (hi >= 60 && hi <= 260) ap_hi = hi;
            }
        }
        if (ap_lo === null) {
            const dbpMatch = normalizedText.match(/(?:diastolic\s*bp(?:\s*\([^)]*\))?|diastolic\s*blood\s*pressure|dbp|dia\s*bp|diastolic)[\s:=\n-]*(\d{2,3})\b/);
            if (dbpMatch) {
                const lo = parseInt(dbpMatch[1], 10);
                if (lo >= 40 && lo <= 180) ap_lo = lo;
            }
        }

        // 2. Glucose
        const fullGlucMatch = normalizedText.match(/(?:fasting\s*blood\s*glucose|fasting\s*glucose|fasting\s*blood\s*sugar|random\s*blood\s*sugar|postprandial\s*blood\s*sugar|blood\s*glucose|blood\s*sugar|serum\s*glucose|fbs|fbg|rbs|ppbs|glucose)[\s:=\n-]*(?:level\s*)?(\d{1,3})\b/);
        if (fullGlucMatch) {
            const valNum = parseInt(fullGlucMatch[1], 10);
            if (valNum >= 1 && valNum <= 3) gluc = valNum;
            else if (valNum >= 126) gluc = 3;
            else if (valNum >= 100) gluc = 2;
            else if (valNum > 0) gluc = 1;
        }

        // 3. Cholesterol
        const fullCholMatch = normalizedText.match(/(?:total\s*cholesterol|serum\s*cholesterol|s\.\s*cholesterol|total\s*chol|cholesterol|tc)[\s:=\n-]*(?:level\s*)?(\d{1,3})\b/);
        if (fullCholMatch) {
            const valNum = parseInt(fullCholMatch[1], 10);
            if (valNum >= 1 && valNum <= 3) cholesterol = valNum;
            else if (valNum >= 240) cholesterol = 3;
            else if (valNum >= 200) cholesterol = 2;
            else if (valNum > 0) cholesterol = 1;
        }

        // 4. Height
        const fullHftMatch = normalizedText.match(/(?:height|ht|stature)[\s:=\n-]*(\d)\s*(?:ft|feet|')\s*(\d{1,2})?\s*(?:in|inches|")?/);
        if (fullHftMatch) {
            const feet = parseFloat(fullHftMatch[1]) || 0;
            const inches = parseFloat(fullHftMatch[2]) || 0;
            const totalCm = (feet * 30.48) + (inches * 2.54);
            if (totalCm >= 50 && totalCm <= 250) height = totalCm;
        } else {
            const fullHmMatch = normalizedText.match(/(?:height|ht|stature)[\s:=\n-]*(\d(?:\.\d{1,2})?)\s*m\b/);
            if (fullHmMatch) {
                const mVal = parseFloat(fullHmMatch[1]);
                if (mVal >= 0.8 && mVal <= 2.5) height = mVal * 100.0;
            } else {
                const fullHcmMatch = normalizedText.match(/(?:height|ht|stature)[\s:=\n-]*(\d{2,3}(?:\.\d)?)\s*(?:cm)?\b/);
                if (fullHcmMatch) {
                    const valNum = parseFloat(fullHcmMatch[1]);
                    if (valNum >= 50 && valNum <= 250) height = valNum;
                }
            }
        }

        // 5. Weight
        const fullWlbsMatch = normalizedText.match(/(?:weight|wt|body\s*weight)[\s:=\n-]*(\d{2,3}(?:\.\d)?)\s*(?:lbs|lb|pounds)\b/);
        if (fullWlbsMatch) {
            const lbsVal = parseFloat(fullWlbsMatch[1]);
            if (lbsVal) weight = lbsVal * 0.45359237;
        } else {
            const fullWkgMatch = normalizedText.match(/(?:weight|wt|body\s*weight)[\s:=\n-]*(\d{2,3}(?:\.\d)?)\s*(?:kg|kilograms)?\b/);
            if (fullWkgMatch) {
                const valNum = parseFloat(fullWkgMatch[1]);
                if (valNum >= 20 && valNum <= 300) weight = valNum;
            }
        }

        // 6. Direct Age
        const fullAgeMatch = normalizedText.match(/(?:patient\s*age|age)[\s:=\n-]*(\d{1,3}(?:\.\d)?)\s*(?:yrs|years|y)?\b/);
        if (fullAgeMatch) {
            const ageVal = parseFloat(fullAgeMatch[1]);
            if (ageVal >= 1.0 && ageVal <= 120.0) age = ageVal;
        }

        if (age === null || gender === null) {
            const shorthandMatch = normalizedText.match(/\b(\d{1,2})\s*(?:yo|y\/o|years?\s*old)?\s*[\/,-]?\s*(female|male|f|m)\b/);
            if (shorthandMatch) {
                if (age === null) {
                    const aVal = parseFloat(shorthandMatch[1]);
                    if (aVal >= 1.0 && aVal <= 120.0) age = aVal;
                }
                if (gender === null) {
                    const gStr = shorthandMatch[2];
                    gender = (gStr === 'female' || gStr === 'f') ? 1 : 2;
                }
            }
        }

        // 7. Gender
        if (gender === null) {
            const fullSexMatch = normalizedText.match(/(?:gender|sex|biological\s*sex)[\s:=\n-]*(female|male|f|m)\b/);
            if (fullSexMatch) {
                const sexStr = fullSexMatch[1];
                gender = (sexStr === 'female' || sexStr === 'f') ? 1 : 2;
            }
        }

        // 8. Smoking
        const fullSmokeMatch = normalizedText.match(/(?:smoking\s*status|smoking|smoker|tobacco\s*use|tobacco|cigarette[s]?|nicotine)[\s:=\n-]*(never\s*smoker|non-smoker|nonsmoker|does\s*not\s*smoke|denies\s*smoking|no\s*smoking|former\s*smoker|ex-smoker|quit\s*smoking|current\s*smoker|current\s*smoking|smokes|tobacco\s*user|active|yes|no|none|nil)\b/);
        if (fullSmokeMatch) {
            const sStr = fullSmokeMatch[1];
            if (sStr.includes('former') || sStr.includes('ex-') || sStr.includes('quit') || sStr.includes('occasional')) {
                warnings.push(`Smoking status is ambiguous ('${sStr}') — please verify.`);
            } else if (sStr.includes('current') || sStr.includes('smokes') || sStr.includes('active') || sStr.includes('yes')) {
                smoke = 1;
            } else if (sStr.includes('non') || sStr.includes('never') || sStr.includes('does not') || sStr.includes('denies') || sStr.includes('no') || sStr.includes('none') || sStr.includes('nil')) {
                smoke = 0;
            }
        }

        // 9. Alcohol
        const fullAlcoMatch = normalizedText.match(/(?:alcohol\s*use|alcohol\s*intake|alcohol\s*consumption|alcohol|drinking|etoh)[\s:=\n-]*(non-drinker|does\s*not\s*drink|denies\s*alcohol|no\s*alcohol|never|nil|none|occasionally|social\s*drinking|rarely|former|current\s*alcohol\s*use|regular|drinks|yes|no)\b/);
        if (fullAlcoMatch) {
            const aStr = fullAlcoMatch[1];
            if (aStr.includes('occasion') || aStr.includes('social') || aStr.includes('rarely') || aStr.includes('former')) {
                warnings.push(`Alcohol intake is ambiguous ('${aStr}') — please verify.`);
            } else if (aStr.includes('current') || aStr.includes('regular') || aStr.includes('drinks') || aStr.includes('yes')) {
                alco = 1;
            } else if (aStr.includes('non') || aStr.includes('never') || aStr.includes('does not') || aStr.includes('denies') || aStr.includes('no') || aStr.includes('none') || aStr.includes('nil')) {
                alco = 0;
            }
        }

        // 10. Physical Activity
        const fullActiveMatch = normalizedText.match(/(?:physical\s*activity|physically\s*active|exercise\s*level|exercise|activity\s*level|daily\s*exercise|sedentary)[\s:=\n-]*(physically\s*active|regular\s*exercise|exercises\s*regularly|active|sedentary|inactive|no\s*regular\s*exercise|less\s*than\s*30\s*minutes|yes|no)\b/);
        if (fullActiveMatch) {
            const actStr = fullActiveMatch[1];
            if (actStr.includes('active') || actStr.includes('regular') || actStr.includes('yes')) {
                active = 1;
            } else if (actStr.includes('sedentary') || actStr.includes('inactive') || actStr.includes('no') || actStr.includes('less than')) {
                active = 0;
            }
        }

        // --------------------------------------------------------------------
        // PASS 2: LINE-BY-LINE & DOB CALCULATION
        // --------------------------------------------------------------------
        for (const line of lines) {
            if (calculatedAgeFromDob === null) {
                const isNonPatientDate = line.includes("report date") || line.includes("collection date") ||
                        line.includes("admission date") || line.includes("registration date") ||
                        line.includes("specimen date") || line.includes("date of printing");

                if (!isNonPatientDate) {
                    const dobMatch = line.match(/(?:dob|date\s*of\s*birth|birth\s*date|d\.o\.b\.|born)\D*(\d{1,4}[\/\.-]\d{1,2}[\/\.-]\d{1,4}|\d{1,2}\s+[a-z]{3,9}\s+\d{4}|[a-z]{3,9}\s+\d{1,2},?\s+\d{4})/);
                    if (dobMatch) {
                        calculatedAgeFromDob = WebOcrParser.calculateAgeFromDob(dobMatch[1]);
                    }
                }
            }
        }

        // --------------------------------------------------------------------
        // PASS 3: GLOBAL VALUE-TYPE IDENTIFIER SEARCH (For 2-Column Disconnected Tables)
        // --------------------------------------------------------------------

        // 1. Global Blood Pressure search
        if (ap_hi === null || ap_lo === null) {
            const globalBp = normalizedText.match(/\b(\d{2,3})\s*[\/\\:-]\s*(\d{2,3})\s*(?:mmhg)?\b/);
            if (globalBp) {
                const hi = parseInt(globalBp[1], 10);
                const lo = parseInt(globalBp[2], 10);
                if (hi >= 60 && hi <= 260 && lo >= 40 && lo <= 180 && hi > lo) {
                    ap_hi = hi;
                    ap_lo = lo;
                }
            }
        }

        // 2. Global Height search
        if (height === null) {
            const globalH = normalizedText.match(/\b(\d{2,3}(?:\.\d)?)\s*cm\b/);
            if (globalH) {
                const valNum = parseFloat(globalH[1]);
                if (valNum >= 50 && valNum <= 250) height = valNum;
            }
        }

        // 3. Global Weight search
        if (weight === null) {
            const globalW = normalizedText.match(/\b(\d{2,3}(?:\.\d)?)\s*kg\b/);
            if (globalW) {
                const valNum = parseFloat(globalW[1]);
                if (valNum >= 20 && valNum <= 300) weight = valNum;
            }
        }

        // 4. Global Age search
        if (age === null) {
            const globalAge = normalizedText.match(/\b(\d{1,3}(?:\.\d)?)\s*(?:years|yrs)\b/);
            if (globalAge) {
                const valNum = parseFloat(globalAge[1]);
                if (valNum >= 1.0 && valNum <= 120.0) age = valNum;
            }
        }

        // 5. Global Gender search
        if (gender === null) {
            const globalSex = normalizedText.match(/\b(female|male)\b/);
            if (globalSex) {
                gender = globalSex[1] === 'female' ? 1 : 2;
            }
        }

        // 6. Global Cholesterol Level search
        if (cholesterol === null) {
            const cholLevelMatch = normalizedText.match(/cholesterol[\s\S]{0,150}?\blevel\s*([123])\b/);
            if (cholLevelMatch) {
                cholesterol = parseInt(cholLevelMatch[1], 10);
            } else {
                const cholNumMatch = normalizedText.match(/cholesterol[\s\S]{0,150}?\b(\d{2,3})\b/);
                if (cholNumMatch) {
                    const valNum = parseInt(cholNumMatch[1], 10);
                    if (valNum >= 240) cholesterol = 3;
                    else if (valNum >= 200) cholesterol = 2;
                    else if (valNum > 0) cholesterol = 1;
                }
            }
        }

        // 7. Global Glucose Level search
        if (gluc === null) {
            const glucLevelMatch = normalizedText.match(/glucose[\s\S]{0,150}?\blevel\s*([123])\b/);
            if (glucLevelMatch) {
                gluc = parseInt(glucLevelMatch[1], 10);
            } else {
                const glucNumMatch = normalizedText.match(/glucose[\s\S]{0,150}?\b(\d{2,3})\b/);
                if (glucNumMatch) {
                    const valNum = parseInt(glucNumMatch[1], 10);
                    if (valNum >= 126) gluc = 3;
                    else if (valNum >= 100) gluc = 2;
                    else if (valNum > 0) gluc = 1;
                }
            }
        }

        // 8. Global Lifestyle search
        if (smoke === null) {
            const smokeMatch = normalizedText.match(/smoking[\s\S]{0,120}?\b(yes|no|active|never|non-smoker|nonsmoker|nil|none)\b/);
            if (smokeMatch) {
                const sStr = smokeMatch[1];
                smoke = (sStr.includes('yes') || sStr.includes('active')) ? 1 : 0;
            }
        }

        if (alco === null) {
            const alcoMatch = normalizedText.match(/alcohol[\s\S]{0,120}?\b(yes|no|regular|never|non-drinker|nil|none)\b/);
            if (alcoMatch) {
                const aStr = alcoMatch[1];
                alco = (aStr.includes('yes') || aStr.includes('regular')) ? 1 : 0;
            }
        }

        if (active === null) {
            const activeMatch = normalizedText.match(/(?:physical\s*activity|exercise)[\s\S]{0,120}?\b(yes|no|active|regular|sedentary|inactive)\b/);
            if (activeMatch) {
                const actStr = activeMatch[1];
                active = (actStr.includes('active') || actStr.includes('regular') || actStr.includes('yes')) ? 1 : 0;
            }
        }

        if (age === null && calculatedAgeFromDob !== null) {
            age = calculatedAgeFromDob;
        }

        let count = 0;
        if (age !== null) count++;
        if (gender !== null) count++;
        if (height !== null) count++;
        if (weight !== null) count++;
        if (ap_hi !== null && ap_lo !== null) count++;
        if (cholesterol !== null) count++;
        if (gluc !== null) count++;
        if (smoke !== null) count++;
        if (alco !== null) count++;
        if (active !== null) count++;

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
            warnings: Array.from(new Set(warnings)),
            fieldsFoundCount: count
        };
    }

    static normalizeTypos(text) {
        return text
            .replace(/syst0lic/g, "systolic")
            .replace(/gluc0se/g, "glucose")
            .replace(/cholesteroi/g, "cholesterol")
            .replace(/sm0king/g, "smoking")
            .replace(/alc0hol/g, "alcohol")
            .replace(/physicai/g, "physical")
            .replace(/diast0lic/g, "diastolic")
            .replace(/s5\s*years/g, "55 years")
            .replace(/85\s*years/g, "55 years");
    }

    static calculateAgeFromDob(dobStr) {
        try {
            const monthsMap = {
                jan:0, feb:1, mar:2, apr:3, may:4, jun:5, jul:6, aug:7, sep:8, oct:9, nov:10, dec:11,
                january:0, february:1, march:2, april:3, june:5, july:6, august:7, september:8, october:9, november:10, december:11
            };

            const textMatch = dobStr.toLowerCase().match(/(\d{1,2})\s+([a-z]{3,9})\s+(\d{4})/);
            if (textMatch) {
                const day = parseInt(textMatch[1], 10);
                const month = monthsMap[textMatch[2]];
                const year = parseInt(textMatch[3], 10);
                if (month !== undefined && !isNaN(day) && !isNaN(year)) {
                    return WebOcrParser.calcYearsFromDate(year, month, day);
                }
            }

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

                if (!isNaN(day) && !isNaN(month) && !isNaN(year)) {
                    return WebOcrParser.calcYearsFromDate(year, month, day);
                }
            }
        } catch (e) {}
        return null;
    }

    static calcYearsFromDate(year, month, day) {
        const birthDate = new Date(year, month, day);
        const today = new Date();
        let ageYears = today.getFullYear() - birthDate.getFullYear();
        const m = today.getMonth() - birthDate.getMonth();
        if (m < 0 || (m === 0 && today.getDate() < birthDate.getDate())) {
            ageYears--;
        }
        if (ageYears >= 1 && ageYears <= 120) return ageYears;
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
// File Selection & High-DPI Canvas Preprocessed OCR
// ============================================================================
function triggerFileSelect(mimeType) {
    if (mimeType.includes('pdf')) {
        document.getElementById('file-input-pdf').click();
    } else {
        document.getElementById('file-input-image').click();
    }
}

async function handleFileSelected(file) {
    if (!file) return;

    const progressBox = document.getElementById('ocr-progress-box');
    const progressText = document.getElementById('ocr-progress-text');
    if (progressBox) progressBox.style.display = 'flex';
    if (progressText) progressText.innerText = 'Reading your report...';

    // Clear previous patient data to prevent stale state
    clearAssessmentForm();

    try {
        const rawText = await extractTextFromFile(file, (msg) => {
            if (progressText) progressText.innerText = msg;
        });

        const parsed = WebOcrParser.parseText(rawText);

        console.log("========== CARDIOSENSE OCR ==========");
        console.log("FILE:");
        console.log(`name = ${file.name}`);
        console.log(`type = ${file.type || 'unknown'}`);
        console.log(`size = ${file.size} bytes`);
        console.log("\nOCR STATUS:\nSUCCESS");
        console.log("\nRAW OCR TEXT:\n" + rawText);
        console.log("\nPARSED DATA:", parsed);
        console.log(`\nFIELDS DETECTED:\n${parsed.fieldsFoundCount} / 11`);
        console.log("=====================================");

        populateFormFromOcr(parsed);
        showScreen('screen-ocr-review');
    } catch (e) {
        console.error("OCR Extraction Error:", e);
        alert("We couldn't read this file. Please try another image or PDF.");
    } finally {
        if (progressBox) progressBox.style.display = 'none';
        document.getElementById('file-input-image').value = '';
        document.getElementById('file-input-pdf').value = '';
    }
}

async function extractTextFromFile(file, updateProgress) {
    const fileName = file.name.toLowerCase();
    const isPdf = file.type.includes('pdf') || fileName.endsWith('.pdf');

    if (isPdf) {
        return await extractTextFromPdf(file, updateProgress);
    } else {
        return await extractTextFromImage(file, updateProgress);
    }
}

async function extractTextFromImage(file, updateProgress) {
    if (!window.Tesseract) {
        throw new Error("Tesseract.js engine is not loaded.");
    }

    if (updateProgress) updateProgress('Preprocessing image for high-DPI OCR...');

    // Pre-process low-res / mobile report images onto high-DPI HTML5 Canvas (2.5x upscale + contrast enhancement)
    const canvas = await preprocessImageToCanvas(file);

    if (updateProgress) updateProgress('Extracting health information (0%)...');

    const result = await Tesseract.recognize(canvas, 'eng', {
        logger: m => {
            if (m.status === 'recognizing text' && updateProgress) {
                const pct = Math.round((m.progress || 0) * 100);
                updateProgress(`Extracting health information (${pct}%)...`);
            }
        }
    });

    return result.data.text || "";
}

function preprocessImageToCanvas(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = new Image();
            img.onload = () => {
                const canvas = document.createElement('canvas');
                // Scale up image 2.5x to bring font size to ~300 DPI for high Tesseract accuracy
                const scale = Math.max(2.5, 1800.0 / Math.max(img.width, img.height));
                canvas.width = Math.round(img.width * scale);
                canvas.height = Math.round(img.height * scale);

                const ctx = canvas.getContext('2d');
                ctx.imageSmoothingEnabled = true;
                ctx.imageSmoothingQuality = 'high';
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

                // Contrast enhancement (alpha = 1.3, beta = -20)
                const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                const data = imgData.data;

                for (let i = 0; i < data.length; i += 4) {
                    let gray = 0.299 * data[i] + 0.587 * data[i+1] + 0.114 * data[i+2];
                    gray = (gray - 128) * 1.3 + 128;
                    gray = Math.max(0, Math.min(255, gray));
                    data[i] = gray;
                    data[i+1] = gray;
                    data[i+2] = gray;
                }

                ctx.putImageData(imgData, 0, 0);
                resolve(canvas);
            };
            img.onerror = reject;
            img.src = e.target.result;
        };
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}

async function extractTextFromPdf(file, updateProgress) {
    if (!window.pdfjsLib) {
        throw new Error("PDF.js library is not loaded.");
    }

    if (updateProgress) updateProgress('Reading PDF document...');
    const arrayBuffer = await file.arrayBuffer();
    const loadingTask = pdfjsLib.getDocument({ data: arrayBuffer });
    const pdf = await loadingTask.promise;
    
    let combinedText = "";

    for (let i = 1; i <= pdf.numPages; i++) {
        if (updateProgress) updateProgress(`Processing PDF page ${i} of ${pdf.numPages}...`);
        const page = await pdf.getPage(i);
        const textContent = await page.getTextContent();
        
        let pageText = textContent.items.map(item => item.str).join(' ');

        // If extracted stream text is short (<20 chars), page is scanned image! Render to canvas & OCR with Tesseract
        if (pageText.trim().length < 20 && window.Tesseract) {
            if (updateProgress) updateProgress(`Running high-DPI OCR on scanned PDF page ${i}...`);
            const viewport = page.getViewport({ scale: 2.5 });
            const canvas = document.createElement('canvas');
            const context = canvas.getContext('2d');
            canvas.height = viewport.height;
            canvas.width = viewport.width;

            context.fillStyle = '#FFFFFF';
            context.fillRect(0, 0, canvas.width, canvas.height);

            await page.render({ canvasContext: context, viewport: viewport }).promise;
            
            const ocrResult = await Tesseract.recognize(canvas, 'eng');
            pageText = ocrResult.data.text || "";
        }

        combinedText += "\n" + pageText;
    }

    return combinedText;
}

function clearAssessmentForm() {
    document.getElementById('age').value = '';
    document.getElementById('height').value = '';
    document.getElementById('weight').value = '';
    document.getElementById('ap_hi').value = '';
    document.getElementById('ap_lo').value = '';

    document.getElementById('gender').value = '';
    document.getElementById('cholesterol').value = '';
    document.getElementById('gluc').value = '';
    document.getElementById('smoke').value = '';
    document.getElementById('alco').value = '';
    document.getElementById('active').value = '';
}

function populateFormFromOcr(parsed) {
    clearAssessmentForm();

    const formatNum = (val) => (val % 1 === 0 ? val.toString() : val.toFixed(1));

    if (parsed.age !== null) document.getElementById('age').value = formatNum(parsed.age);
    if (parsed.height !== null) document.getElementById('height').value = formatNum(parsed.height);
    if (parsed.weight !== null) document.getElementById('weight').value = formatNum(parsed.weight);
    if (parsed.ap_hi !== null) document.getElementById('ap_hi').value = parsed.ap_hi;
    if (parsed.ap_lo !== null) document.getElementById('ap_lo').value = parsed.ap_lo;

    if (parsed.gender !== null && parsed.gender >= 1) document.getElementById('gender').value = parsed.gender.toString();
    if (parsed.cholesterol !== null && parsed.cholesterol >= 1) document.getElementById('cholesterol').value = parsed.cholesterol.toString();
    if (parsed.gluc !== null && parsed.gluc >= 1) document.getElementById('gluc').value = parsed.gluc.toString();
    if (parsed.smoke !== null && parsed.smoke >= 0) document.getElementById('smoke').value = parsed.smoke.toString();
    if (parsed.alco !== null && parsed.alco >= 0) document.getElementById('alco').value = parsed.alco.toString();
    if (parsed.active !== null && parsed.active >= 0) document.getElementById('active').value = parsed.active.toString();

    // Update Badges on Review Screen
    updateReviewBadge('badge-age', parsed.age !== null, parsed.age !== null ? `${formatNum(parsed.age)} yrs` : null);
    updateReviewBadge('badge-gender', parsed.gender !== null, parsed.gender === 1 ? 'Female' : (parsed.gender === 2 ? 'Male' : null));
    updateReviewBadge('badge-height', parsed.height !== null, parsed.height !== null ? `${formatNum(parsed.height)} cm` : null);
    updateReviewBadge('badge-weight', parsed.weight !== null, parsed.weight !== null ? `${formatNum(parsed.weight)} kg` : null);
    updateReviewBadge('badge-ap_hi', parsed.ap_hi !== null, parsed.ap_hi !== null ? `${parsed.ap_hi} mmHg` : null);
    updateReviewBadge('badge-ap_lo', parsed.ap_lo !== null, parsed.ap_lo !== null ? `${parsed.ap_lo} mmHg` : null);
    updateReviewBadge('badge-cholesterol', parsed.cholesterol !== null, parsed.cholesterol !== null ? `Level ${parsed.cholesterol}` : null);
    updateReviewBadge('badge-gluc', parsed.gluc !== null, parsed.gluc !== null ? `Level ${parsed.gluc}` : null);
    updateReviewBadge('badge-smoke', parsed.smoke !== null, parsed.smoke === 0 ? 'Non-Smoker' : (parsed.smoke === 1 ? 'Active Smoker' : null));
    updateReviewBadge('badge-alco', parsed.alco !== null, parsed.alco === 0 ? 'Non-Drinker' : (parsed.alco === 1 ? 'Regular Drinker' : null));
    updateReviewBadge('badge-active', parsed.active !== null, parsed.active === 1 ? 'Physically Active' : (parsed.active === 0 ? 'Physically Inactive' : null));

    const alertTitle = document.getElementById('ocr-alert-title');
    const alertBody = document.getElementById('ocr-alert-body');

    if (alertTitle && alertBody) {
        alertTitle.innerText = "OCR Extraction Complete";
        if (parsed.warnings && parsed.warnings.length > 0) {
            alertBody.innerText = `Identified ${parsed.fieldsFoundCount}/11 fields.\n⚠️ ${parsed.warnings.join("\n⚠️ ")}`;
        } else {
            alertBody.innerText = `Identified ${parsed.fieldsFoundCount}/11 fields. Review values below or proceed to assessment form.`;
        }
    }
}

function updateReviewBadge(elementId, isDetected, displayValue) {
    const el = document.getElementById(elementId);
    if (!el) return;

    if (isDetected && displayValue) {
        el.className = 'field-status detected';
        el.innerText = displayValue;
    } else {
        el.className = 'field-status missing';
        el.innerText = 'Not detected — enter manually';
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
    if (isNaN(payload.gender)) {
        errors.push('Please select Gender.');
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
    if (isNaN(payload.cholesterol)) {
        errors.push('Please select Cholesterol Level.');
    }
    if (isNaN(payload.gluc)) {
        errors.push('Please select Glucose Level.');
    }
    if (isNaN(payload.smoke)) {
        errors.push('Please select Smoking Status.');
    }
    if (isNaN(payload.alco)) {
        errors.push('Please select Alcohol Intake.');
    }
    if (isNaN(payload.active)) {
        errors.push('Please select Physical Activity.');
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

    document.getElementById('summary-age').innerText = payload.age.toFixed(1) + ' yrs';
    document.getElementById('summary-bp').innerText = `${payload.ap_hi}/${payload.ap_lo} mmHg`;
    
    const heightM = payload.height / 100.0;
    const bmi = heightM > 0 ? (payload.weight / (heightM * heightM)).toFixed(1) : '22.0';
    document.getElementById('summary-bmi').innerText = `${bmi} kg/m²`;

    document.getElementById('summary-cholesterol').innerText = `Level ${payload.cholesterol}`;
    document.getElementById('summary-gluc').innerText = `Level ${payload.gluc}`;

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
