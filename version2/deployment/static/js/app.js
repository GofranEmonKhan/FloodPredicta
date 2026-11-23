// Main Dashboard Application Controller
let currentLang = 'bn'; // Default to authentic Bengali
let stationMetadata = {};

document.addEventListener('DOMContentLoaded', async () => {
    // 1. Initialize Theme from localStorage
    const savedTheme = localStorage.getItem('floodpredicta_theme') || 'light';
    setTheme(savedTheme);

    // 2. Fetch Station Metadata
    try {
        const res = await fetch('/api/stations');
        stationMetadata = await res.json();
        populateStationDropdown();
    } catch (err) {
        console.error('Failed to load station metadata:', err);
    }

    // 3. Populate Months
    populateMonthDropdown();

    // 4. Setup Event Listeners
    setupEventListeners();

    // 5. Update Language UI
    updateLanguageUI();

    // 6. Pre-select Sylhet for demonstration
    const stSelect = document.getElementById('stationSelect');
    if (stSelect) {
        stSelect.value = 'Sylhet';
        onStationOrMonthChange();
    }
});

// Theme Toggle
function setTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('floodpredicta_theme', theme);
    const themeBtnText = document.getElementById('themeBtnText');
    const themeIcon = document.getElementById('themeIcon');
    if (theme === 'dark') {
        if (themeBtnText) themeBtnText.setAttribute('data-i18n', 'night_mode');
        if (themeIcon) themeIcon.textContent = '🌙';
    } else {
        if (themeBtnText) themeBtnText.setAttribute('data-i18n', 'day_mode');
        if (themeIcon) themeIcon.textContent = '☀️';
    }
    updateLanguageUI();
}

function toggleTheme() {
    const current = document.documentElement.getAttribute('data-theme') || 'light';
    setTheme(current === 'light' ? 'dark' : 'light');
}

// Language Switch
function toggleLanguage() {
    currentLang = currentLang === 'en' ? 'bn' : 'en';
    updateLanguageUI();
    populateStationDropdown();
    populateMonthDropdown();
}

window.updateLanguageUI = function() {
    const dict = translations[currentLang] || translations.en;
    
    // Update elements with data-i18n
    document.querySelectorAll('[data-i18n]').forEach(el => {
        const key = el.getAttribute('data-i18n');
        if (dict[key]) {
            if (el.tagName === 'INPUT' && el.getAttribute('placeholder')) {
                el.placeholder = dict[key];
            } else {
                el.textContent = dict[key];
            }
        }
    });

    const langBtnText = document.getElementById('langBtnText');
    if (langBtnText) langBtnText.textContent = dict.lang_switch;
};

// Populate Stations
function populateStationDropdown() {
    const select = document.getElementById('stationSelect');
    if (!select || !stationMetadata) return;

    const currentVal = select.value || 'Sylhet';
    select.innerHTML = '';

    const sortedKeys = Object.keys(stationMetadata).sort();
    sortedKeys.forEach(stKey => {
        const opt = document.createElement('option');
        opt.value = stKey;
        const bnName = stationNamesBn[stKey] || stKey;
        opt.textContent = currentLang === 'bn' ? `${bnName} (${stKey})` : `${stKey} (${bnName})`;
        select.appendChild(opt);
    });

    select.value = currentVal;
}

// Populate Months
function populateMonthDropdown() {
    const select = document.getElementById('monthSelect');
    if (!select) return;

    const currentVal = select.value || '7'; // Default to July
    select.innerHTML = '';

    const list = monthNames[currentLang] || monthNames.en;
    list.forEach((mStr, idx) => {
        const opt = document.createElement('option');
        opt.value = idx + 1;
        opt.textContent = mStr;
        select.appendChild(opt);
    });

    select.value = currentVal;
}

// Event Listeners
function setupEventListeners() {
    // Theme Button
    document.getElementById('themeToggleBtn')?.addEventListener('click', toggleTheme);
    
    // Language Button
    document.getElementById('langToggleBtn')?.addEventListener('click', toggleLanguage);
    
    // River Audio Button
    document.getElementById('audioToggleBtn')?.addEventListener('click', () => {
        window.riverSound?.toggle();
    });

    // Station & Month Change
    document.getElementById('stationSelect')?.addEventListener('change', onStationOrMonthChange);
    document.getElementById('monthSelect')?.addEventListener('change', onStationOrMonthChange);

    // Auto-fill Lags button
    document.getElementById('btnAutofillLags')?.addEventListener('click', autofillLags);

    // Form Submit
    document.getElementById('predictionForm')?.addEventListener('submit', onFormSubmit);

    // Preset Chips
    document.getElementById('preset1998')?.addEventListener('click', () => applyPreset('1998'));
    document.getElementById('presetSylhet')?.addEventListener('click', () => applyPreset('sylhet'));
    document.getElementById('presetWinter')?.addEventListener('click', () => applyPreset('winter'));
}

function onStationOrMonthChange() {
    const st = document.getElementById('stationSelect')?.value;
    const m = document.getElementById('monthSelect')?.value || '7';
    if (!st || !stationMetadata[st]) return;

    const meta = stationMetadata[st];
    
    // Update Micro-meta badges
    const altEl = document.getElementById('stationAlt');
    const coordsEl = document.getElementById('stationCoords');
    if (altEl) altEl.textContent = `${meta.alt} m`;
    if (coordsEl) coordsEl.textContent = `${meta.latitude}° N, ${meta.longitude}° E`;

    // Normal monthly average
    const normalRain = meta.monthly_avg_rainfall?.[m] || 350;
    const hintEl = document.getElementById('rainfallNormalVal');
    if (hintEl) hintEl.textContent = `${normalRain} mm`;
}

function autofillLags() {
    const st = document.getElementById('stationSelect')?.value;
    const m = parseInt(document.getElementById('monthSelect')?.value || '7');
    if (!st || !stationMetadata[st]) return;

    const meta = stationMetadata[st];
    const prev1 = m === 1 ? 12 : m - 1;
    const prev2 = prev1 === 1 ? 12 : prev1 - 1;

    const lag1Val = meta.monthly_avg_rainfall?.[String(prev1)] || 280;
    const lag2Val = meta.monthly_avg_rainfall?.[String(prev2)] || 220;

    document.getElementById('inputLag1').value = lag1Val;
    document.getElementById('inputLag2').value = lag2Val;
}

// Preset Scenarios
function applyPreset(type) {
    if (type === '1998') {
        // Replicate historic mega-monsoon deluge
        document.getElementById('stationSelect').value = 'Dhaka';
        document.getElementById('monthSelect').value = '8'; // August
        onStationOrMonthChange();
        document.getElementById('inputRainfall').value = '680';
        document.getElementById('inputLag1').value = '550';
        document.getElementById('inputLag2').value = '480';
        document.getElementById('inputMaxTemp').value = '32.5';
        document.getElementById('inputMinTemp').value = '26.0';
        document.getElementById('inputHumidity').value = '92';
        document.getElementById('inputWind').value = '2.4';
        document.getElementById('inputCloud').value = '7.6';
        document.getElementById('inputSun').value = '2.8';
    } else if (type === 'sylhet') {
        // Sylhet haor flash flood
        document.getElementById('stationSelect').value = 'Sylhet';
        document.getElementById('monthSelect').value = '7'; // July
        onStationOrMonthChange();
        document.getElementById('inputRainfall').value = '820';
        document.getElementById('inputLag1').value = '710';
        document.getElementById('inputLag2').value = '620';
        document.getElementById('inputMaxTemp').value = '31.8';
        document.getElementById('inputMinTemp').value = '25.2';
        document.getElementById('inputHumidity').value = '94';
        document.getElementById('inputWind').value = '3.0';
        document.getElementById('inputCloud').value = '7.8';
        document.getElementById('inputSun').value = '1.9';
    } else if (type === 'winter') {
        // Dry winter baseline
        document.getElementById('stationSelect').value = 'Dinajpur';
        document.getElementById('monthSelect').value = '1'; // January
        onStationOrMonthChange();
        document.getElementById('inputRainfall').value = '2';
        document.getElementById('inputLag1').value = '0';
        document.getElementById('inputLag2').value = '5';
        document.getElementById('inputMaxTemp').value = '24.0';
        document.getElementById('inputMinTemp').value = '10.5';
        document.getElementById('inputHumidity').value = '68';
        document.getElementById('inputWind').value = '1.2';
        document.getElementById('inputCloud').value = '0.8';
        document.getElementById('inputSun').value = '8.2';
    }

    // Trigger auto-submit to see preset result
    document.getElementById('btnSubmit')?.click();
}

// Form Submission & Prediction
async function onFormSubmit(e) {
    e.preventDefault();

    const btnSubmit = document.getElementById('btnSubmit');
    const btnText = document.getElementById('btnSubmitText');
    const dict = translations[currentLang] || translations.en;
    
    if (btnText) btnText.textContent = dict.btn_predicting;
    if (btnSubmit) btnSubmit.disabled = true;

    const payload = {
        station_name: document.getElementById('stationSelect').value,
        month: parseInt(document.getElementById('monthSelect').value),
        rainfall: parseFloat(document.getElementById('inputRainfall').value),
        rainfall_lag1: parseFloat(document.getElementById('inputLag1').value),
        rainfall_lag2: parseFloat(document.getElementById('inputLag2').value),
        max_temp: parseFloat(document.getElementById('inputMaxTemp').value),
        min_temp: parseFloat(document.getElementById('inputMinTemp').value),
        relative_humidity: parseFloat(document.getElementById('inputHumidity').value),
        wind_speed: parseFloat(document.getElementById('inputWind').value),
        cloud_coverage: parseFloat(document.getElementById('inputCloud').value),
        bright_sunshine: parseFloat(document.getElementById('inputSun').value),
        threshold_mode: document.querySelector('input[name="thresholdMode"]:checked')?.value || 'balanced'
    };

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await response.json();
        if (data.status === 'success') {
            renderAssessmentResults(data);
        } else {
            alert('Prediction error: ' + (data.message || 'Unknown'));
        }
    } catch (err) {
        console.error('API Error:', err);
        alert('Failed to connect to FloodPredicta server.');
    } finally {
        if (btnText) btnText.textContent = dict.btn_predict;
        if (btnSubmit) btnSubmit.disabled = false;
    }
}

// Render Results
function renderAssessmentResults(data) {
    document.getElementById('emptyStateBox').style.display = 'none';
    document.getElementById('resultsActiveBox').style.display = 'block';

    const probVal = data.probability;
    const probEl = document.getElementById('gaugeProbability');
    if (probEl) probEl.textContent = `${probVal}%`;

    // Animate Needle (0 to 180 deg)
    const angle = (probVal / 100) * 180 - 90;
    const needle = document.getElementById('gaugeNeedle');
    if (needle) {
        needle.style.transform = `rotate(${angle}deg)`;
        needle.style.transformOrigin = '100px 95px';
    }

    // Status Badge
    const statusPill = document.getElementById('statusPill');
    if (statusPill) {
        statusPill.textContent = currentLang === 'bn' ? data.badge_bn : data.badge_en;
        statusPill.style.backgroundColor = data.badge_color;
        statusPill.style.color = '#ffffff';
    }

    // Drivers
    const driversList = document.getElementById('driversList');
    if (driversList) {
        driversList.innerHTML = '';
        data.drivers.forEach(d => {
            const div = document.createElement('div');
            div.className = 'driver-item';
            div.textContent = currentLang === 'bn' ? d.bn : d.en;
            driversList.appendChild(div);
        });
    }

    // Advisory
    const advEl = document.getElementById('advisoryText');
    if (advEl) {
        advEl.textContent = currentLang === 'bn' ? data.advisory_bn : data.advisory_en;
    }

    // Model Info
    const modelEl = document.getElementById('activeModelName');
    if (modelEl) {
        modelEl.textContent = `${data.model_name} (${data.model_regime})`;
    }
}
