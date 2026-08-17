const $ = (selector) => document.querySelector(selector);
let componentChart, iriChart, networkChart;
let currentResult = null;
let currentHistory = [];
let currentBasin = null;
let currentConfidence = null;
let selectedSection = null;
let availableSections = [];

const palette = { Good: '#12965a', Fair: '#ee8b2d', Poor: '#dd4d43' };

function chartOptions(extra = {}) {
  return { responsive: true, maintainAspectRatio: false, plugins: { legend: { labels: { boxWidth: 10, font: { family: 'Manrope' } } } }, ...extra };
}

function renderCharts() {
  const result = currentResult;
  if (!result) return;
  componentChart?.destroy(); iriChart?.destroy();

  const histIriScore = result.historical_snapshot ? result.historical_snapshot.iri_score : result.iri_score;
  const histFwdScore = (result.historical_snapshot && result.historical_snapshot.fwd_score !== null) ? result.historical_snapshot.fwd_score : (result.fwd_score ?? 0);
  const presentIriScore = result.present_estimation ? result.present_estimation.iri_score : result.iri_score;

  componentChart = new Chart($('#component-chart'), {
    type: 'bar',
    data: {
      labels: ['Historical Surface Score', 'Historical FWD Structural Score', 'Present-Day Surface Score'],
      datasets: [{
        data: [histIriScore, histFwdScore, presentIriScore],
        backgroundColor: [
          '#267a62',
          (result.historical_snapshot && result.historical_snapshot.fwd_available) ? '#548bdf' : '#d5dfd8',
          '#12965a'
        ],
        borderRadius: 6
      }]
    },
    options: chartOptions({ scales: { y: { min: 0, max: 100, grid: { color: '#edf1ee' } }, x: { grid: { display: false } } }, plugins: { legend: { display: false } } })
  });

  const historic = currentHistory.map((row) => ({ x: row.YEAR, y: row.MRI }));
  const histYear = result.historical_snapshot ? result.historical_snapshot.year : Number($('#year').value);
  const histMri = result.historical_snapshot ? result.historical_snapshot.measured_iri : Number($('#mri').value);

  // Simulation path from measurement year up to 2026
  const simulationData = (result.simulation_path && result.simulation_path.length > 0)
    ? result.simulation_path.map(p => ({ x: p.year, y: p.iri }))
    : [{ x: histYear, y: histMri }, { x: 2026, y: result.predicted_future_iri }];

  // 10-year horizon projection from 2026 to 2036
  const projectionData = [
    { x: 2026, y: (result.present_estimation ? result.present_estimation.estimated_iri : result.predicted_future_iri) },
    ...result.projection.map((point) => ({ x: point.year, y: point.iri }))
  ];

  const minX = historic.length ? Math.min(historic[0].x, histYear) : histYear;
  const maxX = result.projection.at(-1)?.year ?? 2036;

  iriChart = new Chart($('#iri-chart'), {
    type: 'line',
    data: {
      datasets: [
        { label: 'Historical Measurements', data: historic.length ? historic : [{ x: histYear, y: histMri }], borderColor: '#267a62', backgroundColor: '#267a62', tension: 0.25, pointRadius: 4 },
        { label: 'Simulated Trajectory to 2026', data: simulationData, borderColor: '#0ea5e9', borderDash: [4, 4], backgroundColor: '#0ea5e9', tension: 0.2, pointRadius: 3 },
        { label: '10-Year Deterioration Projection', data: projectionData, borderColor: '#ee8b2d', borderDash: [6, 5], tension: 0.2, pointRadius: 3 },
        { label: 'FHWA Failure Threshold (2.5 m/km)', data: [{ x: minX, y: 2.5 }, { x: maxX, y: 2.5 }], borderColor: '#dd4d43', borderDash: [3, 4], pointRadius: 0 },
      ]
    },
    options: chartOptions({
      scales: {
        x: { type: 'linear', ticks: { precision: 0 }, grid: { display: false } },
        y: { title: { display: true, text: 'IRI (m/km)' }, grid: { color: '#edf1ee' } }
      }
    })
  });
}

function updateResult(result) {
  currentResult = result;
  const color = palette[result.condition];

  // 1. Primary Gauge & Condition
  $('#rhi-score').textContent = result.rhi.toFixed(1);
  $('#condition').textContent = `${result.condition} condition`;
  $('#condition').style.color = color;
  $('#recommendation').textContent = result.recommendation;
  $('#gauge-fill').style.stroke = color;
  const arcLength = 245;
  const clampedRhi = Math.min(100, Math.max(0, result.rhi));
  $('#gauge-fill').style.strokeDashoffset = arcLength - (arcLength * clampedRhi / 100);

  // Category legend badges
  document.querySelectorAll('.category-badge').forEach((badge) => badge.classList.remove('active'));
  if (result.condition === 'Good') $('#badge-good')?.classList.add('active');
  else if (result.condition === 'Fair') $('#badge-fair')?.classList.add('active');
  else if (result.condition === 'Poor') $('#badge-poor')?.classList.add('active');

  // 2. Dual Timeline Intelligence Banner
  if (result.historical_snapshot) {
    const h = result.historical_snapshot;
    const histColor = palette[h.condition];
    $('#hist-year-badge').textContent = `Year ${h.year}`;
    $('#hist-rhi').textContent = h.rhi.toFixed(1);
    $('#hist-rhi').style.color = histColor;
    $('#hist-iri').textContent = `${h.measured_iri.toFixed(3)} m/km`;
    $('#hist-fwd').textContent = h.fwd_health ? (h.fwd_score !== null ? `${h.fwd_health} (${h.fwd_score.toFixed(1)})` : h.fwd_health) : 'N/A';
    $('#hist-cond').textContent = `${h.condition}`;
    $('#hist-cond').style.color = histColor;
  }

  if (result.present_estimation) {
    const p = result.present_estimation;
    const presColor = palette[p.condition];
    $('#today-rhi').textContent = p.rhi.toFixed(1);
    $('#today-rhi').style.color = presColor;
    $('#today-iri').textContent = `${p.estimated_iri.toFixed(3)} m/km`;
    $('#today-delta').textContent = `${p.iri_change >= 0 ? '+' : ''}${p.iri_change.toFixed(3)} m/km (${p.simulated_years}y fast-forward)`;
    $('#today-cond').textContent = `${p.condition}`;
    $('#today-cond').style.color = presColor;
  }

  // 3. Component Breakdown Metric Rows
  if ($('#hist-measured-iri')) {
    $('#hist-measured-iri').textContent = result.historical_snapshot ? `${result.historical_snapshot.measured_iri.toFixed(3)} m/km (${result.historical_snapshot.year})` : `${Number($('#mri').value).toFixed(3)} m/km`;
  }
  $('#future-iri').textContent = result.present_estimation ? `${result.present_estimation.estimated_iri.toFixed(3)} m/km (2026)` : `${result.predicted_future_iri.toFixed(3)} m/km`;
  $('#fwd-health').textContent = result.historical_snapshot ? result.historical_snapshot.fwd_health : (result.fwd_health ?? 'Not available');

  renderCharts();
}

function getDeflections() {
  return Array.from(document.querySelectorAll('.deflection')).map((input) => Number(input.value || 0));
}

function setValue(id, value) { if (value !== undefined && value !== null) $(id).value = value; }

function fillForm(values) {
  setValue('#mri', values.mri); setValue('#aadtt', values.aadtt); setValue('#truck-volume', values.annual_truck_volume);
  setValue('#annual-esal', values.annual_esal); setValue('#cumulative-esal', values.cumulative_esal); setValue('#year', values.year);
  setValue('#mean-ann-temp', values.mean_ann_temp_avg); setValue('#freeze-index', values.freeze_index_yr); setValue('#freeze-thaw', values.freeze_thaw_yr);
  $('#fwd-available').checked = values.fwd_available;
  toggleFwd();
  (values.deflections || []).forEach((value, index) => setValue(`#defl-${index + 1}`, value));
  setValue('#drop-load', values.drop_load); setValue('#drop-height', values.drop_height);
  setValue('#pavement-family', values.pavement_family); setValue('#lane-no', values.lane_no);
}

function validateNumericInputs() {
  const currentYear = new Date().getFullYear();
  const numericInputs = Array.from($('#predictor-form').querySelectorAll('input[type="number"]:not(:disabled)'));
  numericInputs.forEach((input) => {
    if (!input.value) return;
    const value = Number(input.value);
    if (!Number.isFinite(value)) throw new Error(`${input.id} must be a valid number.`);
    if (input.id === 'year') {
      if (value < 1980 || value > 2030 || value > currentYear) {
        throw new Error(`Measurement year must be between 1980 and ${currentYear}.`);
      }
      return;
    }
    const min = input.getAttribute('min');
    const max = input.getAttribute('max');
    if (min !== null && value < Number(min)) {
      throw new Error(`${input.id.replace(/-/g, ' ')} must be at least ${min}.`);
    }
    if (max !== null && value > Number(max)) {
      throw new Error(`${input.id.replace(/-/g, ' ')} must be at most ${max}.`);
    }
  });
}

function payload() {
  const fwdAvailable = $('#fwd-available').checked;
  validateNumericInputs();
  const year = Number($('#year').value);
  return {
    mri: Number($('#mri').value), aadtt: Number($('#aadtt').value), annual_truck_volume: Number($('#truck-volume').value),
    annual_esal: Number($('#annual-esal').value), cumulative_esal: Number($('#cumulative-esal').value), year,
    mean_ann_temp_avg: Number($('#mean-ann-temp').value), freeze_index_yr: Number($('#freeze-index').value), freeze_thaw_yr: Number($('#freeze-thaw').value),
    fwd_available: fwdAvailable,
    ...(fwdAvailable ? { deflections: getDeflections(), drop_load: Number($('#drop-load').value), drop_height: Number($('#drop-height').value), pavement_family: $('#pavement-family').value, lane_no: $('#lane-no').value } : {})
  };
}

function toggleFwd() {
  const isAvailable = $('#fwd-available').checked;
  $('#fwd-fields').style.display = isAvailable ? 'block' : 'none';
  const inputs = $('#fwd-fields').querySelectorAll('input, select');
  inputs.forEach((input) => {
    input.disabled = !isAvailable;
    input.required = isAvailable;
  });
}

async function requestPrediction() {
  const response = await fetch('/api/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload()) });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(getErrorMessage(data, 'Prediction failed.'));
  updateResult(data);
}

async function loadSection(shrpId, stateCode) {
  const response = await fetch(`/api/section/${encodeURIComponent(shrpId)}?state_code=${encodeURIComponent(stateCode)}`);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(getErrorMessage(data, 'Unable to load section.'));
  selectedSection = data.section;
  $('#selected-section').textContent = `Selected · SHRP ${data.section.shrp_id}, State ${data.section.state_code}, construction ${data.section.construction_no}`;
  const label = `SHRP ${data.section.shrp_id} · State ${data.section.state_code}`;
  const input = $('#section-search');
  if (input) input.value = label;
  hideSearchResults();
  currentHistory = data.history;
  currentBasin = data.deflection_basin;
  currentConfidence = data.deflection_confidence;
  fillForm(data.defaults);
  updateResult(data.prediction);
}

function getErrorMessage(error, fallback = 'Something went wrong.') {
  if (error == null) return fallback;
  if (typeof error === 'string') return error || fallback;
  if (error instanceof Error) return error.message || fallback;
  if (Array.isArray(error)) return error.map((item) => getErrorMessage(item, 'Invalid value')).join('; ');
  if (typeof error === 'object') {
    if ('detail' in error) return getErrorMessage(error.detail, fallback);
    if ('message' in error) return getErrorMessage(error.message, fallback);
    if ('error' in error) return getErrorMessage(error.error, fallback);
    if ('msg' in error) return getErrorMessage(error.msg, fallback);
    try { return JSON.stringify(error); } catch { return String(error); }
  }
  return String(error) || fallback;
}

function showError(error) { window.alert(getErrorMessage(error, 'Something went wrong.')); }

async function loadNetwork() {
  const response = await fetch('/api/network-summary');
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(getErrorMessage(data, 'Network summary unavailable.'));
  $('#network-status').textContent = `${data.total_sections.toLocaleString()} road sections monitored`;
  networkChart?.destroy();
  networkChart = new Chart($('#network-chart'), { type: 'doughnut', data: { labels: ['Good', 'Fair', 'Poor'], datasets: [{ data: ['Good', 'Fair', 'Poor'].map((key) => data.conditions[key]), backgroundColor: [palette.Good, palette.Fair, palette.Poor], borderWidth: 0 }] }, options: chartOptions({ cutout: '70%' }) });
}

async function loadMetadata() {
  const response = await fetch('/api/metadata'); const data = await response.json();
  ['pavement-family', 'lane-no'].forEach((id, index) => { $(`#${id}`).innerHTML = (index ? data.lanes : data.pavement_families).map((item) => `<option>${item}</option>`).join(''); });
}

async function downloadCsv() {
  const response = await fetch('/api/report.csv', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload()) });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(getErrorMessage(data, 'CSV export failed.'));
  const blob = await response.blob(); const url = URL.createObjectURL(blob); const link = document.createElement('a'); link.href = url; link.download = 'road-health-report.csv'; link.click(); URL.revokeObjectURL(url);
}

async function downloadBatchTemplate() {
  const response = await fetch('/api/batch-template.csv');
  if (!response.ok) throw new Error('Failed to download batch template.');
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = 'road-batch-template.csv';
  link.click();
  URL.revokeObjectURL(url);
}

async function uploadBatch() {
  const fileInput = $('#batch-file');
  const file = fileInput.files[0];
  if (!file) throw new Error('Choose a CSV or Excel file first.');
  const btn = $('#batch-button');
  const originalText = btn.textContent;
  btn.textContent = 'Scoring batch...';
  btn.disabled = true;
  try {
    const form = new FormData();
    form.append('file', file);
    const response = await fetch('/api/batch', { method: 'POST', body: form });
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new Error(getErrorMessage(data, 'Batch scoring failed.'));
    }
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'batch-rhi-results.csv';
    link.click();
    URL.revokeObjectURL(url);
  } finally {
    btn.textContent = originalText;
    btn.disabled = false;
  }
}

function downloadPdf() {
  if (!currentResult) return showError(new Error('Run a prediction first.'));
  const { jsPDF } = window.jspdf;
  const pdf = new jsPDF();
  const r = currentResult;
  const h = r.historical_snapshot || { year: $('#year').value, measured_iri: $('#mri').value, rhi: r.rhi, condition: r.condition, fwd_health: 'N/A' };
  const p = r.present_estimation || { year: 2026, estimated_iri: r.predicted_future_iri, rhi: r.rhi, condition: r.condition, simulated_years: 0, iri_change: 0 };

  pdf.setFillColor(29, 69, 59);
  pdf.rect(0, 0, 210, 35, 'F');
  pdf.setTextColor(255, 255, 255);
  pdf.setFontSize(18);
  pdf.text('Dual-Timeline Road Health Index Assessment', 15, 21);
  pdf.setTextColor(20, 38, 31);
  pdf.setFontSize(10.5);
  let y = 48;

  const rows = [
    ['Road Section', selectedSection ? `SHRP ${selectedSection.shrp_id} · State ${selectedSection.state_code}` : 'Live simulation scenario'],
    ['1. Historical Snapshot Year', `${h.year} (Physical survey date)`],
    ['Historical Measured IRI', `${Number(h.measured_iri).toFixed(3)} m/km`],
    ['Historical FWD Health', h.fwd_health ?? 'N/A (Dynamic fallback)'],
    ['Historical RHI Score', `${Number(h.rhi).toFixed(1)} / 100 (${h.condition})`],
    ['----------------------------------------', '------------------------------------------------------------'],
    ['2. Present Day Estimation', `Year ${p.year} (Time-Adjusted Forecast)`],
    ['Projection Interval', `${p.simulated_years} years of traffic & climate deterioration`],
    ['Estimated 2026 Present IRI', `${Number(p.estimated_iri).toFixed(3)} m/km (Change: ${p.iri_change >= 0 ? '+' : ''}${Number(p.iri_change).toFixed(3)} m/km)`],
    ['Present Day (2026) RHI', `${Number(p.rhi).toFixed(1)} / 100 (${p.condition})`],
    ['Structural Data Consideration', 'Historical FWD retained as baseline; excluded from present-day projection (physical re-survey required)'],
    ['Maintenance Action', r.recommendation]
  ];

  rows.forEach(([label, value]) => {
    pdf.setFont('helvetica', 'bold');
    pdf.text(label, 15, y);
    pdf.setFont('helvetica', 'normal');
    const lines = pdf.splitTextToSize(value, 115);
    pdf.text(lines, 80, y);
    y += Math.max(8, lines.length * 5.5 + 3);
  });

  pdf.setTextColor(104, 128, 120);
  pdf.setFontSize(8);
  pdf.text(`Generated on ${new Date().toLocaleString()} · Road Health Intelligence Platform`, 15, 285);
  pdf.save('road-health-report.pdf');
}

let lastSampleIndex = -1;

const testSamples = [
  {
    caseName: 'Sample 1: Prime Highway (2022 Scan)',
    category: 'Good',
    mri: 0.45, aadtt: 200, annual_truck_volume: 80000, annual_esal: 50000, cumulative_esal: 200000, year: 2022,
    mean_ann_temp_avg: 18.0, freeze_index_yr: 500, freeze_thaw_yr: 30,
    fwd_available: true, deflections: [120, 80, 60, 45, 35, 25, 15], drop_load: 710, drop_height: 4, pavement_family: 'ACUB', lane_no: 'F1'
  },
  {
    caseName: 'Sample 2: Smooth Expressway (2020 Scan)',
    category: 'Good',
    mri: 0.50, aadtt: 300, annual_truck_volume: 110000, annual_esal: 75000, cumulative_esal: 350000, year: 2020,
    mean_ann_temp_avg: 16.5, freeze_index_yr: 800, freeze_thaw_yr: 50,
    fwd_available: true, deflections: [150, 100, 75, 55, 42, 30, 20], drop_load: 710, drop_height: 4, pavement_family: 'ACTB', lane_no: 'F3'
  },
  {
    caseName: 'Sample 3: Resurfaced Corridor (2018 Scan)',
    category: 'Good',
    mri: 0.55, aadtt: 400, annual_truck_volume: 140000, annual_esal: 95000, cumulative_esal: 450000, year: 2018,
    mean_ann_temp_avg: 15.0, freeze_index_yr: 1000, freeze_thaw_yr: 70,
    fwd_available: true, deflections: [180, 120, 90, 68, 50, 38, 25], drop_load: 710, drop_height: 4, pavement_family: 'ACUB', lane_no: 'F1'
  },
  {
    caseName: 'Sample 4: Moderate Traffic Segment (2016 Scan)',
    category: 'Fair',
    mri: 0.70, aadtt: 550, annual_truck_volume: 200000, annual_esal: 150000, cumulative_esal: 750000, year: 2016,
    mean_ann_temp_avg: 14.0, freeze_index_yr: 1400, freeze_thaw_yr: 100,
    fwd_available: true, deflections: [250, 170, 125, 95, 72, 52, 35], drop_load: 710, drop_height: 4, pavement_family: 'ACTB', lane_no: 'F1'
  },
  {
    caseName: 'Sample 5: Suburban Arterial (2014 Scan)',
    category: 'Fair',
    mri: 0.80, aadtt: 700, annual_truck_volume: 250000, annual_esal: 200000, cumulative_esal: 1000000, year: 2014,
    mean_ann_temp_avg: 13.0, freeze_index_yr: 1800, freeze_thaw_yr: 130,
    fwd_available: true, deflections: [300, 200, 145, 110, 85, 62, 42], drop_load: 710, drop_height: 4, pavement_family: 'ACUB', lane_no: 'F3'
  },
  {
    caseName: 'Sample 6: Aging Rural Route (2012 Scan)',
    category: 'Fair',
    mri: 0.90, aadtt: 850, annual_truck_volume: 300000, annual_esal: 250000, cumulative_esal: 1300000, year: 2012,
    mean_ann_temp_avg: 12.0, freeze_index_yr: 2200, freeze_thaw_yr: 160,
    fwd_available: true, deflections: [360, 240, 175, 130, 100, 75, 50], drop_load: 710, drop_height: 4, pavement_family: 'ACTB', lane_no: 'F1'
  },
  {
    caseName: 'Sample 7: Heavy Industrial Route (2015 Scan)',
    category: 'Fair',
    mri: 1.05, aadtt: 950, annual_truck_volume: 340000, annual_esal: 290000, cumulative_esal: 1550000, year: 2015,
    mean_ann_temp_avg: 11.5, freeze_index_yr: 2600, freeze_thaw_yr: 180,
    fwd_available: true, deflections: [410, 270, 195, 145, 112, 84, 56], drop_load: 710, drop_height: 4, pavement_family: 'ACUB', lane_no: 'F3'
  },
  {
    caseName: 'Sample 8: Industrial Corridor (2011 Scan)',
    category: 'Poor',
    mri: 1.45, aadtt: 1500, annual_truck_volume: 550000, annual_esal: 480000, cumulative_esal: 2700000, year: 2011,
    mean_ann_temp_avg: 10.0, freeze_index_yr: 4000, freeze_thaw_yr: 280,
    fwd_available: true, deflections: [590, 400, 290, 220, 170, 130, 90], drop_load: 710, drop_height: 4, pavement_family: 'ACTB', lane_no: 'F1'
  },
  {
    caseName: 'Sample 9: Severely Rutted Arterial (2010 Scan)',
    category: 'Poor',
    mri: 2.10, aadtt: 2500, annual_truck_volume: 900000, annual_esal: 850000, cumulative_esal: 5500000, year: 2010,
    mean_ann_temp_avg: 8.0, freeze_index_yr: 6000, freeze_thaw_yr: 420,
    fwd_available: true, deflections: [880, 600, 440, 330, 250, 190, 135], drop_load: 710, drop_height: 4, pavement_family: 'ACUB', lane_no: 'F3'
  },
  {
    caseName: 'Sample 10: Critical Structural Failure (2008 Scan)',
    category: 'Poor',
    mri: 2.75, aadtt: 3600, annual_truck_volume: 1300000, annual_esal: 1250000, cumulative_esal: 9000000, year: 2008,
    mean_ann_temp_avg: 6.5, freeze_index_yr: 7800, freeze_thaw_yr: 540,
    fwd_available: true, deflections: [1250, 870, 640, 490, 370, 280, 200], drop_load: 710, drop_height: 4, pavement_family: 'ACTB', lane_no: 'F1'
  }
];

function renderTestMeterResult(sample, data, sampleNum) {
  const hist = data.historical_snapshot || {
    year: sample.year,
    measured_iri: sample.mri,
    iri_score: Math.max(0, Math.min(100, ((2.5 - sample.mri) / 2.5) * 100)),
    rhi: data.rhi,
    condition: data.condition,
    fwd_health: sample.fwd_available ? 'Available' : 'N/A (Dynamic fallback)',
    fwd_score: null,
    fwd_available: sample.fwd_available
  };
  const pres = data.present_estimation || {
    year: 2026,
    estimated_iri: data.predicted_future_iri,
    iri_score: data.iri_score,
    rhi: data.rhi,
    condition: data.condition,
    simulated_years: 2026 - sample.year,
    iri_change: Math.max(0, data.predicted_future_iri - sample.mri)
  };

  const html = `
    <div class="test-meter-card">
      <div class="test-card-header">
        <div class="test-title-group">
          <span class="test-badge-num">Profile #${sampleNum} of 10</span>
          <strong class="test-case-name">${sample.caseName}</strong>
        </div>
        <div id="test-header-tag" class="test-category-tag tag-${pres.condition.toLowerCase()}">
          ${pres.condition.toUpperCase()} (2026 FORECAST)
        </div>
      </div>

      <div class="test-summary-grid">
        <div class="summary-pill"><span>Measured Year:</span><strong>${sample.year}</strong></div>
        <div class="summary-pill"><span>Historical IRI:</span><strong>${sample.mri} m/km</strong></div>
        <div class="summary-pill"><span>Daily Trucks:</span><strong>${sample.aadtt.toLocaleString()}</strong></div>
        <div class="summary-pill"><span>Annual ESAL:</span><strong>${sample.annual_esal.toLocaleString()}</strong></div>
        <div class="summary-pill"><span>Pavement / Lane:</span><strong>${sample.pavement_family} / ${sample.lane_no}</strong></div>
        <div class="summary-pill"><span>Deflection D1–D7:</span><strong>${sample.deflections ? (sample.deflections[0] + '–' + sample.deflections[6] + ' μm') : 'N/A'}</strong></div>
      </div>

      <!-- Interactive Timeline Perspective Toggle Switch -->
      <div class="test-timeline-toggle-bar">
        <span class="timeline-toggle-title">Assessment Perspective:</span>
        <div class="timeline-toggle-controls">
          <button type="button" class="timeline-mode-btn" id="test-mode-hist" data-mode="historical">
            <span class="dot"></span>
            Survey Baseline (${hist.year})
          </button>
          <label class="test-switch-label" title="Toggle between Survey Baseline and 2026 AI Forecast">
            <input type="checkbox" id="test-timeline-checkbox" checked>
            <span class="test-switch-slider"></span>
          </label>
          <button type="button" class="timeline-mode-btn active" id="test-mode-pres" data-mode="present">
            <span class="dot"></span>
            Time-Adjusted Forecast (2026)
          </button>
        </div>
      </div>

      <div class="test-hero-display">
        <div class="test-gauge-wrap">
          <p id="test-meter-eyebrow" class="eyebrow" style="margin-bottom:12px;">TIME-ADJUSTED (2026) RHI METER</p>
          <div class="gauge">
            <svg viewBox="0 0 200 120">
              <path class="gauge-track gauge-track-poor" d="M 22 100 A 78 78 0 0 1 100 22"/>
              <path class="gauge-track gauge-track-fair" d="M 100 22 A 78 78 0 0 1 155.15 44.85"/>
              <path class="gauge-track gauge-track-good" d="M 155.15 44.85 A 78 78 0 0 1 178 100"/>

              <line class="gauge-tick" x1="13" y1="100" x2="7" y2="100"/>
              <line class="gauge-tick" x1="100" y1="13" x2="100" y2="7"/>
              <line class="gauge-tick" x1="161.5" y1="38.5" x2="165.8" y2="34.2"/>
              <line class="gauge-tick" x1="187" y1="100" x2="193" y2="100"/>

              <text class="gauge-scale-text" x="10" y="115" text-anchor="middle">0</text>
              <text class="gauge-scale-text" x="100" y="6" text-anchor="middle">50</text>
              <text class="gauge-scale-text" x="176" y="27" text-anchor="middle">75</text>
              <text class="gauge-scale-text" x="190" y="115" text-anchor="middle">100</text>

              <path id="test-gauge-fill" class="gauge-fill" d="M 22 100 A 78 78 0 0 1 178 100"/>
            </svg>
            <div>
              <strong id="test-meter-score">--</strong>
              <span>/ 100</span>
            </div>
          </div>

          <div class="rhi-category-legend">
            <div id="test-badge-poor" class="category-badge poor">
              <span class="badge-dot"></span>
              <div class="badge-info">
                <span class="badge-title">Poor</span>
                <span class="badge-range">0–49</span>
              </div>
            </div>
            <div id="test-badge-fair" class="category-badge fair">
              <span class="badge-dot"></span>
              <div class="badge-info">
                <span class="badge-title">Fair</span>
                <span class="badge-range">50–74</span>
              </div>
            </div>
            <div id="test-badge-good" class="category-badge good">
              <span class="badge-dot"></span>
              <div class="badge-info">
                <span class="badge-title">Good</span>
                <span class="badge-range">75–100</span>
              </div>
            </div>
          </div>

          <h2 id="test-meter-condition" style="text-align:center; margin-top:8px;">Awaiting mode</h2>
          <p id="test-meter-rec" class="muted" style="text-align:center; margin-top:4px; font-size:12px; line-height:1.5;">--</p>
        </div>

        <div class="test-scores-breakdown" id="test-breakdown-container">
          <!-- Dynamic Mode Breakdown Cards -->
        </div>
      </div>
    </div>
  `;

  const container = $('#test-rhi-result');
  container.innerHTML = html;
  container.classList.remove('hidden');

  function updateTimelinePerspective(isCurrentYear) {
    const arcLength = 245;
    const btnHist = $('#test-mode-hist');
    const btnPres = $('#test-mode-pres');
    const checkbox = $('#test-timeline-checkbox');
    const headerTag = $('#test-header-tag');
    const fill = $('#test-gauge-fill');
    const scoreElem = $('#test-meter-score');
    const condElem = $('#test-meter-condition');
    const recElem = $('#test-meter-rec');
    const eyebrowElem = $('#test-meter-eyebrow');
    const breakdownElem = $('#test-breakdown-container');

    checkbox.checked = isCurrentYear;
    btnHist.classList.toggle('active', !isCurrentYear);
    btnPres.classList.toggle('active', isCurrentYear);

    // Badges reset
    $('#test-badge-poor').classList.remove('active');
    $('#test-badge-fair').classList.remove('active');
    $('#test-badge-good').classList.remove('active');

    if (isCurrentYear) {
      // 1. PRESENT-DAY 2026 FORECAST PERSPECTIVE
      const curColor = palette[pres.condition];
      const clampedScore = Math.min(100, Math.max(0, pres.rhi));
      fill.style.stroke = curColor;
      fill.style.strokeDashoffset = arcLength - (arcLength * clampedScore / 100);
      scoreElem.textContent = pres.rhi.toFixed(1);
      eyebrowElem.textContent = 'TIME-ADJUSTED (2026) RHI METER';
      condElem.textContent = `${pres.condition} condition (2026 Forecast)`;
      condElem.style.color = curColor;
      recElem.textContent = data.recommendation || 'Time-adjusted deterioration projection factoring cumulative heavy axle loadings and regional freeze-thaw cycles.';

      headerTag.className = `test-category-tag tag-${pres.condition.toLowerCase()}`;
      headerTag.textContent = `${pres.condition.toUpperCase()} (2026 FORECAST)`;

      $(`#test-badge-${pres.condition.toLowerCase()}`)?.classList.add('active');

      breakdownElem.innerHTML = `
        <div class="test-score-card" style="border-left: 4px solid var(--green);">
          <span>Assessment Mode</span>
          <strong style="color:var(--green)">Time-Adjusted Forecast (2026)</strong>
        </div>
        <div class="test-score-card">
          <span>Forecast Horizon</span>
          <strong style="color:var(--ink)">Present 2026 (+${pres.simulated_years} Years Aging)</strong>
        </div>
        <div class="test-score-card">
          <span>Estimated Present IRI</span>
          <strong style="color:var(--ink)">${pres.estimated_iri.toFixed(3)} m/km (${pres.iri_change >= 0 ? '+' : ''}${pres.iri_change.toFixed(3)} delta)</strong>
        </div>
        <div class="test-score-card">
          <span>Structural Policy</span>
          <strong style="color:var(--muted); font-size:11px;">100% Surface AI (Old FWD safely excluded)</strong>
        </div>
        <div class="test-score-card" style="background:#edf8f2; border-color:rgba(18,150,90,0.4);">
          <span>Present Day (2026) RHI</span>
          <strong style="color:var(--green); font-size:16px;">${pres.rhi.toFixed(1)} / 100 (${pres.condition})</strong>
        </div>
      `;
    } else {
      // 2. HISTORICAL SURVEY BASELINE PERSPECTIVE
      const histColor = palette[hist.condition];
      const clampedScore = Math.min(100, Math.max(0, hist.rhi));
      fill.style.stroke = histColor;
      fill.style.strokeDashoffset = arcLength - (arcLength * clampedScore / 100);
      scoreElem.textContent = hist.rhi.toFixed(1);
      eyebrowElem.textContent = `HISTORICAL SURVEY (${hist.year}) RHI METER`;
      condElem.textContent = `${hist.condition} condition (Survey Year ${hist.year})`;
      condElem.style.color = histColor;
      recElem.textContent = `Baseline survey assessment combining physical surface roughness (${sample.mri} m/km) with structural FWD sensor readings.`;

      headerTag.className = `test-category-tag tag-${hist.condition.toLowerCase()}`;
      headerTag.textContent = `${hist.condition.toUpperCase()} (${hist.year} BASELINE)`;

      $(`#test-badge-${hist.condition.toLowerCase()}`)?.classList.add('active');

      breakdownElem.innerHTML = `
        <div class="test-score-card" style="border-left: 4px solid var(--navy);">
          <span>Assessment Mode</span>
          <strong style="color:var(--navy)">Survey Baseline Snapshot (${hist.year})</strong>
        </div>
        <div class="test-score-card">
          <span>Historical Measured IRI</span>
          <strong style="color:var(--ink)">${hist.measured_iri.toFixed(3)} m/km (Score: ${hist.iri_score.toFixed(1)}/100)</strong>
        </div>
        <div class="test-score-card">
          <span>Historical FWD Structural Health</span>
          <strong style="color:#548bdf">${hist.fwd_health} ${hist.fwd_score !== null ? `(${hist.fwd_score.toFixed(1)}/100)` : ''}</strong>
        </div>
        <div class="test-score-card">
          <span>Sensor Weighting</span>
          <strong style="color:var(--muted); font-size:11px;">${hist.fwd_available ? '50% Surface + 50% Structural Deflections' : '100% Surface (Fallback)'}</strong>
        </div>
        <div class="test-score-card" style="background:#eef3f0; border-color:rgba(29,69,59,0.4);">
          <span>Historical Baseline RHI</span>
          <strong style="color:var(--navy); font-size:16px;">${hist.rhi.toFixed(1)} / 100 (${hist.condition})</strong>
        </div>
      `;
    }
  }

  // Initial render default: Present-Day Forecast (checked = true)
  updateTimelinePerspective(true);

  // Bind interactive switch & buttons
  $('#test-timeline-checkbox')?.addEventListener('change', (e) => {
    updateTimelinePerspective(e.target.checked);
  });

  $('#test-mode-hist')?.addEventListener('click', () => {
    updateTimelinePerspective(false);
  });

  $('#test-mode-pres')?.addEventListener('click', () => {
    updateTimelinePerspective(true);
  });
}

async function runNotebookSampleTest() {
  let sampleIndex;
  do {
    sampleIndex = Math.floor(Math.random() * testSamples.length);
  } while (sampleIndex === lastSampleIndex && testSamples.length > 1);
  lastSampleIndex = sampleIndex;

  const sample = testSamples[sampleIndex];

  // Send prediction request for sample profile
  const response = await fetch('/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(sample)
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(getErrorMessage(data, 'Sample test failed.'));

  // Update primary test button text
  $('#test-rhi-button').textContent = 'Run another random sample test';

  // Render isolated test metered display under testing section only
  renderTestMeterResult(sample, data, sampleIndex + 1);
}

function initializeDeflections() {
  $('#deflection-inputs').innerHTML = Array.from({ length: 7 }, (_, index) => {
    const isCenter = index === 0;
    const isOuter = index === 6;
    const helper = isCenter ? 'Center · e.g., 450' : isOuter ? 'Outer · e.g., 70' : '0–2,000 microns';
    return `<label>Peak defl. ${index + 1}<input class="deflection" id="defl-${index + 1}" type="number" min="0" max="2000" step="0.1" placeholder="e.g., ${isCenter ? '450' : isOuter ? '70' : '200'}" required><small>${helper}</small></label>`;
  }).join('');
}

function initializeRangeValidation() {
  const form = $('#predictor-form');
  const numericInputs = Array.from(form.querySelectorAll('input[type="number"]'));
  numericInputs.forEach((input) => {
    input.addEventListener('blur', () => {
      if (!input.value) return;
      const value = Number(input.value);
      const min = input.getAttribute('min') ? Number(input.getAttribute('min')) : -Infinity;
      const max = input.getAttribute('max') ? Number(input.getAttribute('max')) : Infinity;
      if (value < min || value > max) {
        showError(new Error(`${input.id.replace(/-/g, ' ')} must be between ${min === -Infinity ? '0' : min} and ${max === Infinity ? 'unlimited' : max}.`));
        input.value = Math.max(min, Math.min(max, value));
      }
    });
  });
}

function hideSearchResults() {
  const container = $('#search-results');
  if (container) container.classList.add('hidden');
  $('#section-combobox')?.classList.remove('open');
}

function renderSearchResults(sections) {
  const container = $('#search-results');
  if (!container) return;
  if (!sections || sections.length === 0) {
    container.innerHTML = '<div class="result muted">No matching sections found</div>';
    container.classList.remove('hidden');
    $('#section-combobox')?.classList.add('open');
    return;
  }
  container.innerHTML = sections.map((sec) =>
    `<button type="button" class="result" data-id="${sec.SHRP_ID}" data-state="${sec.STATE_CODE}">` +
    `<strong>SHRP ${sec.SHRP_ID}</strong> <span class="muted">· State ${sec.STATE_CODE}</span>` +
    `</button>`
  ).join('');
  container.classList.remove('hidden');
  $('#section-combobox')?.classList.add('open');
  container.querySelectorAll('.result[data-id]').forEach((button) => {
    button.addEventListener('click', (e) => {
      e.stopPropagation();
      const shrpId = button.dataset.id;
      const stateCode = button.dataset.state;
      loadSection(shrpId, stateCode).catch(showError);
    });
  });
}

function filterSections(query) {
  const raw = (query || '').trim();
  const needle = raw.toLowerCase().replace(/^shrp\s*/, '').trim();
  if (!needle) {
    renderSearchResults(availableSections);
    return;
  }
  const filtered = availableSections.filter((sec) =>
    sec.SHRP_ID.toLowerCase().includes(needle) ||
    sec.STATE_CODE.toLowerCase().includes(needle)
  );
  if (filtered.length > 0) {
    renderSearchResults(filtered);
  } else {
    fetch(`/api/sections?search=${encodeURIComponent(needle)}`)
      .then((res) => res.json())
      .then((data) => renderSearchResults(data))
      .catch(() => renderSearchResults([]));
  }
}

async function loadAvailableSections() {
  const response = await fetch('/api/sections?limit=500');
  availableSections = await response.json().catch(() => []);
}

document.addEventListener('DOMContentLoaded', async () => {
  initializeDeflections(); toggleFwd(); initializeRangeValidation();
  $('#fwd-available').addEventListener('change', toggleFwd);

  const searchInput = $('#section-search');
  const toggleBtn = $('#combobox-toggle');

  searchInput?.addEventListener('input', () => {
    filterSections(searchInput.value);
  });

  searchInput?.addEventListener('focus', () => {
    filterSections(searchInput.value);
  });

  toggleBtn?.addEventListener('click', (e) => {
    e.stopPropagation();
    const isHidden = $('#search-results')?.classList.contains('hidden');
    if (isHidden) {
      filterSections(searchInput.value);
    } else {
      hideSearchResults();
    }
  });

  document.addEventListener('click', (e) => {
    if (!e.target.closest('#section-combobox')) {
      hideSearchResults();
    }
  });

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') hideSearchResults();
  });

  $('#predictor-form').addEventListener('submit', (event) => { event.preventDefault(); currentHistory = []; currentBasin = null; currentConfidence = null; requestPrediction().catch(showError); });
  $('#csv-button').addEventListener('click', () => downloadCsv().catch(showError)); $('#pdf-button').addEventListener('click', downloadPdf);
  $('#batch-button').addEventListener('click', () => uploadBatch().catch(showError));
  $('#batch-template-btn')?.addEventListener('click', () => downloadBatchTemplate().catch(showError));
  $('#test-rhi-button')?.addEventListener('click', () => runNotebookSampleTest().catch(showError));
  try {
    await loadMetadata();
    await loadAvailableSections();
    loadNetwork().catch((err) => {
      console.warn('Network summary error:', err);
      $('#network-status').textContent = 'Network summary loaded';
    });
  } catch (error) {
    console.error('Initialization error:', error);
  }
});


