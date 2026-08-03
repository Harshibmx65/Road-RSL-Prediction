const $ = (selector) => document.querySelector(selector);
let componentChart, iriChart, networkChart;
let currentResult = null;
let currentHistory = [];
let currentBasin = null;
let currentConfidence = null;
let selectedSection = null;

const palette = { Good: '#12965a', Fair: '#ee8b2d', Poor: '#dd4d43' };

function chartOptions(extra = {}) {
  return { responsive: true, maintainAspectRatio: false, plugins: { legend: { labels: { boxWidth: 10, font: { family: 'Manrope' } } } }, ...extra };
}

function renderCharts() {
  const result = currentResult;
  if (!result) return;
  componentChart?.destroy(); iriChart?.destroy();
  componentChart = new Chart($('#component-chart'), {
    type: 'bar', data: { labels: ['IRI surface score', 'FWD structural score'], datasets: [{ data: [result.iri_score, result.fwd_score ?? 0], backgroundColor: ['#267a62', result.fwd_score === null ? '#d5dfd8' : '#548bdf'], borderRadius: 6 }] },
    options: chartOptions({ scales: { y: { min: 0, max: 100, grid: { color: '#edf1ee' } }, x: { grid: { display: false } } }, plugins: { legend: { display: false } } })
  });
  const historic = currentHistory.map((row) => ({ x: row.YEAR, y: row.MRI }));
  const finalYear = historic.length ? historic[historic.length - 1].x : Number($('#year').value);
  iriChart = new Chart($('#iri-chart'), {
    type: 'line', data: { datasets: [
      { label: 'Historical IRI', data: historic, borderColor: '#267a62', backgroundColor: '#267a62', tension: .25, pointRadius: 3 },
      { label: 'Projected IRI', data: [{ x: finalYear, y: Number($('#mri').value) }, ...result.projection.map((point) => ({ x: point.year, y: point.iri }))], borderColor: '#ee8b2d', borderDash: [6, 5], tension: .2, pointRadius: 3 },
      { label: 'Failure threshold', data: [{ x: historic[0]?.x ?? finalYear - 2, y: 2.5 }, { x: result.projection.at(-1)?.year ?? finalYear + 10, y: 2.5 }], borderColor: '#dd4d43', borderDash: [3, 4], pointRadius: 0 },
    ] }, options: chartOptions({ scales: { x: { type: 'linear', ticks: { precision: 0 }, grid: { display: false } }, y: { title: { display: true, text: 'IRI (m/km)' }, grid: { color: '#edf1ee' } } } })
  });
}

function updateResult(result) {
  currentResult = result;
  const color = palette[result.condition];
  $('#rhi-score').textContent = result.rhi.toFixed(1);
  $('#condition').textContent = `${result.condition} condition`;
  $('#condition').style.color = color;
  $('#recommendation').textContent = result.recommendation;
  $('#future-iri').textContent = `${result.predicted_future_iri.toFixed(3)} m/km`;
  $('#fwd-health').textContent = result.fwd_health ?? 'Not available';
  $('#gauge-fill').style.stroke = color;
  $('#gauge-fill').style.strokeDashoffset = 267 - (267 * result.rhi / 100);
  $('#fallback-badge').classList.toggle('hidden', !result.fallback_engaged);
  renderCharts();
}

function getDeflections() {
  return Array.from(document.querySelectorAll('.deflection')).map((input) => Number(input.value || 0));
}

function setValue(id, value) { if (value !== undefined && value !== null) $(id).value = value; }

function fillForm(values) {
  setValue('#mri', values.mri); setValue('#aadtt', values.aadtt); setValue('#truck-volume', values.annual_truck_volume);
  setValue('#annual-esal', values.annual_esal); setValue('#cumulative-esal', values.cumulative_esal); setValue('#year', values.year);
  $('#fwd-available').checked = values.fwd_available;
  toggleFwd();
  (values.deflections || []).forEach((value, index) => setValue(`#defl-${index + 1}`, value));
  setValue('#drop-load', values.drop_load); setValue('#drop-height', values.drop_height);
  setValue('#pavement-family', values.pavement_family); setValue('#lane-no', values.lane_no);
}

function payload() {
  const fwdAvailable = $('#fwd-available').checked;
  return {
    mri: Number($('#mri').value), aadtt: Number($('#aadtt').value), annual_truck_volume: Number($('#truck-volume').value),
    annual_esal: Number($('#annual-esal').value), cumulative_esal: Number($('#cumulative-esal').value), year: Number($('#year').value),
    fwd_available: fwdAvailable,
    ...(fwdAvailable ? { deflections: getDeflections(), drop_load: Number($('#drop-load').value), drop_height: Number($('#drop-height').value), pavement_family: $('#pavement-family').value, lane_no: $('#lane-no').value } : {})
  };
}

function toggleFwd() { $('#fwd-fields').style.display = $('#fwd-available').checked ? 'block' : 'none'; }

async function requestPrediction() {
  const response = await fetch('/api/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload()) });
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || 'Prediction failed');
  updateResult(data);
}

async function loadSection(shrpId, stateCode) {
  const response = await fetch(`/api/section/${encodeURIComponent(shrpId)}?state_code=${encodeURIComponent(stateCode)}`);
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || 'Unable to load section');
  selectedSection = data.section;
  $('#selected-section').textContent = `Selected · SHRP ${data.section.shrp_id}, State ${data.section.state_code}, construction ${data.section.construction_no}`;
  $('#search-results').innerHTML = '';
  currentHistory = data.history;
  currentBasin = data.deflection_basin;
  currentConfidence = data.deflection_confidence;
  fillForm(data.defaults);
  updateResult(data.prediction);
}

async function searchSections() {
  const query = $('#section-search').value.trim();
  if (!query) { $('#search-results').innerHTML = ''; return; }
  const response = await fetch(`/api/sections?search=${encodeURIComponent(query)}`);
  const sections = await response.json();
  $('#search-results').innerHTML = sections.map((section) => `<button class="result" data-id="${section.SHRP_ID}" data-state="${section.STATE_CODE}">SHRP ${section.SHRP_ID} <span class="muted">· State ${section.STATE_CODE}</span></button>`).join('') || '<div class="result muted">No matching sections</div>';
  document.querySelectorAll('.result[data-id]').forEach((button) => button.addEventListener('click', () => loadSection(button.dataset.id, button.dataset.state).catch(showError)));
}

function showError(error) { window.alert(error.message || String(error)); }

async function loadNetwork() {
  const response = await fetch('/api/network-summary');
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || 'Network summary unavailable');
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
  if (!response.ok) throw new Error((await response.json()).detail || 'CSV export failed');
  const blob = await response.blob(); const url = URL.createObjectURL(blob); const link = document.createElement('a'); link.href = url; link.download = 'road-health-report.csv'; link.click(); URL.revokeObjectURL(url);
}

async function uploadBatch() {
  const file = $('#batch-file').files[0];
  if (!file) throw new Error('Choose a CSV or Excel file first.');
  const form = new FormData(); form.append('file', file);
  const response = await fetch('/api/batch', { method: 'POST', body: form });
  if (!response.ok) throw new Error((await response.json()).detail || 'Batch scoring failed');
  const blob = await response.blob(); const url = URL.createObjectURL(blob); const link = document.createElement('a'); link.href = url; link.download = 'batch-rhi-results.csv'; link.click(); URL.revokeObjectURL(url);
}

function downloadPdf() {
  if (!currentResult) return showError(new Error('Run a prediction first.'));
  const { jsPDF } = window.jspdf; const pdf = new jsPDF(); const r = currentResult;
  pdf.setFillColor(29, 69, 59); pdf.rect(0, 0, 210, 35, 'F'); pdf.setTextColor(255, 255, 255); pdf.setFontSize(19); pdf.text('Road Health Index Assessment', 15, 21);
  pdf.setTextColor(20, 38, 31); pdf.setFontSize(11); let y = 52;
  const rows = [['Section', selectedSection ? `SHRP ${selectedSection.shrp_id} · State ${selectedSection.state_code}` : 'Live what-if scenario'], ['Final RHI', `${r.rhi.toFixed(1)} / 100 (${r.condition})`], ['IRI score', `${r.iri_score.toFixed(2)} / 100`], ['FWD structural score', r.fwd_score === null ? 'Not available — fallback used' : `${r.fwd_score} / 100 (${r.fwd_health})`], ['Predicted future IRI', `${r.predicted_future_iri.toFixed(3)} m/km`], ['Recommendation', r.recommendation]];
  rows.forEach(([label, value]) => { pdf.setFont('helvetica', 'bold'); pdf.text(label, 15, y); pdf.setFont('helvetica', 'normal'); const lines = pdf.splitTextToSize(value, 125); pdf.text(lines, 70, y); y += Math.max(10, lines.length * 6 + 4); });
  pdf.setTextColor(104, 128, 120); pdf.setFontSize(8); pdf.text(`Generated ${new Date().toLocaleString()}`, 15, 285); pdf.save('road-health-report.pdf');
}

function initializeDeflections() {
  $('#deflection-inputs').innerHTML = Array.from({ length: 7 }, (_, index) => {
    const isCenter = index === 0;
    const isOuter = index === 6;
    const helper = isCenter ? 'Center · e.g., 450' : isOuter ? 'Outer · e.g., 70' : '0–2,000 microns';
    return `<label>Peak defl. ${index + 1}<input class="deflection" id="defl-${index + 1}" type="number" min="0" max="2000" step="0.1" placeholder="e.g., ${isCenter ? '450' : isOuter ? '70' : '200'}" required><small>${helper}</small></label>`;
  }).join('');
}

document.addEventListener('DOMContentLoaded', async () => {
  initializeDeflections(); toggleFwd();
  $('#fwd-available').addEventListener('change', toggleFwd);
  $('#section-search').addEventListener('input', () => { clearTimeout(window.searchTimer); window.searchTimer = setTimeout(searchSections, 220); });
  $('#predictor-form').addEventListener('submit', (event) => { event.preventDefault(); currentHistory = []; currentBasin = null; currentConfidence = null; requestPrediction().catch(showError); });
  $('#csv-button').addEventListener('click', () => downloadCsv().catch(showError)); $('#pdf-button').addEventListener('click', downloadPdf);
  $('#batch-button').addEventListener('click', () => uploadBatch().catch(showError));
  try { await Promise.all([loadMetadata(), loadNetwork()]); } catch (error) { showError(error); }
});
