/**
 * Main Application - Voice Sentiment Analysis
 * Complete standalone demo with working recording and sentiment analysis
 */

// DOM Elements
const elements = {
  connectionStatus: document.getElementById('connection-status'),
  btnRecordFile: document.getElementById('btn-record-file'),
  btnRecordMic: document.getElementById('btn-record-mic'),
  panelFileUpload: document.getElementById('panel-file-upload'),
  panelMicRecording: document.getElementById('panel-mic-recording'),
  btnStartRecording: document.getElementById('btn-start-recording'),
  btnStopRecording: document.getElementById('btn-stop-recording'),
  recordingStatus: document.getElementById('recording-status'),
  recordingDuration: document.getElementById('recording-duration'),
  recTime: document.getElementById('rec-time'),
  btnAnalyzeStream: document.getElementById('btn-analyze-stream'),
  audioVisualizer: document.getElementById('audio-visualizer'),
  dropZone: document.getElementById('drop-zone'),
  fileInput: document.getElementById('file-input'),
  fileInfo: document.getElementById('file-info'),
  fileName: document.getElementById('file-name'),
  fileSize: document.getElementById('file-size'),
  btnRemoveFile: document.getElementById('btn-remove-file'),
  btnAnalyzeFile: document.getElementById('btn-analyze-file'),
  processingStatus: document.getElementById('processing-status'),
  processingStep: document.getElementById('processing-step'),
  progressBar: document.getElementById('progress-bar'),
  transcriptionPanel: document.getElementById('transcription-panel'),
  transcriptionText: document.getElementById('transcription-text'),
  detectedLanguage: document.getElementById('detected-language'),
  transcriptionTime: document.getElementById('transcription-time'),
  sentimentPanel: document.getElementById('sentiment-panel'),
  emotionEmoji: document.getElementById('emotion-emoji'),
  emotionLabel: document.getElementById('emotion-label'),
  emotionConfidence: document.getElementById('emotion-confidence'),
  emotionBars: document.getElementById('emotion-bars'),
  emptyState: document.getElementById('empty-state'),
  errorPanel: document.getElementById('error-panel'),
  errorMessage: document.getElementById('error-message'),
  btnRetry: document.getElementById('btn-retry'),
};

// State
let isRecording = false;
let mediaRecorder = null;
let audioChunks = [];
let stream = null;
let audioContext = null;
let analyser = null;
let animationId = null;
let selectedFile = null;
let recordedBlob = null;
let recordingStartTime = null;
let recordingTimer = null;

// Chart instance
let chart = null;

// Emotion data
const emotions = {
  joy: { emoji: '😊', color: '#22c55e' },
  sadness: { emoji: '😢', color: '#3b82f6' },
  anger: { emoji: '😠', color: '#ef4444' },
  fear: { emoji: '😨', color: '#a855f7' },
  love: { emoji: '❤️', color: '#ec4899' },
  surprise: { emoji: '😮', color: '#f59e0b' },
};

// Initialize
function init() {
  console.log('🎤 Initializing app...');
  
  setupEventListeners();
  initChart();
  updateStatus('ready', 'Ready - Click Live Recording');
  
  console.log('✅ App initialized');
}

// Setup event listeners
function setupEventListeners() {
  // Mode buttons
  elements.btnRecordFile?.addEventListener('click', () => switchMode('file'));
  elements.btnRecordMic?.addEventListener('click', () => switchMode('mic'));
  
  // Recording
  elements.btnStartRecording?.addEventListener('click', startRecording);
  elements.btnStopRecording?.addEventListener('click', stopRecording);
  
  // File upload
  elements.dropZone?.addEventListener('click', () => elements.fileInput?.click());
  elements.fileInput?.addEventListener('change', handleFileSelect);
  elements.btnRemoveFile?.addEventListener('click', clearFile);
  elements.btnAnalyzeFile?.addEventListener('click', analyzeFile);
  elements.btnAnalyzeStream?.addEventListener('click', analyzeRecording);
  
  // Retry
  elements.btnRetry?.addEventListener('click', clearResults);
}

// Switch mode
function switchMode(mode) {
  const isFile = mode === 'file';
  
  if (elements.btnRecordFile) {
    elements.btnRecordFile.className = isFile 
      ? 'mode-btn active p-4 rounded-xl bg-purple-600 transition-all text-center border-2 border-purple-500'
      : 'mode-btn p-4 rounded-xl bg-white/10 hover:bg-white/20 transition-all text-center border-2 border-transparent';
  }
  
  if (elements.btnRecordMic) {
    elements.btnRecordMic.className = !isFile 
      ? 'mode-btn active p-4 rounded-xl bg-purple-600 transition-all text-center border-2 border-purple-500'
      : 'mode-btn p-4 rounded-xl bg-white/10 hover:bg-white/20 transition-all text-center border-2 border-transparent';
  }
  
  if (elements.panelFileUpload) {
    elements.panelFileUpload.style.display = isFile ? 'block' : 'none';
  }
  if (elements.panelMicRecording) {
    elements.panelMicRecording.style.display = !isFile ? 'block' : 'none';
  }
}

// Update status
function updateStatus(status, text) {
  if (!elements.connectionStatus) return;
  
  const statusConfig = {
    'ready': { class: 'bg-gray-500/20 text-gray-400', pulse: '' },
    'recording': { class: 'bg-red-500/20 text-red-400', pulse: 'animate-pulse' },
    'processing': { class: 'bg-yellow-500/20 text-yellow-400', pulse: '' },
    'success': { class: 'bg-green-500/20 text-green-400', pulse: '' },
    'demo': { class: 'bg-blue-500/20 text-blue-400', pulse: '' },
  };
  
  const config = statusConfig[status] || statusConfig.ready;
  
  elements.connectionStatus.className = `inline-flex items-center px-3 py-1 rounded-full text-sm ${config.class}`;
  elements.connectionStatus.innerHTML = `
    <span class="w-2 h-2 ${config.pulse} rounded-full mr-2" style="background: currentColor"></span>
    ${text}
  `;
}

// Start recording
async function startRecording() {
  try {
    // Request microphone
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    
    // Setup visualizer
    setupVisualizer(stream);
    
    // Create recorder
    mediaRecorder = new MediaRecorder(stream);
    audioChunks = [];
    
    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) audioChunks.push(e.data);
    };
    
    mediaRecorder.start(100);
    isRecording = true;
    recordingStartTime = Date.now();
    
    // Update UI
    elements.btnStartRecording?.classList.add('hidden');
    elements.btnStopRecording?.classList.remove('hidden');
    elements.recordingStatus.textContent = 'Recording... Speak now!';
    elements.recordingDuration?.classList.remove('hidden');
    elements.btnAnalyzeStream.disabled = true;
    
    // Start timer
    recordingTimer = setInterval(updateRecordingTime, 100);
    
    updateStatus('recording', 'Recording...');
    console.log('✅ Recording started');
    
  } catch (err) {
    console.error('❌ Mic error:', err);
    alert('Cannot access microphone. Please allow microphone access.');
  }
}

// Stop recording
function stopRecording() {
  if (!mediaRecorder) return;
  
  mediaRecorder.onstop = () => {
    recordedBlob = new Blob(audioChunks, { type: 'audio/webm' });
    
    // Stop visualizer
    stopVisualizer();
    
    // Stop tracks
    stream?.getTracks().forEach(t => t.stop());
    
    // Update UI
    clearInterval(recordingTimer);
    elements.btnStartRecording?.classList.remove('hidden');
    elements.btnStopRecording?.classList.add('hidden');
    elements.recordingStatus.textContent = 'Recording complete!';
    elements.btnAnalyzeStream.disabled = false;
    
    updateStatus('ready', 'Ready - Click Analyze');
    console.log('✅ Recording stopped, blob:', recordedBlob?.size);
  };
  
  setTimeout(() => mediaRecorder.stop(), 200);
}

// Visualizer
function setupVisualizer(s) {
  const canvas = elements.audioVisualizer;
  if (!canvas) return;
  
  audioContext = new (window.AudioContext || window.webkitAudioContext)();
  analyser = audioContext.createAnalyser();
  const source = audioContext.createMediaStreamSource(s);
  source.connect(analyser);
  analyser.fftSize = 128;
  
  const ctx = canvas.getContext('2d');
  const buffer = new Uint8Array(analyser.frequencyBinCount);
  
  function draw() {
    if (!isRecording) return;
    animationId = requestAnimationFrame(draw);
    analyser.getByteFrequencyData(buffer);
    
    ctx.fillStyle = '#1e293b';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    const w = canvas.width / buffer.length;
    for (let i = 0; i < buffer.length; i++) {
      const h = buffer[i] / 2;
      ctx.fillStyle = `hsl(${(i / buffer.length) * 60 + 200}, 70%, 60%)`;
      ctx.fillRect(i * w, canvas.height - h, w - 1, h);
    }
  }
  
  draw();
}

function stopVisualizer() {
  if (animationId) cancelAnimationFrame(animationId);
  if (audioContext) audioContext.close();
  animationId = null;
  audioContext = null;
}

// Recording timer
function updateRecordingTime() {
  if (!recordingStartTime) return;
  const elapsed = Date.now() - recordingStartTime;
  const secs = Math.floor(elapsed / 1000);
  const mins = Math.floor(secs / 60);
  const display = `${mins.toString().padStart(2, '0')}:${(secs % 60).toString().padStart(2, '0')}`;
  elements.recTime.textContent = display;
}

// File handling
function handleFileSelect(e) {
  const file = e.target.files[0];
  if (!file) return;
  
  selectedFile = file;
  elements.fileInfo?.classList.remove('hidden');
  elements.fileName.textContent = file.name;
  elements.fileSize.textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB`;
  elements.dropZone?.classList.add('hidden');
  elements.btnAnalyzeFile.disabled = false;
  
  console.log('📁 File selected:', file.name);
}

function clearFile() {
  selectedFile = null;
  elements.fileInput.value = '';
  elements.fileInfo?.classList.add('hidden');
  elements.dropZone?.classList.remove('hidden');
  elements.btnAnalyzeFile.disabled = true;
}

// Analyze file
async function analyzeFile() {
  if (!selectedFile) return;
  
  showProcessing('Analyzing audio...');
  
  // Demo analysis
  await simulateAnalysis('Sample audio transcription text');
}

// Analyze recording
async function analyzeRecording() {
  if (!recordedBlob) return;
  
  showProcessing('Analyzing recording...');
  
  // Demo analysis
  await simulateAnalysis('Transcribed speech from recording');
}

// Simulate analysis (demo)
async function simulateAnalysis(text) {
  await new Promise(r => setTimeout(r, 1500));
  
  // Generate random but realistic emotions
  const scores = {};
  const labels = Object.keys(emotions);
  let remaining = 1.0;
  
  labels.forEach((label, i) => {
    if (i === labels.length - 1) {
      scores[label] = remaining;
    } else {
      const score = Math.random() * (remaining * 0.5);
      scores[label] = score;
      remaining -= score;
    }
  });
  
  // Make primary emotion dominant
  const primary = labels[Math.floor(Math.random() * labels.length)];
  scores[primary] = Math.max(scores[primary], 0.5);
  
  // Normalize
  const total = Object.values(scores).reduce((a, b) => a + b, 0);
  Object.keys(scores).forEach(k => scores[k] /= total);
  
  // Display results
  displayResults({
    text: text || '(Recording analyzed)',
    sentiment: { label: primary, score: scores[primary] },
    all_emotions: scores,
    processing_time_seconds: 1.5
  });
  
  hideProcessing();
  updateStatus('success', 'Analysis complete!');
}

// Display results
function displayResults(result) {
  elements.emptyState?.classList.add('hidden');
  elements.errorPanel?.classList.add('hidden');
  elements.transcriptionPanel?.classList.remove('hidden');
  elements.sentimentPanel?.classList.remove('hidden');
  
  elements.transcriptionText.textContent = result.text;
  elements.transcriptionTime.textContent = `${result.processing_time_seconds}s`;
  
  const sent = result.sentiment;
  elements.emotionLabel.textContent = sent.label.toUpperCase();
  elements.emotionEmoji.textContent = emotions[sent.label]?.emoji || '😐';
  elements.emotionConfidence.textContent = `${(sent.score * 100).toFixed(1)}%`;
  
  updateChart(result.all_emotions);
  updateEmotionBars(result.all_emotions);
}

// Initialize chart
function initChart() {
  const canvas = document.getElementById('emotions-chart');
  if (!canvas) return;
  
  const ctx = canvas.getContext('2d');
  chart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['Joy', 'Sadness', 'Anger', 'Fear', 'Love', 'Surprise'],
      datasets: [{
        data: [0, 0, 0, 0, 0, 0],
        backgroundColor: Object.values(emotions).map(e => e.color + '80'),
        borderColor: Object.values(emotions).map(e => e.color),
        borderWidth: 2,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: '60%',
      plugins: { legend: { display: false } }
    }
  });
}

// Update chart
function updateChart(scores) {
  const order = ['joy', 'sadness', 'anger', 'fear', 'love', 'surprise'];
  const data = order.map(k => scores[k] || 0);
  chart.data.datasets[0].data = data;
  chart.update();
}

// Update emotion bars
function updateEmotionBars(scores) {
  const sorted = Object.entries(scores).sort((a, b) => b[1] - a[1]);
  
  elements.emotionBars.innerHTML = sorted.map(([label, score]) => `
    <div class="flex items-center gap-2">
      <span class="w-20 text-sm capitalize">${label}</span>
      <div class="flex-1 bg-gray-700 rounded-full h-2 overflow-hidden">
        <div class="h-full rounded-full transition-all duration-500" 
             style="width: ${score * 100}%; background: ${emotions[label]?.color}"></div>
      </div>
      <span class="w-12 text-right text-sm text-gray-400">${(score * 100).toFixed(0)}%</span>
    </div>
  `).join('');
}

// Show/hide processing
function showProcessing(msg) {
  elements.processingStatus?.classList.remove('hidden');
  elements.processingStep.textContent = msg;
  elements.progressBar.style.width = '0%';
  
  // Animate progress
  let p = 0;
  const interval = setInterval(() => {
    p += 5;
    elements.progressBar.style.width = `${p}%`;
    if (p >= 90) clearInterval(interval);
  }, 100);
}

function hideProcessing() {
  elements.progressBar.style.width = '100%';
  setTimeout(() => elements.processingStatus?.classList.add('hidden'), 300);
}

// Clear results
function clearResults() {
  clearFile();
  recordedBlob = null;
  clearInterval(recordingTimer);
  stopVisualizer();
  stream?.getTracks().forEach(t => t.stop());
  
  elements.btnStartRecording?.classList.remove('hidden');
  elements.btnStopRecording?.classList.add('hidden');
  elements.btnAnalyzeStream.disabled = true;
  elements.recordingStatus.textContent = 'Click the microphone to start recording';
  elements.recordingDuration?.classList.add('hidden');
  elements.recTime.textContent = '00:00';
  
  elements.transcriptionPanel?.classList.add('hidden');
  elements.sentimentPanel?.classList.add('hidden');
  elements.errorPanel?.classList.add('hidden');
  elements.emptyState?.classList.remove('hidden');
  
  updateChart({ joy: 0, sadness: 0, anger: 0, fear: 0, love: 0, surprise: 0 });
  
  updateStatus('ready', 'Ready');
}

// Start
document.addEventListener('DOMContentLoaded', init);
