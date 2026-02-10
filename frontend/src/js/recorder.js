/**
 * Audio Recorder with Visualizer - Voice Sentiment Analysis
 * Handles microphone recording with real-time audio visualization
 */

let mediaRecorder = null;
let audioChunks = [];
let audioContext = null;
let analyser = null;
let microphone = null;
let canvas = null;
let canvasCtx = null;
let animationId = null;
let stream = null;

// Recording state
export const recordingState = {
  isRecording: false,
  startTime: null,
  duration: 0,
};

/**
 * Initialize recorder with visualizer
 * @param {HTMLCanvasElement} visualizerCanvas - Canvas element for visualization
 */
export async function initRecorder(visualizerCanvas = null) {
  console.log('🎤 Initializing recorder...');
  
  canvas = visualizerCanvas;
  
  if (canvas) {
    canvasCtx = canvas.getContext('2d');
    setupVisualizer();
  }
  
  // Check support
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    console.warn('MediaRecorder not supported');
    return false;
  }
  
  console.log('✅ Recorder initialized');
  return true;
}

/**
 * Setup audio visualizer
 */
function setupVisualizer() {
  if (!canvas) return;
  
  // Set canvas dimensions
  canvas.width = canvas.offsetWidth || 300;
  canvas.height = canvas.offsetHeight || 80;
}

/**
 * Start audio visualization
 */
function startVisualization(stream) {
  if (!canvasCtx) return;
  
  audioContext = new (window.AudioContext || window.webkitAudioContext)();
  analyser = audioContext.createAnalyser();
  microphone = audioContext.createMediaStreamSource(stream);
  
  microphone.connect(analyser);
  analyser.fftSize = 256;
  
  const bufferLength = analyser.frequencyBinCount;
  const dataArray = new Uint8Array(bufferLength);
  
  function draw() {
    if (!recordingState.isRecording) return;
    
    animationId = requestAnimationFrame(draw);
    
    analyser.getByteFrequencyData(dataArray);
    
    // Clear canvas
    canvasCtx.fillStyle = 'rgb(30, 41, 59)';
    canvasCtx.fillRect(0, 0, canvas.width, canvas.height);
    
    const barWidth = (canvas.width / bufferLength) * 2.5;
    let barHeight;
    let x = 0;
    
    for (let i = 0; i < bufferLength; i++) {
      barHeight = dataArray[i] / 2;
      
      // Color gradient based on height
      const r = barHeight + 25 * (i / bufferLength);
      const g = 250 * (i / bufferLength);
      const b = 50;
      
      canvasCtx.fillStyle = `rgb(${r},${g},${b})`;
      canvasCtx.fillRect(x, canvas.height - barHeight, barWidth, barHeight);
      
      x += barWidth + 1;
    }
  }
  
  draw();
}

/**
 * Stop visualization
 */
function stopVisualization() {
  if (animationId) {
    cancelAnimationFrame(animationId);
    animationId = null;
  }
  
  if (audioContext) {
    audioContext.close();
    audioContext = null;
  }
  
  // Clear canvas
  if (canvasCtx && canvas) {
    canvasCtx.fillStyle = 'rgb(30, 41, 59)';
    canvasCtx.fillRect(0, 0, canvas.width, canvas.height);
  }
}

/**
 * Start recording audio
 * @returns {Promise<{recorder: MediaRecorder, stream: MediaStream}>}
 */
export async function startRecording() {
  console.log('🎤 Requesting microphone access...');
  
  try {
    // Request microphone with optimal settings
    stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        channelCount: 1,
        sampleRate: 48000,
      }
    });
    
    console.log('✅ Microphone access granted');
    
    // Create MediaRecorder
    mediaRecorder = new MediaRecorder(stream, {
      mimeType: 'audio/webm;codecs=opus'
    });
    
    audioChunks = [];
    
    // Collect audio data
    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        audioChunks.push(event.data);
        console.log(`📦 Audio chunk received: ${event.data.size} bytes`);
      }
    };
    
    // Handle errors
    mediaRecorder.onerror = (event) => {
      console.error('❌ MediaRecorder error:', event);
    };
    
    // Handle stop
    mediaRecorder.onstop = () => {
      console.log('🛑 Recording stopped');
      stopVisualization();
    };
    
    // Start recording (collect data every 100ms)
    mediaRecorder.start(100);
    
    // Update state
    recordingState.isRecording = true;
    recordingState.startTime = Date.now();
    
    // Start visualization
    startVisualization(stream);
    
    console.log('✅ Recording started');
    
    return { recorder: mediaRecorder, stream };
    
  } catch (error) {
    console.error('❌ Failed to start recording:', error);
    throw error;
  }
}

/**
 * Stop recording and return audio blob
 * @returns {Promise<Blob>}
 */
export async function stopRecording() {
  return new Promise((resolve, reject) => {
    if (!mediaRecorder) {
      reject(new Error('No active recording'));
      return;
    }
    
    console.log('🛑 Stopping recording...');
    
    // Update state
    recordingState.isRecording = false;
    recordingState.duration = Date.now() - recordingState.startTime;
    
    mediaRecorder.onstop = () => {
      // Stop all tracks
      if (stream) {
        stream.getTracks().forEach(track => {
          track.stop();
          console.log(`🛑 Stopped track: ${track.kind}`);
        });
      }
      
      // Create blob from all chunks
      const blob = new Blob(audioChunks, { type: 'audio/webm' });
      
      console.log(`✅ Recording complete: ${blob.size} bytes, ${recordingState.duration}ms`);
      
      if (blob.size === 0) {
        reject(new Error('No audio recorded'));
      } else {
        resolve(blob);
      }
    };
    
    mediaRecorder.onerror = (event) => {
      reject(new Error('Recording error'));
    };
    
    // Stop after a small delay to ensure final chunk is captured
    setTimeout(() => {
      mediaRecorder.stop();
    }, 200);
  });
}

/**
 * Check if recording is supported
 * @returns {boolean}
 */
export function isRecordingSupported() {
  return !!(
    navigator.mediaDevices && 
    navigator.mediaDevices.getUserMedia &&
    window.MediaRecorder
  );
}

/**
 * Get current recording duration in seconds
 * @returns {number}
 */
export function getRecordingDuration() {
  if (!recordingState.isRecording) {
    return recordingState.duration / 1000;
  }
  return (Date.now() - recordingState.startTime) / 1000;
}

/**
 * Format duration as MM:SS
 * @param {number} seconds
 * @returns {string}
 */
export function formatDuration(seconds) {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

/**
 * Clean up resources
 */
export function cleanup() {
  stopVisualization();
  
  if (stream) {
    stream.getTracks().forEach(track => track.stop());
    stream = null;
  }
  
  mediaRecorder = null;
  audioChunks = [];
  recordingState.isRecording = false;
  recordingState.startTime = null;
}

/**
 * Resume audio context (needed for some browsers)
 */
export function resumeAudioContext() {
  if (audioContext && audioContext.state === 'suspended') {
    audioContext.resume();
  }
}
