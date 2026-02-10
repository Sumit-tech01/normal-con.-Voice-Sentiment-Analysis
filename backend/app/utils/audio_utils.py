"""
Enhanced Audio Processing Utilities for Voice Sentiment Analysis.
Handles audio format conversion, preprocessing, noise reduction, VAD, and validation.

Features:
- Noise reduction using spectral gating
- Proper normalization (peak + RMS)
- 16kHz sample rate enforcement
- Voice Activity Detection (VAD)
- Audio quality validation
- Debug mode for troubleshooting
"""

import os
import logging
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Union, Dict, Any
import subprocess
import struct

import numpy as np
import scipy.io.wavfile as wav
from scipy import signal
from scipy.ndimage import binary_opening, binary_closing

logger = logging.getLogger(__name__)


# Supported audio formats with their MIME types
AUDIO_FORMATS = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".webm": "audio/webm",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
}

# Target sample rate for processing
TARGET_SAMPLE_RATE = 16000

# Max audio duration in seconds
MAX_AUDIO_DURATION = 300  # 5 minutes

# Audio quality thresholds
MIN_AUDIO_DURATION = 0.5  # Minimum 500ms
MAX_SILENCE_RATIO = 0.95  # Max 95% silence
MIN_SAMPLE_RATE = 8000  # Minimum supported sample rate
MAX_AMPLITUDE = 32767  # Max value for 16-bit audio

# Noise reduction parameters
NOISE_ESTIMATE_DURATION = 0.1  # Seconds for noise estimation
NOISE_THRESHOLD_FACTOR = 2.0  # Multiplier for noise gate
SPECTRAL_FLOOR = 0.001  # Minimum spectral value

# VAD parameters
VAD_FRAME_DURATION_MS = 30  # Frame duration for VAD
VAD_HOP_DURATION_MS = 10  # Hop between frames
VAD_THRESHOLD = 0.5  # Voice activity threshold


class AudioProcessingError(Exception):
    """Exception raised for audio processing errors."""
    pass


class AudioQualityError(Exception):
    """Exception raised for audio quality validation failures."""
    pass


def validate_audio_file(
    file_path: Union[str, Path],
    max_size_mb: int = 50,
    allowed_formats: Optional[list] = None,
) -> Tuple[bool, str]:
    """
    Validate an audio file.
    
    Args:
        file_path: Path to the audio file
        max_size_mb: Maximum file size in MB
        allowed_formats: List of allowed file extensions
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if allowed_formats is None:
        allowed_formats = list(AUDIO_FORMATS.keys())

    file_path = Path(file_path)

    # Check if file exists
    if not file_path.exists():
        return False, f"File not found: {file_path}"

    # Check file extension
    suffix = file_path.suffix.lower()
    if suffix not in allowed_formats:
        return False, f"Unsupported format: {suffix}. Allowed: {allowed_formats}"

    # Check file size
    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    if file_size_mb > max_size_mb:
        return False, f"File too large: {file_size_mb:.1f}MB > {max_size_mb}MB limit"

    return True, ""


def convert_to_wav(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    sample_rate: int = TARGET_SAMPLE_RATE,
    mono: bool = True,
    enable_normalization: bool = True,
) -> Tuple[Path, float]:
    """
    Convert audio file to WAV format with proper settings.
    
    Args:
        input_path: Path to input audio file
        output_path: Path for output WAV file (auto-generated if None)
        sample_rate: Target sample rate
        mono: Convert to mono channel
        enable_normalization: Apply normalization during conversion
        
    Returns:
        Tuple of (output_path, duration_seconds)
    """
    input_path = Path(input_path)
    
    if output_path is None:
        output_path = Path(tempfile.gettempdir()) / f"{input_path.stem}_converted.wav"
    else:
        output_path = Path(output_path)

    try:
        # Use ffmpeg for conversion with proper settings
        try:
            cmd = [
                "ffmpeg", "-y",
                "-i", str(input_path),
                "-ar", str(sample_rate),
                "-ac", "1" if mono else "2",
                "-codec:pcm_s16le",
            ]
            
            if enable_normalization:
                cmd.extend(["-af", "dynaudnorm=f=75:g=15"])
            
            cmd.append(str(output_path))
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )
            
            if result.returncode != 0:
                raise AudioProcessingError(f"FFmpeg error: {result.stderr}")

        except FileNotFoundError:
            # Fallback to scipy if ffmpeg not available
            logger.warning("FFmpeg not found, using scipy for conversion")
            _convert_with_scipy(input_path, output_path, sample_rate, mono)

    except Exception as e:
        logger.error(f"Audio conversion failed: {e}")
        raise AudioProcessingError(f"Failed to convert audio: {e}")

    # Get duration
    duration = get_audio_duration(output_path)
    
    return output_path, duration


def _convert_with_scipy(
    input_path: Path,
    output_path: Path,
    sample_rate: int,
    mono: bool,
) -> None:
    """Convert audio using scipy (limited format support)."""
    try:
        import soundfile as sf
        import librosa

        # Load audio
        audio, orig_sr = sf.read(str(input_path))
        
        # Resample if needed
        if orig_sr != sample_rate:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sample_rate)
        
        # Convert to mono
        if mono and len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Normalize to prevent clipping
        if audio.max() > 0:
            audio = audio / audio.max()
        
        # Save as WAV
        sf.write(str(output_path), audio, sample_rate, subtype="PCM_16")

    except Exception as e:
        raise AudioProcessingError(f"Scipy conversion failed: {e}")


def get_audio_duration(file_path: Union[str, Path]) -> float:
    """
    Get the duration of an audio file in seconds.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Duration in seconds
    """
    file_path = Path(file_path)
    
    try:
        # Try using scipy
        try:
            import soundfile as sf
            info = sf.info(str(file_path))
            return float(info.duration)
        except ImportError:
            pass
        
        # Fallback: read with scipy
        sample_rate, data = wav.read(str(file_path))
        duration = len(data) / sample_rate
        return duration
        
    except Exception as e:
        logger.warning(f"Could not get audio duration: {e}")
        return 0.0


def estimate_noise_floor(
    audio_data: np.ndarray,
    sample_rate: int,
    duration: float = NOISE_ESTIMATE_DURATION,
) -> np.ndarray:
    """
    Estimate the noise floor from the beginning of the audio.
    
    Args:
        audio_data: Audio samples
        sample_rate: Sample rate
        duration: Duration in seconds to use for estimation
        
    Returns:
        Noise floor array (frequency bins)
    """
    # Use first portion of audio for noise estimation
    noise_samples = int(sample_rate * duration)
    noise_signal = audio_data[:noise_samples] if len(audio_data) > noise_samples else audio_data
    
    if len(noise_signal) == 0:
        return np.zeros(1025)  # Default noise floor
    
    # Compute FFT
    n_fft = 2048
    hop_length = 512
    
    # Pad if needed
    if len(noise_signal) < n_fft:
        noise_signal = np.pad(noise_signal, (0, n_fft - len(noise_signal)))
    
    # Compute power spectral density
    _, _, S = signal.stft(noise_signal, fs=sample_rate, nperseg=n_fft, noverlap=n_fft - hop_length)
    noise_floor = np.mean(np.abs(S), axis=1) + SPECTRAL_FLOOR
    
    return noise_floor


def spectral_gate(
    audio_data: np.ndarray,
    sample_rate: int,
    threshold_factor: float = NOISE_THRESHOLD_FACTOR,
    floor: float = SPECTRAL_FLOOR,
) -> np.ndarray:
    """
    Apply spectral gating noise reduction.
    
    Args:
        audio_data: Audio samples
        sample_rate: Sample rate
        threshold_factor: Multiplier for noise gate threshold
        floor: Minimum spectral value
        
    Returns:
        Noise-reduced audio
    """
    # Estimate noise floor
    noise_floor = estimate_noise_floor(audio_data, sample_rate)
    
    # STFT parameters
    n_fft = 2048
    hop_length = 512
    
    # Pad audio for STFT
    pad_length = n_fft // 2
    audio_padded = np.pad(audio_data, (pad_length, pad_length), mode='reflect')
    
    # Compute STFT
    _, _, Z = signal.stft(audio_padded, fs=sample_rate, nperseg=n_fft, noverlap=n_fft - hop_length)
    
    # Apply gate
    magnitude = np.abs(Z)
    threshold = noise_floor[np.newaxis, :] * threshold_factor
    
    # Soft thresholding (spectral subtraction)
    gated_magnitude = np.maximum(magnitude - threshold, floor * magnitude)
    
    # Ensure minimum magnitude
    gated_magnitude = np.maximum(gated_magnitude, floor * np.ones_like(gated_magnitude))
    
    # Reconstruct with original phase
    phase = np.angle(Z)
    Z_gated = gated_magnitude * np.exp(1j * phase)
    
    # ISTFT
    _, audio_cleaned = signal.istft(Z_gated, fs=sample_rate, nperseg=n_fft, noverlap=n_fft - hop_length)
    
    # Remove padding
    audio_cleaned = audio_cleaned[pad_length:pad_length + len(audio_data)]
    
    return audio_cleaned


def normalize_audio(
    audio_data: np.ndarray,
    target_peak: float = 0.95,
    target_rms: float = 0.1,
) -> np.ndarray:
    """
    Normalize audio with peak and RMS normalization.
    
    Args:
        audio_data: Audio samples
        target_peak: Target peak amplitude (0-1)
        target_rms: Target RMS level (0-1)
        
    Returns:
        Normalized audio
    """
    audio_data = audio_data.astype(np.float64)
    
    # Calculate current peak and RMS
    current_peak = np.max(np.abs(audio_data))
    current_rms = np.sqrt(np.mean(audio_data ** 2))
    
    # Peak normalization
    if current_peak > 0:
        peak_gain = target_peak / current_peak
    else:
        peak_gain = 1.0
    
    audio_normalized = audio_data * peak_gain
    
    # RMS normalization
    new_rms = np.sqrt(np.mean(audio_normalized ** 2))
    if new_rms > 0:
        rms_gain = target_rms / new_rms
        # Apply gentle RMS scaling (don't over-compress)
        rms_gain = min(rms_gain, 2.0)  # Limit gain
        audio_normalized = audio_normalized * (0.7 + 0.3 * rms_gain)
    
    # Final peak check
    final_peak = np.max(np.abs(audio_normalized))
    if final_peak > target_peak:
        audio_normalized = audio_normalized * (target_peak / final_peak)
    
    return audio_normalized.astype(np.float32)


def remove_silence(
    audio_data: np.ndarray,
    sample_rate: int = TARGET_SAMPLE_RATE,
    threshold_db: float = -40,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """
    Remove silence from beginning and end of audio.
    
    Args:
        audio_data: NumPy array of audio samples
        sample_rate: Sample rate of audio
        threshold_db: Silence threshold in dB
        frame_length: FFT window size
        hop_length: Hop length between frames
        
    Returns:
        Audio with silence removed
    """
    # Convert threshold from dB to amplitude
    threshold = 10 ** (threshold_db / 20)
    
    # Calculate envelope using Hilbert transform with median filtering
    envelope = np.abs(
        signal.hilbert(signal.medfilt(np.abs(audio_data), 101))
    )
    
    # Find non-silent regions
    non_silent = envelope > threshold
    
    # Get indices of non-silent regions
    indices = np.where(non_silent)[0]
    
    if len(indices) == 0:
        return audio_data
    
    # Get start and end points with some padding
    start = max(0, indices[0] - frame_length)
    end = min(len(audio_data), indices[-1] + frame_length)
    
    return audio_data[start:end]


def voice_activity_detection(
    audio_data: np.ndarray,
    sample_rate: int,
    frame_duration_ms: int = VAD_FRAME_DURATION_MS,
    hop_duration_ms: int = VAD_HOP_DURATION_MS,
    threshold: float = VAD_THRESHOLD,
) -> Dict[str, Any]:
    """
    Voice Activity Detection using energy-based detection.
    
    Args:
        audio_data: Audio samples
        sample_rate: Sample rate
        frame_duration_ms: Frame duration in milliseconds
        hop_duration_ms: Hop between frames
        threshold: Energy threshold (0-1)
        
    Returns:
        Dictionary with VAD results
    """
    frame_length = int(sample_rate * frame_duration_ms / 1000)
    hop_length = int(sample_rate * hop_duration_ms / 1000)
    
    # Calculate frame energies
    num_frames = max(1, (len(audio_data) - frame_length) // hop_length + 1)
    energies = []
    
    for i in range(num_frames):
        start = i * hop_length
        end = min(start + frame_length, len(audio_data))
        frame = audio_data[start:end]
        
        if len(frame) > 0:
            # RMS energy
            energy = np.sqrt(np.mean(frame ** 2))
            energies.append(energy)
        else:
            energies.append(0.0)
    
    energies = np.array(energies)
    
    # Normalize energies
    max_energy = np.max(energies) if np.max(energies) > 0 else 1.0
    normalized_energies = energies / max_energy
    
    # Smooth with moving average
    window_size = 5
    if len(normalized_energies) >= window_size:
        smoothed = np.convolve(normalized_energies, np.ones(window_size)/window_size, mode='same')
    else:
        smoothed = normalized_energies
    
    # Voice detection
    is_voice = smoothed > threshold
    
    # Find voice segments
    voice_segments = []
    in_segment = False
    segment_start = 0
    
    for i, voice in enumerate(is_voice):
        if voice and not in_segment:
            in_segment = True
            segment_start = i * hop_duration_ms
        elif not voice and in_segment:
            in_segment = False
            segment_end = i * hop_duration_ms
            voice_segments.append({
                "start_ms": segment_start,
                "end_ms": segment_end,
                "confidence": float(np.mean(smoothed[max(0,i-window_size):i+1]))
            })
    
    # Add final segment if ongoing
    if in_segment:
        voice_segments.append({
            "start_ms": segment_start,
            "end_ms": len(audio_data) / sample_rate * 1000,
            "confidence": float(np.mean(smoothed[-window_size:])) if len(smoothed) > 0 else 0.0
        })
    
    # Calculate statistics
    total_duration_ms = len(audio_data) / sample_rate * 1000
    voice_duration_ms = sum(seg["end_ms"] - seg["start_ms"] for seg in voice_segments)
    voice_ratio = voice_duration_ms / total_duration_ms if total_duration_ms > 0 else 0.0
    
    # Detect if mostly silence
    is_speech = len(voice_segments) > 0 and voice_ratio > 0.05
    
    return {
        "is_speech": bool(is_speech),
        "is_silence": bool(not is_speech),
        "voice_ratio": float(voice_ratio),
        "voice_duration_ms": float(voice_duration_ms),
        "total_duration_ms": float(total_duration_ms),
        "voice_segments": voice_segments,
        "confidence": float(np.max(smoothed)),
    }


def validate_audio_quality(
    audio_data: np.ndarray,
    sample_rate: int,
    min_duration: float = MIN_AUDIO_DURATION,
    max_silence_ratio: float = MAX_SILENCE_RATIO,
) -> Tuple[bool, str]:
    """
    Validate audio quality for sentiment analysis.
    
    Args:
        audio_data: Audio samples
        sample_rate: Sample rate
        min_duration: Minimum duration in seconds
        max_silence_ratio: Maximum allowed silence ratio
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    duration = len(audio_data) / sample_rate
    
    # Check minimum duration
    if duration < min_duration:
        return False, f"Audio too short: {duration:.2f}s < {min_duration}s minimum"
    
    # Check sample rate
    if sample_rate < MIN_SAMPLE_RATE:
        return False, f"Sample rate too low: {sample_rate}Hz < {MIN_SAMPLE_RATE}Hz minimum"
    
    # Check for clipping
    max_val = np.max(np.abs(audio_data))
    if max_val > 0.99:
        logger.warning(f"Audio may be clipped: max amplitude = {max_val:.3f}")
    
    # Check silence ratio using VAD
    vad_result = voice_activity_detection(audio_data, sample_rate)
    silence_ratio = 1.0 - vad_result["voice_ratio"]
    
    if silence_ratio > max_silence_ratio:
        return False, f"Too much silence: {silence_ratio*100:.1f}% > {max_silence_ratio*100:.1f}%"
    
    # Check for low signal (near silence)
    if vad_result["confidence"] < 0.01:
        return False, "Audio appears to be silent or nearly silent"
    
    return True, ""


def preprocess_audio(
    file_path: Union[str, Path],
    target_sample_rate: int = TARGET_SAMPLE_RATE,
    normalize: bool = True,
    remove_silence_flag: bool = True,
    enable_noise_reduction: bool = True,
    validate_quality: bool = True,
    debug: bool = False,
) -> Tuple[np.ndarray, int, Dict[str, Any]]:
    """
    Enhanced preprocessing pipeline for audio files.
    
    Args:
        file_path: Path to audio file
        target_sample_rate: Target sample rate
        normalize: Apply peak + RMS normalization
        remove_silence_flag: Remove leading/trailing silence
        enable_noise_reduction: Apply spectral gating noise reduction
        validate_quality: Validate audio quality after processing
        debug: Enable debug output
        
    Returns:
        Tuple of (audio_data, sample_rate, debug_info)
    """
    file_path = Path(file_path)
    debug_info = {
        "steps": [],
        "original_duration": 0.0,
        "final_duration": 0.0,
    }
    
    try:
        # Step 1: Read audio file
        step_info = {"step": "read", "status": "starting"}
        debug_info["steps"].append(step_info)
        
        try:
            import soundfile as sf
            audio_data, sample_rate = sf.read(str(file_path))
        except ImportError:
            sample_rate, audio_data = wav.read(str(file_path))
        
        debug_info["original_duration"] = len(audio_data) / sample_rate
        step_info["status"] = "complete"
        step_info["duration"] = debug_info["original_duration"]
        step_info["sample_rate"] = sample_rate
        
        if debug:
            logger.debug(f"Read audio: {len(audio_data)} samples at {sample_rate}Hz")
        
        # Step 2: Convert to mono if stereo
        step_info = {"step": "mono_conversion", "status": "starting"}
        debug_info["steps"].append(step_info)
        
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        step_info["status"] = "complete"
        
        # Step 3: Resample to target sample rate
        step_info = {"step": "resample", "status": "starting"}
        debug_info["steps"].append(step_info)
        
        if sample_rate != target_sample_rate:
            import librosa
            audio_data = librosa.resample(
                audio_data,
                orig_sr=sample_rate,
                target_sr=target_sample_rate
            )
            sample_rate = target_sample_rate
        
        step_info["status"] = "complete"
        step_info["new_sample_rate"] = sample_rate
        
        # Step 4: Noise reduction
        if enable_noise_reduction:
            step_info = {"step": "noise_reduction", "status": "starting"}
            debug_info["steps"].append(step_info)
            
            audio_data = spectral_gate(audio_data, sample_rate)
            
            step_info["status"] = "complete"
            
            if debug:
                logger.debug(f"After noise reduction: {len(audio_data)} samples")
        
        # Step 5: Normalization
        if normalize:
            step_info = {"step": "normalization", "status": "starting"}
            debug_info["steps"].append(step_info)
            
            audio_data = normalize_audio(audio_data)
            
            step_info["status"] = "complete"
            
            if debug:
                logger.debug(f"After normalization: max={np.max(np.abs(audio_data)):.3f}")
        
        # Step 6: Remove silence
        if remove_silence_flag:
            step_info = {"step": "silence_removal", "status": "starting"}
            debug_info["steps"].append(step_info)
            
            original_len = len(audio_data)
            audio_data = remove_silence(audio_data, sample_rate)
            
            step_info["removed_samples"] = original_len - len(audio_data)
            step_info["status"] = "complete"
        
        debug_info["final_duration"] = len(audio_data) / sample_rate
        
        # Step 7: Quality validation
        if validate_quality:
            step_info = {"step": "quality_validation", "status": "starting"}
            debug_info["steps"].append(step_info)
            
            is_valid, error_msg = validate_audio_quality(audio_data, sample_rate)
            
            step_info["is_valid"] = is_valid
            step_info["message"] = error_msg
            
            if not is_valid:
                logger.warning(f"Audio quality check failed: {error_msg}")
            
            step_info["status"] = "complete"
        
        # Step 8: Final VAD check
        step_info = {"step": "vad_check", "status": "starting"}
        debug_info["steps"].append(step_info)
        
        vad_result = voice_activity_detection(audio_data, sample_rate)
        debug_info["vad"] = vad_result
        
        step_info["status"] = "complete"
        
        return audio_data, sample_rate, debug_info
        
    except Exception as e:
        logger.error(f"Audio preprocessing failed: {e}")
        raise AudioProcessingError(f"Failed to preprocess audio: {e}")


def chunk_audio(
    audio_data: np.ndarray,
    sample_rate: int,
    chunk_duration_ms: int = 250,
) -> list:
    """
    Split audio into chunks for streaming processing.
    
    Args:
        audio_data: NumPy array of audio samples
        sample_rate: Sample rate of audio
        chunk_duration_ms: Duration of each chunk in milliseconds
        
    Returns:
        List of audio chunks as numpy arrays
    """
    chunk_samples = int(sample_rate * chunk_duration_ms / 1000)
    chunks = []
    
    for i in range(0, len(audio_data), chunk_samples):
        chunk = audio_data[i : i + chunk_samples]
        if len(chunk) > 0:
            # Pad last chunk if needed
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
            chunks.append(chunk)
    
    return chunks


def audio_to_base64(audio_data: np.ndarray, sample_rate: int) -> str:
    """
    Convert numpy audio array to base64 encoded WAV string.
    
    Args:
        audio_data: NumPy array of audio samples
        sample_rate: Sample rate
        
    Returns:
        Base64 encoded WAV string
    """
    import base64
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav.write(tmp.name, sample_rate, audio_data.astype(np.int16))
        
        with open(tmp.name, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("utf-8")
        
        os.unlink(tmp.name)
    
    return audio_b64


def base64_to_audio(base64_string: str) -> Tuple[np.ndarray, int]:
    """
    Convert base64 encoded audio to numpy array.
    
    Args:
        base64_string: Base64 encoded audio string
        
    Returns:
        Tuple of (audio_data, sample_rate)
    """
    import base64
    import io
    
    audio_bytes = base64.b64decode(base64_string)
    
    # Read as WAV
    sample_rate, audio_data = wav.read(io.BytesIO(audio_bytes))
    
    return audio_data, sample_rate


def get_audio_info(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get detailed information about an audio file.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Dictionary with audio information
    """
    file_path = Path(file_path)
    
    info = {
        "file_path": str(file_path),
        "file_size_bytes": file_path.stat().st_size,
        "file_extension": file_path.suffix.lower(),
    }
    
    try:
        # Get duration
        info["duration_seconds"] = get_audio_duration(file_path)
        
        # Read audio
        try:
            import soundfile as sf
            audio_data, sample_rate = sf.read(str(file_path))
        except ImportError:
            sample_rate, audio_data = wav.read(str(file_path))
        
        info["sample_rate"] = sample_rate
        info["num_channels"] = audio_data.shape[1] if len(audio_data.shape) > 1 else 1
        info["num_samples"] = len(audio_data)
        info["dtype"] = str(audio_data.dtype)
        
        # Audio statistics
        audio_float = audio_data.astype(np.float32)
        info["statistics"] = {
            "min": float(np.min(audio_float)),
            "max": float(np.max(audio_float)),
            "mean": float(np.mean(audio_float)),
            "std": float(np.std(audio_float)),
            "rms": float(np.sqrt(np.mean(audio_float ** 2))),
            "peak": float(np.max(np.abs(audio_float))),
        }
        
        # VAD result
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        info["vad"] = voice_activity_detection(audio_data.astype(np.float32), sample_rate)
        
    except Exception as e:
        info["error"] = str(e)
    
    return info


class AudioBuffer:
    """
    Ring buffer for streaming audio processing.
    
    Maintains a rolling window of audio samples.
    """
    
    def __init__(
        self,
        sample_rate: int = TARGET_SAMPLE_RATE,
        max_duration: float = 10.0,
    ):
        """
        Initialize audio buffer.
        
        Args:
            sample_rate: Sample rate in Hz
            max_duration: Maximum buffer duration in seconds
        """
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * max_duration)
        self.buffer: np.ndarray = np.zeros(0)
        self._lock = False
    
    def append(self, data: np.ndarray) -> None:
        """Append new audio data to buffer."""
        if self._lock:
            return
        
        self.buffer = np.concatenate([self.buffer, data])
        
        # Trim if exceeding max size
        if len(self.buffer) > self.max_samples:
            self.buffer = self.buffer[-self.max_samples:]
    
    def get_chunk(
        self,
        start_offset: float = 0.0,
        duration: float = 2.0,
    ) -> np.ndarray:
        """
        Get a chunk of audio from the buffer.
        
        Args:
            start_offset: Start offset from end of buffer (seconds)
            duration: Duration of chunk (seconds)
            
        Returns:
            Audio chunk as numpy array
        """
        start_sample = int((len(self.buffer) - start_offset * self.sample_rate))
        end_sample = int(start_sample + duration * self.sample_rate)
        
        start_sample = max(0, start_sample)
        end_sample = min(len(self.buffer), end_sample)
        
        if start_sample >= end_sample:
            return np.zeros(0)
        
        return self.buffer[start_sample:end_sample]
    
    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer = np.zeros(0)
    
    def get_all(self) -> np.ndarray:
        """Get all buffered audio."""
        return self.buffer.copy()
    
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return len(self.buffer) == 0
    
    def length_seconds(self) -> float:
        """Get current buffer duration in seconds."""
        return len(self.buffer) / self.sample_rate

