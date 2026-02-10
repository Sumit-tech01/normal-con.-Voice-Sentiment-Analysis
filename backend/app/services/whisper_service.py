"""
Enhanced Whisper Service for Speech-to-Text transcription.
Provides audio transcription using OpenAI's Whisper model with improvements.

Features:
- Hallucination detection
- Better language detection
- Minimum audio duration handling
- Post-processing for common errors
- Enhanced confidence scoring
- Debug mode
"""

import os
import logging
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional, Dict, Any, Union, List, Tuple
from dataclasses import dataclass

import numpy as np
import torch
import whisper
from scipy.io import wavfile

from ..config import config
from ..models.schemas import TranscriptionResult

logger = logging.getLogger(__name__)


# Minimum audio duration in seconds for reliable transcription
MIN_AUDIO_DURATION = 0.5  # 500ms
MAX_AUDIO_DURATION = 300  # 5 minutes

# Hallucination detection thresholds
MIN_CONFIDENCE_THRESHOLD = 0.15  # Below this is likely hallucination
MAX_REPETITION_RATIO = 0.5  # Max allowed repetition in text
MIN_WORD_COUNT = 2  # Minimum words for valid transcription

# Common transcription errors to fix
COMMON_CORRECTIONS = [
    (r'\bi\'m\b', "I'm"),
    (r'\bi\'ll\b', "I'll"),
    (r'\bi\'ve\b', "I've"),
    (r'\bdon\'t\b', "don't"),
    (r'\bcan\'t\b', "can't"),
    (r'\bwont\b', "won't"),
    (r'\bcant\b', "can't"),
    (r'\bthats\b', "that's"),
    (r'\bwhats\b', "what's"),
    (r'\bhes\b', "he's"),
    (r'\bshes\b', "she's"),
    (r'\bit\'s\b', "it's"),
    (r'\b  +', " "),  # Multiple spaces
    (r'\.\.+', "..."),  # Multiple periods
]


@dataclass
class WhisperDebugInfo:
    """Debug information for whisper processing."""
    audio_duration: float
    language_detected: str
    language_confidence: Optional[float]
    confidence_score: float
    is_hallucination: bool
    hallucination_reasons: List[str]
    word_count: int
    processing_time: float
    model_size: str
    device: str


class WhisperService:
    """
    Service for speech-to-text transcription using OpenAI Whisper.
    
    Features:
    - Load and cache Whisper model
    - Transcribe audio files
    - Support for multiple model sizes
    - Device optimization (CPU/CUDA)
    - Hallucination detection
    - Post-processing for common errors
    """

    _instance: Optional["WhisperService"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "WhisperService":
        """Singleton pattern to avoid reloading model."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the Whisper service."""
        if self._initialized:
            return

        self.model: Optional[whisper.Whisper] = None
        self.model_size: str = config.WHISPER_MODEL_SIZE
        self.device: str = config.WHISPER_DEVICE
        self._load_time: Optional[float] = None
        self._model_loaded: bool = False
        self._language_cache: Dict[str, float] = {}
        
        # Debug mode
        self.debug_mode: bool = False
        
        # Minimum audio duration check
        self.min_audio_duration: float = MIN_AUDIO_DURATION

        self._initialized = True
        logger.info(f"WhisperService initialized with model size: {self.model_size}")

    def load_model(self, force_reload: bool = False) -> None:
        """
        Load the Whisper model.
        
        Args:
            force_reload: If True, reload even if already loaded
        """
        if self._model_loaded and not force_reload:
            logger.info("Whisper model already loaded, skipping reload")
            return

        start_time = time.time()

        try:
            # Determine device
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            logger.info(f"Loading Whisper '{self.model_size}' model on {self.device}")

            # Load model with specified options
            model_kwargs = {
                "model": self.model_size,
                "device": self.device,
                "download_root": str(config.MODEL_CACHE_DIR_ABSOLUTE),
            }

            # Add compute type for CUDA optimization
            if self.device == "cuda":
                compute_type = config.WHISPER_COMPUTE_TYPE
                if compute_type == "default":
                    compute_type = "float16" if torch.cuda.is_available() else "float32"
                model_kwargs["compute_type"] = compute_type

            self.model = whisper.load_model(**model_kwargs)

            self._load_time = time.time() - start_time
            self._model_loaded = True

            logger.info(
                f"Whisper model loaded successfully in {self._load_time:.2f}s"
            )

        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise RuntimeError(f"Whisper model loading failed: {e}")

    def set_min_audio_duration(self, duration: float) -> None:
        """
        Set minimum audio duration for transcription.
        
        Args:
            duration: Minimum duration in seconds
        """
        self.min_audio_duration = max(0.1, min(duration, 2.0))

    def detect_language(
        self,
        audio_path: Union[str, Path],
    ) -> Tuple[str, float]:
        """
        Detect language of audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (language_code, confidence)
        """
        if not self._model_loaded:
            self.load_model()

        audio_path = Path(audio_path)
        
        try:
            # Load audio using whisper
            audio = whisper.load_audio(str(audio_path))
            
            # Use whisper's language detection
            # The model has built-in language detection
            _, probs = self.model.detect_language(audio)
            
            # Get most likely language
            if probs:
                detected_lang = max(probs, key=probs.get)
                confidence = probs[detected_lang]
            else:
                detected_lang = "en"
                confidence = 0.5
            
            return detected_lang, confidence
            
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return "en", 0.5

    def _detect_hallucination(
        self,
        text: str,
        confidence: Optional[float],
        audio_duration: float,
    ) -> Tuple[bool, List[str]]:
        """
        Detect if transcription is likely a hallucination.
        
        Args:
            text: Transcribed text
            confidence: Confidence score from Whisper
            audio_duration: Duration of audio in seconds
            
        Returns:
            Tuple of (is_hallucination, reasons)
        """
        reasons = []
        
        # Check confidence
        if confidence is not None and confidence < MIN_CONFIDENCE_THRESHOLD:
            reasons.append(f"Low confidence ({confidence:.2f} < {MIN_CONFIDENCE_THRESHOLD})")
        
        # Check word count
        words = text.split()
        word_count = len(words)
        if word_count < MIN_WORD_COUNT:
            reasons.append(f"Too few words ({word_count} < {MIN_WORD_COUNT})")
        
        # Check for excessive repetition
        if word_count > 0:
            unique_words = len(set(word.lower() for word in words))
            repetition_ratio = 1 - (unique_words / word_count)
            if repetition_ratio > MAX_REPETITION_RATIO:
                reasons.append(f"Excessive repetition ({repetition_ratio:.2%})")
        
        # Check for nonsensical patterns
        if len(text) > 0:
            # Check for repeated single characters
            if re.search(r'(.)\1{5,}', text.lower()):
                reasons.append("Repeated character patterns detected")
            
            # Check for all caps (likely error)
            if text.isupper() and len(text) > 10:
                reasons.append("All caps text (likely error)")
        
        # Check duration vs text length mismatch
        if audio_duration > 0:
            words_per_second = word_count / audio_duration
            if words_per_second > 10:  # Too fast for natural speech
                reasons.append(f"Abnormal speech rate ({words_per_second:.1f} words/sec)")
        
        is_hallucination = len(reasons) > 1 or (
            confidence is not None and confidence < MIN_CONFIDENCE_THRESHOLD / 2
        )
        
        return is_hallucination, reasons

    def _post_process_text(self, text: str) -> str:
        """
        Apply post-processing corrections to transcribed text.
        
        Args:
            text: Raw transcribed text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Apply common corrections
        for pattern, replacement in COMMON_CORRECTIONS:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Remove speaker markers if present
        text = re.sub(r'^\[?[Ss]p[e]?a?k?e?r?\s*\d*\]?\s*:?\s*', '', text)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove leading/trailing punctuation
        text = text.strip('.,;:!?')
        
        return text

    def _validate_audio_duration(self, audio_duration: float) -> Tuple[bool, str]:
        """
        Validate audio duration for transcription.
        
        Args:
            audio_duration: Duration in seconds
            
        Returns:
            Tuple of (is_valid, message)
        """
        if audio_duration < self.min_audio_duration:
            return False, f"Audio too short ({audio_duration:.2f}s < {self.min_audio_duration}s minimum)"
        
        if audio_duration > MAX_AUDIO_DURATION:
            return False, f"Audio too long ({audio_duration:.1f}s > {MAX_AUDIO_DURATION}s maximum)"
        
        return True, ""

    def transcribe(
        self,
        audio_path: Union[str, Path],
        language: Optional[str] = None,
        verbose: bool = False,
        no_speech_threshold: float = 0.6,
    ) -> Tuple[TranscriptionResult, Optional[WhisperDebugInfo]]:
        """
        Transcribe an audio file with hallucination detection and post-processing.
        
        Args:
            audio_path: Path to the audio file
            language: Language code (e.g., 'en'). Auto-detected if None
            verbose: Enable verbose logging
            no_speech_threshold: Threshold for no-speech detection
            
        Returns:
            Tuple of (TranscriptionResult, DebugInfo)
        """
        start_time = time.time()

        # Ensure model is loaded
        if not self._model_loaded:
            self.load_model()

        # Convert to Path if string
        audio_path = Path(audio_path)

        # Validate file exists
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info(f"Transcribing audio file: {audio_path.name}")

        try:
            # Get audio duration
            duration = self._get_audio_duration(audio_path)
            
            # Validate duration
            is_valid, message = self._validate_audio_duration(duration)
            if not is_valid:
                logger.warning(f"Audio duration validation: {message}")
            
            # Run transcription
            options = {
                "no_speech_threshold": no_speech_threshold,
                "logprob_threshold": -1.0,
                "compression_ratio_threshold": 2.4,
            }
            
            if language:
                options["language"] = language

            result = self.model.transcribe(
                str(audio_path),
                verbose=verbose,
                **options
            )

            # Extract text and clean
            text = result.get("text", "").strip()
            text = self._post_process_text(text)
            
            # Get language detection info
            language_detected = result.get("language", language or "unknown")
            language_confidence = None
            
            # Calculate confidence from log probabilities if available
            confidence = None
            if "log_probs" in result and result["log_probs"] is not None:
                # Average log probability converted to confidence
                avg_log_prob = np.mean(result["log_probs"])
                confidence = float(np.exp(avg_log_prob)) if avg_log_prob is not None else None
            
            # Detect hallucination
            is_hallucination, hallucination_reasons = self._detect_hallucination(
                text, confidence, duration
            )
            
            # Handle hallucination
            if is_hallucination and len(text) > 0:
                logger.warning(f"Possible hallucination detected: {hallucination_reasons}")
                # Still return the result but mark it
                if confidence is not None:
                    confidence = min(confidence, confidence * 0.5)
            
            processing_time = time.time() - start_time
            word_count = len(text.split()) if text else 0
            
            logger.info(
                f"Transcription complete: {word_count} words in {processing_time:.2f}s"
            )

            # Create debug info
            debug_info = WhisperDebugInfo(
                audio_duration=duration,
                language_detected=language_detected,
                language_confidence=language_confidence,
                confidence_score=confidence or 0.0,
                is_hallucination=is_hallucination,
                hallucination_reasons=hallucination_reasons,
                word_count=word_count,
                processing_time=processing_time,
                model_size=self.model_size,
                device=self.device,
            )

            return TranscriptionResult(
                text=text,
                language=language_detected,
                confidence=confidence,
                duration=duration,
            ), debug_info

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise RuntimeError(f"Transcription error: {e}")

    def transcribe_numpy(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = None,
    ) -> Tuple[TranscriptionResult, Optional[WhisperDebugInfo]]:
        """
        Transcribe audio from numpy array.
        
        Args:
            audio_data: NumPy array of audio samples
            sample_rate: Sample rate of audio
            language: Language code
            
        Returns:
            Tuple of (TranscriptionResult, DebugInfo)
        """
        start_time = time.time()

        if not self._model_loaded:
            self.load_model()

        duration = len(audio_data) / sample_rate
        logger.info(f"Transcribing numpy array: shape={audio_data.shape}, duration={duration:.2f}s")

        try:
            # Create temporary file for whisper
            with tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False
            ) as tmp_file:
                temp_path = tmp_file.name

            try:
                # Save numpy array to WAV file
                wavfile.write(temp_path, sample_rate, audio_data.astype(np.float32))

                # Transcribe using the saved file
                result, debug_info = self.transcribe(temp_path, language)

                return result, debug_info

            finally:
                # Cleanup temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        except Exception as e:
            logger.error(f"NumPy transcription failed: {e}")
            raise RuntimeError(f"NumPy transcription error: {e}")

    def _get_audio_duration(self, audio_path: Path) -> float:
        """
        Get audio duration in seconds.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Duration in seconds
        """
        try:
            import soundfile as sf
            info = sf.info(str(audio_path))
            return float(info.duration)
        except (ImportError, Exception):
            # Fallback
            try:
                sample_rate, data = wavfile.read(str(audio_path))
                return len(data) / sample_rate
            except Exception:
                return 0.0

    def transcribe_with_retry(
        self,
        audio_path: Union[str, Path],
        language: Optional[str] = None,
        max_retries: int = 2,
    ) -> Tuple[TranscriptionResult, Optional[WhisperDebugInfo]]:
        """
        Transcribe with automatic retry on failure.
        
        Args:
            audio_path: Path to audio file
            language: Language code
            max_retries: Maximum retry attempts
            
        Returns:
            Tuple of (TranscriptionResult, DebugInfo)
        """
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                return self.transcribe(audio_path, language)
            except Exception as e:
                last_error = e
                logger.warning(f"Transcription attempt {attempt + 1} failed: {e}")
                
                if attempt < max_retries:
                    # Wait before retry
                    time.sleep(0.5 * (attempt + 1))
        
        raise last_error

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self._model_loaded:
            return {"status": "not_loaded"}

        info = {
            "model_size": self.model_size,
            "device": self.device,
            "is_loaded": self._model_loaded,
            "load_time_seconds": self._load_time,
            "num_parameters": self.model.num_parameters if self.model else None,
            "min_audio_duration": self.min_audio_duration,
        }

        # Add CUDA info if available
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_mb"] = torch.cuda.get_device_properties(
                0
            ).total_memory / (1024**2)

        return info

    def unload_model(self) -> None:
        """Unload the model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
            self._model_loaded = False

            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("Whisper model unloaded")


# Global instance
whisper_service = WhisperService()

