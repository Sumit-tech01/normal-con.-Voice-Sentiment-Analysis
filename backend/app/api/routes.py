"""
REST API routes for Voice Sentiment Analysis application.
Provides endpoints for audio upload, analysis, and health checks.
With enhanced debug mode and integration for improved sentiment analysis.
"""

import logging
import uuid
import time
from pathlib import Path
from typing import Optional

from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename

from ..config import config
from ..models.schemas import (
    AudioUploadRequest,
    AudioAnalysisResponse,
    ErrorResponse,
    HealthResponse,
    SentimentLabel,
)
from ..services.whisper_service import whisper_service
from ..services.sentiment_service import sentiment_service
from ..utils.audio_utils import (
    validate_audio_file,
    convert_to_wav,
    preprocess_audio,
    AudioProcessingError,
)

logger = logging.getLogger(__name__)

api_bp = Blueprint("api", __name__, url_prefix="/api/v1")


# ==================== Health & Status Endpoints ====================

@api_bp.route("/health", methods=["GET"])
def health_check() -> tuple[dict, int]:
    """
    Health check endpoint.
    
    Returns:
        JSON response with service health status
    """
    try:
        whisper_info = whisper_service.get_model_info()
        sentiment_info = sentiment_service.get_model_info()

        services = {
            "whisper": {
                "status": "healthy" if whisper_info.get("is_loaded") else "loading",
                "model": whisper_info.get("model_size", "unknown"),
                "device": whisper_info.get("device", "unknown"),
            },
            "sentiment": {
                "status": "healthy" if sentiment_info.get("is_loaded") else "loading",
                "model": sentiment_info.get("model_name", "unknown"),
                "device": sentiment_info.get("device", "unknown"),
            },
        }

        # Check if all services are ready
        all_healthy = all(
            s.get("status") == "healthy" for s in services.values()
        )

        response = HealthResponse(
            status="healthy" if all_healthy else "degraded",
            version=config.APP_VERSION,
            services=services,
        )

        return response.model_dump(), 200

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return ErrorResponse(
            error="Health check failed",
            error_code="HEALTH_CHECK_ERROR",
            details={"exception": str(e)},
        ).model_dump(), 500


@api_bp.route("/models", methods=["GET"])
def get_model_info() -> tuple[dict, int]:
    """
    Get information about loaded models.
    
    Returns:
        JSON response with model information
    """
    try:
        whisper_info = whisper_service.get_model_info()
        sentiment_info = sentiment_service.get_model_info()

        return {
            "whisper": whisper_info,
            "sentiment": sentiment_info,
        }, 200

    except Exception as e:
        logger.error(f"Model info request failed: {e}")
        return ErrorResponse(
            error="Failed to get model info",
            error_code="MODEL_INFO_ERROR",
        ).model_dump(), 500


@api_bp.route("/config", methods=["GET"])
def get_config() -> tuple[dict, int]:
    """
    Get current configuration settings.
    
    Returns:
        JSON response with configuration
    """
    try:
        return {
            "confidence_threshold": sentiment_service.confidence_threshold,
            "neutral_threshold": sentiment_service.neutral_threshold,
            "low_confidence_threshold": sentiment_service.low_confidence_threshold,
            "min_audio_duration": whisper_service.min_audio_duration,
            "debug_mode": sentiment_service.debug_mode,
        }, 200

    except Exception as e:
        logger.error(f"Config request failed: {e}")
        return ErrorResponse(
            error="Failed to get config",
            error_code="CONFIG_ERROR",
        ).model_dump(), 500


@api_bp.route("/config", methods=["PUT"])
def update_config() -> tuple[dict, int]:
    """
    Update configuration settings.
    
    Request Body (JSON):
        {
            "confidence_threshold": 0.7,
            "neutral_threshold": 0.25,
            "low_confidence_threshold": 0.4,
            "min_audio_duration": 0.5,
            "debug_mode": false
        }
    
    Returns:
        JSON response with updated configuration
    """
    try:
        data = request.get_json()
        
        if not data:
            return ErrorResponse(
                error="No data provided",
                error_code="NO_DATA",
            ).model_dump(), 400

        # Update sentiment thresholds
        if "confidence_threshold" in data:
            sentiment_service.set_confidence_thresholds(
                confidence=data["confidence_threshold"]
            )
        if "neutral_threshold" in data:
            sentiment_service.set_confidence_thresholds(
                neutral=data["neutral_threshold"]
            )
        if "low_confidence_threshold" in data:
            sentiment_service.set_confidence_thresholds(
                low_confidence=data["low_confidence_threshold"]
            )
        
        # Update min audio duration
        if "min_audio_duration" in data:
            whisper_service.set_min_audio_duration(data["min_audio_duration"])
        
        # Update debug mode
        if "debug_mode" in data:
            sentiment_service.debug_mode = bool(data["debug_mode"])
            whisper_service.debug_mode = bool(data["debug_mode"])

        return {
            "success": True,
            "message": "Configuration updated",
            "config": {
                "confidence_threshold": sentiment_service.confidence_threshold,
                "neutral_threshold": sentiment_service.neutral_threshold,
                "low_confidence_threshold": sentiment_service.low_confidence_threshold,
                "min_audio_duration": whisper_service.min_audio_duration,
                "debug_mode": sentiment_service.debug_mode,
            }
        }, 200

    except Exception as e:
        logger.error(f"Config update failed: {e}")
        return ErrorResponse(
            error="Failed to update config",
            error_code="CONFIG_UPDATE_ERROR",
        ).model_dump(), 500


# ==================== Analysis Endpoints ====================

@api_bp.route("/analyze/upload", methods=["POST"])
def analyze_upload() -> tuple[dict, int]:
    """
    Analyze uploaded audio file with enhanced processing.
    
    Request:
        - multipart/form-data with 'audio' field
        - Optional 'language' field for language hint
        - Optional 'enable_debug' field for debug mode
        
    Returns:
        JSON response with transcription and sentiment analysis
    """
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()

    # Validate file presence
    if "audio" not in request.files:
        return ErrorResponse(
            error="No audio file provided",
            error_code="NO_AUDIO_FILE",
            details={
                "hint": "Include 'audio' field in multipart/form-data request"
            },
        ).model_dump(), 400

    audio_file = request.files["audio"]
    
    if audio_file.filename == "":
        return ErrorResponse(
            error="No file selected",
            error_code="EMPTY_FILENAME",
        ).model_dump(), 400

    # Validate file
    filename = secure_filename(audio_file.filename)
    temp_path = Path(config.UPLOAD_FOLDER_ABSOLUTE) / f"{request_id}_{filename}"
    
    # Check for debug mode
    enable_debug = request.form.get("enable_debug", "").lower() in ["true", "1", "yes"]
    
    try:
        # Save uploaded file
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        audio_file.save(str(temp_path))
        
        logger.info(f"Processing upload: {filename}")

        # Validate audio file
        is_valid, error_msg = validate_audio_file(temp_path)
        if not is_valid:
            raise AudioProcessingError(error_msg)

        # Convert to WAV if needed
        if temp_path.suffix.lower() != ".wav":
            wav_path, duration = convert_to_wav(temp_path)
            temp_path.unlink()  # Remove original
            temp_path = wav_path
        else:
            duration = 0.0

        # Get language from request or auto-detect
        lang_hint = request.form.get("language") or None

        # Load models if not loaded
        if not whisper_service._model_loaded:
            whisper_service.load_model()
        if not sentiment_service._model_loaded:
            sentiment_service.load_model()

        # Enhanced audio preprocessing with debug info
        audio_data, sample_rate, audio_debug = preprocess_audio(
            temp_path,
            target_sample_rate=16000,
            normalize=True,
            remove_silence_flag=True,
            enable_noise_reduction=True,
            validate_quality=True,
            debug=enable_debug,
        )

        # Transcribe with enhanced Whisper
        transcription, whisper_debug = whisper_service.transcribe(
            temp_path,
            language=lang_hint,
        )

        # Analyze sentiment with enhanced service
        dominant, sentiment_debug = sentiment_service.analyze(
            transcription.text,
            return_debug=enable_debug,
        )
        all_emotions = sentiment_service.analyze_with_all_scores(transcription.text)

        # Calculate processing time
        processing_time = time.time() - start_time

        # Build response
        response_data = {
            "success": True,
            "request_id": request_id,
            "transcription": {
                "text": transcription.text,
                "language": transcription.language,
                "confidence": transcription.confidence,
                "duration": transcription.duration,
            },
            "sentiment": {
                "label": dominant.label.value,
                "score": dominant.score,
            },
            "all_emotions": {
                label.value: score for label, score in all_emotions.items()
            },
            "processing_time_seconds": round(processing_time, 3),
        }

        # Add debug info if enabled
        if enable_debug:
            response_data["debug"] = {
                "audio_processing": audio_debug,
                "whisper": {
                    "audio_duration": whisper_debug.audio_duration,
                    "language_detected": whisper_debug.language_detected,
                    "confidence_score": whisper_debug.confidence_score,
                    "is_hallucination": whisper_debug.is_hallucination,
                    "hallucination_reasons": whisper_debug.hallucination_reasons,
                    "word_count": whisper_debug.word_count,
                    "processing_time": whisper_debug.processing_time,
                },
                "sentiment": {
                    "original_text": sentiment_debug.original_text,
                    "cleaned_text": sentiment_debug.cleaned_text,
                    "sentences": sentiment_debug.sentences,
                    "confidence_threshold": sentiment_debug.confidence_threshold,
                    "warnings": sentiment_debug.warnings,
                    "context": getattr(sentiment_debug, 'context', None),
                },
            }

        return response_data, 200

    except AudioProcessingError as e:
        logger.warning(f"Audio processing error: {e}")
        return ErrorResponse(
            error=str(e),
            error_code="AUDIO_PROCESSING_ERROR",
        ).model_dump(), 400

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        return ErrorResponse(
            error="Analysis failed",
            error_code="ANALYSIS_ERROR",
            details={"exception": str(e)},
        ).model_dump(), 500

    finally:
        # Cleanup temp file
        try:
            if temp_path.exists():
                temp_path.unlink()
        except Exception:
            pass


@api_bp.route("/analyze/text", methods=["POST"])
def analyze_text() -> tuple[dict, int]:
    """
    Analyze sentiment of provided text with enhanced processing.
    
    Request:
        - JSON body with 'text' field
        - Optional 'enable_debug' field for debug mode
        
    Returns:
        JSON response with sentiment analysis
    """
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()

    try:
        data = request.get_json()
        
        if not data or "text" not in data:
            return ErrorResponse(
                error="No text provided",
                error_code="NO_TEXT",
                details={"hint": "Include 'text' field in JSON body"},
            ).model_dump(), 400

        text = data["text"].strip()
        
        if not text:
            return ErrorResponse(
                error="Empty text provided",
                error_code="EMPTY_TEXT",
            ).model_dump(), 400

        # Check for debug mode
        enable_debug = data.get("enable_debug", False)

        # Load model if not loaded
        if not sentiment_service._model_loaded:
            sentiment_service.load_model()

        # Analyze sentiment with enhanced service
        dominant, debug_info = sentiment_service.analyze(text, return_debug=enable_debug)
        all_emotions = sentiment_service.analyze_with_all_scores(text)
        processing_time = time.time() - start_time

        response_data = {
            "success": True,
            "request_id": request_id,
            "text": text,
            "sentiment": {
                "label": dominant.label.value,
                "score": dominant.score,
            },
            "all_emotions": {
                label.value: score for label, score in all_emotions.items()
            },
            "processing_time_seconds": round(processing_time, 3),
        }

        # Add debug info if enabled
        if enable_debug and debug_info:
            response_data["debug"] = {
                "original_text": debug_info.original_text,
                "cleaned_text": debug_info.cleaned_text,
                "sentences": debug_info.sentences,
                "confidence_threshold": debug_info.confidence_threshold,
                "warnings": debug_info.warnings,
                "context": getattr(debug_info, 'context', None),
            }

        return response_data, 200

    except Exception as e:
        logger.error(f"Text analysis failed: {e}", exc_info=True)
        return ErrorResponse(
            error="Text analysis failed",
            error_code="ANALYSIS_ERROR",
        ).model_dump(), 500


@api_bp.route("/analyze/stream/start", methods=["POST"])
def start_streaming_session() -> tuple[dict, int]:
    """
    Start a streaming analysis session.
    
    Returns:
        JSON response with session ID and configuration
    """
    session_id = str(uuid.uuid4())[:8]
    
    try:
        data = request.get_json() or {}
        language = data.get("language") or None

        # Initialize session data
        session_data = {
            "session_id": session_id,
            "language": language,
            "created_at": time.time(),
            "chunks_processed": 0,
            "status": "active",
        }

        return {
            "success": True,
            "session_id": session_id,
            "language": language,
            "websocket_url": f"/socket.io/?session_id={session_id}",
            "config": {
                "chunk_duration_ms": config.AUDIO_CHUNK_DURATION_MS,
                "max_duration_seconds": config.AUDIO_MAX_DURATION,
            },
        }, 200

    except Exception as e:
        logger.error(f"Failed to start streaming session: {e}")
        return ErrorResponse(
            error="Failed to start session",
            error_code="SESSION_ERROR",
        ).model_dump(), 500


# ==================== Documentation Endpoint ====================

@api_bp.route("/docs", methods=["GET"])
def get_api_docs() -> tuple[dict, int]:
    """
    Return API documentation.
    
    Returns:
        JSON response with API documentation
    """
    docs = {
        "title": "Voice Sentiment Analysis API",
        "version": config.APP_VERSION,
        "endpoints": {
            "GET /api/v1/health": {
                "description": "Health check endpoint",
                "response": "HealthResponse with service statuses",
            },
            "GET /api/v1/models": {
                "description": "Get information about loaded models",
                "response": "Model information",
            },
            "GET /api/v1/config": {
                "description": "Get current configuration",
                "response": "Configuration settings",
            },
            "PUT /api/v1/config": {
                "description": "Update configuration",
                "request": {
                    "content-type": "application/json",
                    "body": {
                        "confidence_threshold": "float (0-1)",
                        "neutral_threshold": "float (0-1)",
                        "low_confidence_threshold": "float (0-1)",
                        "min_audio_duration": "float",
                        "debug_mode": "boolean",
                    },
                },
                "response": "Updated configuration",
            },
            "POST /api/v1/analyze/upload": {
                "description": "Upload and analyze audio file",
                "request": {
                    "content-type": "multipart/form-data",
                    "fields": {
                        "audio": "Audio file (WAV, MP3, WebM, OGG, FLAC)",
                        "language": "(optional) Language hint (e.g., 'en')",
                        "enable_debug": "(optional) Enable debug output",
                    },
                },
                "response": "AudioAnalysisResponse with debug info if enabled",
            },
            "POST /api/v1/analyze/text": {
                "description": "Analyze sentiment of text",
                "request": {
                    "content-type": "application/json",
                    "body": {
                        "text": "Text to analyze",
                        "enable_debug": "(optional) Enable debug output",
                    },
                },
                "response": "Sentiment analysis results with debug info if enabled",
            },
            "POST /api/v1/analyze/stream/start": {
                "description": "Start a streaming session",
                "response": "Session ID and WebSocket URL",
            },
        },
        "websocket": {
            "url": "ws://localhost:5000/socket.io/",
            "events": {
                "audio_chunk": "Send base64-encoded audio chunks",
                "sentiment_update": "Receive incremental results",
                "sentiment_complete": "Receive final results",
            },
        },
        "supported_emotions": [e.value for e in SentimentLabel],
        "features": {
            "audio_preprocessing": {
                "noise_reduction": "Spectral gating noise reduction",
                "normalization": "Peak and RMS normalization",
                "silence_removal": "Leading/trailing silence removal",
                "vad": "Voice activity detection",
                "quality_validation": "Audio quality checks",
            },
            "whisper_enhancements": {
                "hallucination_detection": "Detects potential transcription errors",
                "postprocessing": "Text corrections and cleaning",
                "language_detection": "Automatic language detection",
            },
            "sentiment_enhancements": {
                "text_preprocessing": "Filler word and artifact removal",
                "sentence_analysis": "Sentence-level sentiment analysis",
                "weighted_aggregation": "Word-count weighted emotion aggregation",
                "confidence_thresholds": "Configurable confidence thresholds",
                "neutral_detection": "Automatic neutral classification",
            },
        },
    }

    return docs, 200


# ==================== Error Handlers ====================

@api_bp.errorhandler(400)
def bad_request(error) -> tuple[dict, int]:
    """Handle 400 Bad Request errors."""
    return ErrorResponse(
        error=str(error.description),
        error_code="BAD_REQUEST",
    ).model_dump(), 400


@api_bp.errorhandler(404)
def not_found(error) -> tuple[dict, int]:
    """Handle 404 Not Found errors."""
    return ErrorResponse(
        error="Endpoint not found",
        error_code="NOT_FOUND",
    ).model_dump(), 404


@api_bp.errorhandler(500)
def internal_error(error) -> tuple[dict, int]:
    """Handle 500 Internal Server errors."""
    return ErrorResponse(
        error="Internal server error",
        error_code="INTERNAL_ERROR",
    ).model_dump(), 500

