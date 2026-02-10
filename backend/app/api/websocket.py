"""
WebSocket handlers for real-time streaming audio analysis.
Provides WebSocket endpoints for live audio processing.
With enhanced integration for improved sentiment analysis.
"""

import json
import logging
import time
import uuid
from typing import Dict, Optional
import threading

from flask import Blueprint, request
from flask_socketio import SocketIO, emit, disconnect

from ..config import config
from ..services.whisper_service import whisper_service
from ..services.sentiment_service import sentiment_service
from ..models.schemas import SentimentLabel
from ..utils.audio_utils import (
    base64_to_audio,
    AudioBuffer,
    AudioProcessingError,
)

logger = logging.getLogger(__name__)

websocket_bp = Blueprint("websocket", __name__)

# Initialize SocketIO with async mode
socketio: Optional[SocketIO] = None

# Active streaming sessions
sessions: Dict[str, dict] = {}
sessions_lock = threading.Lock()


def init_socketio(app, cors_allowed_origins: str = "*") -> SocketIO:
    """
    Initialize SocketIO with the Flask app.
    
    Args:
        app: Flask application instance
        cors_allowed_origins: CORS allowed origins
        
    Returns:
        SocketIO instance
    """
    global socketio
    
    socketio = SocketIO(
        app,
        cors_allowed_origins=cors_allowed_origins,
        async_mode="threading",
        logger=True,
        engineio_logger=False,
    )
    
    register_handlers(socketio)
    
    return socketio


def register_handlers(sio: SocketIO) -> None:
    """Register WebSocket event handlers."""
    
    @sio.on("connect")
    def handle_connect(auth: dict = None):
        """Handle new WebSocket connection."""
        client_id = request.sid
        logger.info(f"Client connected: {client_id}")
        
        emit("connected", {
            "client_id": client_id,
            "message": "Connected to Voice Sentiment Analysis server",
            "server_version": config.APP_VERSION,
        })
    
    @sio.on("disconnect")
    def handle_disconnect():
        """Handle client disconnection."""
        client_id = request.sid
        logger.info(f"Client disconnected: {client_id}")
        
        # Clean up session
        with sessions_lock:
            for session_id in list(sessions.keys()):
                if sessions[session_id].get("client_id") == client_id:
                    del sessions[session_id]
                    logger.info(f"Cleaned up session: {session_id}")
    
    @sio.on("start_session")
    def handle_start_session(data: dict):
        """
        Start a new streaming session.
        
        Expected data:
        {
            "language": "en"  (optional)
        }
        """
        client_id = request.sid
        session_id = str(uuid.uuid4())[:8]
        language = data.get("language") if data else None
        
        logger.info(f"Starting session {session_id} for client {client_id}")
        
        # Create session
        session = {
            "session_id": session_id,
            "client_id": client_id,
            "language": language,
            "created_at": time.time(),
            "chunks": [],
            "full_text": "",
            "audio_buffer": AudioBuffer(
                sample_rate=config.AUDIO_SAMPLE_RATE,
                max_duration=config.STREAMING_BUFFER_SECONDS,
            ),
            "status": "active",
        }
        
        with sessions_lock:
            sessions[session_id] = session
        
        emit("session_started", {
            "session_id": session_id,
            "config": {
                "chunk_duration_ms": config.AUDIO_CHUNK_DURATION_MS,
                "max_duration_seconds": config.AUDIO_MAX_DURATION,
            },
        })
    
    @sio.on("audio_chunk")
    def handle_audio_chunk(data: dict):
        """
        Process streaming audio chunk with enhanced analysis.
        
        Expected data:
        {
            "data": "<base64 encoded audio>",
            "chunk_id": 1,
            "is_final": false,
            "timestamp": 1234567890.123
        }
        """
        start_time = time.time()
        client_id = request.sid
        
        # Find session for this client
        session_id = data.get("session_id")
        if not session_id:
            # Try to find by client_id
            with sessions_lock:
                for sid, session in sessions.items():
                    if session.get("client_id") == client_id:
                        session_id = sid
                        break
        
        if not session_id or session_id not in sessions:
            emit("error", {
                "error": "No active session",
                "error_code": "NO_SESSION",
            })
            return
        
        session = sessions[session_id]
        
        try:
            # Decode base64 audio
            audio_b64 = data.get("data")
            if not audio_b64:
                raise ValueError("No audio data provided")
            
            audio_data, sample_rate = base64_to_audio(audio_b64)
            
            # Store in buffer
            session["audio_buffer"].append(audio_data)
            
            # Process chunk
            chunk_id = data.get("chunk_id", 0)
            is_final = data.get("is_final", False)
            
            # Load models if needed
            if not whisper_service._model_loaded:
                whisper_service.load_model()
            if not sentiment_service._model_loaded:
                sentiment_service.load_model()
            
            # Get accumulated audio for transcription
            accumulated_audio = session["audio_buffer"].get_all()
            
            if len(accumulated_audio) > 0:
                # Run transcription on accumulated audio
                transcription, whisper_debug = whisper_service.transcribe_numpy(
                    accumulated_audio,
                    sample_rate=sample_rate,
                    language=session.get("language"),
                )
                
                # Enhanced sentiment analysis with debug info
                dominant, sentiment_debug = sentiment_service.analyze(
                    transcription.text,
                    return_debug=True,
                )
                all_emotions = sentiment_service.analyze_with_all_scores(transcription.text)
                
                # Build response
                processing_time_ms = (time.time() - start_time) * 1000
                
                response = {
                    "type": "sentiment_update",
                    "chunk_id": chunk_id,
                    "timestamp": time.time(),
                    "text_partial": transcription.text,
                    "current_emotions": {
                        label.value: score 
                        for label, score in all_emotions.items()
                    },
                    "dominant_sentiment": {
                        "label": dominant.label.value,
                        "score": dominant.score,
                    },
                    "is_final": is_final,
                    "processing_time_ms": round(processing_time_ms, 2),
                    "confidence": transcription.confidence,
                    "language": transcription.language,
                }
                
                # Add debug info
                if sentiment_debug:
                    response["debug"] = {
                        "cleaned_text": sentiment_debug.cleaned_text,
                        "num_sentences": len(sentiment_debug.sentences),
                        "warnings": sentiment_debug.warnings,
                    }
                
                if whisper_debug:
                    response["debug"] = response.get("debug", {})
                    response["debug"]["whisper"] = {
                        "is_hallucination": whisper_debug.is_hallucination,
                        "confidence": whisper_debug.confidence_score,
                        "word_count": whisper_debug.word_count,
                    }
                
                if is_final:
                    response["type"] = "sentiment_complete"
                    response["total_chunks"] = chunk_id + 1
                    response["total_processing_time_seconds"] = round(
                        time.time() - session["created_at"], 3
                    )
                    response["language"] = transcription.language
                    response["text_complete"] = transcription.text
                    session["status"] = "completed"
                
                emit(response["type"], response)
            
        except Exception as e:
            logger.error(f"Error processing chunk: {e}", exc_info=True)
            emit("error", {
                "error": str(e),
                "error_code": "PROCESSING_ERROR",
                "chunk_id": data.get("chunk_id"),
            })
    
    @sio.on("end_session")
    def handle_end_session(data: dict):
        """
        End a streaming session.
        
        Expected data:
        {
            "session_id": "abc123"
        }
        """
        session_id = data.get("session_id") if data else None
        
        with sessions_lock:
            if session_id and session_id in sessions:
                session = sessions.pop(session_id)
                logger.info(f"Session ended: {session_id}")
                
                emit("session_ended", {
                    "session_id": session_id,
                    "status": "completed",
                })
            else:
                emit("error", {
                    "error": "Session not found",
                    "error_code": "SESSION_NOT_FOUND",
                })
    
    @sio.on("ping")
    def handle_ping():
        """Handle ping for connection health check."""
        emit("pong", {"timestamp": time.time()})
    
    @sio.on("get_status")
    def handle_get_status(data: dict = None):
        """Get status of active sessions."""
        with sessions_lock:
            active_sessions = [
                {
                    "session_id": s["session_id"],
                    "created_at": s["created_at"],
                    "status": s["status"],
                    "client_id": s["client_id"],
                }
                for s in sessions.values()
            ]
        
        emit("status", {
            "active_sessions": len(active_sessions),
            "sessions": active_sessions,
        })
    
    @sio.on("configure")
    def handle_configure(data: dict):
        """
        Configure analysis parameters.
        
        Expected data:
        {
            "confidence_threshold": 0.7,
            "neutral_threshold": 0.25,
            "enable_debug": false
        }
        """
        try:
            if "confidence_threshold" in data:
                sentiment_service.set_confidence_thresholds(
                    confidence=data["confidence_threshold"]
                )
            
            if "neutral_threshold" in data:
                sentiment_service.set_confidence_thresholds(
                    neutral=data["neutral_threshold"]
                )
            
            emit("configured", {
                "success": True,
                "config": {
                    "confidence_threshold": sentiment_service.confidence_threshold,
                    "neutral_threshold": sentiment_service.neutral_threshold,
                }
            })
        except Exception as e:
            emit("error", {
                "error": str(e),
                "error_code": "CONFIG_ERROR",
            })

