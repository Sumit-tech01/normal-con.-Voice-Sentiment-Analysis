"""
Pydantic models/schemas for Voice Sentiment Analysis API.
Defines request/response schemas for all API endpoints.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, field_validator
from enum import Enum


class SentimentLabel(str, Enum):
    """Enum for sentiment labels."""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    LOVE = "love"
    SURPRISE = "surprise"
    NEUTRAL = "neutral"


# ==================== Request Schemas ====================

class AudioUploadRequest(BaseModel):
    """Request model for audio file upload."""
    enable_streaming: bool = Field(
        default=False,
        description="Enable real-time streaming analysis"
    )
    language: Optional[str] = Field(
        default=None,
        description="Language code (e.g., 'en' for English)"
    )

    @field_validator("language")
    @classmethod
    def validate_language(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and len(v) != 2:
            raise ValueError("Language must be a 2-character code")
        return v


class StreamingStartRequest(BaseModel):
    """Request to start a streaming session."""
    language: Optional[str] = Field(
        default=None,
        description="Language code for transcription"
    )


class AudioChunkRequest(BaseModel):
    """Request for streaming audio chunk."""
    chunk_id: int = Field(..., description="Sequential chunk ID")
    is_final: bool = Field(
        default=False,
        description="Whether this is the final chunk"
    )
    timestamp: float = Field(
        ...,
        description="Client-side timestamp when chunk was recorded"
    )


# ==================== Response Schemas ====================

class SentimentResult(BaseModel):
    """Sentiment analysis result for a single emotion."""
    label: SentimentLabel
    score: float = Field(..., ge=0, le=1)


class TranscriptionResult(BaseModel):
    """Transcription result from Whisper."""
    text: str
    language: str
    confidence: Optional[float] = None
    duration: Optional[float] = None


class AudioAnalysisResponse(BaseModel):
    """Response model for complete audio analysis."""
    success: bool
    request_id: str = Field(..., description="Unique request identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Transcription
    transcription: TranscriptionResult
    
    # Sentiment Analysis
    sentiment: SentimentResult = Field(
        ...,
        description="Primary dominant sentiment"
    )
    all_emotions: Dict[SentimentLabel, float] = Field(
        ...,
        description="All emotion scores"
    )
    
    # Performance metrics
    processing_time_seconds: float = Field(
        ...,
        description="Total processing time"
    )
    model_load_time_seconds: Optional[float] = None
    
    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class StreamingUpdateResponse(BaseModel):
    """Response for streaming audio chunk update."""
    type: str = Field(default="sentiment_update")
    chunk_id: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Partial transcription (grows with each chunk)
    text_partial: Optional[str] = None
    
    # Current emotion state
    current_emotions: Dict[SentimentLabel, float] = Field(
        default_factory=dict
    )
    
    # Processing info
    is_final: bool = False
    processing_time_ms: Optional[float] = None


class StreamingCompleteResponse(BaseModel):
    """Response for streaming session completion."""
    type: str = Field(default="sentiment_complete")
    success: bool
    request_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Complete transcription
    text_complete: str
    language: str
    
    # Final sentiment
    dominant_sentiment: SentimentResult
    all_emotions: Dict[SentimentLabel, float]
    
    # Session metrics
    total_chunks: int
    total_processing_time_seconds: float


class ErrorResponse(BaseModel):
    """Standard error response."""
    success: bool = False
    error: str
    error_code: str = Field(..., description="Machine-readable error code")
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="overall health status")
    version: str
    services: Dict[str, Dict[str, Any]] = Field(
        ...,
        description="Status of individual services"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class ModelInfo(BaseModel):
    """Information about loaded models."""
    whisper_model: str
    whisper_device: str
    whisper_parameters: int
    sentiment_model: str
    sentiment_device: str
    loaded_at: datetime
    memory_usage_mb: Optional[float] = None

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


# ==================== WebSocket Message Schemas ====================

class WebSocketMessage(BaseModel):
    """Base schema for WebSocket messages."""
    type: str
    data: Dict[str, Any]


class WebSocketAudioChunk(BaseModel):
    """Audio chunk message from client."""
    type: str = "audio_chunk"
    data: str = Field(..., description="Base64-encoded audio data")
    chunk_id: int
    is_final: bool = False
    timestamp: float


class WebSocketResponse(BaseModel):
    """Response message to client."""
    type: str
    chunk_id: Optional[int] = None
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}

