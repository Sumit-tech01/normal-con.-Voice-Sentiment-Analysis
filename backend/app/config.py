"""
Configuration management for Voice Sentiment Analysis application.
Loads settings from environment variables and provides sensible defaults.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Application configuration class."""

    # Application Settings
    APP_NAME: str = "Voice Sentiment Analysis"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    TESTING: bool = False

    # Server Configuration
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "5000"))
    SECRET_KEY: str = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")

    # WebSocket Configuration
    SOCKETIO_PATH: str = "/socket.io"
    SOCKETIO_CORS_ALLOWED_ORIGINS: Optional[str] = os.getenv(
        "SOCKETIO_CORS_ORIGINS", "*"
    )

    # Whisper Model Settings
    WHISPER_MODEL_SIZE: str = os.getenv("WHISPER_MODEL_SIZE", "base")  # tiny, base, small, medium, large
    WHISPER_DEVICE: str = os.getenv("WHISPER_DEVICE", "auto")  # auto, cpu, cuda
    WHISPER_COMPUTE_TYPE: str = os.getenv("WHISPER_COMPUTE_TYPE", "default")  # default, int8, int16

    # Sentiment Model Settings
    SENTIMENT_MODEL_NAME: str = os.getenv(
        "SENTIMENT_MODEL_NAME",
        "j-hartmann/emotion-english-distilroberta-base"
    )
    SENTIMENT_DEVICE: str = os.getenv("SENTIMENT_DEVICE", "auto")
    SENTIMENT_TOP_K: int = int(os.getenv("SENTIMENT_TOP_K", "1"))  # Number of top emotions to return

    # Audio Processing Settings
    AUDIO_SAMPLE_RATE: int = 16000
    AUDIO_CHUNK_DURATION_MS: int = 250  # For streaming (Phase 2)
    AUDIO_MAX_DURATION_SEC: int = 300  # 5 minutes max
    AUDIO_SUPPORTED_FORMATS: tuple = (".wav", ".mp3", ".webm", ".ogg", ".flac")
    AUDIO_MAX_FILE_SIZE_MB: int = 50

    # Streaming Settings (Phase 2)
    STREAMING_CHUNK_SIZE: int = 1024 * 4  # 4KB chunks
    STREAMING_BUFFER_SECONDS: float = 2.0
    STREAMING_MAX_CONNECTIONS: int = 10

    # Celery Settings (Phase 2)
    CELERY_BROKER_URL: str = os.getenv(
        "CELERY_BROKER_URL", "redis://localhost:6379/0"
    )
    CELERY_RESULT_BACKEND: str = os.getenv(
        "CELERY_RESULT_BACKEND", "redis://localhost:6379/0"
    )

    # Redis Settings (Phase 2)
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))

    # File Upload Settings
    UPLOAD_FOLDER: Path = Path(os.getenv("UPLOAD_FOLDER", "./uploads"))
    MAX_CONTENT_LENGTH: int = AUDIO_MAX_FILE_SIZE_MB * 1024 * 1024

    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = os.getenv(
        "LOG_FORMAT",
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # CORS Settings
    CORS_ORIGINS: list = os.getenv("CORS_ORIGINS", "*").split(",")
    CORS_SUPPORTS_CREDENTIALS: bool = os.getenv("CORS_CREDENTIALS", "False").lower() == "true"

    # Model Cache Settings
    MODEL_CACHE_DIR: Path = Path(os.getenv("MODEL_CACHE_DIR", "./models"))
    
    # Paths (computed)
    @property
    def UPLOAD_FOLDER_ABSOLUTE(self) -> Path:
        """Get absolute path for uploads."""
        if self.UPLOAD_FOLDER.is_absolute():
            return self.UPLOAD_FOLDER
        return Path(__file__).parent.parent / self.UPLOAD_FOLDER

    @property
    def MODEL_CACHE_DIR_ABSOLUTE(self) -> Path:
        """Get absolute path for model cache."""
        if self.MODEL_CACHE_DIR.is_absolute():
            return self.MODEL_CACHE_DIR
        return Path(__file__).parent.parent / self.MODEL_CACHE_DIR

    def validate(self) -> None:
        """Validate configuration settings."""
        # Check for required environment variables in production
        if not self.DEBUG and not self.TESTING:
            if self.SECRET_KEY == "dev-secret-key-change-in-production":
                raise ValueError(
                    "SECRET_KEY must be set in production environment"
                )

        # Validate model size
        valid_models = ["tiny", "base", "small", "medium", "large"]
        if self.WHISPER_MODEL_SIZE not in valid_models:
            raise ValueError(
                f"WHISPER_MODEL_SIZE must be one of {valid_models}, "
                f"got '{self.WHISPER_MODEL_SIZE}'"
            )

        # Validate audio formats
        if not all(fmt.startswith(".") for fmt in self.AUDIO_SUPPORTED_FORMATS):
            raise ValueError("AUDIO_SUPPORTED_FORMATS must start with '.'")


class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG: bool = True
    TESTING: bool = False
    LOG_LEVEL: str = "DEBUG"


class TestingConfig(Config):
    """Testing configuration."""
    DEBUG: bool = True
    TESTING: bool = True
    LOG_LEVEL: str = "DEBUG"
    # Use smaller models for faster tests
    WHISPER_MODEL_SIZE: str = "tiny"
    SENTIMENT_MODEL_NAME: str = "j-hartmann/emotion-english-distilroberta-base"


class ProductionConfig(Config):
    """Production configuration."""
    DEBUG: bool = False
    TESTING: bool = False
    LOG_LEVEL: str = "INFO"


def get_config(env: Optional[str] = None) -> Config:
    """
    Get configuration based on environment.

    Args:
        env: Environment name ('development', 'testing', 'production')
             If None, reads from ENVIRONMENT env var or defaults to development

    Returns:
        Config instance for the specified environment
    """
    if env is None:
        env = os.getenv("ENVIRONMENT", "development").lower()

    configs = {
        "development": DevelopmentConfig,
        "testing": TestingConfig,
        "production": ProductionConfig,
        "dev": DevelopmentConfig,
        "test": TestingConfig,
        "prod": ProductionConfig,
    }

    config_class = configs.get(env, DevelopmentConfig)
    config = config_class()
    
    # Validate configuration
    try:
        config.validate()
    except ValueError as e:
        raise ValueError(f"Configuration validation failed: {e}")

    return config


# Global config instance
config = get_config()

