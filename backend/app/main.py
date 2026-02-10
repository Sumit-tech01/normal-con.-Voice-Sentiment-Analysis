"""
Main Flask application entry point.
Voice Sentiment Analysis API with REST and WebSocket support.
"""

import logging
import sys
from pathlib import Path

from flask import Flask, jsonify
from flask_cors import CORS

from .config import config
from .api.routes import api_bp
from .api.websocket import init_socketio
from .services.whisper_service import whisper_service
from .services.sentiment_service import sentiment_service

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def create_app() -> tuple[Flask, any]:
    """
    Application factory for creating Flask app.
    
    Returns:
        Tuple of (Flask app, SocketIO instance)
    """
    # Create Flask app
    app = Flask(
        __name__,
        static_folder="../frontend/dist",
        static_url_path="/",
        template_folder="../frontend",
    )
    
    # Configure app
    app.config["SECRET_KEY"] = config.SECRET_KEY
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH
    
    # Enable CORS
    CORS(
        app,
        origins=config.CORS_ORIGINS,
        supports_credentials=config.CORS_SUPPORTS_CREDENTIALS,
    )
    
    # Register blueprints
    app.register_blueprint(api_bp)
    
    # Initialize SocketIO
    socketio = init_socketio(
        app,
        cors_allowed_origins=config.SOCKETIO_CORS_ALLOWED_ORIGINS or "*",
    )
    
    # Register error handlers
    register_error_handlers(app)
    
    # Load models on startup (optional, can be lazy loaded)
    @app.before_request
    def before_first_request():
        """Load models on first request for faster responses."""
        pass  # Models are lazy loaded in services
    
    # Root route - serve frontend
    @app.route("/")
    def index():
        """Serve the main frontend."""
        return app.send_static_file("index.html")
    
    # API documentation route
    @app.route("/api")
    def api_docs():
        """Return API documentation redirect."""
        return jsonify({
            "message": "Voice Sentiment Analysis API",
            "version": config.APP_VERSION,
            "docs": "/api/v1/docs",
            "health": "/api/v1/health",
        })
    
    logger.info(f"Flask app created in {config.APP_NAME}")
    
    return app, socketio


def register_error_handlers(app: Flask) -> None:
    """Register global error handlers."""
    
    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({
            "success": False,
            "error": "Bad request",
            "error_code": "BAD_REQUEST",
        }), 400
    
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            "success": False,
            "error": "Endpoint not found",
            "error_code": "NOT_FOUND",
        }), 404
    
    @app.errorhandler(413)
    def payload_too_large(error):
        return jsonify({
            "success": False,
            "error": "File too large",
            "error_code": "FILE_TOO_LARGE",
            "details": {
                "max_size_mb": config.AUDIO_MAX_FILE_SIZE_MB,
            },
        }), 413
    
    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Internal server error: {error}", exc_info=True)
        return jsonify({
            "success": False,
            "error": "Internal server error",
            "error_code": "INTERNAL_ERROR",
        }), 500


def init_models(load_whisper: bool = True, load_sentiment: bool = True) -> None:
    """
    Pre-load ML models for faster first request.
    
    Args:
        load_whisper: Load Whisper model
        load_sentiment: Load sentiment model
    """
    logger.info("Initializing ML models...")
    
    if load_whisper:
        logger.info("Loading Whisper model...")
        whisper_service.load_model()
    
    if load_sentiment:
        logger.info("Loading Sentiment model...")
        sentiment_service.load_model()
    
    logger.info("ML models initialized successfully")


def main():
    """Main entry point for running the application."""
    import argparse
    import uvicorn
    
    parser = argparse.ArgumentParser(
        description="Voice Sentiment Analysis API"
    )
    parser.add_argument(
        "--host",
        default=config.HOST,
        help="Host to bind to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=config.PORT,
        help="Port to bind to",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes",
    )
    parser.add_argument(
        "--init-models",
        action="store_true",
        help="Pre-load ML models on startup",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )
    
    args = parser.parse_args()
    
    # Update config with command line args
    config.HOST = args.host
    config.PORT = args.port
    config.DEBUG = args.debug
    
    # Create app
    app, socketio = create_app()
    
    # Pre-load models if requested
    if args.init_models or args.debug:
        init_models()
    
    # Run with SocketIO
    logger.info(f"Starting server on {args.host}:{args.port}")
    
    if args.debug or args.reload:
        socketio.run(
            app,
            host=args.host,
            port=args.port,
            debug=True,
            use_reloader=args.reload,
            allow_unsafe_werkzeug=True,
        )
    else:
        socketio.run(
            app,
            host=args.host,
            port=args.port,
            workers=args.workers,
        )


# Application instance for WSGI servers
app, socketio = create_app()


if __name__ == "__main__":
    main()

