# Voice Sentiment Analysis - Enhanced Version

A Flask-based voice sentiment analysis application with improved accuracy for casual/conversational speech.

## 🚀 Features

### Audio Preprocessing (`audio_utils.py`)
- **Noise Reduction**: Spectral gating for background noise removal
- **Normalization**: Peak + RMS normalization for consistent audio levels
- **16kHz Enforcement**: Automatic sample rate conversion
- **Voice Activity Detection (VAD)**: Detects speech vs silence
- **Audio Quality Validation**: Validates duration, silence ratio, and signal quality
- **Debug Mode**: Shows all preprocessing steps

### Whisper Enhancements (`whisper_service.py`)
- **Hallucination Detection**: Identifies potential transcription errors
- **Text Post-processing**: Fixes common transcription errors
- **Language Detection**: Automatic language detection
- **Confidence Scoring**: Improved confidence metrics
- **Min Duration Handling**: Validates audio length before transcription

### Sentiment Analysis (`sentiment_service.py`)
- **Text Preprocessing**: Removes filler words (um, uh, like, you know)
- **Artifact Cleaning**: Removes transcription artifacts ([laughter], etc.)
- **Sentence-Level Analysis**: Analyzes each sentence separately
- **Weighted Aggregation**: Word-count weighted emotion aggregation
- **Confidence Thresholds**: Configurable thresholds (default: >70%)
- **Neutral Detection**: Automatic neutral classification for ambiguous text
- **Context Awareness**: Detects greetings, questions, negations

## 📦 Installation

```bash
# Navigate to backend directory
cd backend

# Install dependencies
pip install -r requirements.txt

# Install development dependencies for testing
pip install pytest pytest-cov
```

## 🧪 Testing

### Run All Tests
```bash
cd backend
python test_cases.py
```

### Run with Verbose Output
```bash
cd backend
python test_cases.py --verbose
```

### Run with Debug Mode
```bash
cd backend
python test_cases.py --debug
```

### Run Example Usage
```bash
cd backend
python test_cases.py --example
```

### Run Specific Category
```bash
# Audio tests only
python test_cases.py --category audio

# Sentiment tests only
python test_cases.py --category sentiment

# Integration tests
python test_cases.py --category integration
```

### JSON Output
```bash
python test_cases.py --json
```

## 🔧 API Usage

### Health Check
```bash
curl http://localhost:5000/api/v1/health
```

### Get Model Info
```bash
curl http://localhost:5000/api/v1/models
```

### Get Configuration
```bash
curl http://localhost:5000/api/v1/config
```

### Update Configuration
```bash
curl -X PUT http://localhost:5000/api/v1/config \
  -H "Content-Type: application/json" \
  -d '{
    "confidence_threshold": 0.8,
    "neutral_threshold": 0.3,
    "debug_mode": true
  }'
```

### Analyze Text with Debug
```bash
curl -X POST http://localhost:5000/api/v1/analyze/text \
  -H "Content-Type: application/json" \
  -d '{
    "text": "I um, you know, really love this feature!",
    "enable_debug": true
  }'
```

### Analyze Audio File with Debug
```bash
curl -X POST http://localhost:5000/api/v1/analyze/upload \
  -F "audio=@audio.wav" \
  -F "enable_debug=true"
```

### Response with Debug Info
```json
{
  "success": true,
  "request_id": "abc123",
  "text": "I um, you know, really love this feature!",
  "cleaned_text": "I really love this feature",
  "sentences": [
    {
      "text": "I really love this feature",
      "sentiment": "joy",
      "score": 0.85,
      "word_count": 5,
      "low_confidence": false
    }
  ],
  "sentiment": {
    "label": "joy",
    "score": 0.85
  },
  "all_emotions": {
    "joy": 0.85,
    "sadness": 0.02,
    "anger": 0.01,
    "fear": 0.02,
    "love": 0.08,
    "surprise": 0.02,
    "neutral": 0.00
  },
  "debug": {
    "warnings": [],
    "context": {
      "is_greeting": false,
      "is_question": false,
      "has_intensifiers": true,
      "has_negation": false,
      "is_casual": true
    }
  }
}
```

## 📊 Confidence Thresholds

| Threshold | Default | Description |
|-----------|---------|-------------|
| `confidence_threshold` | 0.70 | Minimum score for confident prediction |
| `neutral_threshold` | 0.25 | Score above which neutral is considered |
| `low_confidence_threshold` | 0.40 | Below this triggers low confidence warning |

### Setting Custom Thresholds
```python
from app.services.sentiment_service import sentiment_service

# More strict (higher confidence required)
sentiment_service.set_confidence_thresholds(
    confidence=0.8,
    neutral=0.3,
    low_confidence=0.5
)

# More lenient (lower confidence accepted)
sentiment_service.set_confidence_thresholds(
    confidence=0.5,
    neutral=0.2,
    low_confidence=0.3
)
```

## 🎯 Expected Behaviors

### Normal Conversation (Casual Speech)
**Input**: "I um, you know, I'm kind of feeling okay about this"
**Output**: `neutral` or `joy` with moderate confidence

### Emotionally Charged Speech
**Input**: "I am SO happy and excited about this!"
**Output**: `joy` with high confidence

### Negative Sentiment
**Input**: "I'm really disappointed and frustrated"
**Output**: `anger` or `sadness` with high confidence

### Questions
**Input**: "What do you think about this?"
**Output**: `neutral` (questions are typically neutral)

### Mixed Emotions
**Input**: "I'm happy about the good news, but worried about the results"
**Output**: Weighted blend of `joy` and `fear`/`sadness`

## 🐛 Troubleshooting

### Low Confidence Scores

**Problem**: Getting many low confidence scores for normal speech.

**Solutions**:
1. Increase audio quality (reduce background noise)
2. Adjust `confidence_threshold` lower
3. Use longer audio samples (minimum 2-3 seconds)

### Incorrect Emotions for Normal Speech

**Problem**: Normal conversations showing wrong emotions.

**Solutions**:
1. Check VAD results - audio may have too much silence
2. Verify audio preprocessing is enabled
3. Review debug info for preprocessing issues
4. Try adjusting confidence thresholds

### Hallucination Detection Triggered

**Problem**: Whisper indicating potential hallucination.

**Solutions**:
1. Check audio quality (too noisy)
2. Verify audio duration is sufficient
3. Check for repeated audio regions

### Silent Audio

**Problem**: Audio validation fails due to silence.

**Solutions**:
1. Check microphone input levels
2. Verify VAD threshold settings
3. Ensure audio has sufficient volume

## 📁 Project Structure

```
backend/
├── app/
│   ├── api/
│   │   ├── routes.py          # REST API endpoints
│   │   └── websocket.py       # WebSocket handlers
│   ├── config.py              # Configuration
│   ├── models/
│   │   └── schemas.py         # Pydantic schemas
│   ├── services/
│   │   ├── whisper_service.py # Enhanced Whisper transcription
│   │   └── sentiment_service.py # Enhanced sentiment analysis
│   └── utils/
│       └── audio_utils.py     # Enhanced audio preprocessing
├── test_cases.py              # Comprehensive tests
├── requirements.txt
└── README.md
```

## 🔧 Configuration

All configuration is managed through environment variables or the `/api/v1/config` endpoint.

### Key Settings
```python
# Audio Processing
AUDIO_SAMPLE_RATE = 16000
MIN_AUDIO_DURATION = 0.5  # seconds

# Whisper
WHISPER_MODEL_SIZE = "base"
MIN_CONFIDENCE_THRESHOLD = 0.15

# Sentiment
CONFIDENCE_THRESHOLD = 0.70
NEUTRAL_THRESHOLD = 0.25
LOW_CONFIDENCE_THRESHOLD = 0.40

# Text Preprocessing
FILLER_WORDS = {'um', 'uh', 'like', 'you know', ...}
```

## 📝 License

MIT License

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Run tests to ensure compatibility
5. Submit a pull request

