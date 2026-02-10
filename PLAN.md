# 🎤 Voice Sentiment Analysis - AI-Assisted Development Brief

## 1. ROLE & CONTEXT

You are a senior full-stack engineer using VS Code with AI assistance to build a voice sentiment analysis application. The team is **solo developer** and we're targeting an **MVP in 2 weeks**.

**Environment:**
- VS Code 1.85+ with extensions: Python, Docker, ESLint, Prettier, REST Client
- AI Tool: Claude for architecture decisions, code generation, debugging
- **Stack:** Flask 3.0 + Whisper (speech-to-text) + Transformers (sentiment) + Chart.js

## 2. OBJECTIVE & DELIVERABLES

**Primary Goal:** Build a real-time voice sentiment analysis web application that:
- Records audio from user's microphone
- Transcribes speech to text using OpenAI Whisper
- Analyzes sentiment using DistilBERT (joy, sadness, anger, fear, love, surprise)
- Displays results with live visualizations
- Supports both real-time streaming and file upload

**Deliverables:**
- [x] Flask backend with REST API and WebSocket support
- [x] Whisper integration for speech-to-text
- [x] Transformers integration for sentiment analysis
- [x] Real-time audio streaming with WebSocket
- [x] Interactive frontend with Chart.js visualization
- [x] Responsive UI with Tailwind CSS
- [x] Docker setup for deployment
- [x] Unit tests (>80% coverage)
- [x] README with setup instructions

## 3. CONSTRAINTS & SCOPE

### Technical Constraints

| Component | Version/Requirement |
|-----------|-------------------|
| Python | 3.10+ |
| Flask | 3.0+ |
| Flask-SocketIO | 5.3+ |
| OpenAI Whisper | Latest (base model) |
| Transformers | 4.30+ |
| Node.js | 18+ (for frontend tooling) |
| Browser Support | Chrome, Firefox, Safari (last 2 versions) |

### Performance Requirements

- **Latency (Streaming):** < 500ms from audio chunk to sentiment result
- **Latency (Upload):** < 3s for 30-second audio file
- **Memory:** < 500MB for model loading, streaming process
- **Model Size:** Whisper base (~74MB), DistilBERT (~250MB)

### Scope Boundaries

**IN SCOPE:**
- Audio recording from microphone
- Speech-to-text transcription
- Sentiment analysis (6 emotions)
- Real-time WebSocket streaming
- File upload (WAV, MP3, WebM)
- Chart.js visualizations
- Responsive web interface
- Docker containerization
- Unit and integration tests

**OUT OF SCOPE:**
- Mobile app (web only)
- Multiple languages (English only MVP)
- User authentication
- Database persistence
- Cloud deployment (local only)

### Safety & Quality Requirements

- **Code Quality:** ESLint + Black formatter, type hints required
- **Security:** No hardcoded API keys, input validation on all uploads
- **Documentation:** Inline docstrings, API documentation
- **Error Handling:** Graceful degradation, user-friendly error messages

## 4. OUTPUT REQUIREMENTS

### Project Structure

```
voice-sentiment-app/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py              # Flask app entry point
│   │   ├── config.py            # Configuration management
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── routes.py        # REST API endpoints
│   │   │   └── websocket.py     # WebSocket handlers
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── whisper_service.py     # Speech-to-text
│   │   │   ├── sentiment_service.py   # Sentiment analysis
│   │   │   └── streaming_service.py   # Real-time processing
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   └── schemas.py       # Pydantic schemas
│   │   └── utils/
│   │       ├── __init__.py
│   │       └── audio_utils.py   # Audio processing utilities
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── test_api.py
│   │   ├── test_services.py
│   │   └── conftest.py
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── index.html
│   │   ├── css/
│   │   │   └── styles.css
│   │   ├── js/
│   │   │   ├── app.js           # Main application logic
│   │   │   ├── recorder.js     # Audio recording
│   │   │   ├── websocket.js    # WebSocket client
│   │   │   ├── chart.js        # Visualization
│   │   │   └── api.js          # REST API client
│   │   └── assets/
│   ├── package.json
│   └── vite.config.js
├── docker-compose.yml
├── .gitignore
└── README.md
```

### Code Style & Patterns

- **Naming Convention:** snake_case for Python, camelCase for JavaScript
- **Architecture:** Layered (API → Services → Models)
- **Patterns to Use:** Repository pattern for services, Factory pattern for models
- **Patterns to Avoid:** Global state, circular dependencies
- **Formatting:** Black (Python), Prettier (JavaScript)

### Documentation Requirements

- [x] README.md with setup instructions
- [x] API documentation (inline docstrings)
- [x] Architecture overview
- [x] Deployment guide

## 5. PHASED IMPLEMENTATION

### Phase 1: MVP (Week 1)

**Goals:**
- [x] Flask backend with basic API
- [x] Whisper integration for transcription
- [x] Sentiment analysis with Transformers
- [x] File upload endpoint
- [x] Basic frontend UI
- [x] Chart.js visualization for results

**Success Criteria:**
- ✅ Upload audio file → get transcription + sentiment
- ✅ Transcription accuracy > 90% on clear audio
- ✅ Sentiment confidence > 85% on strong emotions
- ✅ Page load < 2s
- ✅ API response < 3s for 30s audio

### Phase 2: Enhancement (Week 2)

**Goals:**
- [x] WebSocket real-time streaming
- [x] Microphone recording in browser
- [x] Chunked audio processing (250ms chunks)
- [x] Live visualization updates
- [x] Celery async processing
- [ ] Comprehensive testing (>80% coverage)
- [ ] Error handling and graceful degradation

**Success Criteria:**
- ✅ Real-time streaming with < 500ms latency
- ✅ Progressive results as audio is processed
- ✅ Smooth UI during processing (no freezing)
- ✅ All tests pass

### Phase 3: Production (Week 3+)

**Goals:**
- [ ] Docker containerization
- [ ] CI/CD pipeline
- [ ] Performance optimization
- [ ] Security hardening
- [ ] Documentation complete
- [ ] Deployment scripts

**Success Criteria:**
- ✅ Docker image < 2GB
- ✅ Memory usage < 500MB
- ✅ Production-ready deployment

## 6. EXAMPLES & REFERENCES

### Input Example

**Audio Upload Request:**
```bash
POST /api/v1/analyze/upload
Content-Type: multipart/form-data

audio: [WAV/MP3 file]
```

 Response**Expected:**
```json
{
  "success": true,
  "data": {
    "id": "abc-123",
    "text": "I'm so happy to see you today!",
    "sentiment": {
      "label": "joy",
      "confidence": 0.94
    },
    "emotions": {
      "joy": 0.94,
      "sadness": 0.02,
      "anger": 0.01,
      "fear": 0.01,
      "love": 0.01,
      "surprise": 0.01
    },
    "processing_time": 2.3,
    "timestamp": "2024-02-08T12:00:00Z"
  }
}
```

### Streaming Example

**WebSocket Message Flow:**
```javascript
// Client sends audio chunk
{
  "type": "audio_chunk",
  "data": "<base64 encoded audio>",
  "chunk_id": 1,
  "is_final": false
}

// Server responds with incremental results
{
  "type": "sentiment_update",
  "chunk_id": 1,
  "text_partial": "I'm so happy...",
  "emotions": {
    "joy": 0.92,
    "sadness": 0.03,
    "anger": 0.02,
    "fear": 0.01,
    "love": 0.01,
    "surprise": 0.01
  },
  "timestamp": "2024-02-08T12:00:00.100Z"
}

// Final chunk with complete results
{
  "type": "sentiment_complete",
  "chunk_id": 10,
  "is_final": true,
  "text_complete": "I'm so happy to see you today!",
  "emotions": {...},
  "total_processing_time": 3.2
}
```

### Reference Implementations

- **Whisper:** https://github.com/openai/whisper
- **Transformers:** https://huggingface.co/docs/transformers
- **Flask-SocketIO:** https://flask-socketio.readthedocs.io
- **Sentiment Analysis:** https://huggingface.co/j-hartmann/emotion-english-distilroberta-base

## 7. INTERACTION PROTOCOL

### When to Ask for Clarification

**ASK BEFORE:**
- Choosing between different Whisper model sizes (base vs small)
- Changing the sentiment analysis model architecture
- Modifying WebSocket protocol
- Adding new API endpoints
- Changing the frontend framework

### When to Make Decisions

**DECIDE AUTONOMOUSLY:**
- Component structure within services
- Utility function implementation
- CSS styling and UI details
- Error message wording
- Test case scenarios
- Configuration values (unless security-related)

### Iteration Preferences

- **Review frequency:** After each major feature (Phase 1, 2, 3)
- **Feedback format:** Written summary with code review notes
- **Iteration cycle:** Complete phase, review, then proceed

---

## 🎯 Quick Start Commands

```bash
# Clone and setup
git clone <repo-url>
cd voice-sentiment-app

# Backend setup
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r backend/requirements.txt

# Frontend setup
cd frontend
npm install

# Run with Docker
docker-compose up --build

# Run locally
# Terminal 1: Backend
cd backend
python -m uvicorn app.main:app --reload

# Terminal 2: Frontend
cd frontend
npm run dev
```

## 📊 Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Transcription Accuracy | > 90% | Whisper benchmark |
| Sentiment Accuracy | > 85% | Ground truth comparison |
| Streaming Latency | < 500ms | End-to-end timing |
| API Response Time | < 3s | For 30s audio file |
| Test Coverage | > 80% | pytest coverage |
| Memory Usage | < 500MB | Model loading |
| Bundle Size | < 500KB | Frontend (gzipped) |

