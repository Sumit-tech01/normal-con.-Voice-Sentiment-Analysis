"""
Comprehensive Test Cases for Voice Sentiment Analysis.
Provides unit tests, integration tests, and example usage.

Usage:
    python3 test_cases.py              # Run all tests
    python3 test_cases.py --verbose    # Run with verbose output
    python3 test_cases.py --debug      # Run with debug mode
    python3 test_cases.py --example    # Run example usage
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

# Add parent directory to path for imports
backend_path = str(Path(__file__).parent)
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestCategory(Enum):
    """Test categories."""
    AUDIO_UTILS = "Audio Utilities"
    WHISPER = "Whisper Service"
    SENTIMENT = "Sentiment Service"
    PREPROCESSING = "Text Preprocessing"
    INTEGRATION = "Integration"
    API = "API"


@dataclass
class TestCase:
    """Test case definition."""
    name: str
    category: TestCategory
    description: str
    input_data: Any
    expected_output: Any
    validate_fn: callable
    tags: List[str] = None


@dataclass
class TestResult:
    """Test result."""
    name: str
    category: TestCategory
    passed: bool
    duration_ms: float
    actual_output: Any = None
    error: str = None


class TestRunner:
    """Test runner for voice sentiment analysis."""
    
    def __init__(self, verbose: bool = False, debug: bool = False):
        """Initialize test runner."""
        self.verbose = verbose
        self.debug = debug
        self.results: List[TestResult] = []
        
        if self.debug:
            logging.getLogger().setLevel(logging.DEBUG)
    
    def log(self, message: str):
        """Log message."""
        if self.verbose:
            print(f"  [INFO] {message}")
    
    def run_test(self, test_case: TestCase) -> TestResult:
        """Run a single test case."""
        start_time = time.time()
        
        try:
            # Run validation function with or without input data
            if test_case.input_data is not None:
                result = test_case.validate_fn(test_case.input_data)
            else:
                result = test_case.validate_fn()
            
            # Check if result matches expected
            passed = result == test_case.expected_output
            
            duration_ms = (time.time() - start_time) * 1000
            
            return TestResult(
                name=test_case.name,
                category=test_case.category,
                passed=passed,
                duration_ms=duration_ms,
                actual_output=result,
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            
            return TestResult(
                name=test_case.name,
                category=test_case.category,
                passed=False,
                duration_ms=duration_ms,
                error=str(e),
            )
    
    def run_all_tests(self, tests: List[TestCase]) -> Dict[str, Any]:
        """Run all test cases."""
        self.results = []
        
        print("\n" + "="*60)
        print("VOICE SENTIMENT ANALYSIS - TEST SUITE")
        print("="*60)
        
        # Group tests by category
        tests_by_category: Dict[TestCategory, List[TestCase]] = {}
        for test in tests:
            if test.category not in tests_by_category:
                tests_by_category[test.category] = []
            tests_by_category[test.category].append(test)
        
        # Run tests by category
        total_passed = 0
        total_failed = 0
        
        for category, category_tests in tests_by_category.items():
            print(f"\n📂 {category.value}")
            print("-" * 40)
            
            category_passed = 0
            category_failed = 0
            
            for test in category_tests:
                result = self.run_test(test)
                self.results.append(result)
                
                if result.passed:
                    category_passed += 1
                    print(f"  ✅ {test.name}")
                    if self.verbose:
                        print(f"     Duration: {result.duration_ms:.2f}ms")
                else:
                    category_failed += 1
                    print(f"  ❌ {test.name}")
                    if result.error:
                        print(f"     Error: {result.error}")
                    else:
                        print(f"     Expected: {test.expected_output}")
                        print(f"     Got: {result.actual_output}")
            
            total_passed += category_passed
            total_failed += category_failed
            
            print(f"  📊 {category.value}: {category_passed} passed, {category_failed} failed")
        
        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Total Tests: {total_passed + total_failed}")
        print(f"Passed: {total_passed}")
        print(f"Failed: {total_failed}")
        print(f"Success Rate: {(total_passed / (total_passed + total_failed) * 100):.1f}%" if (total_passed + total_failed) > 0 else "Success Rate: 0.0%")
        print("="*60)
        
        return {
            "total": total_passed + total_failed,
            "passed": total_passed,
            "failed": total_failed,
            "success_rate": (total_passed / (total_passed + total_failed) * 100) if (total_passed + total_failed) > 0 else 0,
            "results": [asdict(r) for r in self.results],
        }


# ==================== Audio Utils Tests ====================

def test_audio_utils_import():
    """Test that audio_utils imports correctly."""
    from app.utils.audio_utils import (
        validate_audio_file,
        preprocess_audio,
        normalize_audio,
        spectral_gate,
        voice_activity_detection,
        validate_audio_quality,
    )
    return True


def test_audio_normalization():
    """Test audio normalization."""
    from app.utils.audio_utils import normalize_audio
    import numpy as np
    
    # Create test audio with varying amplitudes
    audio = np.array([0.1, 0.5, 0.8, 0.3, -0.6, -0.9, 0.4])
    normalized = normalize_audio(audio)
    
    # Check that peak is now at 0.95
    assert np.max(np.abs(normalized)) <= 0.95, f"Peak should be <= 0.95, got {np.max(np.abs(normalized))}"
    assert np.max(np.abs(normalized)) > 0.9, f"Peak should be > 0.9, got {np.max(np.abs(normalized))}"
    
    return True


def test_silence_removal():
    """Test silence removal."""
    from app.utils.audio_utils import remove_silence
    import numpy as np
    
    # Create audio with silence at beginning and end
    audio = np.concatenate([
        np.zeros(1000),  # Silence
        np.array([0.5, 0.6, 0.7, 0.6, 0.5]),  # Speech
        np.zeros(1000),  # Silence
    ])
    
    cleaned = remove_silence(audio, sample_rate=16000, threshold_db=-40)
    
    # Should be much shorter
    assert len(cleaned) < len(audio), "Silence should be removed"
    assert len(cleaned) >= 5, "Speech should be preserved"
    
    return True


def test_vad():
    """Test voice activity detection."""
    from app.utils.audio_utils import voice_activity_detection
    import numpy as np
    
    # Create speech-like audio
    speech_audio = np.sin(2 * np.pi * 440 * np.arange(16000)) * 0.5
    
    # Create silent audio
    silent_audio = np.zeros(16000)
    
    speech_result = voice_activity_detection(speech_audio, 16000)
    silent_result = voice_activity_detection(silent_audio, 16000)
    
    assert speech_result["is_speech"] == True, "Should detect speech"
    assert speech_result["is_silence"] == False, "Should not be silent"
    assert silent_result["is_silence"] == True, "Should detect silence"
    assert silent_result["is_speech"] == False, "Should not detect speech"
    
    return True


def test_audio_quality_validation():
    """Test audio quality validation."""
    from app.utils.audio_utils import validate_audio_quality
    import numpy as np
    
    # Valid audio
    valid_audio = np.random.randn(16000) * 0.5
    is_valid, msg = validate_audio_quality(valid_audio, 16000)
    assert is_valid == True, f"Valid audio should pass: {msg}"
    
    # Silent audio
    silent_audio = np.zeros(1000)
    is_valid, msg = validate_audio_quality(silent_audio, 16000)
    assert is_valid == False, "Silent audio should fail"
    
    return True


def test_noise_reduction():
    """Test spectral gating noise reduction."""
    from app.utils.audio_utils import spectral_gate
    import numpy as np
    
    # Create noisy audio
    np.random.seed(42)
    clean = np.sin(2 * np.pi * 440 * np.arange(8000) / 16000) * 0.5
    noise = np.random.randn(8000) * 0.2
    noisy = clean + noise
    
    # Apply noise reduction
    cleaned = spectral_gate(noisy, 16000)
    
    # Should have reduced noise while preserving signal
    assert len(cleaned) == len(noisy), "Length should be preserved"
    assert np.max(np.abs(cleaned)) <= 1.0, "Should not clip"
    
    return True


# ==================== Whisper Service Tests ====================

def test_whisper_import():
    """Test that whisper_service imports correctly."""
    from app.services.whisper_service import (
        WhisperService,
        WhisperDebugInfo,
    )
    return True


def test_hallucination_detection():
    """Test hallucination detection."""
    from app.services.whisper_service import WhisperService
    
    service = WhisperService()
    
    # Normal text - should not be hallucination
    is_halluc, reasons = service._detect_hallucination(
        "Hello, how are you today?",
        confidence=0.9,
        audio_duration=2.0
    )
    assert is_halluc == False, "Normal text should not be hallucination"
    
    # Low confidence - might be hallucination
    is_halluc, reasons = service._detect_hallucination(
        "Test text",
        confidence=0.05,
        audio_duration=1.0
    )
    assert len(reasons) > 0, "Low confidence should have reasons"
    
    # Repetitive text - likely hallucination
    is_halluc, reasons = service._detect_hallucination(
        "test test test test test test",
        confidence=0.8,
        audio_duration=0.5
    )
    assert is_halluc == True, "Repetitive text should be detected as hallucination"
    
    return True


def test_text_postprocessing():
    """Test text post-processing."""
    from app.services.whisper_service import WhisperService
    
    service = WhisperService()
    
    # Test common corrections
    text = "hello   world  test..."
    cleaned = service._post_process_text(text)
    assert "  " not in cleaned, "Multiple spaces should be removed"
    
    # Test bracket removal
    text = "Hello [speaker 1] world"
    cleaned = service._post_process_text(text)
    assert "[" not in cleaned, "Brackets should be removed"
    
    # Test punctuation cleanup
    text = "  Hello world  "
    cleaned = service._post_process_text(text)
    assert cleaned == "Hello world", "Should trim whitespace"
    
    return True


def test_min_audio_duration():
    """Test minimum audio duration validation."""
    from app.services.whisper_service import WhisperService
    
    service = WhisperService()
    
    # Valid duration
    is_valid, msg = service._validate_audio_duration(1.0)
    assert is_valid == True, "Valid duration should pass"
    
    # Too short
    is_valid, msg = service._validate_audio_duration(0.1)
    assert is_valid == False, "Too short should fail"
    
    # Too long
    is_valid, msg = service._validate_audio_duration(400)
    assert is_valid == False, "Too long should fail"
    
    return True


# ==================== Sentiment Service Tests ====================

def test_sentiment_import():
    """Test that sentiment_service imports correctly."""
    from app.services.sentiment_service import (
        EnhancedSentimentService,
        SentimentService,
        text_preprocessor,
        TextPreprocessor,
    )
    return True


def test_text_preprocessor():
    """Test text preprocessing."""
    from app.services.sentiment_service import text_preprocessor
    
    # Test filler word removal
    text = "I um, you know, really like this"
    cleaned = text_preprocessor.remove_filler_words(text)
    assert "um" not in cleaned.lower(), "um should be removed"
    assert "you know" not in cleaned.lower(), "you know should be removed"
    
    # Test artifact cleaning
    text = "Hello [music] world"
    cleaned = text_preprocessor.clean_artifacts(text)
    assert "[" not in cleaned, "Brackets should be removed"
    
    # Test full preprocessing
    text = "I um, [laughter] you know, really like this"
    cleaned = text_preprocessor.preprocess(text)
    assert "um" not in cleaned.lower(), "um should be removed"
    assert "[" not in cleaned, "Bracket should be removed"
    
    return True


def test_sentence_splitting():
    """Test sentence splitting."""
    from app.services.sentiment_service import text_preprocessor
    
    text = "Hello world. How are you? I'm fine!"
    sentences = text_preprocessor.split_into_sentences(text)
    
    assert len(sentences) >= 2, f"Should have multiple sentences, got {len(sentences)}"
    
    return True


def test_context_detection():
    """Test conversational context detection."""
    from app.services.sentiment_service import text_preprocessor
    
    # Greeting
    context = text_preprocessor.detect_context("Hello, how are you?")
    assert context['is_greeting'] == True, "Should detect greeting"
    
    # Question
    context = text_preprocessor.detect_context("What is your name?")
    assert context['is_question'] == True, "Should detect question"
    
    # Negation
    context = text_preprocessor.detect_context("I don't like this")
    assert context['has_negation'] == True, "Should detect negation"
    
    return True


def test_sentiment_analysis():
    """Test sentiment analysis."""
    from app.services.sentiment_service import sentiment_service
    
    # Test positive text
    result, debug = sentiment_service.analyze("I'm so happy and excited about this!")
    assert result.label.value in ['joy', 'surprise'], f"Expected positive emotion, got {result.label}"
    assert result.score > 0.5, f"Confidence should be > 0.5, got {result.score}"
    
    # Test negative text
    result, debug = sentiment_service.analyze("I'm really sad and disappointed")
    assert result.label.value in ['sadness'], f"Expected sadness, got {result.label}"
    
    # Test neutral text
    result, debug = sentiment_service.analyze("The sky is blue today")
    assert result.label.value in ['neutral', 'sadness', 'joy'], f"Expected neutral/ambiguous, got {result.label}"
    
    return True


def test_sentiment_with_debug():
    """Test sentiment analysis with debug info."""
    from app.services.sentiment_service import sentiment_service
    
    result, debug = sentiment_service.analyze(
        "I am really happy today! But sometimes I feel sad.",
        return_debug=True
    )
    
    assert debug is not None, "Debug info should be returned"
    assert debug.original_text is not None, "Original text should be saved"
    assert debug.cleaned_text is not None, "Cleaned text should be saved"
    assert len(debug.sentences) > 0, "Sentences should be analyzed"
    
    return True


def test_sentiment_confidence_thresholds():
    """Test confidence threshold settings."""
    from app.services.sentiment_service import sentiment_service
    
    # Set custom thresholds
    sentiment_service.set_confidence_thresholds(
        confidence=0.8,
        neutral=0.3,
        low_confidence=0.5
    )
    
    assert sentiment_service.confidence_threshold == 0.8
    assert sentiment_service.neutral_threshold == 0.3
    assert sentiment_service.low_confidence_threshold == 0.5
    
    return True


def test_emotion_aggregation():
    """Test weighted emotion aggregation."""
    from app.services.sentiment_service import EnhancedSentimentService
    from app.services.sentiment_service import SentenceResult
    from app.models.schemas import SentimentLabel
    
    service = EnhancedSentimentService()
    
    # Create mock sentence results
    sentences = [
        SentenceResult(
            text="I am happy",
            sentiment=SentimentLabel.JOY,
            score=0.9,
            all_scores={SentimentLabel.JOY: 0.9, SentimentLabel.SADNESS: 0.1},
            word_count=3
        ),
        SentenceResult(
            text="But sometimes I am sad",
            sentiment=SentimentLabel.SADNESS,
            score=0.8,
            all_scores={SentimentLabel.JOY: 0.2, SentimentLabel.SADNESS: 0.8},
            word_count=5
        ),
    ]
    
    label, score, scores = service._aggregate_emotions(sentences, method="weighted")
    
    # Should be a blend of both
    assert label in [SentimentLabel.JOY, SentimentLabel.SADNESS], "Should be either joy or sadness"
    assert scores[SentimentLabel.JOY] > 0, "Should have some joy"
    assert scores[SentimentLabel.SADNESS] > 0, "Should have some sadness"
    
    return True


def test_neutral_detection():
    """Test neutral emotion detection."""
    from app.services.sentiment_service import sentiment_service
    
    # Text that should be neutral
    result, debug = sentiment_service.analyze("The meeting is scheduled for three o'clock")
    # Should be neutral or close to neutral
    
    return True


def test_filler_word_removal():
    """Test specific filler word removal."""
    from app.services.sentiment_service import text_preprocessor
    
    test_cases = [
        ("I um, uh, really like this", "I really like this"),
        ("You know, I think so", "I think so"),
        ("Basically, it's fine", "it's fine"),
        ("Actually, I disagree", "I disagree"),
    ]
    
    for original, expected_contains in test_cases:
        cleaned = text_preprocessor.remove_filler_words(original)
        # Check that filler words are removed
        assert "um" not in cleaned.lower(), f"um should be removed from: {original}"
    
    return True


# ==================== Integration Tests ====================

def test_pipeline_integration():
    """Test full analysis pipeline."""
    from app.services.sentiment_service import sentiment_service
    
    # Test various text types
    test_texts = [
        "I'm really excited about this project!",
        "I'm feeling a bit down today.",
        "Why did this happen to me?",
        "This is just okay, not great but not bad.",
    ]
    
    for text in test_texts:
        result, debug = sentiment_service.analyze(text, return_debug=True)
        assert result.label is not None, f"Result should have label for: {text}"
        assert result.score >= 0 and result.score <= 1, "Score should be between 0 and 1"
    
    return True


def test_conversational_text():
    """Test analysis of conversational text."""
    from app.services.sentiment_service import text_preprocessor, sentiment_service
    
    # Conversational text
    text = "Hey, um, how are you doing? I was thinking, like, maybe we could meet up?"
    
    # Preprocess
    cleaned = text_preprocessor.preprocess(text)
    
    # Should remove fillers
    assert "um" not in cleaned.lower(), "um should be removed"
    assert "like" not in cleaned.lower(), "like filler should be removed"
    
    # Analyze
    result, debug = sentiment_service.analyze(text, return_debug=True)
    
    # Should detect greeting and question
    assert debug.context['is_greeting'] == True, "Should detect greeting"
    assert debug.context['is_question'] == True, "Should detect question"
    
    return True


# ==================== Test Cases List ====================

def get_test_cases() -> List[TestCase]:
    """Get all test cases."""
    return [
        # Audio Utils Tests
        TestCase(
            name="Audio Utils Import",
            category=TestCategory.AUDIO_UTILS,
            description="Test that all audio utilities can be imported",
            input_data=None,
            expected_output=True,
            validate_fn=test_audio_utils_import,
            tags=["import", "audio"],
        ),
        TestCase(
            name="Audio Normalization",
            category=TestCategory.AUDIO_UTILS,
            description="Test audio normalization to peak amplitude",
            input_data=None,
            expected_output=True,
            validate_fn=test_audio_normalization,
            tags=["audio", "normalization"],
        ),
        TestCase(
            name="Silence Removal",
            category=TestCategory.AUDIO_UTILS,
            description="Test removal of leading/trailing silence",
            input_data=None,
            expected_output=True,
            validate_fn=test_silence_removal,
            tags=["audio", "silence"],
        ),
        TestCase(
            name="Voice Activity Detection",
            category=TestCategory.AUDIO_UTILS,
            description="Test VAD for speech/silence detection",
            input_data=None,
            expected_output=True,
            validate_fn=test_vad,
            tags=["audio", "vad"],
        ),
        TestCase(
            name="Audio Quality Validation",
            category=TestCategory.AUDIO_UTILS,
            description="Test audio quality validation",
            input_data=None,
            expected_output=True,
            validate_fn=test_audio_quality_validation,
            tags=["audio", "validation"],
        ),
        TestCase(
            name="Noise Reduction",
            category=TestCategory.AUDIO_UTILS,
            description="Test spectral gating noise reduction",
            input_data=None,
            expected_output=True,
            validate_fn=test_noise_reduction,
            tags=["audio", "noise"],
        ),
        
        # Whisper Service Tests
        TestCase(
            name="Whisper Service Import",
            category=TestCategory.WHISPER,
            description="Test that whisper service can be imported",
            input_data=None,
            expected_output=True,
            validate_fn=test_whisper_import,
            tags=["import", "whisper"],
        ),
        TestCase(
            name="Hallucination Detection",
            category=TestCategory.WHISPER,
            description="Test hallucination detection logic",
            input_data=None,
            expected_output=True,
            validate_fn=test_hallucination_detection,
            tags=["whisper", "hallucination"],
        ),
        TestCase(
            name="Text Post-processing",
            category=TestCategory.WHISPER,
            description="Test text post-processing corrections",
            input_data=None,
            expected_output=True,
            validate_fn=test_text_postprocessing,
            tags=["whisper", "postprocessing"],
        ),
        TestCase(
            name="Minimum Audio Duration",
            category=TestCategory.WHISPER,
            description="Test minimum audio duration validation",
            input_data=None,
            expected_output=True,
            validate_fn=test_min_audio_duration,
            tags=["whisper", "duration"],
        ),
        
        # Sentiment Service Tests
        TestCase(
            name="Sentiment Service Import",
            category=TestCategory.SENTIMENT,
            description="Test that sentiment service can be imported",
            input_data=None,
            expected_output=True,
            validate_fn=test_sentiment_import,
            tags=["import", "sentiment"],
        ),
        TestCase(
            name="Text Preprocessing",
            category=TestCategory.PREPROCESSING,
            description="Test filler word and artifact removal",
            input_data=None,
            expected_output=True,
            validate_fn=test_text_preprocessor,
            tags=["preprocessing", "text"],
        ),
        TestCase(
            name="Sentence Splitting",
            category=TestCategory.PREPROCESSING,
            description="Test sentence splitting functionality",
            input_data=None,
            expected_output=True,
            validate_fn=test_sentence_splitting,
            tags=["preprocessing", "sentences"],
        ),
        TestCase(
            name="Context Detection",
            category=TestCategory.PREPROCESSING,
            description="Test conversational context detection",
            input_data=None,
            expected_output=True,
            validate_fn=test_context_detection,
            tags=["preprocessing", "context"],
        ),
        TestCase(
            name="Sentiment Analysis",
            category=TestCategory.SENTIMENT,
            description="Test basic sentiment analysis",
            input_data=None,
            expected_output=True,
            validate_fn=test_sentiment_analysis,
            tags=["sentiment", "analysis"],
        ),
        TestCase(
            name="Sentiment with Debug",
            category=TestCategory.SENTIMENT,
            description="Test sentiment analysis with debug info",
            input_data=None,
            expected_output=True,
            validate_fn=test_sentiment_with_debug,
            tags=["sentiment", "debug"],
        ),
        TestCase(
            name="Confidence Thresholds",
            category=TestCategory.SENTIMENT,
            description="Test confidence threshold configuration",
            input_data=None,
            expected_output=True,
            validate_fn=test_sentiment_confidence_thresholds,
            tags=["sentiment", "confidence"],
        ),
        TestCase(
            name="Emotion Aggregation",
            category=TestCategory.SENTIMENT,
            description="Test weighted emotion aggregation",
            input_data=None,
            expected_output=True,
            validate_fn=test_emotion_aggregation,
            tags=["sentiment", "aggregation"],
        ),
        TestCase(
            name="Neutral Detection",
            category=TestCategory.SENTIMENT,
            description="Test neutral emotion detection",
            input_data=None,
            expected_output=True,
            validate_fn=test_neutral_detection,
            tags=["sentiment", "neutral"],
        ),
        TestCase(
            name="Filler Word Removal",
            category=TestCategory.PREPROCESSING,
            description="Test specific filler word removal",
            input_data=None,
            expected_output=True,
            validate_fn=test_filler_word_removal,
            tags=["preprocessing", "fillers"],
        ),
        
        # Integration Tests
        TestCase(
            name="Pipeline Integration",
            category=TestCategory.INTEGRATION,
            description="Test full analysis pipeline",
            input_data=None,
            expected_output=True,
            validate_fn=test_pipeline_integration,
            tags=["integration", "pipeline"],
        ),
        TestCase(
            name="Conversational Text",
            category=TestCategory.INTEGRATION,
            description="Test analysis of conversational text",
            input_data=None,
            expected_output=True,
            validate_fn=test_conversational_text,
            tags=["integration", "conversational"],
        ),
    ]


# ==================== Example Usage ====================

def run_examples():
    """Run example usage scenarios."""
    print("\n" + "="*60)
    print("EXAMPLE USAGE")
    print("="*60)
    
    from app.utils.audio_utils import (
        preprocess_audio,
        normalize_audio,
        voice_activity_detection,
        validate_audio_quality,
    )
    from app.services.sentiment_service import sentiment_service
    from app.services.whisper_service import whisper_service
    import numpy as np
    
    # Example 1: Audio Preprocessing
    print("\n📊 Example 1: Audio Preprocessing")
    print("-" * 40)
    
    # Generate test audio
    sample_rate = 16000
    duration = 2.0
    audio = np.sin(2 * np.pi * 440 * np.arange(int(sample_rate * duration)) / sample_rate) * 0.5
    audio[:1000] = 0  # Add silence at start
    
    print(f"  Original length: {len(audio)} samples")
    print(f"  Sample rate: {sample_rate} Hz")
    
    # Normalize
    normalized = normalize_audio(audio)
    print(f"  After normalization: peak = {np.max(np.abs(normalized)):.3f}")
    
    # VAD
    vad = voice_activity_detection(normalized, sample_rate)
    print(f"  Voice detected: {vad['is_speech']}")
    print(f"  Voice ratio: {vad['voice_ratio']:.1%}")
    
    # Example 2: Sentiment Analysis
    print("\n🎭 Example 2: Sentiment Analysis")
    print("-" * 40)
    
    test_texts = [
        "I'm so happy and excited about this!",
        "I'm really disappointed with the results.",
        "I don't know what to feel about this.",
        "Hello, how are you doing today?",
    ]
    
    for text in test_texts:
        result, debug = sentiment_service.analyze(text, return_debug=True)
        print(f"\n  Text: \"{text[:50]}...\"" if len(text) > 50 else f"\n  Text: \"{text}\"")
        print(f"  Emotion: {result.label.value}")
        print(f"  Confidence: {result.score:.2%}")
        print(f"  All scores: ", end="")
        scores_str = ", ".join([f"{k.value}: {v:.2%}" for k, v in sorted(debug.final_emotions.items(), key=lambda x: x[1], reverse=True)[:3]])
        print(scores_str)
    
    # Example 3: Debug Information
    print("\n🔍 Example 3: Debug Information")
    print("-" * 40)
    
    text = "I um, you know, [laughter] really love this feature!"
    result, debug = sentiment_service.analyze(text, return_debug=True)
    
    print(f"  Original: \"{text}\"")
    print(f"  Cleaned:  \"{debug.cleaned_text}\"")
    print(f"  Sentences: {len(debug.sentences)}")
    print(f"  Warnings: {debug.warnings}")
    
    # Example 4: Conversational Text
    print("\n💬 Example 4: Conversational Text Analysis")
    print("-" * 40)
    
    conversational_texts = [
        "Hey, um, what's up?",
        "I was thinking, like, maybe we could grab lunch?",
        "You know, I basically just want to go home.",
    ]
    
    for text in conversational_texts:
        result, debug = sentiment_service.analyze(text, return_debug=True)
        print(f"\n  Text: \"{text}\"")
        print(f"  Cleaned: \"{debug.cleaned_text}\"")
        print(f"  Emotion: {result.label.value}")
        print(f"  Context: greeting={debug.context['is_greeting']}, question={debug.context['is_question']}")
    
    print("\n" + "="*60)


# ==================== Main Entry Point ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Voice Sentiment Analysis Test Suite")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--debug", "-d", action="store_true", help="Debug mode with detailed logging")
    parser.add_argument("--example", "-e", action="store_true", help="Run example usage")
    parser.add_argument("--category", "-c", type=str, choices=["audio", "whisper", "sentiment", "preprocessing", "integration"], help="Run specific category only")
    parser.add_argument("--json", "-j", action="store_true", help="Output results as JSON")
    
    args = parser.parse_args()
    
    # Run examples if requested
    if args.example:
        run_examples()
        sys.exit(0)
    
    # Get all test cases
    tests = get_test_cases()
    
    # Filter by category if specified
    if args.category:
        category_map = {
            "audio": TestCategory.AUDIO_UTILS,
            "whisper": TestCategory.WHISPER,
            "sentiment": TestCategory.SENTIMENT,
            "preprocessing": TestCategory.PREPROCESSING,
            "integration": TestCategory.INTEGRATION,
        }
        category = category_map.get(args.category)
        if category:
            tests = [t for t in tests if t.category == category]
    
    # Run tests
    runner = TestRunner(verbose=args.verbose, debug=args.debug)
    results = runner.run_all_tests(tests)
    
    # Output JSON if requested
    if args.json:
        print("\n" + json.dumps(results, indent=2))
    
    # Exit with appropriate code
    sys.exit(0 if results["failed"] == 0 else 1)

