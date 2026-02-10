"""
Enhanced Sentiment Analysis Service using Transformers.
Provides emotion detection with text preprocessing, sentence-level analysis, 
and improved accuracy for conversational text.

Features:
- Text preprocessing (filler word removal, artifact cleaning)
- Sentence-level analysis with weighted aggregation
- Confidence thresholds with neutral detection
- Context awareness for conversational text
- Debug mode for troubleshooting
"""

import logging
import re
import threading
import time
from pathlib import Path
from typing import Dict, Optional, List, Any, Tuple
from dataclasses import dataclass
from enum import Enum

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)

from ..config import config
from ..models.schemas import SentimentLabel, SentimentResult

logger = logging.getLogger(__name__)


# Confidence thresholds
CONFIDENCE_THRESHOLD = 0.70  # 70% threshold for confident predictions
NEUTRAL_THRESHOLD = 0.25  # Threshold for neutral classification
LOW_CONFIDENCE_THRESHOLD = 0.40  # Below this is low confidence

# Text preprocessing patterns
FILLER_WORDS = {
    'um', 'uh', 'like', 'you know', 'basically', 'actually', 
    'literally', 'sort of', 'kind of', 'i mean', 'right', 
    'okay', 'so yeah', 'you see', 'i guess', 'i suppose',
    'oh', 'well', 'hmm', 'er', 'ah', 'erm'
}

# Sentence ending patterns
SENTENCE_ENDINGS = r'[.!?]+'
SENTENCE_MIN_LENGTH = 3  # Minimum words for a valid sentence

# Conversational indicators
CONVERSATIONAL_INDICATORS = [
    'hey', 'hi', 'hello', 'good morning', 'good afternoon',
    'thanks', 'thank you', 'please', 'sorry', 'excuse me',
    'goodbye', 'bye', 'see you', 'talk to you later'
]

# Emotion weights for different contexts
EMOTION_WEIGHTS = {
    'joy': 1.0,
    'sadness': 1.0,
    'anger': 1.0,
    'fear': 1.0,
    'love': 1.0,
    'surprise': 1.0,
    'neutral': 0.8,
}


@dataclass
class SentenceResult:
    """Result for a single sentence."""
    text: str
    sentiment: SentimentLabel
    score: float
    all_scores: Dict[SentimentLabel, float]
    word_count: int


@dataclass
class SentimentDebugInfo:
    """Debug information for sentiment analysis."""
    original_text: str
    cleaned_text: str
    sentences: List[Dict[str, Any]]
    confidence_threshold: float
    low_confidence_sentences: List[int]
    aggregation_method: str
    final_emotions: Dict[str, float]
    warnings: List[str]


class TextPreprocessor:
    """Text preprocessing utilities for sentiment analysis."""
    
    # Common transcription artifacts
    ARTIFACTS = [
        r'\[[^\]]*\]',  # Bracketed text
        r'\([^)]*\)',  # Parenthesized text
        r'\{[^{}]*\}',  # Braced text
        r'<[^>]*>',  # HTML tags
        r'\*+\s*\*+',  # Multiple asterisks
        r'\s+',  # Multiple spaces
    ]
    
    # Emotional intensifiers
    INTENSIFIERS = {
        'very': 1.3,
        'really': 1.25,
        'extremely': 1.5,
        'super': 1.4,
        'totally': 1.2,
        'absolutely': 1.4,
        'quite': 1.1,
        'pretty': 1.1,
        'so ': 1.2,
        'incredibly': 1.5,
    }
    
    # Negation words that flip sentiment
    NEGATIONS = {
        'not', "n't", 'never', 'no ', 'none', "n't", 'neither',
        'nobody', 'nothing', 'nowhere', 'hardly', 'barely', 'scarcely'
    }
    
    def __init__(self):
        """Initialize preprocessor."""
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for efficiency."""
        self._filler_pattern = self._create_filler_pattern()
        self._artifact_patterns = [re.compile(p) for p in self.ARTIFACTS]
        self._sentence_pattern = re.compile(SENTENCE_ENDINGS)
    
    def _create_filler_pattern(self) -> re.Pattern:
        """Create regex pattern for filler words."""
        escaped_words = [re.escape(word) for word in FILLER_WORDS]
        pattern = r'\b(' + '|'.join(escaped_words) + r')\b'
        return re.compile(pattern, re.IGNORECASE)
    
    def remove_filler_words(self, text: str) -> str:
        """
        Remove filler words from text.
        
        Args:
            text: Input text
            
        Returns:
            Text with filler words removed
        """
        # Remove common fillers
        text = self._filler_pattern.sub('', text)
        
        # Clean up multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing spaces
        text = text.strip()
        
        return text
    
    def clean_artifacts(self, text: str) -> str:
        """
        Remove transcription artifacts from text.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        for pattern in self._artifact_patterns:
            text = pattern.sub('', text)
        
        # Clean up special characters
        text = re.sub(r'[^\w\s\.\!\?\,\'\"]', '', text)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def preprocess(self, text: str) -> str:
        """
        Full preprocessing pipeline.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        # Remove artifacts first
        text = self.clean_artifacts(text)
        
        # Remove filler words
        text = self.remove_filler_words(text)
        
        return text
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Split by sentence endings
        sentences = self._sentence_pattern.split(text)
        
        # Filter out empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Filter by minimum length
        sentences = [s for s in sentences if len(s.split()) >= SENTENCE_MIN_LENGTH]
        
        return sentences
    
    def detect_context(self, text: str) -> Dict[str, bool]:
        """
        Detect conversational context.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with context flags
        """
        text_lower = text.lower()
        
        return {
            'is_greeting': any(g in text_lower for g in ['hello', 'hi', 'hey', 'good morning']),
            'is_farewell': any(g in text_lower for g in ['bye', 'goodbye', 'see you', 'later']),
            'is_question': '?' in text,
            'has_intensifiers': any(i in text_lower for i in self.INTENSIFIERS.keys()),
            'has_negation': any(n in text_lower for n in self.NEGATIONS),
            'is_casual': any(c in text_lower for c in ['yeah', 'okay', 'cool', 'nice']),
        }


# Global preprocessor instance
text_preprocessor = TextPreprocessor()


class EnhancedSentimentService:
    """
    Enhanced service for sentiment/emotion analysis using Transformers.
    
    Features:
    - j-hartmann/emotion-english-distilroberta-base model
    - Text preprocessing
    - Sentence-level analysis
    - Weighted emotion aggregation
    - Confidence thresholds
    - Neutral emotion detection
    """

    _instance: Optional["EnhancedSentimentService"] = None
    _lock: threading.Lock = threading.Lock()

    # Mapping from model labels to our enum
    EMOTION_MAPPING = {
        "joy": SentimentLabel.JOY,
        "sadness": SentimentLabel.SADNESS,
        "anger": SentimentLabel.ANGER,
        "fear": SentimentLabel.FEAR,
        "love": SentimentLabel.LOVE,
        "surprise": SentimentLabel.SURPRISE,
        "neutral": SentimentLabel.NEUTRAL,
    }

    # Reverse mapping for aggregation
    LABEL_TO_STRING = {
        SentimentLabel.JOY: "joy",
        SentimentLabel.SADNESS: "sadness",
        SentimentLabel.ANGER: "anger",
        SentimentLabel.FEAR: "fear",
        SentimentLabel.LOVE: "love",
        SentimentLabel.SURPRISE: "surprise",
        SentimentLabel.NEUTRAL: "neutral",
    }

    def __new__(cls) -> "EnhancedSentimentService":
        """Singleton pattern to avoid reloading model."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the sentiment service."""
        if self._initialized:
            return

        self.model_name: str = config.SENTIMENT_MODEL_NAME
        self.device: str = config.SENTIMENT_DEVICE
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForSequenceClassification] = None
        self.classifier: Optional[Any] = None
        self._load_time: Optional[float] = None
        self._model_loaded: bool = False
        self._labels: List[str] = []
        
        # Confidence settings
        self.confidence_threshold: float = CONFIDENCE_THRESHOLD
        self.neutral_threshold: float = NEUTRAL_THRESHOLD
        self.low_confidence_threshold: float = LOW_CONFIDENCE_THRESHOLD
        
        # Debug mode
        self.debug_mode: bool = False
        
        # Preprocessor
        self.preprocessor: TextPreprocessor = text_preprocessor

        self._initialized = True
        logger.info(f"EnhancedSentimentService initialized with model: {self.model_name}")

    def load_model(self, force_reload: bool = False) -> None:
        """
        Load the sentiment analysis model.
        
        Args:
            force_reload: If True, reload even if already loaded
        """
        if self._model_loaded and not force_reload:
            logger.info("Sentiment model already loaded, skipping reload")
            return

        start_time = time.time()

        try:
            # Determine device
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            logger.info(
                f"Loading sentiment model '{self.model_name}' on {self.device}"
            )

            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            )
            self.model.to(self.device)
            self.model.eval()

            # Get labels from model config
            self._labels = list(self.model.config.id2label.values())

            # Create pipeline for easier inference
            self.classifier = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device if self.device != "cpu" else -1,
                top_k=None,
                truncation=True,
                max_length=512,
            )

            self._load_time = time.time() - start_time
            self._model_loaded = True

            logger.info(
                f"Sentiment model loaded successfully in {self._load_time:.2f}s"
            )

        except Exception as e:
            logger.error(f"Failed to load sentiment model: {e}")
            raise RuntimeError(f"Sentiment model loading failed: {e}")

    def _analyze_single_sentence(
        self,
        sentence: str,
    ) -> Tuple[SentimentLabel, float, Dict[SentimentLabel, float]]:
        """
        Analyze sentiment of a single sentence.
        
        Args:
            sentence: Input sentence
            
        Returns:
            Tuple of (dominant_label, dominant_score, all_scores)
        """
        try:
            # Run classification
            results = self.classifier(sentence)

            # Handle different return formats
            if isinstance(results, list) and len(results) > 0:
                if isinstance(results[0], list):
                    results = results[0]

                # Extract scores
                scores = {
                    r["label"]: r["score"]
                    for r in sorted(results, key=lambda x: x["score"], reverse=True)
                }
            else:
                scores = {}

            # Find dominant emotion
            if scores:
                dominant_label = max(scores, key=scores.get)
                dominant_score = scores[dominant_label]
            else:
                dominant_label = "neutral"
                dominant_score = 1.0

            # Map to our enum
            mapped_label = self.EMOTION_MAPPING.get(
                dominant_label, SentimentLabel.NEUTRAL
            )

            # Map all scores
            all_scores = {}
            for label, score in scores.items():
                mapped = self.EMOTION_MAPPING.get(label, SentimentLabel.NEUTRAL)
                all_scores[mapped] = score

            # Ensure all labels present
            for label in SentimentLabel:
                if label not in all_scores:
                    all_scores[label] = 0.0

            return mapped_label, dominant_score, all_scores

        except Exception as e:
            logger.error(f"Single sentence analysis failed: {e}")
            return SentimentLabel.NEUTRAL, 1.0, {label: 0.0 for label in SentimentLabel}

    def _aggregate_emotions(
        self,
        sentence_results: List[SentenceResult],
        method: str = "weighted",
    ) -> Tuple[SentimentLabel, float, Dict[SentimentLabel, float]]:
        """
        Aggregate sentence-level emotions into final result.
        
        Args:
            sentence_results: List of sentence results
            method: Aggregation method ('weighted', 'average', 'majority')
            
        Returns:
            Tuple of (dominant_label, dominant_score, all_scores)
        """
        if not sentence_results:
            return SentimentLabel.NEUTRAL, 1.0, {label: 0.0 for label in SentimentLabel}

        if len(sentence_results) == 1:
            sr = sentence_results[0]
            return sr.sentiment, sr.score, sr.all_scores

        # Calculate weights based on sentence length
        total_words = sum(sr.word_count for sr in sentence_results)
        weights = [sr.word_count / total_words for sr in sentence_results] if total_words > 0 else [1/len(sentence_results)] * len(sentence_results)

        if method == "weighted":
            # Weighted average of scores
            aggregated_scores: Dict[SentimentLabel, float] = {}
            for i, sr in enumerate(sentence_results):
                weight = weights[i]
                for label, score in sr.all_scores.items():
                    if label not in aggregated_scores:
                        aggregated_scores[label] = 0.0
                    aggregated_scores[label] += score * weight

        elif method == "average":
            # Simple average
            aggregated_scores = {}
            num_sentences = len(sentence_results)
            for sr in sentence_results:
                for label, score in sr.all_scores.items():
                    if label not in aggregated_scores:
                        aggregated_scores[label] = 0.0
                    aggregated_scores[label] += score / num_sentences

        elif method == "majority":
            # Vote by dominant emotion
            votes = {label: 0 for label in SentimentLabel}
            for i, sr in enumerate(sentence_results):
                votes[sr.sentiment] += weights[i]

            # Return highest voted
            dominant_label = max(votes, key=votes.get)
            dominant_score = votes[dominant_label] / sum(weights)
            aggregated_scores = {label: 0.0 for label in SentimentLabel}

            return dominant_label, dominant_score, aggregated_scores

        else:
            # Default to weighted
            return self._aggregate_emotions(sentence_results, "weighted")

        # Normalize scores
        total_score = sum(aggregated_scores.values())
        if total_score > 0:
            for label in aggregated_scores:
                aggregated_scores[label] /= total_score

        # Find dominant
        if aggregated_scores:
            dominant_label = max(aggregated_scores, key=aggregated_scores.get)
            dominant_score = aggregated_scores[dominant_label]
        else:
            dominant_label = SentimentLabel.NEUTRAL
            dominant_score = 0.0

        return dominant_label, dominant_score, aggregated_scores

    def _apply_confidence_threshold(
        self,
        label: SentimentLabel,
        score: float,
        all_scores: Dict[SentimentLabel, float],
    ) -> Tuple[SentimentLabel, float, List[str]]:
        """
        Apply confidence threshold and determine if neutral.
        
        Args:
            label: Dominant label
            score: Confidence score
            all_scores: All emotion scores
            
        Returns:
            Tuple of (final_label, final_score, warnings)
        """
        warnings = []
        
        if score < self.low_confidence_threshold:
            warnings.append(f"Low confidence score: {score:.2f}")
            # Increase neutral tendency for low confidence
            all_scores[SentimentLabel.NEUTRAL] = max(
                all_scores.get(SentimentLabel.NEUTRAL, 0),
                (self.low_confidence_threshold - score) * 0.5
            )
        
        # Check if neutral should override
        neutral_score = all_scores.get(SentimentLabel.NEUTRAL, 0)
        if neutral_score > self.neutral_threshold:
            if score < self.confidence_threshold:
                warnings.append(f"Neutral overriding due to high neutral score: {neutral_score:.2f}")
                return SentimentLabel.NEUTRAL, max(neutral_score, score), warnings
        
        # If below confidence threshold, suggest neutral
        if score < self.confidence_threshold:
            warnings.append(f"Score below threshold: {score:.2f} < {self.confidence_threshold}")
            # Don't override, but note it
        
        return label, score, warnings

    def _detect_conversational_context(
        self,
        sentences: List[str],
    ) -> Dict[str, Any]:
        """
        Detect conversational context and adjust processing.
        
        Args:
            sentences: List of sentences
            
        Returns:
            Context information
        """
        full_text = ' '.join(sentences)
        context = self.preprocessor.detect_context(full_text)
        
        # Count questions
        question_count = sum(1 for s in sentences if '?' in s)
        
        # Detect exclamation usage
        exclamation_count = sum(1 for s in sentences if '!' in s)
        
        return {
            **context,
            'question_count': question_count,
            'exclamation_count': exclamation_count,
            'num_sentences': len(sentences),
            'avg_sentence_length': sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0,
        }

    def analyze(
        self,
        text: str,
        return_debug: bool = False,
    ) -> Tuple[SentimentResult, Optional[SentimentDebugInfo]]:
        """
        Analyze sentiment/emotion in text with full preprocessing.
        
        Args:
            text: Input text to analyze
            return_debug: If True, return debug information
            
        Returns:
            Tuple of (SentimentResult, DebugInfo)
        """
        if not self._model_loaded:
            self.load_model()

        if not text or not text.strip():
            result = SentimentResult(
                label=SentimentLabel.NEUTRAL,
                score=1.0,
            )
            debug_info = SentimentDebugInfo(
                original_text=text,
                cleaned_text="",
                sentences=[],
                confidence_threshold=self.confidence_threshold,
                low_confidence_sentences=[],
                aggregation_method="none",
                final_emotions={},
                warnings=["Empty text provided"],
            ) if return_debug else None
            return result, debug_info

        debug_info = SentimentDebugInfo(
            original_text=text,
            cleaned_text="",
            sentences=[],
            confidence_threshold=self.confidence_threshold,
            low_confidence_sentences=[],
            aggregation_method="weighted",
            final_emotions={},
            warnings=[],
        ) if return_debug else None

        # Step 1: Preprocess text
        cleaned_text = self.preprocessor.preprocess(text)
        if debug_info:
            debug_info.cleaned_text = cleaned_text

        if not cleaned_text.strip():
            result = SentimentResult(
                label=SentimentLabel.NEUTRAL,
                score=1.0,
            )
            if debug_info:
                debug_info.warnings.append("Text became empty after preprocessing")
            return result, debug_info

        # Step 2: Split into sentences
        sentences = self.preprocessor.split_into_sentences(cleaned_text)
        if not sentences:
            # Use whole text as single sentence
            sentences = [cleaned_text]

        # Step 3: Detect context
        context = self._detect_conversational_context(sentences)
        
        # Step 4: Analyze each sentence
        sentence_results: List[SentenceResult] = []
        low_confidence_indices = []

        for i, sentence in enumerate(sentences):
            if len(sentence.split()) < 2:
                continue

            label, score, all_scores = self._analyze_single_sentence(sentence)

            sr = SentenceResult(
                text=sentence,
                sentiment=label,
                score=score,
                all_scores=all_scores,
                word_count=len(sentence.split()),
            )
            sentence_results.append(sr)

            if score < self.low_confidence_threshold:
                low_confidence_indices.append(i)

        # Step 5: Aggregate results
        dominant_label, dominant_score, aggregated_scores = self._aggregate_emotions(
            sentence_results,
            method="weighted",
        )

        # Step 6: Apply confidence thresholds
        final_label, final_score, warnings = self._apply_confidence_threshold(
            dominant_label,
            dominant_score,
            aggregated_scores,
        )

        # Step 7: Context-aware adjustments
        if context['is_question'] and final_label in [SentimentLabel.JOY, SentimentLabel.SURPRISE]:
            # Questions with positive emotion might be inquiries
            warnings.append("Question detected - context may affect interpretation")
        
        if context['exclamation_count'] > 0:
            # Increase weight for surprise/joy in excited speech
            if final_label == SentimentLabel.SURPRISE:
                final_score = min(final_score * 1.1, 1.0)

        # Build result
        result = SentimentResult(
            label=final_label,
            score=final_score,
        )

        # Build debug info
        if debug_info:
            debug_info.sentences = [
                {
                    "index": i,
                    "text": sr.text,
                    "sentiment": sr.sentiment.value,
                    "score": sr.score,
                    "word_count": sr.word_count,
                    "low_confidence": i in low_confidence_indices,
                }
                for i, sr in enumerate(sentence_results)
            ]
            debug_info.low_confidence_sentences = low_confidence_indices
            debug_info.final_emotions = {
                self.LABEL_TO_STRING[k]: v 
                for k, v in aggregated_scores.items()
            }
            debug_info.warnings = warnings
            debug_info.context = context

        return result, debug_info

    def analyze_with_all_scores(
        self,
        text: str,
    ) -> Dict[SentimentLabel, float]:
        """
        Analyze text and return scores for all emotions.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary mapping emotion labels to confidence scores
        """
        result, _ = self.analyze(text)
        
        # Get all scores by analyzing again
        if not self._model_loaded:
            self.load_model()

        if not text or not text.strip():
            return {label: 0.0 for label in SentimentLabel}

        try:
            results = self.classifier(text)

            # Handle multiple inputs
            if isinstance(results, list):
                results = results[0]

            # Extract all scores
            all_scores: Dict[SentimentLabel, float] = {}
            for result_item in results:
                label = result_item.get("label", "unknown")
                score = result_item.get("score", 0.0)
                mapped = self.EMOTION_MAPPING.get(label, SentimentLabel.NEUTRAL)
                all_scores[mapped] = score

            # Ensure all labels present
            for label in SentimentLabel:
                if label not in all_scores:
                    all_scores[label] = 0.0

            return all_scores

        except Exception as e:
            logger.error(f"Full sentiment analysis failed: {e}")
            return {label: 0.0 for label in SentimentLabel}

    def analyze_complete(
        self,
        text: str,
    ) -> Tuple[SentimentResult, Dict[SentimentLabel, float]]:
        """
        Complete sentiment analysis with all scores.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Tuple of (dominant_result, all_scores_dict)
        """
        dominant, debug_info = self.analyze(text)
        all_scores = self.analyze_with_all_scores(text)
        return dominant, all_scores

    def set_confidence_thresholds(
        self,
        confidence: float = None,
        neutral: float = None,
        low_confidence: float = None,
    ) -> None:
        """
        Set confidence thresholds.
        
        Args:
            confidence: Main confidence threshold (0-1)
            neutral: Neutral classification threshold (0-1)
            low_confidence: Low confidence threshold (0-1)
        """
        if confidence is not None:
            self.confidence_threshold = max(0.0, min(1.0, confidence))
        if neutral is not None:
            self.neutral_threshold = max(0.0, min(1.0, neutral))
        if low_confidence is not None:
            self.low_confidence_threshold = max(0.0, min(1.0, low_confidence))

        logger.info(
            f"Confidence thresholds set: "
            f"confidence={self.confidence_threshold}, "
            f"neutral={self.neutral_threshold}, "
            f"low_confidence={self.low_confidence_threshold}"
        )

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self._model_loaded:
            return {"status": "not_loaded"}

        info = {
            "model_name": self.model_name,
            "device": self.device,
            "is_loaded": self._model_loaded,
            "load_time_seconds": self._load_time,
            "labels": self._labels,
            "confidence_threshold": self.confidence_threshold,
            "neutral_threshold": self.neutral_threshold,
        }

        # Add memory info
        if torch.cuda.is_available():
            info["gpu_memory_mb"] = torch.cuda.get_device_properties(
                0
            ).total_memory / (1024**2)

        return info

    def unload_model(self) -> None:
        """Unload the model to free memory."""
        if self.model is not None:
            del self.model
            del self.tokenizer
            del self.classifier
            self.model = None
            self.tokenizer = None
            self.classifier = None
            self._model_loaded = False

            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("Sentiment model unloaded")


# Keep old service name for compatibility
SentimentService = EnhancedSentimentService

# Global instance
sentiment_service = EnhancedSentimentService()

