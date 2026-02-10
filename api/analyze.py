"""
Vercel API Route - Audio Analysis
Note: For production, ML models should run on a dedicated backend service
(e.g., Railway, Render, or AWS) due to Vercel's execution time limits.
"""

import json
import base64
import io
import os

def validate_audio(data, max_size_mb=10):
    """Validate base64 audio data."""
    if not data:
        return False, "No audio data provided"
    
    # Estimate size
    size_bytes = len(data) * 3/4  # base64 encoding overhead
    size_mb = size_bytes / (1024 * 1024)
    
    if size_mb > max_size_mb:
        return False, f"Audio too large: {size_mb:.1f}MB > {max_size_mb}MB limit"
    
    return True, None

def analyze_text_sentiment(text):
    """
    Simple sentiment analysis using keyword matching.
    For production, use a proper ML model via API.
    """
    # Simple keyword-based sentiment (fallback for demo)
    positive_words = ['happy', 'joy', 'love', 'great', 'good', 'excited', 'wonderful', 'amazing', 'beautiful', 'fantastic']
    negative_words = ['sad', 'angry', 'fear', 'terrible', 'bad', 'hate', 'awful', 'horrible', 'upset', 'depressed']
    
    text_lower = text.lower()
    
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    total = positive_count + negative_count
    
    if total == 0:
        return {
            "label": "neutral",
            "score": 0.5,
            "all_emotions": {
                "joy": 0.2,
                "sadness": 0.2,
                "anger": 0.2,
                "fear": 0.2,
                "love": 0.1,
                "surprise": 0.1
            }
        }
    
    joy_score = positive_count / total if positive_count > 0 else 0
    sadness_score = negative_count / total if negative_count > 0 else 0
    
    emotions = {
        "joy": round(joy_score * 0.9, 2),
        "sadness": round(sadness_score * 0.3, 2),
        "anger": round(sadness_score * 0.2, 2),
        "fear": round(sadness_score * 0.2, 2),
        "love": round(joy_score * 0.1, 2),
        "surprise": 0.1
    }
    
    # Normalize to sum to 1
    total_score = sum(emotions.values())
    emotions = {k: round(v / total_score, 2) for k, v in emotions.items()}
    
    dominant = max(emotions, key=emotions.get)
    
    return {
        "label": dominant,
        "score": emotions[dominant],
        "all_emotions": emotions
    }

def main(request):
    """Handle audio analysis requests."""
    # CORS headers
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
        "Content-Type": "application/json"
    }
    
    # Handle preflight
    if request.method == "OPTIONS":
        return json.Response(status_code=204, headers=headers)
    
    if request.method != "POST":
        return json.Response(
            json.dumps({"error": "Method not allowed"}),
            status_code=405,
            headers=headers
        )
    
    try:
        body = json.loads(request.body)
        text = body.get("text", "").strip()
        
        if not text:
            return json.Response(
                json.dumps({"error": "No text provided"}),
                status_code=400,
                headers=headers
            )
        
        # Analyze sentiment
        sentiment = analyze_text_sentiment(text)
        
        response = {
            "success": True,
            "text": text,
            "sentiment": sentiment,
            "processing_time_seconds": 0.01,
            "note": "Demo mode - for full ML analysis, use the Docker backend"
        }
        
        return json.Response(
            json.dumps(response),
            status_code=200,
            headers=headers
        )
        
    except Exception as e:
        return json.Response(
            json.dumps({"error": str(e)}),
            status_code=500,
            headers=headers
        )

def handler(request, context):
    return main(request)
