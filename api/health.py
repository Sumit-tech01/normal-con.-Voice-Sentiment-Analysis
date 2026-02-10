"""
Vercel API Route - Health Check
"""
import json

def main(request):
    """Health check endpoint."""
    return json.Response({
        "status": "healthy",
        "version": "1.0.0",
        "service": "voice-sentiment-analysis"
    })

# Vercel serverless function handler
def handler(request, context):
    return main(request)
