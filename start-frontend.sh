#!/bin/bash

# 🎤 Voice Sentiment Analysis - Start Script
# This script serves only the frontend directory, hiding all project files

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="$SCRIPT_DIR/frontend"
PORT=${1:-8000}

echo ""
echo "=========================================="
echo "🎤 Voice Sentiment Analysis"
echo "=========================================="
echo ""
echo "Starting frontend server..."
echo ""
echo "📍 URL: http://localhost:$PORT"
echo ""
echo "✅ Only the application UI is visible to users"
echo "❌ Project files (backend, README, etc.) are hidden"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=========================================="
echo ""

# Change to frontend directory and start server
cd "$FRONTEND_DIR"
python3 -m http.server "$PORT"

