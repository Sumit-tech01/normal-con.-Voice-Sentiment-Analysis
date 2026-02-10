#!/bin/bash

# 🎤 Voice Sentiment Analysis - Full Stack Start Script
# Starts both frontend and backend servers

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
FRONTEND_DIR="$SCRIPT_DIR/frontend"
BACKEND_PORT=5000
FRONTEND_PORT=8000

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo ""
echo "=========================================="
echo "🎤 Voice Sentiment Analysis"
echo "=========================================="
echo ""

cleanup() {
    echo ""
    echo "Stopping servers..."
    [ -n "$BACKEND_PID" ] && kill $BACKEND_PID 2>/dev/null
    [ -n "$FRONTEND_PID" ] && kill $FRONTEND_PID 2>/dev/null
    echo "Servers stopped."
    exit 0
}

trap cleanup INT

# Check and install dependencies
echo -e "${YELLOW}Checking dependencies...${NC}"
if ! python3 -c "import flask" 2>/dev/null; then
    echo -e "${YELLOW}Installing Python dependencies...${NC}"
    pip install -q -r "$BACKEND_DIR/requirements.txt"
    echo -e "${GREEN}✓ Dependencies installed${NC}"
else
    echo -e "${GREEN}✓ Dependencies already installed${NC}"
fi

# Start Backend
echo ""
echo -e "${YELLOW}Starting backend server (Port $BACKEND_PORT)...${NC}"
# Run from parent directory with correct module path
cd "$SCRIPT_DIR"
/Users/apple/Library/Python/3.9/bin/uvicorn backend.app.main:app --host 0.0.0.0 --port $BACKEND_PORT &
BACKEND_PID=$!
echo -e "${GREEN}✓ Backend running at http://localhost:$BACKEND_PORT${NC}"

# Start Frontend (serves only frontend directory)
echo ""
echo -e "${YELLOW}Starting frontend server (Port $FRONTEND_PORT)...${NC}"
cd "$FRONTEND_DIR"
python3 -m http.server $FRONTEND_PORT &
FRONTEND_PID=$!
echo -e "${GREEN}✓ Frontend running at http://localhost:$FRONTEND_PORT${NC}"

echo ""
echo "=========================================="
echo -e "${GREEN}🎉 Application is running!${NC}"
echo "=========================================="
echo ""
echo -e "📍 Frontend: ${GREEN}http://localhost:$FRONTEND_PORT${NC}"
echo -e "📍 Backend API: ${GREEN}http://localhost:$BACKEND_PORT${NC}"
echo ""
echo -e "${YELLOW}API Endpoints:${NC}"
echo "  • GET  /api/v1/health      - Health check"
echo "  • POST /api/v1/analyze/upload - Upload & analyze audio"
echo "  • POST /api/v1/analyze/text   - Analyze text sentiment"
echo ""
echo "✅ Only frontend UI is visible to users"
echo "✅ All project files are hidden"
echo ""
echo "Press Ctrl+C to stop all servers"
echo "=========================================="

wait
