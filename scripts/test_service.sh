#!/bin/bash
# Testing script for Xynenyx LLM Service
# Run this script to test all Phase 2 criteria

set -e

echo "🧪 Xynenyx LLM Service - Phase 2 Testing"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if poetry is available
if ! command -v poetry &> /dev/null; then
    echo -e "${RED}❌ Poetry not found. Please install Poetry first.${NC}"
    exit 1
fi

# Check if .env file exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}⚠️  .env file not found. Please create it from .env.example${NC}"
    exit 1
fi

echo "📦 Installing dependencies..."
poetry install

echo ""
echo "🔍 Running unit tests..."
poetry run pytest -v --cov=app --cov-report=term-missing --cov-report=html

echo ""
echo "✅ Unit tests completed"
echo ""
echo "🚀 Starting service for integration tests..."
echo "   (Service will start in background on port 8003)"
echo ""

# Start service in background
poetry run uvicorn app.main:app --port 8003 &
SERVICE_PID=$!

# Wait for service to start
sleep 3

# Check if service is running
if ! ps -p $SERVICE_PID > /dev/null; then
    echo -e "${RED}❌ Service failed to start${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Service started (PID: $SERVICE_PID)${NC}"
echo ""

# Test health endpoint
echo "1️⃣  Testing health endpoint..."
HEALTH_RESPONSE=$(curl -s http://localhost:8003/health)
if echo "$HEALTH_RESPONSE" | grep -q "healthy"; then
    echo -e "${GREEN}✅ Health check passed${NC}"
    echo "   Response: $HEALTH_RESPONSE"
else
    echo -e "${RED}❌ Health check failed${NC}"
    echo "   Response: $HEALTH_RESPONSE"
fi
echo ""

# Test readiness endpoint
echo "2️⃣  Testing readiness endpoint..."
READY_RESPONSE=$(curl -s http://localhost:8003/ready)
if echo "$READY_RESPONSE" | grep -q "ready"; then
    echo -e "${GREEN}✅ Readiness check passed${NC}"
    echo "   Response: $READY_RESPONSE"
else
    echo -e "${RED}❌ Readiness check failed${NC}"
    echo "   Response: $READY_RESPONSE"
fi
echo ""

# Test providers endpoint
echo "3️⃣  Testing providers endpoint..."
PROVIDERS_RESPONSE=$(curl -s http://localhost:8003/providers)
if echo "$PROVIDERS_RESPONSE" | grep -q "providers"; then
    echo -e "${GREEN}✅ Providers endpoint passed${NC}"
    echo "   Response: $PROVIDERS_RESPONSE"
else
    echo -e "${RED}❌ Providers endpoint failed${NC}"
    echo "   Response: $PROVIDERS_RESPONSE"
fi
echo ""

# Test completion endpoint (requires X-User-ID header)
echo "4️⃣  Testing completion endpoint..."
COMPLETION_RESPONSE=$(curl -s -X POST http://localhost:8003/complete \
    -H "Content-Type: application/json" \
    -H "X-User-ID: test-user-123" \
    -d '{
        "messages": [{"role": "user", "content": "Hello"}],
        "provider": "openai"
    }')
if echo "$COMPLETION_RESPONSE" | grep -q "content"; then
    echo -e "${GREEN}✅ Completion endpoint passed${NC}"
    echo "   Response preview: $(echo $COMPLETION_RESPONSE | head -c 200)..."
else
    echo -e "${RED}❌ Completion endpoint failed${NC}"
    echo "   Response: $COMPLETION_RESPONSE"
fi
echo ""

# Test embedding endpoint
echo "5️⃣  Testing embedding endpoint..."
EMBEDDING_RESPONSE=$(curl -s -X POST http://localhost:8003/embeddings \
    -H "Content-Type: application/json" \
    -H "X-User-ID: test-user-123" \
    -d '{
        "text": "Startup funding round",
        "provider": "openai"
    }')
if echo "$EMBEDDING_RESPONSE" | grep -q "embedding"; then
    echo -e "${GREEN}✅ Embedding endpoint passed${NC}"
    echo "   Response preview: $(echo $EMBEDDING_RESPONSE | head -c 200)..."
else
    echo -e "${RED}❌ Embedding endpoint failed${NC}"
    echo "   Response: $EMBEDDING_RESPONSE"
fi
echo ""

# Test streaming endpoint
echo "6️⃣  Testing streaming endpoint..."
echo "   (Streaming test - checking for SSE format)..."
STREAM_RESPONSE=$(curl -s -X POST http://localhost:8003/complete/stream \
    -H "Content-Type: application/json" \
    -H "X-User-ID: test-user-123" \
    -d '{
        "messages": [{"role": "user", "content": "Say hello"}],
        "provider": "openai"
    }' | head -c 500)
if echo "$STREAM_RESPONSE" | grep -q "data:"; then
    echo -e "${GREEN}✅ Streaming endpoint passed (SSE format detected)${NC}"
    echo "   Response preview: $(echo $STREAM_RESPONSE | head -c 200)..."
else
    echo -e "${YELLOW}⚠️  Streaming response format unclear${NC}"
    echo "   Response preview: $STREAM_RESPONSE"
fi
echo ""

# Stop service
echo "🛑 Stopping service..."
kill $SERVICE_PID 2>/dev/null || true
wait $SERVICE_PID 2>/dev/null || true

echo ""
echo "=========================================="
echo -e "${GREEN}✅ Testing complete!${NC}"
echo ""
echo "📊 Test Coverage Report:"
echo "   HTML report: htmlcov/index.html"
echo ""
echo "📝 Next steps:"
echo "   1. Review test coverage report"
echo "   2. Check Supabase llm_usage table for usage tracking"
echo "   3. Verify all endpoints respond correctly"
echo ""

