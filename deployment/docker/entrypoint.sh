#!/bin/bash
set -e

echo "🚀 Starting AI Data Analyst Agent..."

# Start FastAPI in background
echo "Starting FastAPI backend on port ${API_PORT:-8000}..."
uvicorn backend.main:app \
  --host "${API_HOST:-0.0.0.0}" \
  --port "${API_PORT:-8000}" \
  --workers "${API_WORKERS:-2}" \
  --log-level info &

FASTAPI_PID=$!

# Wait for API to be ready
echo "Waiting for API to be healthy..."
for i in {1..30}; do
  if curl -sf "http://localhost:${API_PORT:-8000}/health" > /dev/null 2>&1; then
    echo "✅ API is healthy"
    break
  fi
  sleep 2
done

# Start Streamlit frontend
echo "Starting Streamlit frontend on port ${STREAMLIT_PORT:-8501}..."
streamlit run frontend/app.py \
  --server.port "${STREAMLIT_PORT:-8501}" \
  --server.address "0.0.0.0" \
  --server.headless true \
  --server.enableCORS false \
  --browser.gatherUsageStats false &

STREAMLIT_PID=$!

echo "✅ All services started"
echo "   API:       http://0.0.0.0:${API_PORT:-8000}"
echo "   Frontend:  http://0.0.0.0:${STREAMLIT_PORT:-8501}"
echo "   API Docs:  http://0.0.0.0:${API_PORT:-8000}/api/docs"

# Wait for any process to exit
wait -n $FASTAPI_PID $STREAMLIT_PID
EXIT_CODE=$?

echo "⚠️  A service exited with code $EXIT_CODE"
exit $EXIT_CODE
