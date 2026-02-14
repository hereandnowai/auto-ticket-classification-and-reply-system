#!/bin/bash
set -e

echo "🚀 Starting Shanyan AI Backend..."

# Check if model is already trained
MODEL_PATH="/app/training/models/fine_tuned_bert/config.json"

if [ ! -f "$MODEL_PATH" ]; then
    echo "📦 No trained model found. Starting model training..."
    echo "⏳ This may take 10-30 minutes on first run..."
    cd /app
    python training/train.py
    echo "✅ Model training complete!"
else
    echo "✅ Trained model found. Skipping training."
fi

# Start the backend server in the background
echo "🌐 Starting FastAPI server on port 8000..."
cd /app
uvicorn app.main:app --host 0.0.0.0 --port 8000 &
SERVER_PID=$!

# Wait for server to be ready
echo "⏳ Waiting for server to start..."
for i in $(seq 1 30); do
    if curl -s http://localhost:8000/ > /dev/null 2>&1; then
        echo "✅ Server is ready!"
        break
    fi
    sleep 2
done

# Initialize demo users
echo "👥 Initializing demo users..."
curl -s -X POST http://localhost:8000/api/init-users > /dev/null 2>&1 || true
echo "✅ Demo users initialized!"

echo ""
echo "🎉 Backend is running!"
echo "📋 Demo Accounts:"
echo "   Admin:        admin / admin123"
echo "   Client:       client1 / client123 (Rajesh Kumar)"
echo "   Tech Support: tech1 / tech123 (Priya Sharma)"
echo "   Accounting:   acc1 / acc123 (Amit Patel)"
echo "   Sales:        sales1 / sales123 (Sneha Reddy)"
echo ""

# Wait for the server process
wait $SERVER_PID
