#!/bin/bash

# Shanyan AI - Complete System Startup Script

echo "🚀 Starting Shanyan AI System..."
echo ""

# Check if backend server is running
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null ; then
    echo "✅ Backend already running on port 8000"
else
    echo "📦 Starting Backend Server..."
    cd /Users/hnai/Downloads/auto-ticket-classification-and-reply-system/backend
    PYTHONPATH=/Users/hnai/Downloads/auto-ticket-classification-and-reply-system/backend /Users/hnai/Downloads/auto-ticket-classification-and-reply-system/.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload > /tmp/shanyan-backend.log 2>&1 &
    BACKEND_PID=$!
    echo "✅ Backend started (PID: $BACKEND_PID)"
    sleep 5
fi

# Initialize users
echo ""
echo "👥 Initializing Demo Users..."
curl -s -X POST http://localhost:8000/api/init-users > /dev/null
echo "✅ Demo users ready"

echo ""
echo "🎉 System Ready!"
echo ""
echo "📋 Demo Accounts:"
echo "   Admin:       admin / admin123"
echo "   Client:      client1 / client123 (Rajesh Kumar)"
echo "   Tech Support: tech1 / tech123 (Priya Sharma)"
echo "   Accounting:  acc1 / acc123 (Amit Patel)"
echo "   Sales:       sales1 / sales123 (Sneha Reddy)"
echo ""
echo "🌐 Backend API: http://localhost:8000"
echo "📚 API Docs:    http://localhost:8000/docs"
echo ""
echo "💡 To start frontend:"
echo "   cd frontend && npm run dev"
echo ""
echo "Created by Shankar Narayanan, Student, Dr MGR University"
echo ""
