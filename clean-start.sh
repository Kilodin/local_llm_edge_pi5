#!/bin/bash

echo "🧹 Cleaning up existing processes..."

# Kill any existing server helper processes
pkill -f server-helper.js 2>/dev/null || true

# Kill any existing LLM server processes
pkill -f "src/server/index.js" 2>/dev/null || true

# Kill any React development server
pkill -f "react-scripts start" 2>/dev/null || true

# Wait a moment for processes to clean up
sleep 2

echo "🚀 Starting fresh application..."

# Start the server helper
echo "Starting server helper..."
node server-helper.js &
HELPER_PID=$!

# Wait for helper to start
sleep 2

# Start the React app
echo "Starting React app..."
cd src/ui && npm start

# When React app exits, kill the helper
echo "Shutting down server helper..."
kill $HELPER_PID 2>/dev/null || true 