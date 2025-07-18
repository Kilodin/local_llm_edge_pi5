#!/bin/bash

# Start the server helper in the background
echo "Starting server helper..."
node server-helper.js &
HELPER_PID=$!

# Wait a moment for the helper to start
sleep 2

# Start the React app
echo "Starting React app..."
cd src/ui && npm start

# When React app exits, kill the helper
echo "Shutting down server helper..."
kill $HELPER_PID 