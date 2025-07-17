#!/bin/bash

# Set the library path for llama.cpp
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/third_party/llama.cpp/build/bin

# Start the server
echo "Starting LLM server with library path: $LD_LIBRARY_PATH"
PORT=3000 npm start 