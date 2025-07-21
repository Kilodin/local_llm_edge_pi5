# Local LLM Edge Pi5 Inference

A high-performance local LLM inference system optimized for edge devices like the Raspberry Pi 5, using llama.cpp with C++ core and Node.js bindings, featuring a modern React web interface.

## 🚀 Features

- **CPU-Optimized Inference**: Built with llama.cpp for efficient CPU-only inference
- **Multi-Platform Support**: C++ core with Node.js bindings for cross-platform compatibility
- **Modern Web UI**: React-based interface with real-time streaming
- **CLI Interface**: Command-line tools for headless operation
- **REST API**: Full HTTP API for integration with other applications
- **WebSocket Streaming**: Real-time text generation with streaming support
- **Edge Device Optimized**: Configured for resource-constrained devices

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React UI      │    │   Node.js       │    │   C++ Core      │
│   (Frontend)    │◄──►│   (Backend)     │◄──►│   (llama.cpp)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                       │                       │
        │              ┌─────────────────┐              │
        └──────────────►│   WebSocket     │◄─────────────┘
                        │   (Streaming)   │
                        └─────────────────┘
```

## 📋 Prerequisites

- **Node.js** 18+ and npm
- **CMake** 3.16+
- **C++17** compatible compiler (GCC 7+, Clang 5+)
- **Python** 3.8+ (for build scripts)
- **Git**

### System Requirements

- **RAM**: Minimum 4GB, Recommended 8GB+
- **Storage**: 2GB+ for models and dependencies
- **CPU**: Multi-core processor (optimized for ARM64 on Pi5)

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd local_llm_edge_pi5
```

### 2. Install Dependencies

```bash
# Install Node.js dependencies
npm install

# Install React UI dependencies
cd src/ui && npm install && cd ../..

# Install system dependencies (Ubuntu/Debian)
sudo apt update
sudo apt install build-essential cmake git python3
```

### 3. Build the Project

```bash
# Build C++ core and Node.js bindings
npm run build:cpp

# Build React UI
npm run build:ui
```

### 4. Download a Model

Download a GGUF format model (recommended for edge devices):

```bash
# Create models directory
mkdir -p models

# Download a small model (example)
wget -O models/llama-2-7b-chat.Q4_K_M.gguf \
  https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf
```

## 🚀 Usage

### Web Interface

**Recommended: Automatic Startup**
```bash
./start-app.sh
```

This will automatically:
- Start the server helper on port 4000
- Start the React UI on port 3000
- The main LLM server runs on port 3001
- The React UI proxies API requests to port 3001 (see `src/ui/package.json`)
- Automatically attempt to start the LLM server when the page loads
- Show a retry button if the server fails to start

**Alternative: Manual Startup**
```bash
# Start server helper (port 4000)
npm run start:helper

# In another terminal, start the React UI (port 3000)
cd src/ui && npm start
```

Open your browser to `http://localhost:3000`

The app will automatically attempt to start the LLM server (port 3001) when the page loads. If it fails, you'll see a "Retry Connection" button.

### Command Line Interface

```bash
# Initialize a model
npm run cli init -m ./models/llama-2-7b-chat.Q4_K_M.gguf -t 4

# Start interactive chat
npm run cli chat

# Generate text from prompt
npm run cli generate "Hello, how are you?"

# Show system information
npm run cli info

# Update parameters
npm run cli params --temp 0.8 --top-p 0.9

# Start server only
npm run cli server -p 3001
```

### API Usage

```bash
# Initialize model
curl -X POST http://localhost:3001/api/initialize \
  -H "Content-Type: application/json" \
  -d '{
    "modelPath": "./models/llama-2-7b-chat.Q4_K_M.gguf",
    "threads": 4,
    "contextSize": 2048
  }'

# Generate text
curl -X POST http://localhost:3001/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello, how are you?",
    "maxTokens": 256
  }'

# Get model info
curl http://localhost:3001/api/model-info
```

---

**Port Reference:**
- Server Helper: `4000`
- Main LLM Server: `3001`
- React UI: `3000` (proxies API requests to `3001`)

---

## ⚙️ Configuration

### Model Configuration

```javascript
{
  "modelPath": "/path/to/model.gguf",
  "contextSize": 2048,        // Context window size
  "batchSize": 512,           // Batch size for processing
  "threads": 4,               // Number of CPU threads
  "gpuLayers": 0,             // GPU layers (0 for CPU-only)
  "temperature": 0.7,         // Sampling temperature
  "topP": 0.9,                // Top-p sampling
  "topK": 40,                 // Top-k sampling
  "repeatPenalty": 1.1,       // Repeat penalty
  "seed": 42                  // Random seed
}
```

### Performance Tuning

For Raspberry Pi 5 optimization:

```bash
# Use all available cores
npm run cli init -m ./models/model.gguf -t 4

# Reduce context size for memory efficiency
npm run cli init -m ./models/model.gguf -c 1024

# Use smaller batch size
npm run cli init -m ./models/model.gguf -b 256
```

## 📁 Project Structure

```
local_llm_edge_pi5/
├── src/
│   ├── cpp/                    # C++ Core
│   │   ├── model/             # LLM model implementation
│   │   ├── inference/         # Inference engine
│   │   └── bindings/          # Node.js native addon
│   ├── server/                # Express.js server
│   ├── cli/                   # Command-line interface
│   └── ui/                    # React frontend
├── models/                    # Model storage
├── examples/                  # Usage examples
├── docs/                      # Documentation
├── test/                      # Test files
├── CMakeLists.txt            # C++ build configuration
├── binding.gyp               # Node.js native addon build
├── package.json              # Node.js package
└── README.md                 # This file
```

## 🔧 Development

### Building from Source

```bash
# Clean build
rm -rf build
npm run build:cpp

# Development mode
npm run dev
```

### Testing

```bash
# Run tests
npm test

# Test CLI
npm run cli info
```

### Debugging

```bash
# Debug C++ core
gdb build/llm_node

# Debug Node.js
node --inspect src/server/index.js
```

## 🐛 Troubleshooting

### Common Issues

1. **Server Won't Start (Port Already in Use)**
   ```bash
   # Use the clean start script
   ./clean-start.sh
   
   # Or manually clean up processes
   pkill -f server-helper.js
   pkill -f "src/server/index.js"
   pkill -f "react-scripts start"
   ```

2. **Build Failures**
   ```bash
   # Ensure all dependencies are installed
   sudo apt install build-essential cmake
   npm install
   ```

3. **Model Loading Errors**
   ```bash
   # Check model file exists and is valid
   ls -la models/
   file models/your-model.gguf
   ```

4. **Memory Issues**
   ```bash
   # Reduce context size and batch size
   npm run cli init -m ./model.gguf -c 512 -b 128
   ```

5. **Performance Issues**
   ```bash
   # Monitor system resources
   htop
   free -h
   ```

### Logs

```bash
# View server logs
npm start 2>&1 | tee server.log

# View build logs
npm run build:cpp 2>&1 | tee build.log
```

## 📊 Performance

### Benchmarks (Raspberry Pi 5)

| Model | Context | Tokens/sec | Memory Usage |
|-------|---------|------------|--------------|
| Llama-2-7B-Q4 | 2048 | ~2-3 | ~2GB |
| Llama-2-7B-Q4 | 1024 | ~3-4 | ~1.5GB |
| TinyLlama-1.1B | 2048 | ~8-10 | ~800MB |

### Optimization Tips

1. **Use quantized models** (Q4_K_M, Q5_K_M)
2. **Reduce context size** for memory efficiency
3. **Optimize thread count** for your CPU
4. **Use SSD storage** for faster model loading
5. **Close unnecessary applications** to free memory

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- [llama.cpp](https://github.com/ggerganov/llama.cpp) - Core inference engine
- [node-addon-api](https://github.com/nodejs/node-addon-api) - Node.js native addon framework
- [React](https://reactjs.org/) - Frontend framework
- [Express.js](https://expressjs.com/) - Backend framework

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)
- **Documentation**: [Wiki](https://github.com/your-repo/wiki)

---

**Made with ❤️ for edge computing and local AI** 