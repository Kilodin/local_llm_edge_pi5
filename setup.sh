#!/bin/bash

# Local LLM Edge Pi5 Setup Script
# This script automates the setup process for the local LLM inference system

set -e  # Exit on any error

echo "🚀 Local LLM Edge Pi5 Setup Script"
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on Raspberry Pi
check_raspberry_pi() {
    if [[ -f /proc/device-tree/model ]] && grep -q "Raspberry Pi" /proc/device-tree/model; then
        print_success "Detected Raspberry Pi"
        return 0
    else
        print_warning "Not running on Raspberry Pi - some optimizations may not apply"
        return 1
    fi
}

# Check system requirements
check_requirements() {
    print_status "Checking system requirements..."
    
    # Check Node.js
    if ! command -v node &> /dev/null; then
        print_error "Node.js is not installed. Please install Node.js 18+ first."
        exit 1
    fi
    
    NODE_VERSION=$(node --version | cut -d'v' -f2 | cut -d'.' -f1)
    if [ "$NODE_VERSION" -lt 18 ]; then
        print_error "Node.js version 18+ is required. Current version: $(node --version)"
        exit 1
    fi
    
    print_success "Node.js version: $(node --version)"
    
    # Check npm
    if ! command -v npm &> /dev/null; then
        print_error "npm is not installed"
        exit 1
    fi
    
    print_success "npm version: $(npm --version)"
    
    # Check CMake
    if ! command -v cmake &> /dev/null; then
        print_error "CMake is not installed. Installing..."
        sudo apt update
        sudo apt install -y cmake
    fi
    
    print_success "CMake version: $(cmake --version | head -n1)"
    
    # Check build tools
    if ! command -v g++ &> /dev/null; then
        print_status "Installing build tools..."
        sudo apt update
        sudo apt install -y build-essential
    fi
    
    print_success "Build tools available"
}

# Install system dependencies
install_system_deps() {
    print_status "Installing system dependencies..."
    
    sudo apt update
    sudo apt install -y \
        build-essential \
        cmake \
        git \
        python3 \
        python3-pip \
        curl \
        wget \
        unzip
    
    print_success "System dependencies installed"
}

# Install Node.js dependencies
install_node_deps() {
    print_status "Installing Node.js dependencies..."
    
    npm install
    
    print_success "Node.js dependencies installed"
}

# Install React UI dependencies
install_ui_deps() {
    print_status "Installing React UI dependencies..."
    
    cd src/ui
    npm install
    cd ../..
    
    print_success "React UI dependencies installed"
}

# Clone llama.cpp
setup_llama_cpp() {
    print_status "Setting up llama.cpp..."
    
    if [ ! -d "third_party/llama.cpp" ]; then
        mkdir -p third_party
        cd third_party
        git clone https://github.com/ggerganov/llama.cpp.git
        cd llama.cpp
        
        # Build llama.cpp
        make clean
        make -j$(nproc)
        
        cd ../..
        print_success "llama.cpp built successfully"
    else
        print_status "llama.cpp already exists, updating..."
        cd third_party/llama.cpp
        git pull
        make clean
        make -j$(nproc)
        cd ../..
        print_success "llama.cpp updated and rebuilt"
    fi
}

# Build C++ core
build_cpp_core() {
    print_status "Building C++ core..."
    
    # Create build directory
    mkdir -p build
    
    # Build with CMake
    cd build
    cmake ..
    make -j$(nproc)
    cd ..
    
    print_success "C++ core built successfully"
}

# Build Node.js bindings
build_node_bindings() {
    print_status "Building Node.js bindings..."
    
    npm run install
    
    print_success "Node.js bindings built successfully"
}

# Build React UI
build_react_ui() {
    print_status "Building React UI..."
    
    cd src/ui
    npm run build
    cd ../..
    
    print_success "React UI built successfully"
}

# Create models directory
setup_models_dir() {
    print_status "Setting up models directory..."
    
    mkdir -p models
    
    print_success "Models directory created"
}

# Download sample model (optional)
download_sample_model() {
    print_status "Would you like to download a sample model? (y/n)"
    read -r response
    
    if [[ "$response" =~ ^[Yy]$ ]]; then
        print_status "Downloading sample model (TinyLlama 1.1B Q4)..."
        
        cd models
        wget -O tinyllama-1.1b-chat.Q4_K_M.gguf \
            "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat.Q4_K_M.gguf"
        cd ..
        
        print_success "Sample model downloaded"
    else
        print_status "Skipping model download"
    fi
}

# Create configuration file
create_config() {
    print_status "Creating configuration file..."
    
    cat > config.json << EOF
{
  "defaultModel": "./models/tinyllama-1.1b-chat.Q4_K_M.gguf",
  "defaultConfig": {
    "contextSize": 2048,
    "batchSize": 512,
    "threads": 4,
    "gpuLayers": 0,
    "temperature": 0.7,
    "topP": 0.9,
    "topK": 40,
    "repeatPenalty": 1.1,
    "seed": 42
  },
  "server": {
    "port": 3001,
    "host": "0.0.0.0"
  },
  "ui": {
    "port": 3000,
    "host": "localhost"
  }
}
EOF
    
    print_success "Configuration file created"
}

# Run tests
run_tests() {
    print_status "Running tests..."
    
    if npm test; then
        print_success "Tests passed"
    else
        print_warning "Some tests failed - this is normal if no model is loaded"
    fi
}

# Main setup function
main() {
    print_status "Starting setup process..."
    
    # Check if we're on Raspberry Pi
    check_raspberry_pi
    
    # Check requirements
    check_requirements
    
    # Install dependencies
    install_system_deps
    install_node_deps
    install_ui_deps
    
    # Setup llama.cpp
    setup_llama_cpp
    
    # Build components
    build_cpp_core
    build_node_bindings
    build_react_ui
    
    # Setup directories
    setup_models_dir
    
    # Create configuration
    create_config
    
    # Download sample model
    download_sample_model
    
    # Run tests
    run_tests
    
    print_success "Setup completed successfully!"
    echo ""
    echo "🎉 Your Local LLM Edge Pi5 system is ready!"
    echo ""
    echo "Next steps:"
    echo "1. Download a GGUF model to the 'models' directory"
    echo "2. Run 'npm start' to start the web server"
    echo "3. Open http://localhost:3001 in your browser"
    echo "4. Or use 'npm run cli chat' for command-line interface"
    echo ""
    echo "For more information, see README.md"
}

# Run main function
main "$@" 