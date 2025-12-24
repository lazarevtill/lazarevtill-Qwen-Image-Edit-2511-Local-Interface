#!/bin/bash

# Qwen-Image-Edit-2511 Local Interface Setup Script
# Supports: NVIDIA CUDA, Intel XPU (Arc GPUs), AMD ROCm, and CPU
# This script auto-starts if already installed

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse command line arguments
SHARE_MODE=""
HOST="127.0.0.1"

while [[ $# -gt 0 ]]; do
    case $1 in
        --share|--public)
            HOST="0.0.0.0"
            shift
            ;;
        --local)
            HOST="127.0.0.1"
            shift
            ;;
        *)
            shift
            ;;
    esac
done

echo "=================================================="
echo " Qwen-Image-Edit-2511 Local Interface"
echo "=================================================="
echo ""

# Function to print colored messages
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

# Function to check command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to start the application
start_app() {
    echo ""
    echo "=================================================="
    echo " Starting Qwen-Image-Edit-2511"
    if [ "$HOST" = "0.0.0.0" ]; then
        echo " Access from any device on network"
        echo " Local: http://127.0.0.1:7860"
        # Get local IP
        if command_exists hostname; then
            LOCAL_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
            if [ -n "$LOCAL_IP" ]; then
                echo " Network: http://$LOCAL_IP:7860"
            fi
        fi
    else
        echo " Open browser to: http://127.0.0.1:7860"
    fi
    echo "=================================================="
    echo ""
    python app.py --host "$HOST"
}

# =============================================================================
# Check if already installed - AUTO START
# =============================================================================
if [ -d ".venv" ]; then
    print_status "Virtual environment found, checking installation..."

    # Activate virtual environment
    source .venv/bin/activate 2>/dev/null

    # Check if key packages are installed
    if python -c "import gradio; import diffusers; import torch" 2>/dev/null; then
        print_success "All packages installed. Starting application..."
        start_app
        exit 0
    else
        print_status "Some packages missing, continuing with setup..."
    fi
fi

# =============================================================================
# SETUP MODE - Install everything
# =============================================================================

# =============================================================================
# Step 1: Check for Python
# =============================================================================
echo "[1/6] Checking for Python..."

PYTHON_CMD=""

# Check for python3 first, then python
if command_exists python3; then
    PYTHON_CMD="python3"
elif command_exists python; then
    PYTHON_CMD="python"
else
    print_error "Python not found! Please install Python 3.10+"
    echo "       On Ubuntu/Debian: sudo apt install python3 python3-venv python3-pip"
    echo "       On Fedora: sudo dnf install python3 python3-pip"
    echo "       On macOS: brew install python3"
    exit 1
fi

# Get Python version
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

print_status "Found Python $PYTHON_VERSION"

# Check Python version is 3.10+
if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
    print_error "Python 3.10+ required. Found Python $PYTHON_VERSION"
    exit 1
fi

# =============================================================================
# Step 2: Create virtual environment
# =============================================================================
echo ""
echo "[2/6] Creating virtual environment..."

if [ -d ".venv" ]; then
    print_status "Virtual environment already exists, skipping creation."
else
    $PYTHON_CMD -m venv .venv
    if [ $? -ne 0 ]; then
        print_error "Failed to create virtual environment"
        echo "       Try: sudo apt install python3-venv (Ubuntu/Debian)"
        exit 1
    fi
    print_success "Virtual environment created successfully."
fi

# =============================================================================
# Step 3: Activate virtual environment
# =============================================================================
echo ""
echo "[3/6] Activating virtual environment..."

source .venv/bin/activate
if [ $? -ne 0 ]; then
    print_error "Failed to activate virtual environment"
    exit 1
fi
print_success "Virtual environment activated."

# =============================================================================
# Step 4: Upgrade pip
# =============================================================================
echo ""
echo "[4/6] Upgrading pip..."

pip install --upgrade pip --quiet
print_success "pip upgraded."

# =============================================================================
# Step 5: Detect hardware
# =============================================================================
echo ""
echo "[5/6] Detecting hardware..."
echo ""

DEVICE_TYPE="cpu"
TORCH_INDEX_URL=""

# Detect OS
OS_TYPE=$(uname -s)
print_status "Operating System: $OS_TYPE"

# -----------------------------------------------------------------------------
# Check for NVIDIA GPU
# -----------------------------------------------------------------------------
print_status "Checking for NVIDIA GPU..."

if command_exists nvidia-smi; then
    NVIDIA_GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)
    if [ -n "$NVIDIA_GPU" ]; then
        print_success "NVIDIA GPU detected: $NVIDIA_GPU"

        # Get all GPUs
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | while read line; do
            echo "           $line"
        done

        DEVICE_TYPE="cuda"
        TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"
    fi
fi

# -----------------------------------------------------------------------------
# Check for Intel GPU (if no NVIDIA found)
# -----------------------------------------------------------------------------
if [ "$DEVICE_TYPE" = "cpu" ]; then
    print_status "Checking for Intel GPU..."

    INTEL_GPU_FOUND=false

    # Method 1: Check via lspci (Linux)
    if command_exists lspci; then
        INTEL_GPU=$(lspci | grep -i "VGA\|3D\|Display" | grep -i "Intel" | head -n1)
        if [ -n "$INTEL_GPU" ]; then
            print_success "Intel GPU detected via lspci"
            echo "           $INTEL_GPU"
            INTEL_GPU_FOUND=true
        fi
    fi

    # Method 2: Check /sys/class/drm (Linux)
    if [ "$INTEL_GPU_FOUND" = false ] && [ -d "/sys/class/drm" ]; then
        for card in /sys/class/drm/card*/device/vendor; do
            if [ -f "$card" ]; then
                VENDOR=$(cat "$card" 2>/dev/null)
                # Intel vendor ID is 0x8086
                if [ "$VENDOR" = "0x8086" ]; then
                    print_success "Intel GPU detected via sysfs"
                    INTEL_GPU_FOUND=true
                    break
                fi
            fi
        done
    fi

    # Method 3: Check via system_profiler (macOS)
    if [ "$INTEL_GPU_FOUND" = false ] && [ "$OS_TYPE" = "Darwin" ]; then
        INTEL_GPU=$(system_profiler SPDisplaysDataType 2>/dev/null | grep -i "Intel" | head -n1)
        if [ -n "$INTEL_GPU" ]; then
            print_success "Intel GPU detected via system_profiler"
            echo "           $INTEL_GPU"
            INTEL_GPU_FOUND=true
        fi
    fi

    # Check if it's an Arc GPU or newer Intel GPU that supports XPU
    if [ "$INTEL_GPU_FOUND" = true ]; then
        # Check for Arc specifically
        if lspci 2>/dev/null | grep -qi "Arc"; then
            print_success "Intel Arc GPU detected - will use XPU backend"
            DEVICE_TYPE="xpu"
            TORCH_INDEX_URL="https://download.pytorch.org/whl/xpu"
        elif lspci 2>/dev/null | grep -qi "Xe\|Iris Xe\|UHD Graphics 7"; then
            print_success "Intel Xe/Iris GPU detected - will try XPU backend"
            DEVICE_TYPE="xpu"
            TORCH_INDEX_URL="https://download.pytorch.org/whl/xpu"
        else
            print_warning "Intel GPU found but may not support XPU. Will try XPU backend."
            DEVICE_TYPE="xpu"
            TORCH_INDEX_URL="https://download.pytorch.org/whl/xpu"
        fi
    fi
fi

# -----------------------------------------------------------------------------
# Check for AMD GPU (ROCm) - Bonus support
# -----------------------------------------------------------------------------
if [ "$DEVICE_TYPE" = "cpu" ]; then
    print_status "Checking for AMD GPU (ROCm)..."

    if command_exists rocm-smi; then
        AMD_GPU=$(rocm-smi --showproductname 2>/dev/null | grep -i "GPU" | head -n1)
        if [ -n "$AMD_GPU" ]; then
            print_success "AMD GPU with ROCm detected: $AMD_GPU"
            DEVICE_TYPE="rocm"
            TORCH_INDEX_URL="https://download.pytorch.org/whl/rocm6.0"
        fi
    elif command_exists rocminfo; then
        AMD_GPU=$(rocminfo 2>/dev/null | grep "Marketing Name" | head -n1 | cut -d: -f2 | xargs)
        if [ -n "$AMD_GPU" ]; then
            print_success "AMD GPU with ROCm detected: $AMD_GPU"
            DEVICE_TYPE="rocm"
            TORCH_INDEX_URL="https://download.pytorch.org/whl/rocm6.0"
        fi
    fi
fi

# -----------------------------------------------------------------------------
# Fallback to CPU
# -----------------------------------------------------------------------------
if [ "$DEVICE_TYPE" = "cpu" ]; then
    print_warning "No dedicated GPU found, using CPU mode."
    echo "           Note: CPU mode is very slow for image generation."
fi

echo ""
echo "=================================================="
echo " Detected Configuration:"
echo "   Device: $DEVICE_TYPE"
if [ -n "$TORCH_INDEX_URL" ]; then
    echo "   PyTorch Index: $TORCH_INDEX_URL"
else
    echo "   PyTorch Index: Default (CPU)"
fi
echo "=================================================="
echo ""

# =============================================================================
# Step 6: Install packages
# =============================================================================
echo "[6/6] Installing packages..."
echo ""

# Install PyTorch based on detected hardware
print_status "Installing PyTorch for $DEVICE_TYPE..."

case $DEVICE_TYPE in
    "cuda")
        pip install torch torchvision torchaudio --index-url "$TORCH_INDEX_URL" --quiet
        ;;
    "xpu")
        pip install torch torchvision torchaudio --index-url "$TORCH_INDEX_URL" --quiet
        print_status "Installing Intel Extension for PyTorch..."
        pip install intel-extension-for-pytorch --quiet 2>/dev/null || {
            print_warning "Could not install intel-extension-for-pytorch"
            echo "           XPU support will use native PyTorch XPU backend"
        }
        ;;
    "rocm")
        pip install torch torchvision torchaudio --index-url "$TORCH_INDEX_URL" --quiet
        ;;
    *)
        pip install torch torchvision torchaudio --quiet
        ;;
esac

# Install core dependencies
print_status "Installing core dependencies..."
pip install "gradio>=4.0.0" --quiet
pip install "numpy>=1.24.0" --quiet
pip install "pillow>=10.0.0" --quiet
pip install "huggingface_hub>=0.20.0" --quiet

# Install diffusers and related packages
print_status "Installing diffusers and ML packages..."
pip install "diffusers>=0.32.0" --quiet
pip install "transformers>=4.40.0" --quiet
pip install "accelerate>=0.30.0" --quiet
pip install "sentencepiece>=0.1.99" --quiet

# Install GGUF support
print_status "Installing GGUF support..."
pip install "gguf>=0.6.0" --quiet

echo ""
echo "=================================================="
print_success "Installation Complete!"
echo "=================================================="
echo ""
echo " Device configured: $DEVICE_TYPE"
echo ""
echo " First run will download the base model files (~2GB)."
echo " GGUF model weights can be downloaded from the UI."
echo ""

# Start the application
start_app
