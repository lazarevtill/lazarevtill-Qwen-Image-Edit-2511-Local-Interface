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

# Default settings
HOST="127.0.0.1"
PORT="7860"
DO_RESET=false
FORCE_DEVICE=""
FOUND_CUDA=false
FOUND_XPU=false
FOUND_ROCM=false

# Parse command line arguments
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
        --port)
            PORT="$2"
            shift 2
            ;;
        --reset)
            DO_RESET=true
            shift
            ;;
        --cuda)
            FORCE_DEVICE="cuda"
            shift
            ;;
        --xpu)
            FORCE_DEVICE="xpu"
            shift
            ;;
        --rocm)
            FORCE_DEVICE="rocm"
            shift
            ;;
        --cpu)
            FORCE_DEVICE="cpu"
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
    echo -e "${GREEN}[OK]${NC} $1"
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
    echo " URL: http://${HOST}:${PORT}"
    if [ "$HOST" = "0.0.0.0" ]; then
        echo " [Network mode - accessible from other devices]"
        # Get local IP
        if command_exists hostname; then
            LOCAL_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
            if [ -n "$LOCAL_IP" ]; then
                echo " Network: http://$LOCAL_IP:$PORT"
            fi
        fi
    fi
    echo "=================================================="
    echo ""
    echo " Press Ctrl+C to stop the server"
    echo ""
    echo " TIP: If something goes wrong, run: ./setup.sh --reset"
    echo " TIP: Force device with: ./setup.sh --cuda or --xpu or --cpu"
    echo ""
    python app.py --host "$HOST" --port "$PORT"
}

# =============================================================================
# Handle --reset flag
# =============================================================================
if [ "$DO_RESET" = true ]; then
    echo "[RESET] Removing virtual environment..."
    rm -rf .venv 2>/dev/null || true
    echo "       Done. Continuing with fresh install..."
    echo ""
fi

# =============================================================================
# Check if already installed - AUTO START
# =============================================================================
if [ -f ".venv/bin/python" ]; then
    print_status "Checking installation..."

    # Activate virtual environment
    source .venv/bin/activate 2>/dev/null || true

    # Check if key packages are installed
    if .venv/bin/python -c "import gradio; import diffusers; import torch" 2>/dev/null; then
        print_success "All packages installed. Starting application..."
        start_app
        exit 0
    else
        print_status "Some packages missing, continuing with setup..."
        echo ""
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

echo "       Found Python $PYTHON_VERSION"

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
    echo "       Virtual environment exists, reusing it."
else
    $PYTHON_CMD -m venv .venv
    if [ $? -ne 0 ]; then
        print_error "Failed to create virtual environment"
        echo "       Try: sudo apt install python3-venv (Ubuntu/Debian)"
        echo ""
        echo "TIP: If you have issues, try running: ./setup.sh --reset"
        exit 1
    fi
    echo "       Virtual environment created."
fi

# =============================================================================
# Step 3: Activate virtual environment
# =============================================================================
echo ""
echo "[3/6] Activating virtual environment..."

source .venv/bin/activate
if [ $? -ne 0 ]; then
    print_error "Failed to activate virtual environment"
    echo ""
    echo "TIP: Try running: ./setup.sh --reset"
    exit 1
fi
echo "       Activated."

# =============================================================================
# Step 4: Upgrade pip
# =============================================================================
echo ""
echo "[4/6] Upgrading pip..."

pip install --upgrade pip --quiet
echo "       Done."

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
echo "       Operating System: $OS_TYPE"

# -----------------------------------------------------------------------------
# Check for NVIDIA GPU
# -----------------------------------------------------------------------------
echo "       Checking for NVIDIA GPU..."

if command_exists nvidia-smi; then
    # Use nvidia-smi -L which works on all versions
    NVIDIA_GPU=$(nvidia-smi -L 2>/dev/null | head -n1)
    if [ -n "$NVIDIA_GPU" ]; then
        echo "       [FOUND] $NVIDIA_GPU"
        FOUND_CUDA=true
    fi
fi

# -----------------------------------------------------------------------------
# Check for Intel GPU
# -----------------------------------------------------------------------------
echo "       Checking for Intel GPU..."

INTEL_GPU_FOUND=false

# Method 1: Check via lspci (Linux)
if command_exists lspci; then
    INTEL_GPU=$(lspci | grep -i "VGA\|3D\|Display" | grep -i "Intel" | head -n1)
    if [ -n "$INTEL_GPU" ]; then
        echo "       [FOUND] Intel: $INTEL_GPU"
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
                echo "       [FOUND] Intel GPU via sysfs"
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
        echo "       [FOUND] Intel: $INTEL_GPU"
        INTEL_GPU_FOUND=true
    fi
fi

if [ "$INTEL_GPU_FOUND" = true ]; then
    FOUND_XPU=true
fi

# -----------------------------------------------------------------------------
# Check for AMD GPU (ROCm)
# -----------------------------------------------------------------------------
echo "       Checking for AMD GPU (ROCm)..."

if command_exists rocm-smi; then
    AMD_GPU=$(rocm-smi --showproductname 2>/dev/null | grep -i "GPU" | head -n1)
    if [ -n "$AMD_GPU" ]; then
        echo "       [FOUND] AMD: $AMD_GPU"
        FOUND_ROCM=true
    fi
elif command_exists rocminfo; then
    AMD_GPU=$(rocminfo 2>/dev/null | grep "Marketing Name" | head -n1 | cut -d: -f2 | xargs)
    if [ -n "$AMD_GPU" ]; then
        echo "       [FOUND] AMD: $AMD_GPU"
        FOUND_ROCM=true
    fi
fi

# -----------------------------------------------------------------------------
# Determine which device to use
# -----------------------------------------------------------------------------
if [ -n "$FORCE_DEVICE" ]; then
    DEVICE_TYPE="$FORCE_DEVICE"
    echo ""
    echo "       [FORCED] Using $FORCE_DEVICE as requested"
elif [ "$FOUND_CUDA" = true ]; then
    DEVICE_TYPE="cuda"
elif [ "$FOUND_XPU" = true ]; then
    DEVICE_TYPE="xpu"
elif [ "$FOUND_ROCM" = true ]; then
    DEVICE_TYPE="rocm"
else
    DEVICE_TYPE="cpu"
    echo "       [INFO] No supported GPU found, using CPU mode."
    echo "              Note: CPU is very slow for image generation."
fi

# Set PyTorch index URL based on device
case $DEVICE_TYPE in
    "cuda") TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121" ;;
    "xpu") TORCH_INDEX_URL="https://download.pytorch.org/whl/xpu" ;;
    "rocm") TORCH_INDEX_URL="https://download.pytorch.org/whl/rocm6.0" ;;
esac

echo ""
echo "=================================================="
echo " Available GPUs:"
[ "$FOUND_CUDA" = true ] && echo "   - NVIDIA CUDA"
[ "$FOUND_XPU" = true ] && echo "   - Intel XPU"
[ "$FOUND_ROCM" = true ] && echo "   - AMD ROCm"
echo "   - CPU (always available)"
echo ""
echo " Selected Device: $DEVICE_TYPE"
echo "=================================================="
echo ""

# =============================================================================
# Step 6: Install packages
# =============================================================================
echo "[6/6] Installing packages..."
echo ""

# Install PyTorch based on detected hardware
echo "       Installing PyTorch for $DEVICE_TYPE..."

case $DEVICE_TYPE in
    "cuda")
        # Use --index-url to prioritize CUDA wheels over CPU-only from PyPI
        pip install torch torchvision torchaudio --index-url "$TORCH_INDEX_URL" || {
            print_warning "Failed with index-url, trying extra-index-url..."
            pip install torch torchvision torchaudio --extra-index-url "$TORCH_INDEX_URL" || {
                print_warning "Failed with extra-index-url, trying default PyTorch..."
                pip install torch torchvision torchaudio
            }
        }
        ;;
    "xpu")
        # Use --index-url to prioritize XPU wheels
        pip install torch torchvision torchaudio --index-url "$TORCH_INDEX_URL" || {
            print_warning "Failed with index-url, trying extra-index-url..."
            pip install torch torchvision torchaudio --extra-index-url "$TORCH_INDEX_URL" || {
                print_warning "Failed with extra-index-url, trying default PyTorch..."
                pip install torch torchvision torchaudio
            }
        }
        echo ""
        echo "       Installing Intel Extension for PyTorch..."
        pip install intel-extension-for-pytorch 2>/dev/null || {
            print_warning "Could not install intel-extension-for-pytorch"
            echo "           XPU support will use native PyTorch XPU backend"
        }
        ;;
    "rocm")
        # Use --index-url to prioritize ROCm wheels
        pip install torch torchvision torchaudio --index-url "$TORCH_INDEX_URL" || {
            print_warning "Failed with index-url, trying extra-index-url..."
            pip install torch torchvision torchaudio --extra-index-url "$TORCH_INDEX_URL" || {
                print_warning "Failed with extra-index-url, trying default PyTorch..."
                pip install torch torchvision torchaudio
            }
        }
        ;;
    *)
        pip install torch torchvision torchaudio
        ;;
esac

echo ""
echo "       Installing core dependencies..."
pip install gradio numpy pillow huggingface_hub

echo ""
echo "       Installing ML packages..."
pip install diffusers transformers accelerate sentencepiece

echo ""
echo "       Installing GGUF support..."
pip install gguf

# Install bitsandbytes for 8-bit quantization (CUDA only, helps with limited VRAM)
if [ "$DEVICE_TYPE" = "cuda" ]; then
    echo ""
    echo "       Installing bitsandbytes for 8-bit quantization..."
    pip install bitsandbytes 2>/dev/null || {
        print_warning "Could not install bitsandbytes"
        echo "           8-bit text encoder quantization will not be available"
        echo "           This is optional - needed only for GPUs with < 24GB VRAM"
    }
fi

echo ""
echo "=================================================="
echo " Installation Complete!"
echo " Device: $DEVICE_TYPE"
echo "=================================================="
echo ""

# Start the application
start_app
