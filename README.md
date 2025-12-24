# Qwen-Image-Edit-2511 Local Interface

A local Gradio interface for running [Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) image editing models using [GGUF quantized weights](https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF).

![Interface Preview](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/qwen_image_edit_logo.png)

## Features

- **Local inference** - Run entirely on your machine, no cloud required
- **Multiple quantization options** - From Q2_K (7.2GB) to full precision BF16 (40.9GB)
- **Multi-GPU support** - NVIDIA CUDA, Intel XPU (Arc GPUs), AMD ROCm, and CPU
- **Multi-GPU detection** - Detects all available GPUs, allows forcing specific device
- **Model management** - Download, delete, and manage models from the UI
- **Image saving** - Automatically saves all input/output images
- **Network mode** - Share on local network for access from other devices

## Quick Start

### Windows

```batch
:: First run - installs everything and starts the app
setup.bat

:: Network mode (accessible from other devices)
setup.bat --share

:: Force specific GPU
setup.bat --cuda      :: Use NVIDIA GPU
setup.bat --xpu       :: Use Intel Arc GPU
setup.bat --cpu       :: Use CPU only

:: Clean reinstall if something breaks
setup.bat --reset
```

### Linux / macOS

```bash
# Make executable
chmod +x setup.sh

# First run - installs everything and starts the app
./setup.sh

# Network mode
./setup.sh --share

# Force specific GPU
./setup.sh --cuda      # Use NVIDIA GPU
./setup.sh --xpu       # Use Intel Arc GPU
./setup.sh --rocm      # Use AMD GPU (Linux only)
./setup.sh --cpu       # Use CPU only

# Clean reinstall
./setup.sh --reset
```

Then open http://127.0.0.1:7860 in your browser.

## Requirements

- **Python** 3.10 or higher
- **RAM** 16GB+ recommended
- **VRAM** 8-24GB depending on model (or system RAM for CPU mode)
- **Disk** 7-41GB per model

### Supported Hardware

| Device | Backend | Performance |
|--------|---------|-------------|
| NVIDIA GPU (RTX 20/30/40 series) | CUDA | Fast |
| Intel Arc GPU (A770, A750, etc.) | XPU | Moderate |
| Intel Core Ultra (integrated) | XPU | Moderate |
| AMD GPU (with ROCm) | ROCm | Moderate |
| CPU | CPU | Very Slow |

## Available Models

| Model | Size | Quality | Recommended For |
|-------|------|---------|-----------------|
| Q2_K | 7.22 GB | Lower | Testing, low VRAM |
| Q3_K_M | 9.74 GB | Fair | Budget GPUs (8GB) |
| **Q4_K_M** | **13.1 GB** | **Good** | **General use (Recommended)** |
| Q5_K_M | 15 GB | Better | Quality focus |
| Q6_K | 16.8 GB | High | High quality |
| Q8_0 | 21.8 GB | Very High | Near-original |
| BF16/F16 | 40.9 GB | Original | Maximum quality |

## Usage

### Command Line Options

```bash
python app.py [options]

Options:
  --host HOST    Host to bind to (default: 127.0.0.1, use 0.0.0.0 for network)
  --port PORT    Port to bind to (default: 7860)
  --share        Create a public Gradio link
```

### Setup Script Options

```bash
setup.bat [options]   # Windows
./setup.sh [options]  # Linux/macOS

Options:
  --share, --public    Enable network mode (bind to 0.0.0.0)
  --local              Local only mode (default)
  --port PORT          Custom port number
  --reset              Remove .venv and reinstall everything
  --cuda               Force NVIDIA CUDA device
  --xpu                Force Intel XPU device
  --rocm               Force AMD ROCm device (Linux only)
  --cpu                Force CPU mode
```

### Multi-GPU Systems

On systems with multiple GPUs (e.g., NVIDIA + Intel Arc), the setup script will:
1. Detect ALL available GPUs
2. Display them in the console
3. Auto-select by priority: CUDA > XPU > ROCm > CPU

To override the automatic selection, use the force flags:
```bash
# System has both NVIDIA and Intel Arc, but you want to use Intel:
setup.bat --xpu
./setup.sh --xpu
```

## Directory Structure

```
img/
├── app.py              # Main application
├── setup.bat           # Windows setup/run script
├── setup.sh            # Linux/macOS setup/run script
├── requirements.txt    # Python dependencies
├── README.md           # This file
├── AGENTS.md           # AI agent instructions
├── models/             # Downloaded GGUF models
└── images/
    ├── inputs/         # Saved input images
    └── outputs/        # Generated output images
```

## Tips for Best Results

1. **Be specific with prompts**: "Add a red Santa hat to the person" works better than "add hat"
2. **Guidance scale 4-6**: Higher values follow prompts more strictly
3. **40-50 inference steps**: Good balance of quality and speed
4. **Start with Q4_K_M**: Best quality/performance ratio for most GPUs

## Troubleshooting

### Out of memory
- Try a smaller quantization (Q3_K_M, Q2_K)
- Reduce output image dimensions
- Close other GPU-intensive applications

### Slow generation
- Reduce inference steps (20-30 for previews)
- Use a smaller model
- Ensure GPU is being used (check Devices tab)

### Intel XPU not detected
- Install PyTorch with XPU support:
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/xpu
  ```
- Install Intel Extension for PyTorch:
  ```bash
  pip install intel-extension-for-pytorch
  ```

### Only NVIDIA detected on multi-GPU system
- The setup scripts now detect all GPUs automatically
- Use `--xpu` flag to force Intel Arc GPU
- Use `--cpu` flag to force CPU mode

### Virtual environment issues
- Run with `--reset` flag to clean reinstall:
  ```bash
  setup.bat --reset   # Windows
  ./setup.sh --reset  # Linux/macOS
  ```

## Links

- [Qwen-Image-Edit-2511 (Original Model)](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)
- [GGUF Quantized Models](https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF)
- [Qwen-Image GitHub](https://github.com/QwenLM/Qwen-Image)
- [Diffusers GGUF Documentation](https://huggingface.co/docs/diffusers/en/quantization/gguf)
- [Intel Extension for PyTorch](https://github.com/intel/intel-extension-for-pytorch)

## License

This project uses the Qwen-Image-Edit-2511 model which is licensed under Apache 2.0.
