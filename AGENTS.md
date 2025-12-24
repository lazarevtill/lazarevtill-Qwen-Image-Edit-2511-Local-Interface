# AGENTS.md - AI Agent Instructions

This file provides instructions for AI agents (Claude, GPT, Copilot, etc.) working on this codebase.

## Project Overview

This is a local Gradio interface for running Qwen-Image-Edit-2511 GGUF quantized models for image editing and generation. It supports multiple hardware backends (NVIDIA CUDA, Intel XPU, AMD ROCm, CPU).

## Key Files

| File | Purpose |
|------|---------|
| `app.py` | Main Gradio application with UI and inference logic |
| `setup.bat` | Windows setup and run script |
| `setup.sh` | Linux/macOS setup and run script |
| `requirements.txt` | Python dependencies |
| `README.md` | User documentation |
| `AGENTS.md` | This file - AI agent instructions |

## Architecture

### Device Detection (`detect_available_devices()`)
- Detects NVIDIA GPUs via `torch.cuda`
- Detects Intel XPU via `torch.xpu`
- Falls back to CPU if no GPU available
- Returns dict with device info, dtype, and device string

### Setup Script Hardware Detection
The setup scripts detect ALL available GPUs on the system:
- **NVIDIA**: via `nvidia-smi -L` (compatible with older nvidia-smi versions)
- **Intel XPU**: via PowerShell `Get-CimInstance` (Windows) or `lspci`/sysfs (Linux)
- **AMD ROCm**: via `rocm-smi` or `rocminfo` (Linux only)

Scripts display all detected GPUs and select by priority: CUDA > XPU > ROCm > CPU

### Model Loading (`load_pipeline()`)
- Uses `QwenImageTransformer2DModel.from_single_file()` for GGUF loading
- Applies `GGUFQuantizationConfig` for quantized inference
- Caches pipeline globally to avoid reloading
- Supports CPU offload for CUDA, direct `.to()` for XPU

### Image Management
- Input images saved to `./images/inputs/`
- Output images saved to `./images/outputs/`
- Filenames include timestamp and sanitized prompt

## Development Guidelines

### Adding New Device Support
1. Add detection logic in `detect_available_devices()`
2. Add dtype selection in device info dict
3. Handle device-specific pipeline loading in `load_pipeline()`
4. Update `clear_device_cache()` for memory management
5. Update setup scripts for package installation

### Adding New Models
1. Add entry to `AVAILABLE_MODELS` dict
2. **Filename format**: `qwen-image-edit-2511-{QUANT}.gguf` (lowercase prefix!)
3. Models are downloaded from `unsloth/Qwen-Image-Edit-2511-GGUF`

**Important**: Model filenames use lowercase `qwen-image-edit-2511-` prefix, not `Qwen-Image-Edit-2511-`.

### UI Modifications
- UI is created in `create_ui()` function
- Uses Gradio Blocks API with Tabs
- Four tabs: Image Editing, Model Manager, Devices, Help
- CSS should be passed to `launch()`, not `Blocks()` constructor (Gradio 6.0+)

## Common Tasks

### Test GPU Detection
```python
from app import detect_available_devices
devices = detect_available_devices()
for name, info in devices.items():
    print(f"{name}: {info['description']}")
```

### Test Model Loading
```python
from app import load_pipeline
pipe = load_pipeline("Q4_K_M (13.1 GB) - Recommended", "NVIDIA CUDA")
```

### Run Without UI
```python
from app import load_pipeline
from PIL import Image

pipe = load_pipeline("Q4_K_M (13.1 GB) - Recommended", "CPU")
image = Image.open("input.png")
result = pipe(
    image=[image],
    prompt="add a hat",
    num_inference_steps=40,
    true_cfg_scale=4.0,
)
result.images[0].save("output.png")
```

## Dependencies

### Core
- `gradio` - Web UI framework
- `torch` - PyTorch (with CUDA/XPU/ROCm support)
- `diffusers` - Hugging Face diffusion models
- `transformers` - Model tokenizers and configs
- `accelerate` - Model loading optimization

### GGUF Support
- `gguf` - GGUF file format support
- Quantization handled by `GGUFQuantizationConfig`

### Optional
- `intel-extension-for-pytorch` - Intel XPU optimization

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `DIFFUSERS_GGUF_CUDA_KERNELS` | Set to `true` for optimized CUDA GGUF kernels |

## Setup Script Options

### Windows (`setup.bat`)
```batch
setup.bat              # Auto-detect and use best GPU
setup.bat --cuda       # Force NVIDIA CUDA
setup.bat --xpu        # Force Intel XPU
setup.bat --cpu        # Force CPU mode
setup.bat --share      # Network mode (0.0.0.0)
setup.bat --port 8080  # Custom port
setup.bat --reset      # Clean reinstall
```

### Linux/macOS (`setup.sh`)
```bash
./setup.sh              # Auto-detect and use best GPU
./setup.sh --cuda       # Force NVIDIA CUDA
./setup.sh --xpu        # Force Intel XPU
./setup.sh --rocm       # Force AMD ROCm
./setup.sh --cpu        # Force CPU mode
./setup.sh --share      # Network mode (0.0.0.0)
./setup.sh --port 8080  # Custom port
./setup.sh --reset      # Clean reinstall
```

## Testing

### Manual Testing Checklist
- [ ] Fresh install with `setup.bat --reset` / `./setup.sh --reset`
- [ ] GPU detection (NVIDIA/Intel/AMD/CPU)
- [ ] Multi-GPU detection (all GPUs shown)
- [ ] Device force flags (`--cuda`, `--xpu`, `--cpu`)
- [ ] Model download from UI
- [ ] Image editing with uploaded image
- [ ] Image generation without input
- [ ] Model switching
- [ ] Network mode with `--share`

## Troubleshooting for Agents

### "Module not found" errors
- Ensure virtual environment is activated
- Run `pip install -r requirements.txt`

### CUDA out of memory
- Suggest smaller quantization
- Check if `enable_model_cpu_offload()` is called

### Intel XPU issues
- Verify PyTorch XPU build: `torch.xpu.is_available()`
- Check IPEX installation: `import intel_extension_for_pytorch`

### Model download 404 errors
- Verify filename uses lowercase prefix: `qwen-image-edit-2511-*.gguf`
- Check `AVAILABLE_MODELS` dict matches actual filenames on HuggingFace

### Script exits unexpectedly (Windows)
- Check for unescaped special characters in batch file
- Avoid nested parentheses in `if` statements
- Use `!var!` with `EnableDelayedExpansion`
- Don't use `goto` inside `if` blocks with parentheses

### nvidia-smi compatibility
- Use `nvidia-smi -L` instead of `--format=csv,noheader` (works on all versions)

### Only one GPU detected on multi-GPU system
- Ensure script detects ALL GPUs before selecting one
- Use `--cuda`, `--xpu`, or `--cpu` flags to force specific device

## Code Style

- Python: Follow PEP 8
- Type hints encouraged for function signatures
- Docstrings for public functions
- Comments for complex logic
- Batch/Shell: Use descriptive echo messages for user feedback
- Gradio 6.0+: Pass CSS to `launch()`, not `Blocks()` constructor
