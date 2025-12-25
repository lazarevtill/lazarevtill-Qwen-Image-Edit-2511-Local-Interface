"""
Qwen-Image-Edit-2511 Local Interface
A local Gradio interface for running Qwen-Image-Edit-2511 GGUF models.
Supports NVIDIA CUDA, Intel XPU (Arc GPUs), and CPU.
"""

import gradio as gr
import numpy as np
import random
import torch
import os
import platform
from PIL import Image
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from datetime import datetime
from huggingface_hub import hf_hub_download

# Available GGUF quantization options from unsloth/Qwen-Image-Edit-2511-GGUF
# Note: filenames are lowercase on HuggingFace
AVAILABLE_MODELS = {
    "Q2_K (7.22 GB) - Smallest": "qwen-image-edit-2511-Q2_K.gguf",
    "Q3_K_S (9.04 GB)": "qwen-image-edit-2511-Q3_K_S.gguf",
    "Q3_K_M (9.74 GB)": "qwen-image-edit-2511-Q3_K_M.gguf",
    "Q3_K_L (10.4 GB)": "qwen-image-edit-2511-Q3_K_L.gguf",
    "Q4_0 (11.9 GB)": "qwen-image-edit-2511-Q4_0.gguf",
    "Q4_K_S (12.3 GB)": "qwen-image-edit-2511-Q4_K_S.gguf",
    "Q4_1 (12.8 GB)": "qwen-image-edit-2511-Q4_1.gguf",
    "Q4_K_M (13.1 GB) - Recommended": "qwen-image-edit-2511-Q4_K_M.gguf",
    "Q5_K_S (14.3 GB)": "qwen-image-edit-2511-Q5_K_S.gguf",
    "Q5_0 (14.4 GB)": "qwen-image-edit-2511-Q5_0.gguf",
    "Q5_K_M (15 GB)": "qwen-image-edit-2511-Q5_K_M.gguf",
    "Q5_1 (15.4 GB)": "qwen-image-edit-2511-Q5_1.gguf",
    "Q6_K (16.8 GB) - High Quality": "qwen-image-edit-2511-Q6_K.gguf",
    "Q8_0 (21.8 GB)": "qwen-image-edit-2511-Q8_0.gguf",
    "BF16 (40.9 GB) - Full Precision": "qwen-image-edit-2511-BF16.gguf",
    "F16 (40.9 GB) - Full Precision": "qwen-image-edit-2511-F16.gguf",
}

REPO_ID = "unsloth/Qwen-Image-Edit-2511-GGUF"
BASE_MODEL_REPO = "Qwen/Qwen-Image-Edit-2511"
BASE_PIPELINE_DIR = "./models/Qwen-Image-Edit-2511-base"

# Global pipeline cache
current_pipeline = None
current_model_name = None
current_device = None

MAX_SEED = np.iinfo(np.int32).max
MODELS_DIR = "./models"
IMAGES_DIR = "./images"
INPUT_IMAGES_DIR = "./images/inputs"
OUTPUT_IMAGES_DIR = "./images/outputs"


# ============================================================================
# Device Detection and Management
# ============================================================================

def detect_available_devices() -> Dict[str, dict]:
    """Detect all available compute devices."""
    devices = {}

    # Always available: CPU
    devices["CPU"] = {
        "available": True,
        "name": "CPU",
        "device_str": "cpu",
        "description": f"CPU ({platform.processor() or 'Unknown'})",
        "memory": "System RAM",
        "dtype": torch.float32,
    }

    # Check NVIDIA CUDA
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            device_name = torch.cuda.get_device_name(i)
            memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            key = f"CUDA:{i}" if torch.cuda.device_count() > 1 else "NVIDIA CUDA"
            devices[key] = {
                "available": True,
                "name": device_name,
                "device_str": f"cuda:{i}" if torch.cuda.device_count() > 1 else "cuda",
                "description": f"{device_name} ({memory:.1f} GB)",
                "memory": f"{memory:.1f} GB",
                "dtype": torch.bfloat16,
            }

    # Check Intel XPU (Arc GPUs)
    try:
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            for i in range(torch.xpu.device_count()):
                device_name = torch.xpu.get_device_name(i)
                key = f"Intel XPU:{i}" if torch.xpu.device_count() > 1 else "Intel XPU"
                devices[key] = {
                    "available": True,
                    "name": device_name,
                    "device_str": f"xpu:{i}" if torch.xpu.device_count() > 1 else "xpu",
                    "description": f"{device_name}",
                    "memory": "Check Intel GPU",
                    "dtype": torch.float16,  # Intel XPU typically works better with fp16
                }
    except Exception as e:
        print(f"Intel XPU check failed: {e}")

    # Check for Intel Extension for PyTorch (IPEX)
    try:
        import intel_extension_for_pytorch as ipex
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            # Already handled above, but mark IPEX as available
            for key in list(devices.keys()):
                if "Intel XPU" in key:
                    devices[key]["ipex_available"] = True
                    devices[key]["description"] += " (IPEX)"
    except ImportError:
        pass

    return devices


def get_device_choices() -> List[str]:
    """Get list of available device choices for dropdown."""
    devices = detect_available_devices()
    choices = []

    # Prefer GPU options first
    for key in devices:
        if "CUDA" in key or "XPU" in key:
            choices.insert(0, key)
        else:
            choices.append(key)

    return choices


def get_device_info() -> str:
    """Get formatted device information string."""
    devices = detect_available_devices()
    lines = ["## Detected Devices\n"]

    for key, info in devices.items():
        status = "Available" if info["available"] else "Not Available"
        lines.append(f"- **{key}**: {info['description']} - {status}")

    # Add installation hints
    lines.append("\n## Installation Notes\n")

    if not any("CUDA" in k for k in devices):
        lines.append("- **NVIDIA CUDA**: Install [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) and PyTorch with CUDA support")

    if not any("XPU" in k for k in devices):
        lines.append("- **Intel XPU**: Install PyTorch with XPU support: `pip install torch --index-url https://download.pytorch.org/whl/xpu`")
        lines.append("- **Intel IPEX** (optional): `pip install intel-extension-for-pytorch`")

    return "\n".join(lines)


def clear_device_cache(device_str: str):
    """Clear cache for the specified device."""
    if "cuda" in device_str:
        torch.cuda.empty_cache()
    elif "xpu" in device_str:
        try:
            torch.xpu.empty_cache()
        except:
            pass


def get_compute_dtype(device_str: str) -> torch.dtype:
    """Get the appropriate compute dtype for a device."""
    devices = detect_available_devices()

    for key, info in devices.items():
        if info["device_str"] == device_str:
            return info["dtype"]

    # Default fallback
    if "cuda" in device_str:
        return torch.bfloat16
    elif "xpu" in device_str:
        return torch.float16
    else:
        return torch.float32


# ============================================================================
# Directory and Image Management
# ============================================================================

def ensure_directories():
    """Create necessary directories if they don't exist."""
    Path(MODELS_DIR).mkdir(exist_ok=True)
    Path(IMAGES_DIR).mkdir(exist_ok=True)
    Path(INPUT_IMAGES_DIR).mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_IMAGES_DIR).mkdir(parents=True, exist_ok=True)


def save_input_images(images: List[Image.Image], session_id: str) -> List[str]:
    """Save input images to the inputs directory."""
    ensure_directories()
    saved_paths = []

    for i, img in enumerate(images):
        filename = f"{session_id}_input_{i}.png"
        filepath = Path(INPUT_IMAGES_DIR) / filename
        img.save(filepath)
        saved_paths.append(str(filepath))
        print(f"Saved input image: {filepath}")

    return saved_paths


def save_output_images(images: List[Image.Image], session_id: str, prompt: str) -> List[str]:
    """Save output images to the outputs directory."""
    ensure_directories()
    saved_paths = []

    # Sanitize prompt for filename (take first 50 chars, remove special chars)
    safe_prompt = "".join(c if c.isalnum() or c in " -_" else "" for c in prompt[:50]).strip()
    safe_prompt = safe_prompt.replace(" ", "_")

    for i, img in enumerate(images):
        filename = f"{session_id}_{safe_prompt}_{i}.png"
        filepath = Path(OUTPUT_IMAGES_DIR) / filename
        img.save(filepath)
        saved_paths.append(str(filepath))
        print(f"Saved output image: {filepath}")

    return saved_paths


# ============================================================================
# Model Management
# ============================================================================

def get_downloaded_models() -> List[str]:
    """Get list of already downloaded models."""
    models_path = Path(MODELS_DIR)
    if not models_path.exists():
        return []

    downloaded = []
    for model_name, filename in AVAILABLE_MODELS.items():
        if (models_path / filename).exists():
            downloaded.append(model_name)
    return downloaded


def check_model_status(model_name: str) -> str:
    """Check if a model is downloaded."""
    if model_name not in AVAILABLE_MODELS:
        return "Unknown model"

    filename = AVAILABLE_MODELS[model_name]
    local_path = Path(MODELS_DIR) / filename

    if local_path.exists():
        size_gb = local_path.stat().st_size / (1024**3)
        return f"Downloaded ({size_gb:.2f} GB)"
    else:
        return "Not downloaded"


def download_model(model_name: str, progress=gr.Progress()) -> str:
    """Download a specific GGUF model."""
    if model_name not in AVAILABLE_MODELS:
        return f"Error: Unknown model {model_name}"

    filename = AVAILABLE_MODELS[model_name]
    models_path = Path(MODELS_DIR)
    models_path.mkdir(exist_ok=True)

    local_path = models_path / filename

    if local_path.exists():
        size_gb = local_path.stat().st_size / (1024**3)
        return f"Model already downloaded: {filename} ({size_gb:.2f} GB)"

    try:
        progress(0, desc=f"Starting download of {filename}...")

        downloaded_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            local_dir=MODELS_DIR,
        )

        size_gb = Path(downloaded_path).stat().st_size / (1024**3)
        return f"Successfully downloaded: {filename} ({size_gb:.2f} GB)"

    except Exception as e:
        return f"Error downloading {filename}: {str(e)}"


def delete_model(model_name: str) -> str:
    """Delete a downloaded model."""
    global current_pipeline, current_model_name, current_device

    if model_name not in AVAILABLE_MODELS:
        return f"Error: Unknown model {model_name}"

    filename = AVAILABLE_MODELS[model_name]
    local_path = Path(MODELS_DIR) / filename

    if not local_path.exists():
        return f"Model not found: {filename}"

    try:
        # Clear pipeline if this model is currently loaded
        if current_model_name == model_name:
            del current_pipeline
            current_pipeline = None
            current_model_name = None
            if current_device:
                clear_device_cache(current_device)
            current_device = None

        local_path.unlink()
        return f"Successfully deleted: {filename}"
    except Exception as e:
        return f"Error deleting {filename}: {str(e)}"


def get_all_model_status() -> str:
    """Get status of all models."""
    models_path = Path(MODELS_DIR)
    lines = ["## Downloaded Models\n"]

    total_size = 0
    downloaded_count = 0

    # Check base pipeline
    base_pipeline_path = Path(BASE_PIPELINE_DIR)
    if base_pipeline_path.exists() and (base_pipeline_path / "model_index.json").exists():
        # Calculate base pipeline size
        base_size = sum(f.stat().st_size for f in base_pipeline_path.rglob("*") if f.is_file()) / (1024**3)
        lines.append(f"- **Base Pipeline**: {base_size:.2f} GB (tokenizer, scheduler, etc.)")
        total_size += base_size
    else:
        lines.append("- **Base Pipeline**: *Not downloaded*")

    lines.append("")  # Empty line separator

    # Check GGUF models
    for model_name, filename in AVAILABLE_MODELS.items():
        local_path = models_path / filename
        if local_path.exists():
            size_gb = local_path.stat().st_size / (1024**3)
            total_size += size_gb
            downloaded_count += 1
            lines.append(f"- **{model_name}**: {size_gb:.2f} GB")

    if downloaded_count == 0:
        lines.append("*No GGUF models downloaded yet*")
    else:
        lines.append(f"\n**Total**: {downloaded_count} GGUF models + base, {total_size:.2f} GB")

    return "\n".join(lines)


def get_model_path(model_filename: str, models_dir: str = "./models") -> str:
    """Download or locate the GGUF model file."""
    models_path = Path(models_dir)
    models_path.mkdir(exist_ok=True)

    local_path = models_path / model_filename

    if local_path.exists():
        print(f"Using local model: {local_path}")
        return str(local_path)

    print(f"Downloading {model_filename} from {REPO_ID}...")
    downloaded_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=model_filename,
        local_dir=models_dir,
    )
    return downloaded_path


# ============================================================================
# Pipeline Loading
# ============================================================================

def load_pipeline(model_name: str, device_choice: str):
    """Load the pipeline with the specified GGUF model and device."""
    global current_pipeline, current_model_name, current_device

    devices = detect_available_devices()

    if device_choice not in devices:
        raise ValueError(f"Device {device_choice} not available")

    device_info = devices[device_choice]
    device_str = device_info["device_str"]
    compute_dtype = device_info["dtype"]

    # Check if we can reuse the current pipeline
    if (current_model_name == model_name and
        current_device == device_str and
        current_pipeline is not None):
        return current_pipeline

    # Clear previous pipeline from memory
    if current_pipeline is not None:
        del current_pipeline
        current_pipeline = None
        if current_device:
            clear_device_cache(current_device)

    from diffusers import QwenImageTransformer2DModel, GGUFQuantizationConfig, QwenImageEditPlusPipeline

    model_filename = AVAILABLE_MODELS[model_name]
    model_path = get_model_path(model_filename)

    print(f"Loading transformer from {model_path}...")
    print(f"Using device: {device_choice} ({device_str})")
    print(f"Compute dtype: {compute_dtype}")

    # Apply Intel IPEX optimizations if available
    ipex_enabled = False
    if "xpu" in device_str:
        try:
            import intel_extension_for_pytorch as ipex
            ipex_enabled = True
            print("Intel Extension for PyTorch (IPEX) detected")
        except ImportError:
            print("IPEX not installed, using native PyTorch XPU support")

    # Load the transformer with GGUF quantization
    transformer = QwenImageTransformer2DModel.from_single_file(
        model_path,
        quantization_config=GGUFQuantizationConfig(compute_dtype=compute_dtype),
        torch_dtype=compute_dtype,
        config=BASE_MODEL_REPO,
        subfolder="transformer",
    )

    print("Loading full pipeline...")

    # Determine pipeline source - use local directory if available
    local_pipeline_path = Path(BASE_PIPELINE_DIR)
    if local_pipeline_path.exists() and (local_pipeline_path / "model_index.json").exists():
        pipeline_source = str(local_pipeline_path)
        print(f"Using local pipeline: {pipeline_source}")
    else:
        pipeline_source = BASE_MODEL_REPO
        print(f"Using HuggingFace pipeline: {pipeline_source}")

    # Load the full pipeline with the quantized transformer
    # Auto-retry with force_download if cache is corrupted
    pipeline = None

    for attempt in range(2):
        try:
            force_download = (attempt > 0)
            if force_download:
                print("Retrying with force_download=True to fix corrupted cache...")

            # If using local path and force_download needed, download fresh to local dir
            if force_download and pipeline_source != BASE_MODEL_REPO:
                from huggingface_hub import snapshot_download
                snapshot_download(
                    repo_id=BASE_MODEL_REPO,
                    local_dir=BASE_PIPELINE_DIR,
                    ignore_patterns=["transformer/*"],
                    force_download=True,
                )

            pipeline = QwenImageEditPlusPipeline.from_pretrained(
                pipeline_source,
                transformer=transformer,
                torch_dtype=compute_dtype,
                force_download=force_download if pipeline_source == BASE_MODEL_REPO else False,
            )
            break  # Success, exit retry loop
        except (OSError, EnvironmentError) as e:
            error_msg = str(e).lower()
            # Check if it's a corrupted file / consistency check error
            if attempt == 0 and ("consistency check failed" in error_msg or
                                  "corrupted" in error_msg or
                                  "size" in error_msg and "should be" in error_msg):
                print(f"Detected corrupted cache file: {e}")
                continue  # Retry with force_download
            else:
                raise  # Re-raise if not a cache error or already retried

    if pipeline is None:
        raise RuntimeError("Failed to load pipeline after retries")

    # Move to device and enable optimizations
    if device_str == "cpu":
        # CPU mode - no special handling needed
        pass
    elif "cuda" in device_str:
        # NVIDIA CUDA - check VRAM to decide loading strategy
        try:
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except:
            vram_gb = 8  # Assume 8GB if can't detect

        print(f"Detected VRAM: {vram_gb:.1f} GB")

        # Clear any existing CUDA cache
        torch.cuda.empty_cache()

        if vram_gb >= 24:
            # Enough VRAM for full model - load everything to GPU
            pipeline = pipeline.to(device_str)
            print(f"Pipeline moved to {device_str} (full GPU mode)")
        else:
            # Limited VRAM (8-12GB typical) - hybrid mode
            # Keep text encoder on CPU (uses ~15GB RAM), move only transformer to GPU
            print(f"Using hybrid mode: text encoder on CPU (RAM), transformer on GPU")

            # First, remove ALL accelerate hooks - they cause automatic GPU transfers
            # Need to do this recursively on all submodules
            def remove_all_hooks(module):
                """Recursively remove all accelerate hooks from a module."""
                if hasattr(module, '_hf_hook'):
                    delattr(module, '_hf_hook')
                if hasattr(module, '_old_forward'):
                    module.forward = module._old_forward
                    delattr(module, '_old_forward')
                for child in module.children():
                    remove_all_hooks(child)

            try:
                for name, component in pipeline.components.items():
                    if component is not None and hasattr(component, 'children'):
                        remove_all_hooks(component)
                print("  Removed accelerate hooks from all components")
            except Exception as e:
                print(f"  Warning: Could not remove all hooks: {e}")

            # Move transformer to GPU - this is where most computation happens
            if hasattr(pipeline, 'transformer') and pipeline.transformer is not None:
                pipeline.transformer = pipeline.transformer.to(device_str)
                print(f"  Transformer moved to {device_str}")

            # Keep VAE on CPU - it uses 3D convolutions that don't have CUDA float32 kernels
            # The VAE is small and fast even on CPU
            if hasattr(pipeline, 'vae') and pipeline.vae is not None:
                pipeline.vae = pipeline.vae.to("cpu").float()
                print(f"  VAE kept on CPU (3D conv compatibility)")

            # Explicitly keep text encoder on CPU with float32 for stability
            if hasattr(pipeline, 'text_encoder') and pipeline.text_encoder is not None:
                pipeline.text_encoder = pipeline.text_encoder.to("cpu").float()
                pipeline.text_encoder.eval()
                # Disable gradient computation for text encoder
                for param in pipeline.text_encoder.parameters():
                    param.requires_grad = False
                print(f"  Text encoder kept on CPU (~15GB RAM)")

            # Enable memory optimizations
            try:
                pipeline.enable_attention_slicing(slice_size="auto")
                print("  Enabled attention slicing")
            except:
                pass

            # Mark this pipeline as using hybrid mode
            pipeline._hybrid_mode = True
            pipeline._gpu_device = device_str

            # Store the GPU device and dtype for the pipeline
            gpu_device = device_str
            gpu_dtype = compute_dtype  # bfloat16 for CUDA

            # Patch encode_prompt to run on CPU and move results to GPU with correct dtype
            original_encode_prompt = pipeline.encode_prompt

            def patched_encode_prompt(*args, **kwargs):
                kwargs['device'] = "cpu"
                result = original_encode_prompt(*args, **kwargs)
                if isinstance(result, tuple):
                    return tuple(r.to(device=gpu_device, dtype=gpu_dtype) if r is not None and hasattr(r, 'to') else r for r in result)
                return result.to(device=gpu_device, dtype=gpu_dtype) if hasattr(result, 'to') else result

            pipeline.encode_prompt = patched_encode_prompt
            print("  Patched encode_prompt for hybrid mode")

            # Patch prepare_latents to move VAE outputs to GPU with correct dtype
            original_prepare_latents = pipeline.prepare_latents

            def patched_prepare_latents(*args, **kwargs):
                result = original_prepare_latents(*args, **kwargs)
                # Result is (latents, image_latents) - move both to GPU with correct dtype
                if isinstance(result, tuple):
                    return tuple(r.to(device=gpu_device, dtype=gpu_dtype) if r is not None and hasattr(r, 'to') else r for r in result)
                return result.to(device=gpu_device, dtype=gpu_dtype) if hasattr(result, 'to') else result

            pipeline.prepare_latents = patched_prepare_latents
            print("  Patched prepare_latents for hybrid mode")

            # Patch _encode_vae_image to keep VAE on CPU but return GPU tensors with correct dtype
            if hasattr(pipeline, '_encode_vae_image'):
                original_encode_vae = pipeline._encode_vae_image

                def patched_encode_vae(*args, **kwargs):
                    result = original_encode_vae(*args, **kwargs)
                    return result.to(device=gpu_device, dtype=gpu_dtype) if hasattr(result, 'to') else result

                pipeline._encode_vae_image = patched_encode_vae
                print("  Patched _encode_vae_image for hybrid mode")
    elif "xpu" in device_str:
        # Intel XPU
        pipeline = pipeline.to(device_str)

        # Apply IPEX optimization if available
        if ipex_enabled:
            try:
                # IPEX optimization for inference
                pipeline.transformer = ipex.optimize(pipeline.transformer)
                print("Applied IPEX optimization to transformer")
            except Exception as e:
                print(f"IPEX optimization failed: {e}")

    current_pipeline = pipeline
    current_model_name = model_name
    current_device = device_str

    print(f"Pipeline loaded successfully with {model_name} on {device_choice}")
    return pipeline


# ============================================================================
# Image Processing
# ============================================================================

def process_input_images(images) -> List[Image.Image]:
    """Process input images from Gradio gallery."""
    pil_images = []
    if images is not None:
        for item in images:
            try:
                if isinstance(item, tuple):
                    img = item[0]
                else:
                    img = item

                if isinstance(img, Image.Image):
                    pil_images.append(img.convert("RGB"))
                elif isinstance(img, str):
                    pil_images.append(Image.open(img).convert("RGB"))
                elif hasattr(img, "name"):
                    pil_images.append(Image.open(img.name).convert("RGB"))
            except Exception as e:
                print(f"Error processing image: {e}")
                continue
    return pil_images


# ============================================================================
# Inference
# ============================================================================

def infer(
    images,
    prompt: str,
    model_name: str,
    device_choice: str,
    seed: int = 42,
    randomize_seed: bool = False,
    true_guidance_scale: float = 4.0,
    num_inference_steps: int = 40,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_images_per_prompt: int = 1,
    progress=gr.Progress(track_tqdm=True),
) -> Tuple[List[Image.Image], int, str]:
    """Generate edited images using Qwen-Image-Edit pipeline."""

    status_msg = ""

    # Generate session ID for this generation run
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        # Load pipeline
        status_msg = f"Loading model: {model_name} on {device_choice}..."
        pipe = load_pipeline(model_name, device_choice)

        # Process input images
        pil_images = process_input_images(images)

        # Save input images
        if len(pil_images) > 0:
            input_paths = save_input_images(pil_images, session_id)
            print(f"Saved {len(input_paths)} input images")

        if randomize_seed:
            seed = random.randint(0, MAX_SEED)

        # Create generator on appropriate device
        devices = detect_available_devices()
        device_str = devices[device_choice]["device_str"]

        # Generator should be on CPU for offloading modes
        # The pipeline will handle moving tensors as needed
        generator = torch.Generator(device="cpu").manual_seed(seed)

        # Handle default dimensions
        if height == 256 and width == 256:
            height, width = None, None

        print(f"Generating with prompt: '{prompt}'")
        print(f"Device: {device_choice}, Seed: {seed}, Steps: {num_inference_steps}, Guidance: {true_guidance_scale}")

        # Run inference
        with torch.inference_mode():
            output = pipe(
                image=pil_images if len(pil_images) > 0 else None,
                prompt=prompt,
                height=height,
                width=width,
                negative_prompt=" ",
                num_inference_steps=num_inference_steps,
                generator=generator,
                true_cfg_scale=true_guidance_scale,
                num_images_per_prompt=num_images_per_prompt,
            )

        # Save output images
        output_paths = save_output_images(output.images, session_id, prompt)

        status_msg = f"Generation complete! Device: {device_choice}, Seed: {seed}\nSaved {len(output_paths)} output images to {OUTPUT_IMAGES_DIR}"
        return output.images, seed, status_msg

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return [], seed, error_msg


# ============================================================================
# Gradio UI
# ============================================================================

def create_ui():
    """Create the Gradio interface."""

    # Detect available devices at startup
    available_devices = get_device_choices()
    default_device = available_devices[0] if available_devices else "CPU"

    with gr.Blocks(title="Qwen-Image-Edit-2511 Local") as demo:
        with gr.Column(elem_id="col-container"):
            gr.Markdown("# Qwen-Image-Edit-2511 Local Interface")
            gr.Markdown(
                "Local interface for [Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) "
                "using [GGUF quantized models](https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF). "
                "Supports **NVIDIA CUDA**, **Intel XPU (Arc)**, and **CPU**."
            )

            with gr.Tabs():
                # Main Image Editing Tab
                with gr.TabItem("Image Editing"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            # Device selection
                            device_dropdown = gr.Dropdown(
                                choices=available_devices,
                                value=default_device,
                                label="Compute Device",
                                info="Select GPU or CPU for inference"
                            )

                            # Model selection
                            model_dropdown = gr.Dropdown(
                                choices=list(AVAILABLE_MODELS.keys()),
                                value="Q4_K_M (13.1 GB) - Recommended",
                                label="Select Model",
                                info="Choose quantization level. Smaller = faster but lower quality."
                            )

                            # Model status indicator
                            model_status_text = gr.Textbox(
                                label="Model Status",
                                value=check_model_status("Q4_K_M (13.1 GB) - Recommended"),
                                interactive=False,
                            )

                            # Input images
                            input_images = gr.Gallery(
                                label="Input Images",
                                show_label=True,
                                elem_id="input_gallery",
                                columns=2,
                                rows=2,
                                height="auto",
                                object_fit="contain",
                                type="pil",
                                interactive=True,
                            )

                            # Prompt
                            prompt = gr.Textbox(
                                label="Edit Instruction",
                                placeholder="Describe what you want to change (e.g., 'make the sky sunset colored', 'add a hat to the person')",
                                lines=3,
                            )

                            # Run button
                            run_button = gr.Button("Generate Edit", variant="primary", size="lg")

                        with gr.Column(scale=1):
                            # Output
                            result = gr.Gallery(
                                label="Result",
                                show_label=True,
                                elem_id="result_gallery",
                                columns=2,
                                rows=2,
                                height="auto",
                                object_fit="contain",
                                type="pil",
                            )

                            # Status
                            status = gr.Textbox(label="Status", interactive=False, lines=3)

                            # Seed output
                            seed_output = gr.Number(label="Used Seed", interactive=False)

                    # Advanced settings
                    with gr.Accordion("Advanced Settings", open=False):
                        with gr.Row():
                            seed = gr.Slider(
                                label="Seed",
                                minimum=0,
                                maximum=MAX_SEED,
                                step=1,
                                value=42,
                            )
                            randomize_seed = gr.Checkbox(
                                label="Randomize seed",
                                value=True,
                            )

                        with gr.Row():
                            true_guidance_scale = gr.Slider(
                                label="True Guidance Scale",
                                minimum=1.0,
                                maximum=10.0,
                                step=0.1,
                                value=4.0,
                                info="Higher values follow the prompt more strictly"
                            )
                            num_inference_steps = gr.Slider(
                                label="Inference Steps",
                                minimum=1,
                                maximum=100,
                                step=1,
                                value=40,
                                info="More steps = better quality but slower"
                            )

                        with gr.Row():
                            height = gr.Slider(
                                label="Output Height",
                                minimum=256,
                                maximum=2048,
                                step=8,
                                value=256,
                                info="Set to 256 for auto"
                            )
                            width = gr.Slider(
                                label="Output Width",
                                minimum=256,
                                maximum=2048,
                                step=8,
                                value=256,
                                info="Set to 256 for auto"
                            )

                        num_images = gr.Slider(
                            label="Number of Images",
                            minimum=1,
                            maximum=4,
                            step=1,
                            value=1,
                        )

                # Model Management Tab
                with gr.TabItem("Model Manager"):
                    gr.Markdown("## Download & Manage Models")
                    gr.Markdown(
                        "Download GGUF models locally for offline use. "
                        "Models are saved to the `./models` directory."
                    )

                    with gr.Row():
                        with gr.Column(scale=1):
                            # Model selection for download
                            download_model_dropdown = gr.Dropdown(
                                choices=list(AVAILABLE_MODELS.keys()),
                                value="Q4_K_M (13.1 GB) - Recommended",
                                label="Select Model to Download",
                                info="Choose a model quantization to download"
                            )

                            with gr.Row():
                                download_button = gr.Button(
                                    "Download Model",
                                    variant="primary",
                                )
                                delete_button = gr.Button(
                                    "Delete Model",
                                    variant="stop",
                                )

                            download_status = gr.Textbox(
                                label="Download Status",
                                interactive=False,
                                lines=3,
                            )

                        with gr.Column(scale=1):
                            # Downloaded models list
                            downloaded_models_md = gr.Markdown(
                                value=get_all_model_status(),
                                label="Downloaded Models",
                            )

                            refresh_button = gr.Button("Refresh Status")

                    # Model info table
                    gr.Markdown("""
                    ## Available Models

                    | Model | Size | Quality | Best For |
                    |-------|------|---------|----------|
                    | Q2_K | 7.22 GB | Lower | Testing, low VRAM |
                    | Q3_K_S | 9.04 GB | Fair | Budget GPUs |
                    | Q3_K_M | 9.74 GB | Fair | Budget GPUs |
                    | Q3_K_L | 10.4 GB | Fair+ | Budget GPUs |
                    | Q4_0 | 11.9 GB | Good | General use |
                    | Q4_K_S | 12.3 GB | Good | General use |
                    | Q4_1 | 12.8 GB | Good | General use |
                    | **Q4_K_M** | **13.1 GB** | **Good** | **Recommended** |
                    | Q5_K_S | 14.3 GB | Better | Quality focus |
                    | Q5_0 | 14.4 GB | Better | Quality focus |
                    | Q5_K_M | 15 GB | Better | Quality focus |
                    | Q5_1 | 15.4 GB | Better | Quality focus |
                    | Q6_K | 16.8 GB | High | High quality |
                    | Q8_0 | 21.8 GB | Very High | Near-original |
                    | BF16 | 40.9 GB | Original | Maximum quality |
                    | F16 | 40.9 GB | Original | Maximum quality |
                    """)

                # Device Info Tab
                with gr.TabItem("Devices"):
                    device_info_md = gr.Markdown(value=get_device_info())

                    refresh_devices_btn = gr.Button("Refresh Device Info")

                    gr.Markdown("""
                    ## Device Support

                    ### NVIDIA CUDA
                    - **Best performance** for most users
                    - Requires NVIDIA GPU with CUDA support
                    - Install: `pip install torch --index-url https://download.pytorch.org/whl/cu121`

                    ### Intel XPU (Arc GPUs)
                    - Supports Intel Arc A-series GPUs (A770, A750, etc.)
                    - Supports Intel Core Ultra integrated GPUs
                    - Install PyTorch with XPU: `pip install torch --index-url https://download.pytorch.org/whl/xpu`
                    - Optional IPEX for better performance: `pip install intel-extension-for-pytorch`

                    ### CPU
                    - Works on any system
                    - **Very slow** for image generation
                    - Use smaller models (Q2_K, Q3_K_M) for faster results

                    ## Performance Tips

                    | Device | Recommended Model | Expected Speed |
                    |--------|-------------------|----------------|
                    | NVIDIA RTX 3090/4090 | Q4_K_M - Q8_0 | Fast |
                    | NVIDIA RTX 3060/4060 | Q3_K_M - Q4_K_M | Moderate |
                    | Intel Arc A770 | Q4_K_M | Moderate |
                    | Intel Arc A750 | Q3_K_M - Q4_K_M | Moderate |
                    | CPU | Q2_K - Q3_K_M | Very Slow |
                    """)

                # Help Tab
                with gr.TabItem("Help"):
                    gr.Markdown("""
                    ## About Qwen-Image-Edit-2511

                    This is a local interface for the Qwen-Image-Edit-2511 model, a powerful
                    image generation and editing model from Alibaba's Qwen team.

                    ### Features
                    - **Image Editing**: Modify existing images based on text instructions
                    - **Image Generation**: Create new images from text descriptions
                    - **Multiple Quantizations**: Choose from various GGUF quantization levels
                    - **Multi-Device Support**: NVIDIA CUDA, Intel XPU, and CPU

                    ### Requirements
                    - **GPU**: NVIDIA GPU with CUDA or Intel Arc GPU recommended
                    - **VRAM**: 8-24 GB depending on model choice
                    - **Disk Space**: 7-41 GB per model

                    ### Tips for Best Results
                    1. **Be specific**: "Add a red hat to the person" works better than "add hat"
                    2. **Use guidance scale 4-6**: Higher values follow prompts more strictly
                    3. **More steps = better quality**: 40-50 steps is a good balance
                    4. **Start with Q4_K_M**: Best balance of quality and speed

                    ### Troubleshooting
                    - **Out of memory**: Try a smaller quantization (Q3_K_M, Q2_K)
                    - **Slow generation**: Reduce inference steps or try a smaller model
                    - **Poor quality**: Increase guidance scale or try a larger model
                    - **Intel XPU not detected**: Ensure you have the XPU PyTorch build installed

                    ### Image Storage
                    - Input images: `./images/inputs/`
                    - Output images: `./images/outputs/`

                    ### Links
                    - [Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)
                    - [GGUF Models](https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF)
                    - [Qwen-Image GitHub](https://github.com/QwenLM/Qwen-Image)
                    - [Intel Extension for PyTorch](https://github.com/intel/intel-extension-for-pytorch)
                    """)

            # Event handlers for Image Editing tab
            model_dropdown.change(
                fn=check_model_status,
                inputs=[model_dropdown],
                outputs=[model_status_text],
            )

            gr.on(
                triggers=[run_button.click, prompt.submit],
                fn=infer,
                inputs=[
                    input_images,
                    prompt,
                    model_dropdown,
                    device_dropdown,
                    seed,
                    randomize_seed,
                    true_guidance_scale,
                    num_inference_steps,
                    height,
                    width,
                    num_images,
                ],
                outputs=[result, seed_output, status],
            )

            # Event handlers for Model Manager tab
            download_button.click(
                fn=download_model,
                inputs=[download_model_dropdown],
                outputs=[download_status],
            ).then(
                fn=get_all_model_status,
                outputs=[downloaded_models_md],
            ).then(
                fn=check_model_status,
                inputs=[model_dropdown],
                outputs=[model_status_text],
            )

            delete_button.click(
                fn=delete_model,
                inputs=[download_model_dropdown],
                outputs=[download_status],
            ).then(
                fn=get_all_model_status,
                outputs=[downloaded_models_md],
            ).then(
                fn=check_model_status,
                inputs=[model_dropdown],
                outputs=[model_status_text],
            )

            refresh_button.click(
                fn=get_all_model_status,
                outputs=[downloaded_models_md],
            ).then(
                fn=check_model_status,
                inputs=[model_dropdown],
                outputs=[model_status_text],
            )

            # Event handlers for Devices tab
            refresh_devices_btn.click(
                fn=get_device_info,
                outputs=[device_info_md],
            )

    return demo


def download_base_pipeline():
    """Download/cache the base pipeline components (tokenizer, text encoder, scheduler, etc.)."""
    from huggingface_hub import snapshot_download

    print("\nChecking base pipeline components...")
    print(f"Location: {BASE_PIPELINE_DIR}")

    # Ensure models directory exists
    Path(MODELS_DIR).mkdir(exist_ok=True)

    # Try up to 2 times - second time with force_download if cache is corrupted
    for attempt in range(2):
        try:
            force_download = (attempt > 0)
            if force_download:
                print("Retrying with force_download=True to fix corrupted cache...")

            # Download all pipeline components except the transformer (we use GGUF for that)
            # This includes: tokenizer, text_encoder (with weights!), scheduler, vae config, etc.
            # Only exclude transformer/* folder since we load that from GGUF
            # Store in local models directory for clear organization
            snapshot_download(
                repo_id=BASE_MODEL_REPO,
                local_dir=BASE_PIPELINE_DIR,
                ignore_patterns=["transformer/*"],
                force_download=force_download,
            )
            print("Base pipeline components ready.")
            return True

        except (OSError, EnvironmentError) as e:
            error_msg = str(e).lower()
            # Check if it's a corrupted file / consistency check error
            if attempt == 0 and ("consistency check failed" in error_msg or
                                  "corrupted" in error_msg or
                                  "size" in error_msg and "should be" in error_msg):
                print(f"Detected corrupted cache file: {e}")
                continue  # Retry with force_download
            else:
                print(f"Note: Could not pre-download pipeline components: {e}")
                print("Components will be downloaded on first run.")
                return False

        except Exception as e:
            print(f"Note: Could not pre-download pipeline components: {e}")
            print("Components will be downloaded on first run.")
            return False

    return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Qwen-Image-Edit-2511 Local Interface")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="Host to bind to (use 0.0.0.0 for network access)")
    parser.add_argument("--port", type=int, default=7860,
                        help="Port to bind to")
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio link")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip downloading base pipeline on startup")
    args = parser.parse_args()

    # Print detected devices at startup
    print("\n" + "="*50)
    print("Qwen-Image-Edit-2511 Local Interface")
    print("="*50)
    devices = detect_available_devices()
    print("\nDetected devices:")
    for name, info in devices.items():
        print(f"  - {name}: {info['description']}")
    print("="*50)

    # Download base pipeline components on startup
    if not args.skip_download:
        download_base_pipeline()

    if args.host == "0.0.0.0":
        print("\nNetwork mode enabled - accessible from other devices")
        print(f"Local URL: http://127.0.0.1:{args.port}")
        # Try to get local IP
        try:
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            print(f"Network URL: http://{local_ip}:{args.port}")
        except:
            pass
    else:
        print(f"\nLocal URL: http://{args.host}:{args.port}")

    print("="*50 + "\n")

    demo = create_ui()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )
