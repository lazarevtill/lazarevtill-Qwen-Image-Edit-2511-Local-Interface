@echo off
chcp 65001 >nul 2>&1
setlocal EnableDelayedExpansion

:: Default settings
set "HOST=127.0.0.1"
set "PORT=7860"
set "FOUND_CUDA=0"
set "FOUND_XPU=0"
set "DEVICE_TYPE=cpu"
set "TORCH_INDEX_URL="
set "SKIP_SETUP=0"

:: Parse command line arguments
:parse_args
if "%~1"=="" goto main
if /i "%~1"=="--share" set "HOST=0.0.0.0"
if /i "%~1"=="--public" set "HOST=0.0.0.0"
if /i "%~1"=="--local" set "HOST=127.0.0.1"
if /i "%~1"=="--port" set "PORT=%~2" & shift
if /i "%~1"=="--reset" set "DO_RESET=1"
if /i "%~1"=="--cuda" set "FORCE_DEVICE=cuda"
if /i "%~1"=="--xpu" set "FORCE_DEVICE=xpu"
if /i "%~1"=="--cpu" set "FORCE_DEVICE=cpu"
shift
goto parse_args

:main
echo ==================================================
echo   Qwen-Image-Edit-2511 Local Interface
echo ==================================================
echo.

:: Handle --reset flag
if "!DO_RESET!"=="1" (
    echo [RESET] Removing virtual environment...
    if exist .venv rmdir /s /q .venv 2>nul
    echo       Done. Continuing with fresh install...
    echo.
)

:: Check if already installed
if not exist ".venv\Scripts\python.exe" goto do_setup

echo [CHECK] Checking installation...
call .venv\Scripts\activate.bat 2>nul

:: Check if key packages are installed
.venv\Scripts\python.exe -c "import gradio; import diffusers; import torch" >nul 2>&1
if !errorlevel! equ 0 set "SKIP_SETUP=1"

if "!SKIP_SETUP!"=="1" (
    echo [OK] All packages installed. Starting application...
    goto start_application
)

echo [INFO] Some packages missing, continuing with setup...
echo.

:do_setup
echo [1/6] Checking for Python...
where python >nul 2>&1
if !errorlevel! neq 0 (
    echo [ERROR] Python not found!
    echo         Please install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set "PYTHON_VERSION=%%i"
echo       Found Python %PYTHON_VERSION%

:: Create virtual environment
echo.
echo [2/6] Creating virtual environment...
if exist .venv (
    echo       Virtual environment exists, reusing it.
) else (
    python -m venv .venv
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to create virtual environment
        echo.
        echo TIP: If you have issues, try running:
        echo      setup.bat --reset
        pause
        exit /b 1
    )
    echo       Virtual environment created.
)

:: Activate virtual environment
echo.
echo [3/6] Activating virtual environment...
call .venv\Scripts\activate.bat 2>nul
echo       Activated.

:: Upgrade pip
echo.
echo [4/6] Upgrading pip...
python -m pip install --upgrade pip --quiet
echo       Done.

:: Detect hardware
echo.
echo [5/6] Detecting hardware...
echo.

:: Check for NVIDIA GPU
echo       Checking for NVIDIA GPU...
nvidia-smi >nul 2>&1
if !errorlevel! equ 0 (
    for /f "tokens=*" %%i in ('nvidia-smi -L 2^>nul') do (
        echo       [FOUND] %%i
        set "FOUND_CUDA=1"
    )
)

:: Check for Intel GPU using PowerShell
echo       Checking for Intel GPU...
for /f "tokens=*" %%i in ('powershell -NoProfile -Command "Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match 'Intel' -and $_.Name -match 'Arc|Iris|UHD|Xe' } | Select-Object -ExpandProperty Name" 2^>nul') do (
    if not "%%i"=="" (
        echo       [FOUND] Intel: %%i
        set "FOUND_XPU=1"
    )
)

:: Determine which device to use
if defined FORCE_DEVICE (
    set "DEVICE_TYPE=!FORCE_DEVICE!"
    echo.
    echo       [FORCED] Using !FORCE_DEVICE! as requested
) else if "!FOUND_CUDA!"=="1" (
    set "DEVICE_TYPE=cuda"
) else if "!FOUND_XPU!"=="1" (
    set "DEVICE_TYPE=xpu"
) else (
    set "DEVICE_TYPE=cpu"
    echo       [INFO] No supported GPU found, using CPU mode.
    echo              Note: CPU is very slow for image generation.
)

:: Set PyTorch index URL based on device
if "!DEVICE_TYPE!"=="cuda" set "TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124"
if "!DEVICE_TYPE!"=="xpu" set "TORCH_INDEX_URL=https://download.pytorch.org/whl/xpu"

echo.
echo ==================================================
echo   Available GPUs:
if "!FOUND_CUDA!"=="1" echo     - NVIDIA CUDA
if "!FOUND_XPU!"=="1" echo     - Intel XPU
echo     - CPU (always available)
echo.
echo   Selected Device: !DEVICE_TYPE!
echo ==================================================
echo.

echo [6/6] Installing packages...
echo.

:: Install PyTorch based on detected hardware
echo       Installing PyTorch for !DEVICE_TYPE!...
if "!DEVICE_TYPE!"=="cuda" (
    :: Use --index-url to prioritize CUDA wheels over CPU-only from PyPI
    pip install torch torchvision torchaudio --index-url !TORCH_INDEX_URL!
    if !errorlevel! neq 0 (
        echo       [NOTE] Trying CUDA 12.1...
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        if !errorlevel! neq 0 (
            echo       [NOTE] Trying PyTorch nightly with CUDA...
            pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124
            if !errorlevel! neq 0 (
                echo       [WARNING] Could not install CUDA PyTorch. Your Python version may not be supported.
                echo       [WARNING] Consider using Python 3.10-3.12 for CUDA support.
                echo       [NOTE] Falling back to CPU-only PyTorch...
                pip install torch torchvision torchaudio
            )
        )
    )
    goto install_deps
)
if "!DEVICE_TYPE!"=="xpu" (
    :: Use --index-url to prioritize XPU wheels
    pip install torch torchvision torchaudio --index-url !TORCH_INDEX_URL!
    if !errorlevel! neq 0 (
        echo       [NOTE] Trying with extra-index-url...
        pip install torch torchvision torchaudio --extra-index-url !TORCH_INDEX_URL!
    )
    echo.
    echo       Installing Intel Extension for PyTorch...
    pip install intel-extension-for-pytorch 2>nul
    goto install_deps
)
pip install torch torchvision torchaudio

:install_deps
echo.
echo       Installing core dependencies...
pip install gradio numpy pillow huggingface_hub

echo.
echo       Installing ML packages...
pip install diffusers transformers accelerate sentencepiece

echo.
echo       Installing GGUF support...
pip install gguf

:: Install bitsandbytes for 8-bit quantization (CUDA only, helps with limited VRAM)
if "!DEVICE_TYPE!"=="cuda" (
    echo.
    echo       Installing bitsandbytes for 8-bit quantization...
    pip install bitsandbytes-windows 2>nul
    if !errorlevel! neq 0 (
        echo       [NOTE] bitsandbytes not installed - 8-bit quantization unavailable
        echo              This is optional - needed only for GPUs with less than 24GB VRAM
    )
)

echo.
echo ==================================================
echo   Installation Complete!
echo   Device: !DEVICE_TYPE!
echo ==================================================
echo.

:start_application
echo ==================================================
echo   Starting Qwen-Image-Edit-2511
echo   URL: http://!HOST!:!PORT!
if "!HOST!"=="0.0.0.0" echo   [Network mode - accessible from other devices]
echo ==================================================
echo.
echo   Press Ctrl+C to stop the server
echo.
echo   TIP: If something goes wrong, run: setup.bat --reset
echo   TIP: Force device with: setup.bat --cuda or --xpu or --cpu
echo.

python app.py --host !HOST! --port !PORT!

echo.
echo   Server stopped.
pause
