@echo off
chcp 65001 >nul 2>&1
setlocal EnableDelayedExpansion

:: Default settings
set "HOST=127.0.0.1"
set "PORT=7860"
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

:: Check for NVIDIA GPU using nvidia-smi
echo       Checking for NVIDIA GPU...
nvidia-smi --query-gpu=name --format=csv,noheader >nul 2>&1
if !errorlevel! equ 0 (
    for /f "tokens=*" %%i in ('nvidia-smi --query-gpu=name --format=csv,noheader 2^>nul') do (
        echo       [FOUND] NVIDIA: %%i
    )
    set "DEVICE_TYPE=cuda"
    set "TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121"
    goto show_config
)

:: Check for Intel GPU using PowerShell
echo       Checking for Intel GPU...
for /f "tokens=*" %%i in ('powershell -NoProfile -Command "Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match 'Intel' } | Select-Object -ExpandProperty Name" 2^>nul') do (
    if not "%%i"=="" (
        echo       [FOUND] Intel: %%i
        set "DEVICE_TYPE=xpu"
        set "TORCH_INDEX_URL=https://download.pytorch.org/whl/xpu"
    )
)
if "!DEVICE_TYPE!"=="xpu" goto show_config

echo       [INFO] No GPU found, using CPU mode.
echo              Note: CPU is very slow for image generation.
set "DEVICE_TYPE=cpu"
set "TORCH_INDEX_URL="

:show_config
echo.
echo ==================================================
echo   Detected Device: !DEVICE_TYPE!
echo ==================================================
echo.

echo [6/6] Installing packages...
echo.

:: Install PyTorch based on detected hardware
echo       Installing PyTorch for !DEVICE_TYPE!...
if "!DEVICE_TYPE!"=="cuda" (
    pip install torch torchvision torchaudio --index-url !TORCH_INDEX_URL!
    goto install_deps
)
if "!DEVICE_TYPE!"=="xpu" (
    pip install torch torchvision torchaudio --index-url !TORCH_INDEX_URL!
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
echo.

python app.py --host !HOST! --port !PORT!

echo.
echo   Server stopped.
pause
