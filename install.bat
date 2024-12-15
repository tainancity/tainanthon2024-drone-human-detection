@echo off
chcp 65001 >nul
REM Check if Python is installed
WHERE pip >nul 2>&1
IF %ERRORLEVEL% NEQ 0 ECHO [101mPython is not installed. Please install Python first...[0m && PAUSE && EXIT /B 1

REM Install CUDA Toolkit 11.8
WHERE nvcc >nul 2>&1
IF %ERRORLEVEL% EQU 0 (
    ECHO [92mCUDA is already installed...[0m
    goto PIP
    )
IF NOT %ERRORLEVEL% EQU 0 (
    set /p "cuda=CUDA Toolkit isn't installed, install now?(y/n):"
)
IF %cuda% EQU n goto PIP
IF %cuda% EQU N goto PIP

ECHO [92mStart installing CUDA Toolkit 11.8...[0m
start /wait cuda_11.8.0_windows_network.exe

:PIP
set /p "pip=Install all the required Python libraries？(y/n):"

IF %pip% EQU n PAUSE && EXIT /B 1
IF %pip% EQU N PAUSE && EXIT /B 1

ECHO [92mStart installing libraries...[0m
REM Install PyTorch, TorchVision and Torchaudio with CUDA support
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

REM Install other dependencies
pip install -r requirements.txt

echo [92mInstallation is complete...[0m
PAUSE
