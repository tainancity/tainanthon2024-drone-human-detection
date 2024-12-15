@echo off
chcp 65001 >nul
REM Check if Python is installed
WHERE pip >nul 2>&1
IF %ERRORLEVEL% NEQ 0 ECHO [101mPython is not installed. Please install Python first...[0m && PAUSE && EXIT /B 1

REM Check CUDA Toolkit
WHERE nvcc >nul 2>&1
IF %ERRORLEVEL% EQU 0 (
    ECHO [92mCUDA is already installed...[0m
    goto PIP
)
IF NOT %ERRORLEVEL% EQU 0 (
    set /p "cuda=CUDA Toolkit isn't installed, install now?(y/n):"
)
:cuda
IF /I %cuda% EQU n goto PIP
IF /I %cuda% NEQ y (
	echo [101mInvalid option.[0m
	set /p "cuda=CUDA Toolkit isn't installed, install now?(y/n):"
	goto cuda
)
REM Install CUDA Toolkit 11.8
IF exist cuda_11.8.0_windows_network.exe (
	start /wait cuda_11.8.0_windows_network.exe
	goto PIP
) else (
	echo [101mInstaller not found...[0m
	set /p "dl=Would you like to download and install the CUDA Toolkit?(y/n)"
)
:get_cuda
IF /I %dl% EQU n goto PIP
IF /I %dl% NEQ y (
	echo [101mInvalid option.[0m
	set /p "dl=Would you like to download and install the CUDA Toolkit?(y/n)"
	goto get_cuda
)
echo Downloading the cuda toolkit installer...
powershell -Command "Invoke-WebRequest https://developer.download.nvidia.com/compute/cuda/11.8.0/network_installers/cuda_11.8.0_windows_network.exe -Outfile cuda_11.8.0_windows_network.exe"
ECHO [92mStart installing CUDA Toolkit 11.8...[0m
start /wait cuda_11.8.0_windows_network.exe

:PIP
set /p "pip=Install all the required Python libraries？(y/n):"
IF /I %pip% EQU n PAUSE && EXIT /B 1
IF /I %pip% NEQ y (
	echo [101mInvalid option.[0m
	goto PIP
)

ECHO [92mStart installing libraries...[0m
REM Install PyTorch, TorchVision and Torchaudio with CUDA support
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

REM Install other dependencies
pip install -r requirements.txt

IF %ERRORLEVEL% EQU 0 echo [92mInstallation completed...[0m
PAUSE
