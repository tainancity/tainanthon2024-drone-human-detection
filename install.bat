@echo off
chcp 65001 >nul
REM Check if Python is installed
WHERE pip >nul 2>&1
IF %ERRORLEVEL% NEQ 0 ECHO [101mPython is not installed. Please install Python first...[0m && PAUSE && EXIT /B 1

REM Check CUDA Toolkit
set err=0
set cuda=0
WHERE nvcc >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    set err=1
	set /p "cuda=CUDA Toolkit isn't installed, install now?(y/n):"
)
:cuda
IF /I %cuda% EQU y goto install
IF /I %cuda% EQU n goto PIP
IF %err% EQU 1 IF /I %cuda% NEQ y (
	echo [101mInvalid option.[0m
	set /p "cuda=CUDA Toolkit isn't installed, install now?(y/n):"
	goto cuda
)
set ver=0
set ignore=
for /f %%i in ('nvcc --version ^| find "11.8"') do set ver=%%i
IF %ver% NEQ 0 (
    ECHO [92mCUDA is already installed...[0m
    goto PIP
) ELSE (
    ECHO [101mCUDA Toolkit version mismatched![0m
	ECHO It is recommended to uninstall the current CUDA version...
	ECHO [93m && nvcc --version && ECHO [0m
	set /p "ignore=CUDA 11.8 is recommended, do you want to continue the installtion?(y/n):"
)
:mismatch
IF /I %ignore% EQU y goto PIP
IF /I %ignore% EQU n PAUSE && EXIT /B 1
IF /I %ignore% NEQ y (
	echo [101mInvalid option.[0m
	set /p "ignore=CUDA 11.8 is recommended, do you want to continue the installtion?(y/n):"
	goto mismatch
)
:install
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
IF %ERRORLEVEL% NEQ 0 pause && EXIT /B 1

REM Install other dependencies
pip install -r requirements.txt

IF %ERRORLEVEL% EQU 0 echo [92mInstallation completed...[0m
PAUSE