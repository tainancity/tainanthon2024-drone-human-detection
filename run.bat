@echo off
WHERE conda >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo Conda is not installed. Running natively...
    python -m streamlit run main.py --server.maxUploadSize 10000
)
IF %ERRORLEVEL% EQU 0 (
    echo Conda is installed.
    set /p "env=Please enter the Conda environment name to activate:"
)
IF %env% NEQ "" (
    echo Activating Conda environment %env%...
    conda activate rtdetr
    streamlit run main.py --server.maxUploadSize 10000
)