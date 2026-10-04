@echo off
setlocal EnableExtensions
pushd "%~dp0"

set "VENV_PYTHON=%CD%\gaze-env\Scripts\python.exe"
if not exist "work\matplotlib" mkdir "work\matplotlib"
set "MPLCONFIGDIR=%CD%\work\matplotlib"

if not exist "%VENV_PYTHON%" (
    echo The project environment is missing. Starting the safe Atlas setup first...
    call setup.bat --from-run
    if errorlevel 1 (
        echo Setup failed, so the legacy preview was not started.
        popd
        pause
        exit /b 1
    )
)

if not exist "models\neural\gaze_L.onnx" goto :missing_legacy
if not exist "models\neural\gaze_R.onnx" goto :missing_legacy
if not exist "models\neural\response_curve.json" goto :missing_legacy
"%VENV_PYTHON%" -c "import onnxruntime" >nul 2>nul
if errorlevel 1 goto :missing_legacy

"%VENV_PYTHON%" main.py %*
set "RUN_EXIT=%ERRORLEVEL%"
popd
if not "%RUN_EXIT%"=="0" pause
exit /b %RUN_EXIT%

:missing_legacy
echo.
echo Legacy ONNX preview is intentionally not part of the default public setup.
echo Use atlas_preview.bat for the current Personal Lens Atlas route.
echo If you independently confirm the research-model terms, run:
echo   setup.bat --experimental-neural
popd
pause
exit /b 2
