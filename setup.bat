@echo off
setlocal EnableExtensions DisableDelayedExpansion
pushd "%~dp0"

set "PAUSE_AT_END=1"
set "INSTALL_EXPERIMENTAL_NEURAL=0"
:parse_args
if "%~1"=="" goto :args_done
if /I "%~1"=="--from-run" set "PAUSE_AT_END=0"
if /I "%~1"=="--experimental-neural" set "INSTALL_EXPERIMENTAL_NEURAL=1"
shift
goto :parse_args
:args_done

echo.
echo ============================================================
echo  Personal Lens Atlas - safe local setup
echo ============================================================
echo This setup creates a local .\gaze-env and the official MediaPipe face model.
echo It will not uninstall, replace, or downgrade your Python 3.11.
if "%INSTALL_EXPERIMENTAL_NEURAL%"=="1" echo Experimental ONNX support was explicitly requested.
echo.

set "PYTHON_EXE=python"
set "PYTHON_311=%LocalAppData%\Programs\Python\Python311\python.exe"
where python >nul 2>nul
if errorlevel 1 (
    rem Some Windows installations expose python to PowerShell but not cmd.exe.
    rem Check the standard per-user Python 3.11 location before giving up.
    if exist "%PYTHON_311%" (
        set "PYTHON_EXE=%PYTHON_311%"
    ) else (
        echo [ERROR] Python was not found in cmd.exe PATH or the standard
        echo per-user Python 3.11 location.
        echo Install 64-bit Python 3.11, enable "Add python.exe to PATH",
        echo then run this setup again.
        goto :failed
    )
)

set "PYTHON_VERSION=unknown"
"%PYTHON_EXE%" -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)"
if errorlevel 1 if exist "%PYTHON_311%" set "PYTHON_EXE=%PYTHON_311%"
"%PYTHON_EXE%" -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)"
if errorlevel 1 (
    echo [ERROR] Python 3.11 is required. Detected:
    "%PYTHON_EXE%" --version
    echo Nothing was changed. Start this script from a terminal where
    echo "python --version" reports Python 3.11.x.
    goto :failed
)
set "PYTHON_VERSION=3.11"

"%PYTHON_EXE%" -m pip --version >nul 2>nul
if errorlevel 1 (
    echo [ERROR] pip is unavailable for this Python 3.11 installation.
    echo Nothing was changed. Repair pip for Python 3.11, then retry.
    goto :failed
)

echo [OK] Python %PYTHON_VERSION% and pip are available.

set "VENV_PYTHON=%CD%\gaze-env\Scripts\python.exe"
if not exist "work\matplotlib" mkdir "work\matplotlib"
set "MPLCONFIGDIR=%CD%\work\matplotlib"
if not exist "%VENV_PYTHON%" (
    echo Creating isolated environment: gaze-env
    "%PYTHON_EXE%" -m venv "gaze-env"
    if errorlevel 1 goto :failed
) else (
    echo [OK] Reusing existing isolated environment: gaze-env
)

echo Checking pip inside gaze-env...
"%VENV_PYTHON%" -m pip --version
if errorlevel 1 goto :failed

echo Checking MediaPipe, OpenCV, and NumPy inside gaze-env...
"%VENV_PYTHON%" -c "import importlib.metadata as m; import cv2, mediapipe, numpy; assert m.version('mediapipe') == '1.0.1'; assert m.version('opencv-contrib-python') == '5.0.0.93'; assert m.version('numpy') == '2.4.5'" >nul 2>nul
if not errorlevel 1 (
    echo [OK] Compatible dependencies are already installed; no network is needed.
) else (
    echo Installing compatible prebuilt wheels into gaze-env...
    "%VENV_PYTHON%" -u bootstrap_dependencies.py
    if errorlevel 1 goto :failed
)

echo Downloading or checking the official Face Landmarker model...
"%VENV_PYTHON%" download_model.py
if errorlevel 1 goto :failed

echo Checking installed packages...
"%VENV_PYTHON%" -c "import cv2, mediapipe, numpy; print('  MediaPipe:', mediapipe.__version__); print('  OpenCV:', cv2.__version__); print('  NumPy:', numpy.__version__)"
if errorlevel 1 goto :failed

echo Checking package dependency integrity...
"%VENV_PYTHON%" -m pip check
if errorlevel 1 goto :failed

echo Checking MediaPipe Tasks API and the local model file...
"%VENV_PYTHON%" -c "from config import MODEL_PATH; import mediapipe as mp; assert MODEL_PATH.is_file() and MODEL_PATH.stat().st_size > 1000000; assert hasattr(mp.tasks.vision, 'FaceLandmarker'); print('  Face Landmarker API and model file: OK')"
if errorlevel 1 goto :failed

if "%INSTALL_EXPERIMENTAL_NEURAL%"=="1" (
    echo.
    echo Installing optional experimental ONNX Runtime and research-model helper...
    echo Read THIRD_PARTY_NOTICES.md before redistributing or relying on this legacy path.
    "%VENV_PYTHON%" -m pip install --disable-pip-version-check --only-binary=:all: onnxruntime==1.30.0
    if errorlevel 1 goto :failed
    "%VENV_PYTHON%" download_neural_models.py
    if errorlevel 1 goto :failed
    "%VENV_PYTHON%" -c "from pathlib import Path; import onnxruntime as ort; paths=(Path('models/neural/gaze_L.onnx'), Path('models/neural/gaze_R.onnx')); sessions=[ort.InferenceSession(str(path), providers=['CPUExecutionProvider']) for path in paths]; assert all('CPUExecutionProvider' in session.get_providers() for session in sessions); print('  Experimental neural models: OK (CPU)')"
    if errorlevel 1 goto :failed
)

echo Running lightweight self-tests...
"%VENV_PYTHON%" -m unittest discover -s tests -v
if errorlevel 1 goto :failed

echo.
echo ============================================================
echo  Setup completed successfully.
echo  Next: double-click capture_reference.bat
echo  Then: build_reference_atlas.bat and atlas_preview.bat
if "%INSTALL_EXPERIMENTAL_NEURAL%"=="1" echo  Optional legacy ONNX experiment is also available through run.bat.
echo ============================================================
goto :done

:failed
echo.
echo ============================================================
echo  Setup did not finish. Read the error above, then retry.
echo  Your global Python installation was not changed.
echo ============================================================
set "SETUP_EXIT=1"
goto :done

:done
popd
if "%PAUSE_AT_END%"=="1" pause
if defined SETUP_EXIT (
    exit /b %SETUP_EXIT%
)
exit /b 0
