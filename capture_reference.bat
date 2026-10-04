@echo off
setlocal EnableExtensions
pushd "%~dp0"

set "VENV_PYTHON=%CD%\gaze-env\Scripts\python.exe"
if not exist "work\matplotlib" mkdir "work\matplotlib"
set "MPLCONFIGDIR=%CD%\work\matplotlib"
if not exist "%VENV_PYTHON%" (
    echo The project environment is missing. Starting setup first...
    call setup.bat --from-run
    if errorlevel 1 (
        echo Setup failed, so reference capture was not started.
        popd
        pause
        exit /b 1
    )
)

"%VENV_PYTHON%" capture_reference.py %*
set "CAPTURE_EXIT=%ERRORLEVEL%"
popd
if not "%CAPTURE_EXIT%"=="0" pause
exit /b %CAPTURE_EXIT%
