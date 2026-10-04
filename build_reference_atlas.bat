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
        echo Setup failed, so the personal eye atlas was not built.
        popd
        pause
        exit /b 1
    )
)

"%VENV_PYTHON%" build_reference_atlas.py %*
set "BUILD_EXIT=%ERRORLEVEL%"
popd
if not "%BUILD_EXIT%"=="0" pause
exit /b %BUILD_EXIT%
