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
        echo Setup failed, so the lens-reference preview was not started.
        popd
        pause
        exit /b 1
    )
)

if not exist "reference_atlas\active.json" (
    echo Your personal eye reference library has not been built yet.
    echo Starting the local builder now...
    call build_reference_atlas.bat
    if errorlevel 1 (
        echo The personal eye reference library could not be built.
        popd
        pause
        exit /b 1
    )
)

"%VENV_PYTHON%" atlas_preview.py %*
set "PREVIEW_EXIT=%ERRORLEVEL%"
popd
if not "%PREVIEW_EXIT%"=="0" pause
exit /b %PREVIEW_EXIT%
