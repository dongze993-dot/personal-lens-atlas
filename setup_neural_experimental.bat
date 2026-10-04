@echo off
setlocal EnableExtensions
pushd "%~dp0"

echo This optional legacy path downloads research-model artifacts.
echo Read THIRD_PARTY_NOTICES.md and confirm the upstream terms before continuing.
call setup.bat --experimental-neural
set "SETUP_EXIT=%ERRORLEVEL%"
popd
exit /b %SETUP_EXIT%
