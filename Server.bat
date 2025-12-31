chcp 65001 > NUL
@echo off

pushd %~dp0
set "PATH=C:\Program Files\eSpeak NG;%PATH%"
set "PHONEMIZER_ESPEAK_PATH=C:\Program Files\eSpeak NG"
set "PYTHON_CMD=venv\Scripts\python.exe"
if not exist "%PYTHON_CMD%" (
    set "PYTHON_CMD=python"
)

echo Running server_fastapi.py
"%PYTHON_CMD%" server_fastapi.py

if %errorlevel% neq 0 ( pause & popd & exit /b %errorlevel% )

popd
pause