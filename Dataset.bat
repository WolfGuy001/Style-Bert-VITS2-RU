chcp 65001 > NUL
@echo off

pushd %~dp0
set "PATH=C:\Program Files\eSpeak NG;%PATH%"
set "PHONEMIZER_ESPEAK_PATH=C:\Program Files\eSpeak NG"
set "PYTHON_CMD=venv\Scripts\python.exe"
if not exist "%PYTHON_CMD%" (
    set "PYTHON_CMD=python"
)

echo Running gradio_tabs/dataset.py...
"%PYTHON_CMD%" -m gradio_tabs.dataset

if %errorlevel% neq 0 ( pause & popd & exit /b %errorlevel% )

popd
pause