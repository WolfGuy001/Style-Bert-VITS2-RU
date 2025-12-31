chcp 65001 > NUL
@echo off

pushd %~dp0
set "PYTHON_CMD=venv\Scripts\python.exe"
if not exist "%PYTHON_CMD%" (
    set "PYTHON_CMD=python"
)

echo Running gradio_tabs/style_vectors.py...
"%PYTHON_CMD%" -m gradio_tabs.style_vectors

if %errorlevel% neq 0 ( pause & popd & exit /b %errorlevel% )

popd
pause