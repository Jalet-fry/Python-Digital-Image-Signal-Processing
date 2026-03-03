@echo off
echo ==========================================
echo   Installing PythonDSP Dependencies
echo ==========================================
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Something went wrong during installation.
    pause
) else (
    echo.
    echo [SUCCESS] All dependencies are ready!
    pause
)