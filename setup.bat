@echo off
setlocal enabledelayedexpansion

echo Checking Python installation...
python --version 2>nul
if errorlevel 1 (
    echo Python is not installed or not in PATH
    echo Please install Python 3.10 or higher
    pause
    exit /b 1
)

echo Cleaning up old environment...
rmdir /s /q venv 2>nul
del /f /q *.pyc 2>nul

echo Creating new virtual environment...
python -m venv venv
if errorlevel 1 (
    echo Failed to create virtual environment
    echo Please run as administrator
    pause
    exit /b 1
)

echo Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo Failed to activate virtual environment
    pause
    exit /b 1
)

echo Installing core dependencies...
python -m pip install --upgrade pip
python -m pip install wheel==0.40.0
python -m pip install setuptools==68.0.0
python -m pip install virtualenv==20.25.0

echo Installing scientific packages...
pip install --no-cache-dir numpy==1.24.3
pip install --no-cache-dir pandas==2.0.3
pip install --no-cache-dir scikit-learn==1.3.0

echo Installing web packages...
pip install flask==2.3.3
pip install flask-cors==4.0.0
pip install Werkzeug==2.2.3
pip install gunicorn==21.2.0
pip install python-dateutil==2.8.2

echo.
echo Setup completed successfully!
echo.
echo Next steps:
echo 1. Run: python trainmodel.py
echo 2. Run: python app.py
echo.

pause
