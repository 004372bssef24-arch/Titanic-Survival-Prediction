@echo off
echo ========================================
echo   Titanic Survival Project - Setup
echo ========================================
echo.

echo Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate

echo Installing required packages...
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo ========================================
echo   Setup Complete!
echo   Run: python run_pipeline.py
echo ========================================
pause