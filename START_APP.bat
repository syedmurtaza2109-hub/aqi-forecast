@echo off
echo ============================================
echo    AQI Forecast - Starting up...
echo ============================================

:: Activate virtual environment if it exists
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else (
    echo Installing packages first...
    python -m venv venv
    call venv\Scripts\activate.bat
    pip install -r requirements.txt
)

echo.
echo Starting Flask API on port 5000...
start cmd /k "call venv\Scripts\activate.bat && python dashboard/api.py"

timeout /t 3 /nobreak >nul

echo Starting Streamlit dashboard...
start cmd /k "call venv\Scripts\activate.bat && streamlit run dashboard/app.py"

echo.
echo ============================================
echo  App is starting!
echo  Open your browser at: http://localhost:8501
echo ============================================
pause
