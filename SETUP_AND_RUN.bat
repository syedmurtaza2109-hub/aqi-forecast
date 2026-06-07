@echo off
echo ============================================
echo    AQI Forecast - Full Setup
echo    This runs the pipeline end to end
echo ============================================

:: Activate venv
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else (
    echo Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo Installing packages (takes 3-5 mins)...
    pip install -r requirements.txt
)

echo.
echo [1/3] Running feature pipeline (fetching live data)...
python feature_pipeline/fetch_and_store.py
if %errorlevel% neq 0 (
    echo ERROR in feature pipeline. Check your API keys in .env
    pause
    exit /b 1
)

echo.
echo [2/3] Running backfill for 30 days of history...
python feature_pipeline/backfill.py --days 30 --hours-per-day 4
if %errorlevel% neq 0 (
    echo ERROR in backfill. Check logs above.
    pause
    exit /b 1
)

echo.
echo [3/3] Training models...
python training_pipeline/train.py
if %errorlevel% neq 0 (
    echo ERROR in training. Check logs above.
    pause
    exit /b 1
)

echo.
echo ============================================
echo  Setup complete! Now launching dashboard...
echo ============================================
timeout /t 2 /nobreak >nul

start cmd /k "call venv\Scripts\activate.bat && python dashboard/api.py"
timeout /t 3 /nobreak >nul
start cmd /k "call venv\Scripts\activate.bat && streamlit run dashboard/app.py"

echo.
echo Open browser at: http://localhost:8501
pause
