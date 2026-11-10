@echo off
rem One-click start for Windows: sets everything up the first time, then opens the dashboard.
cd /d "%~dp0"
if not exist .venv\Scripts\apex.exe (
  echo First run: installing, this takes a few minutes...
  py -3 -m venv .venv || goto :error
  .venv\Scripts\python -m pip install --upgrade pip >nul
  .venv\Scripts\pip install -e . || goto :error
)
if not exist data\apex.sqlite (
  echo Downloading 9 years of market data...
  .venv\Scripts\apex fetch --days 3300 || goto :error
)
if not exist data\models\paper_model.joblib (
  echo Training the model and opening the paper account...
  .venv\Scripts\apex paper init || goto :error
)
start "" http://127.0.0.1:8000
.venv\Scripts\apex serve
goto :eof
:error
echo.
echo Something went wrong. Copy the text above and send it to Claude.
pause
