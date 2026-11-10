@echo off
rem One-click start for Windows: sets everything up the first time, then opens the dashboard.
rem Uses "python -m apex" because some PCs block unsigned launcher programs such as apex.exe.
cd /d "%~dp0"
set APEX=.venv\Scripts\python.exe -m apex
if not exist .venv\installed3.flag (
  echo First run: installing, this takes a few minutes...
  if not exist .venv\Scripts\python.exe ( py -3 -m venv .venv || goto :error )
  .venv\Scripts\python.exe -m pip uninstall -y polars polars-runtime-32 polars-lts-cpu >nul 2>&1
  .venv\Scripts\python.exe -m pip install -e . || goto :error
  echo ok> .venv\installed3.flag
)
if not exist data\models\paper_model.joblib (
  echo Downloading 9 years of market data...
  %APEX% fetch --days 3300 || goto :error
  echo Training the model and opening the paper account...
  %APEX% paper init || goto :error
)
echo Starting the dashboard. The first start can take a minute while Windows checks the files.
start "" http://127.0.0.1:8000
%APEX% serve
goto :eof
:error
echo.
echo Something went wrong. Copy the text above and send it to Claude.
pause
