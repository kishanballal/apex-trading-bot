@echo off
rem Starts the ACTIVE fake-money bot (intraday, NSE stocks, news). Leave this window open
rem from 9:15 to 3:30 on market days. Your daily bot (run.bat) is separate and unaffected.
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Please run run.bat once first, it installs everything.
  pause
  goto :eof
)
if not exist .venv\finbert1.flag (
  echo One-time setup: installing the FinBERT headline reader. If this fails the bot still runs.
  .venv\Scripts\python.exe -m pip install transformers
  echo ok> .venv\finbert1.flag
)
echo Starting the active bot. The first start downloads 60 days of 5-minute data (about a minute).
start "" http://127.0.0.1:8001
.venv\Scripts\python.exe -m apex.active || goto :error
goto :eof
:error
echo.
echo Something went wrong. Copy the text above and send it to Claude.
pause
