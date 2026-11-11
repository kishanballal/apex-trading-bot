@echo off
rem Tests the Kronos candlestick model on NSE 5-minute data. Does not touch either bot.
rem First run downloads PyTorch (about 3 GB) and the Kronos model, so allow some time.
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Please run run.bat once first, it installs everything.
  pause
  goto :eof
)
if not exist .venv\kronos1.flag (
  echo Installing PyTorch for your NVIDIA GPU, this takes a while...
  .venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128 || goto :error
  .venv\Scripts\python.exe -m pip install einops==0.8.1 huggingface_hub==0.33.1 safetensors tqdm || goto :error
  echo ok> .venv\kronos1.flag
)
.venv\Scripts\python.exe tools\kronos_test.py %* || goto :error
pause
goto :eof
:error
echo.
echo Something went wrong. Copy the text above and send it to Claude.
pause
