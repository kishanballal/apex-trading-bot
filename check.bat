@echo off
rem Diagnoses which compiled libraries Windows allows Python to load.
cd /d "%~dp0"
.venv\Scripts\python.exe -c "import sys, platform; print('Python', sys.version); print(platform.platform(), platform.machine())"
for %%m in (numpy pydantic_core polars._plr polars sklearn) do (
  .venv\Scripts\python.exe -c "import %%m; print('OK   %%m')" 2>&1 | findstr /v /c:"Traceback" /c:"File " /c:"^ "
)
.venv\Scripts\python.exe -m pip show polars polars-runtime-32 2>nul | findstr /b "Name Version"
echo.
echo Done. Take a screenshot of this window and send it to Claude.
pause
