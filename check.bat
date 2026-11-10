@echo off
rem Diagnoses which compiled libraries Windows allows Python to load. Writes check_result.txt.
cd /d "%~dp0"
set OUT=check_result.txt
echo Checking, please wait about a minute...
.venv\Scripts\python.exe -c "import sys, platform; print(sys.version); print(platform.platform())" > %OUT% 2>&1
powershell -NoProfile -Command "try { (Get-MpComputerStatus).SmartAppControlState } catch { 'unknown' }" >> %OUT% 2>&1
.venv\Scripts\python.exe -m pip list >> %OUT% 2>&1
for %%m in (numpy pydantic_core polars.polars pandas scipy sklearn joblib sqlite3) do (
  echo ==== %%m >> %OUT%
  .venv\Scripts\python.exe -X faulthandler -c "import %%m; print('OK')" >> %OUT% 2>&1
)
echo Done. Tell Claude "done".
pause
