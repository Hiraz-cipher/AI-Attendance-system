@echo off
cd /d "%~dp0"
set PORT=5001
echo Starting UPDATED attendance app on http://localhost:5001
echo Folder: %cd%
if exist .venv\Scripts\python.exe (
  .venv\Scripts\python.exe app.py
) else (
  python app.py
)
pause
