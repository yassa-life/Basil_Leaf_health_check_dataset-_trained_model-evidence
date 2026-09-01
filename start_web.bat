@echo off
cd /d "%~dp0"
echo Opening Basil Lab at http://127.0.0.1:8000
echo Keep this window open while using the website. Press Ctrl+C to stop.
".venv\Scripts\python.exe" app.py
pause
