@echo off
rem Serve the local dashboard preview: http://localhost:8765/preview.html (this PC only).
cd /d "%~dp0..\dashboard"
start "" http://localhost:8765/preview.html
python -m http.server 8765 --bind 127.0.0.1
