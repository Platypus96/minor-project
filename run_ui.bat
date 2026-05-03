@echo off
set HTTP_PROXY=
set HTTPS_PROXY=
set http_proxy=
set https_proxy=

echo ========================================
echo   Activating Conda Environment...
echo ========================================
call conda activate speech_pipeline

echo ========================================
echo   Starting Web UI...
echo ========================================
python pipeline\demo.py

pause
