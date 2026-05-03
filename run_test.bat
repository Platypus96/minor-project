@echo off
set HTTP_PROXY=
set HTTPS_PROXY=
set http_proxy=
set https_proxy=
set PYTHONUTF8=1

call conda activate speech_pipeline
python pipeline\pipeline.py pipeline\sample.wav
