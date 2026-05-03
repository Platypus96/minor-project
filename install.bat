@echo off
set HTTP_PROXY=
set HTTPS_PROXY=
set http_proxy=
set https_proxy=

call conda activate speech_pipeline

echo ========================================
echo   Installing PyTorch + TorchAudio
echo ========================================
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

echo ========================================
echo   Installing FunASR + ModelScope
echo ========================================
pip install funasr modelscope

echo ========================================
echo   Installing other dependencies
echo ========================================
pip install gradio requests python-dotenv numpy scipy nltk jiwer

echo ========================================
echo   ALL DONE!
echo ========================================
pause
