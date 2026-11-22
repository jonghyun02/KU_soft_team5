@echo off
chcp 65001 >nul
rem 가상환경 만들기와 패키지 설치. 한 번만 실행한다.
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  py -3.11 -m venv .venv || exit /b 1
)
set PY=%~dp0.venv\Scripts\python.exe
"%PY%" -m pip install --upgrade pip || exit /b 1
"%PY%" -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128 || exit /b 1
"%PY%" -m pip install -r requirements.txt || exit /b 1
"%PY%" -c "import torch, comet, sentence_transformers, panphon, espeakng_loader; print('ok', torch.__version__, torch.cuda.is_available())"
