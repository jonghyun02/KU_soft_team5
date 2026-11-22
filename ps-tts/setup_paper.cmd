@echo off
setlocal
rem ASCII only above "exit /b" (see the Korean notes at the end of this file).
for /f "tokens=2 delims=:" %%a in ('chcp') do set "PSTTS_CP=%%a"
if defined PSTTS_CP set "PSTTS_CP=%PSTTS_CP: =%"
if defined PSTTS_CP set "PSTTS_CP=%PSTTS_CP:.=%"
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
set "PY=%~dp0.venv\Scripts\python.exe"
set "PSTTS_EXIT=0"
set "PSTTS_CONSTRAINTS=%TEMP%\pstts_paper_constraints_%RANDOM%.txt"
if not exist "%PY%" (
  echo .venv\Scripts\python.exe is missing. Run setup.cmd first.
  goto fail
)
echo [1/5] pip freeze to "%PSTTS_CONSTRAINTS%"
"%PY%" -m pip freeze > "%PSTTS_CONSTRAINTS%"
if errorlevel 1 goto fail
echo [2/5] torchaudio 2.11.0 cu128 build
"%PY%" -m pip install -c "%PSTTS_CONSTRAINTS%" torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 goto fail
echo [3/5] requirements-paper.txt
"%PY%" -m pip install -c "%PSTTS_CONSTRAINTS%" -r "%~dp0requirements-paper.txt"
if errorlevel 1 goto fail
echo [4/5] nltk cmudict for g2pK
"%PY%" -c "import nltk, sys, os; d = os.path.join(sys.prefix, 'nltk_data'); ok = nltk.download('cmudict', download_dir=d, quiet=True); print('cmudict', d, ok); sys.exit(0 if ok else 1)"
if errorlevel 1 goto fail
echo [5/5] import check
"%PY%" -c "import phonemizer, g2pk, mecab, jamo, nltk, sklearn, torchaudio, TTS; from pstts.paper import g2p; print('ok', g2p.versions()); print(g2p.tokens('Run!', 'en')); print(g2p.tokens('\uac19\uc774 \uc77d\uace0', 'ko'))"
if errorlevel 1 goto fail
goto end
:fail
set "PSTTS_EXIT=1"
:end
if exist "%PSTTS_CONSTRAINTS%" del "%PSTTS_CONSTRAINTS%"
if defined PSTTS_CP chcp %PSTTS_CP% >nul
exit /b %PSTTS_EXIT%

rem ---- 설명(exit /b 뒤라서 cmd 가 읽지 않는다) ----
rem 논문 그대로 판(pstts.paper)이 더 쓰는 패키지 설치. setup.cmd(기존 판) 다음에 한 번 실행한다.
rem 1) 지금 설치된 판을 pip freeze 로 제약 파일(-c)에 적어 기존 패키지 판이 바뀌지 않게 한다.
rem    pip 가 기존 패키지를 바꿔야 한다고 하면 설치가 멈춘다(그대로 두고 README_paper.md 의 설치 항목을 본다).
rem 2) torchaudio 는 torch 와 같은 CUDA 판을 PyTorch 색인에서 받는다(coqui-tts 가 부른다).
rem 3) requirements-paper.txt: phonemizer, g2pK, python-mecab-ko(+사전), jamo, nltk, coqui-tts, six, torchcodec 등.
rem 4) g2pK 가 쓰는 nltk cmudict 를 .venv\nltk_data 에 받는다(g2p.py 는 이것이 없으면 멈춘다).
rem 5) 가져오기 점검과 G2P 한 번씩.
rem exit /b 위에는 영문만 둔다(65001 에서 한글 줄을 잘못 읽는 일이 있음). 줄 끝은 CRLF.
