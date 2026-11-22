@echo off
setlocal
rem ASCII only above "exit /b" (see the Korean notes at the end of this file).
for /f "tokens=2 delims=:" %%a in ('chcp') do set "PSTTS_CP=%%a"
if defined PSTTS_CP set "PSTTS_CP=%PSTTS_CP: =%"
if defined PSTTS_CP set "PSTTS_CP=%PSTTS_CP:.=%"
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set "PSTTS_ROOT=%~dp0."
if defined PYTHONPATH set "PYTHONPATH=%PSTTS_ROOT%;%PYTHONPATH%"
if not defined PYTHONPATH set "PYTHONPATH=%PSTTS_ROOT%"
"%~dp0.venv\Scripts\python.exe" -m pstts.paper %*
set "PSTTS_EXIT=%ERRORLEVEL%"
if defined PSTTS_CP chcp %PSTTS_CP% >nul
exit /b %PSTTS_EXIT%

rem ---- 설명(exit /b 뒤라서 cmd 가 읽지 않는다) ----
rem PS-TTS 논문 그대로 판 실행(python -m pstts.paper). 설명은 README_paper.md.
rem 예: run_paper.cmd "We have to leave before the storm hits." --seconds 2.1 --device cpu
rem     run_paper.cmd --input examples\lines.csv --out out\paper_demo --device cpu
rem run.cmd 와 같은 규칙: 한글이 깨지지 않게 코드 페이지를 UTF-8(65001)로 바꾸고, 끝나면 부른 창의 코드 페이지로 되돌린다.
rem PYTHONUTF8=1 은 기존 판과 같은 UTF-8 모드다(논문 판 근사 프런트엔드는 실행 중 PanPhon 을 부르지 않아 없어도 돈다). 끝 코드: 0, 1, 2, 130(README_paper.md).
rem exit /b 위에는 영문만 둔다. 65001 에서 cmd 가 한글 줄의 바이트 위치를 잘못 세어 다음 줄을 잘라 읽는 일이 있다.
rem 줄 끝은 CRLF 여야 한다(LF 만 쓰면 cmd 가 줄을 잘못 나눈다).
