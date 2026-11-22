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
"%~dp0.venv\Scripts\python.exe" -m pstts %*
set "PSTTS_EXIT=%ERRORLEVEL%"
if defined PSTTS_CP chcp %PSTTS_CP% >nul
exit /b %PSTTS_EXIT%

rem ---- 설명(exit /b 뒤라서 cmd 가 읽지 않는다) ----
rem PS-TTS 텍스트판 실행. 예: run.cmd "I didn't order two tickets." --register 해요체
rem 한글이 깨지지 않게 코드 페이지를 UTF-8(65001)로 바꾸고, 끝나면 부른 창의 코드 페이지(chcp 출력의 번호)로 되돌린다.
rem setlocal 은 코드 페이지를 되돌리지 않는다. 어느 폴더에서 불러도 pstts 를 찾게 PYTHONPATH 에 이 폴더를 넣는다.
rem exit /b 위에는 영문만 둔다. 65001 에서 cmd 가 한글 줄의 바이트 위치를 잘못 세어 다음 줄을 잘라 읽는 일이 있다
rem (2026-11-22 확인: 한글 rem 줄 조각이 명령으로 실행됨).
