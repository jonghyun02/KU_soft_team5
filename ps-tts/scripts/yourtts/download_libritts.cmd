@echo off
setlocal
rem ASCII only above "exit /b" (see the Korean notes at the end of this file).
set "URL=https://www.openslr.org/resources/60/train-clean-360.tar.gz"
set "SIZE=27504073644"
if "%~1"=="" goto usage
if /i "%~1"=="--check" goto check
set "DEST=%~1"
set "ARCHIVE=%DEST%\train-clean-360.tar.gz"
if not exist "%DEST%" mkdir "%DEST%"
if not exist "%DEST%" exit /b 1
set "GOT=0"
if exist "%ARCHIVE%" for %%F in ("%ARCHIVE%") do set "GOT=%%~zF"
if "%GOT%"=="%SIZE%" goto extract
curl -f -L -C - --retry 10 --retry-delay 30 -o "%ARCHIVE%" "%URL%"
if errorlevel 1 exit /b 1
for %%F in ("%ARCHIVE%") do set "GOT=%%~zF"
if not "%GOT%"=="%SIZE%" (
  echo size mismatch: %GOT% bytes, expected %SIZE%. Run again to resume.
  exit /b 1
)
:extract
tar -xzf "%ARCHIVE%" -C "%DEST%"
if errorlevel 1 exit /b 1
echo done: %DEST%\LibriTTS\train-clean-360
exit /b 0

:check
curl -sSI -L "%URL%"
exit /b %ERRORLEVEL%

:usage
echo usage: download_libritts.cmd TARGET_DIR
echo        download_libritts.cmd --check
exit /b 2

rem ---- 설명(exit /b 뒤라서 cmd 가 읽지 않는다) ----
rem LibriTTS train-clean-360(논문 4.1절 'Libri-TTS-360', 904명)을 OpenSLR 60 에서 이어받기(curl -C -)로 받아 푼다.
rem 쓰는 법: download_libritts.cmd D:\data   ->  D:\data\LibriTTS\train-clean-360\<화자>\<장>\*.wav, *.normalized.txt
rem          download_libritts.cmd --check   ->  HEAD 요청으로 주소만 확인한다(받지 않음).
rem 파일 크기 27504073644 바이트(약 25.6 GiB)는 2026-11-22 HEAD 응답의 Content-Length 다(Last-Modified 2019-04-02).
rem 크기가 맞으면 다시 받지 않고 풀기만 한다. 중간에 끊기면 같은 명령을 다시 실행하면 이어 받는다.
rem 풀면 약 27GB 가 더 든다. 다 풀고 나면 make_manifest.py --libritts-dir D:\data\LibriTTS 로 매니페스트를 만든다.
rem curl.exe 와 tar.exe 는 Windows 10 이후에 들어 있다.
