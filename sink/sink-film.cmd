@echo off
rem Usage: sink-film.cmd SCRIPT VIDEO [options]
rem One film in one go: video, asr, align, final, check, films\FILM_ID\preview.srt
rem Runs this folder's .venv python with -I and passes every argument through unchanged.
rem Quote paths that contain spaces. The two files may come in either order.
setlocal DisableDelayedExpansion
if not exist "%~dp0.venv\Scripts\python.exe" (
  echo sink-film: .venv\Scripts\python.exe not found next to sink-film.cmd 1>&2
  exit /b 2
)
"%~dp0.venv\Scripts\python.exe" -I -m sinkprep film %*
exit /b %ERRORLEVEL%
