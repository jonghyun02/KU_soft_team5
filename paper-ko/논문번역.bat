@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
python "%~dp0paper_ko.py" %*
echo.
pause
