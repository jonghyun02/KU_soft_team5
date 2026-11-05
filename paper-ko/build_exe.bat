@echo off
rem Build dist\<app>.exe from a clean venv (global Python has torch etc. which bloats the exe)
cd /d "%~dp0"
if not exist .venv-build\Scripts\python.exe (
  python -m venv .venv-build
  .venv-build\Scripts\python -m pip install pymupdf tkinterdnd2 pyinstaller pillow
)
if not exist app.ico .venv-build\Scripts\python make_icon.py
.venv-build\Scripts\python -m PyInstaller --noconfirm --onefile --windowed --name "paper-ko" --icon app.ico --add-data "app.ico;." --collect-all tkinterdnd2 gui.py
echo.
echo Done: dist\paper-ko.exe  (rename as you like)
pause
