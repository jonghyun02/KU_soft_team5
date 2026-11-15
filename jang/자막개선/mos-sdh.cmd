@echo off
setlocal
if "%UV_PROJECT_ENVIRONMENT%"=="" set "UV_PROJECT_ENVIRONMENT=%LOCALAPPDATA%\mos-sdh\venv"
if "%MOS_SDH_CACHE_DIR%"=="" set "MOS_SDH_CACHE_DIR=%LOCALAPPDATA%\mos-sdh\cache"
uv run mos-sdh %*
