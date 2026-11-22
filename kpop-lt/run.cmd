@echo off
chcp 65001 >nul
rem 사용법: run.cmd data ^| analysis ^| chain CONFIG ^| nsp ^| infer ^| evaluate
rem CONFIG = line, line_syl, sect, sect_syl
cd /d %~dp0
if not exist logs mkdir logs
set PYTHONIOENCODING=utf-8
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
.venv\Scripts\python.exe -X utf8 -m kplt.pipeline %*
