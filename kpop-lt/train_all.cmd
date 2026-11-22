@echo off
chcp 65001 >nul
rem 네 설정(line, line_syl, sect, sect_syl)을 창 네 개로 동시에 학습한다. 각 창은 general -> nonsingable -> singable.
rem Claude Code 세션과 떨어져 돌도록 창을 따로 띄운다. 끝난 단계는 runs\CONFIG\STAGE\done.json 으로 건너뛴다.
cd /d %~dp0
if not exist logs mkdir logs
for %%C in (line line_syl sect sect_syl) do (
  start "kpop-lt %%C" cmd /c "cd /d %~dp0 && set PYTHONIOENCODING=utf-8&& set HF_HUB_DISABLE_SYMLINKS_WARNING=1&& .venv\Scripts\python.exe -X utf8 -m kplt.pipeline chain %%C >> logs\train_%%C.log 2>&1"
)
