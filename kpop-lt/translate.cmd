@echo off
chcp 65001 >nul
rem 구글 번역 전부 (막히면 기다렸다 이어감). 세션과 떨어져 돌도록 explorer 로 띄운다.
cd /d %~dp0
if not exist logs mkdir logs
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
>> logs\translate.log echo === 시작 %date% %time%
.venv\Scripts\python.exe -X utf8 -m kplt.data.translate_all >> logs\translate.log 2>&1
>> logs\translate.log echo === 끝 %date% %time% 종료코드 %ERRORLEVEL%
