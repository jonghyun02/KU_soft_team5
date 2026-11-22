@echo off
chcp 65001 >nul
rem 학습이 모두 끝난 뒤: NSP 모델, 번역 9개, 표 5·6
cd /d %~dp0
call run.cmd nsp >> logs\nsp.log 2>&1
call run.cmd infer >> logs\infer.log 2>&1
call run.cmd evaluate >> logs\evaluate.log 2>&1
