@echo off
setlocal
cd /d %~dp0
echo VaM ML Skeleton Engine starting...
:: engineフォルダ内のpython.exeを相対パスで指定
.\engine\python.exe CVAE_GEN.py
if %errorlevel% neq 0 pause