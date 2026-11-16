@echo off
setlocal
cd /d "%~dp0"
set "DMS_PYTHON=python"
if exist "D:\Anaconda3\local\envs\yolo\python.exe" set "DMS_PYTHON=D:\Anaconda3\local\envs\yolo\python.exe"
"%DMS_PYTHON%" src\main.py --web %*
if errorlevel 1 pause
