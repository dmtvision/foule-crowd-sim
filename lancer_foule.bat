@echo off
REM Lance l'application Foule (sequences detourees de banque\ -> foules animees)
cd /d "%~dp0"
python -m foule
if errorlevel 1 pause
