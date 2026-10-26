@echo off
chcp 65001 > nul
title MediVision AI - Klinik Karar Destek Sistemi
echo ======================================================================
echo           MediVision AI - Klinik Karar Destek Sistemi (CDSS)
echo           YOLO Segmentasyon ^& VLM Doktor Raporlama Sunucusu
echo ======================================================================
echo.
echo [1/2] Sanal ortam ve modeller hazirlaniyor...
cd /d "%~dp0"

echo [2/2] Web Sunucusu Baslatiliyor (http://localhost:8000)...
start "" http://localhost:8000
.\medikal_env\Scripts\python.exe app.py

pause
