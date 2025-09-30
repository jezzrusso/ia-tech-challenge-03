@echo off
set TITLE=%1
if "%TITLE%"=="" set TITLE=Apple iPhone 14 Pro Max 128GB
python -m src.infer --title "%TITLE%" --question "Descreva o produto"
pause
