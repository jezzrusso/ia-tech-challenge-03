@echo off
REM Exemplo de uso:
REM train.bat 1
REM train.bat 2 resume
set EPOCHS=%1
set RESUME=%2

if "%RESUME%"=="resume" (
    python -m src.train --epochs %EPOCHS% --resume
) else (
    python -m src.train --epochs %EPOCHS%
)
pause
