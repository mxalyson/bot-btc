@echo off
REM Script de início rápido para Windows
REM Execute este arquivo para iniciar o bot

echo ================================================================================
echo BOT BTC SCALPER - INICIO RAPIDO
echo ================================================================================
echo.

REM Ativar ambiente virtual se existir
if exist venv\Scripts\activate.bat (
    echo Ativando ambiente virtual...
    call venv\Scripts\activate.bat
) else (
    echo AVISO: Ambiente virtual nao encontrado
    echo Criando ambiente virtual...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo Instalando dependencias...
    pip install -r requirements.txt
)

echo.
echo Validando ambiente...
python setup.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERRO: Ambiente nao esta pronto!
    echo Corrija os erros acima antes de continuar.
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo INICIANDO BOT...
echo ================================================================================
echo.
echo IMPORTANTE: Para parar o bot, pressione Ctrl+C
echo.

python main.py

pause
