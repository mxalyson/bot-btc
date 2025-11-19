#!/bin/bash
# Script de início rápido para Linux/Mac
# Execute: bash start.sh

echo "================================================================================"
echo "🤖 BOT BTC SCALPER - INÍCIO RÁPIDO"
echo "================================================================================"
echo ""

# Ativar ambiente virtual se existir
if [ -d "venv" ]; then
    echo "Ativando ambiente virtual..."
    source venv/bin/activate
else
    echo "⚠️  Ambiente virtual não encontrado"
    echo "Criando ambiente virtual..."
    python3 -m venv venv
    source venv/bin/activate
    echo "Instalando dependências..."
    pip install -r requirements.txt
fi

echo ""
echo "Validando ambiente..."
python setup.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ ERRO: Ambiente não está pronto!"
    echo "Corrija os erros acima antes de continuar."
    exit 1
fi

echo ""
echo "================================================================================"
echo "🚀 INICIANDO BOT..."
echo "================================================================================"
echo ""
echo "⚠️  IMPORTANTE: Para parar o bot, pressione Ctrl+C"
echo ""

python main.py
