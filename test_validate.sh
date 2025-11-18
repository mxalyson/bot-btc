#!/bin/bash
# Script de teste rápido do validate_optimized_ultra_scalper.py

echo "🧪 TESTANDO VALIDATE_OPTIMIZED_ULTRA_SCALPER.PY"
echo "=" | tr '\n' '='  | head -c 80 && echo

echo ""
echo "Teste 1: Demo mode (30 days)"
python validate_optimized_ultra_scalper.py --demo --days 30

echo ""
echo "="| tr '\n' '=' | head -c 80 && echo
echo "✅ TESTE COMPLETO!"
echo ""
echo "Próximo passo: Rode com seus dados reais:"
echo "  python validate_optimized_ultra_scalper.py --symbol BTCUSDT --days 90"
