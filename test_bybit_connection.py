"""
Script de teste para verificar conexão com Bybit
"""
import ccxt
from datetime import datetime, timedelta

# Tentar diferentes configurações
configs = [
    {
        'name': 'Linear (defaultType)',
        'config': {
            'enableRateLimit': True,
            'options': {'defaultType': 'linear'}
        },
        'symbol': 'BTC/USDT:USDT'
    },
    {
        'name': 'Swap',
        'config': {
            'enableRateLimit': True,
            'options': {'defaultType': 'swap'}
        },
        'symbol': 'BTC/USDT'
    },
    {
        'name': 'Future',
        'config': {
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        },
        'symbol': 'BTC/USDT:USDT'
    },
]

for test in configs:
    print(f"\n{'='*60}")
    print(f"Testando: {test['name']}")
    print(f"{'='*60}")

    try:
        exchange = ccxt.bybit(test['config'])
        print(f"✓ Exchange inicializada")

        # Tentar baixar 10 candles
        symbol = test['symbol']
        print(f"Símbolo: {symbol}")

        since = int((datetime.utcnow() - timedelta(days=1)).timestamp() * 1000)

        candles = exchange.fetch_ohlcv(
            symbol,
            timeframe='15m',
            since=since,
            limit=10
        )

        print(f"✓ {len(candles)} candles baixados!")
        print(f"  Primeiro: {datetime.fromtimestamp(candles[0][0]/1000)}")
        print(f"  Último: {datetime.fromtimestamp(candles[-1][0]/1000)}")
        print(f"  Preço: ${candles[-1][4]:,.2f}")

        # Se funcionou, sair
        print(f"\n✅ SUCESSO com configuração: {test['name']}")
        print(f"   Use: symbol='{symbol}', defaultType='{test['config']['options']['defaultType']}'")
        break

    except Exception as e:
        print(f"✗ Erro: {e}")
        continue
