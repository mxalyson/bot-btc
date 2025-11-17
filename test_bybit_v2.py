"""
Teste Bybit API v5 com categoria linear forçada
"""
import ccxt
from datetime import datetime, timedelta

print("Testando Bybit API v5...")

# Configurar exchange
exchange = ccxt.bybit({
    'enableRateLimit': True,
})

# Listar markets para debug
try:
    print("\nCarregando markets...")
    markets = exchange.load_markets()
    print(f"✓ {len(markets)} markets carregados")

    # Procurar BTC/USDT linear
    linear_btc = [m for m in markets.keys() if 'BTC' in m and 'USDT' in m]
    print(f"\nMarkets BTC/USDT disponíveis: {linear_btc[:10]}")

    # Tentar o primeiro que encontrar
    if linear_btc:
        symbol = linear_btc[0]
        print(f"\nTentando baixar dados de: {symbol}")

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
        print(f"\n✅ SUCESSO! Use symbol: '{symbol}'")

except Exception as e:
    print(f"✗ Erro: {e}")
    import traceback
    traceback.print_exc()
