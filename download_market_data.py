"""
📥 MARKET DATA DOWNLOADER
Download robustamente dados de múltiplas exchanges com cache local

FONTES (em ordem de prioridade):
1. Cache local (CSV) - se recente
2. Binance API - mais confiável
3. Bybit API - fallback
4. Simulated data - última opção
"""

import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional


def download_market_data(
    symbol: str = 'BTCUSDT',
    days: int = 90,
    timeframe: str = '5m',
    force_download: bool = False,
    demo: bool = False
) -> pd.DataFrame:
    """
    Download dados de mercado com múltiplas fontes

    Args:
        symbol: Par de trading
        days: Número de dias históricos
        timeframe: Timeframe (5m, 15m, 1h, etc)
        force_download: Ignorar cache e baixar novamente
        demo: Usar dados simulados

    Returns:
        DataFrame com colunas: open, high, low, close, volume
    """
    print(f"\n📥 Downloading {symbol} data ({days} days, {timeframe})...")

    # Verificar cache local primeiro
    cache_file = f"data_cache_{symbol}_{days}d_{timeframe}.csv"
    if not force_download and not demo and os.path.exists(cache_file):
        try:
            cache_age = time.time() - os.path.getmtime(cache_file)
            if cache_age < 3600:  # 1 hora
                print(f"  💾 Loading from cache ({cache_file}, {cache_age/60:.0f}min old)...")
                data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                if len(data) >= days * 200:  # Mínimo razoável de candles
                    print(f"  ✅ Loaded {len(data)} candles from cache")
                    print(f"  📅 Period: {data.index[0]} to {data.index[-1]}")
                    return data
                else:
                    print(f"  ⚠️  Cache has only {len(data)} candles, downloading fresh data...")
        except Exception as e:
            print(f"  ⚠️  Cache load failed: {e}")

    if demo:
        print("  🎲 Generating simulated data...")
        data = generate_simulated_data(symbol, days, timeframe)
    else:
        # Tentar APIs reais
        data = None

        # TENTATIVA 1: BINANCE
        print("  🔄 Trying Binance API...")
        data = download_from_binance(symbol, days, timeframe)

        # TENTATIVA 2: BYBIT (se Binance falhar)
        if data is None or len(data) < days * 100:
            print("  🔄 Binance insufficient, trying Bybit API...")
            data = download_from_bybit(symbol, days, timeframe)

        # TENTATIVA 3: SIMULAÇÃO (se tudo falhar)
        if data is None or len(data) < days * 50:
            print("  ⚠️  All APIs failed, using simulated data...")
            data = generate_simulated_data(symbol, days, timeframe)
        else:
            # Salvar em cache
            try:
                data.to_csv(cache_file)
                print(f"  💾 Saved to cache: {cache_file}")
            except Exception as e:
                print(f"  ⚠️  Could not save cache: {e}")

    return data


def generate_simulated_data(symbol: str, days: int, timeframe: str) -> pd.DataFrame:
    """Gera dados simulados realistas"""
    # Converter timeframe para minutos
    if timeframe == '1m':
        minutes = 1
    elif timeframe == '5m':
        minutes = 5
    elif timeframe == '15m':
        minutes = 15
    elif timeframe == '1h':
        minutes = 60
    elif timeframe == '4h':
        minutes = 240
    else:
        minutes = 5

    candles_per_day = (24 * 60) // minutes
    n_candles = days * candles_per_day

    dates = pd.date_range(
        end=datetime.now(),
        periods=n_candles,
        freq=f'{minutes}min'
    )

    # Simula preços com GBM (Geometric Brownian Motion)
    base_price = 50000
    drift = 0.0001  # Slight upward drift
    volatility = 0.02

    returns = np.random.normal(drift, volatility, n_candles)
    price_path = base_price * np.exp(np.cumsum(returns))

    # Simula OHLC
    data = pd.DataFrame({
        'open': price_path * (1 + np.random.uniform(-0.001, 0.001, n_candles)),
        'high': price_path * (1 + np.random.uniform(0, 0.003, n_candles)),
        'low': price_path * (1 - np.random.uniform(0, 0.003, n_candles)),
        'close': price_path,
        'volume': np.random.lognormal(10, 1, n_candles)
    }, index=dates)

    # Garantir high >= open/close e low <= open/close
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)

    print(f"  ✅ Generated {len(data)} simulated candles")

    return data


def download_from_binance(symbol: str, days: int, timeframe: str) -> Optional[pd.DataFrame]:
    """Download de Binance Futures (mais líquido e confiável)"""
    try:
        import ccxt

        exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })

        # Calcular quantos candles precisamos
        timeframe_minutes = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '4h': 240, '1d': 1440
        }.get(timeframe, 5)

        candles_per_day = (24 * 60) // timeframe_minutes
        total_needed = days * candles_per_day
        max_per_request = 1500  # Binance allows up to 1500

        start_date = datetime.now() - timedelta(days=days + 1)  # Extra margin
        since = int(start_date.timestamp() * 1000)

        all_data = []
        requests = 0
        max_requests = (total_needed // max_per_request) + 5  # Safety margin

        print(f"    📊 Need ~{total_needed} candles...")

        consecutive_empty = 0

        while len(all_data) < total_needed and requests < max_requests:
            try:
                ohlcv = exchange.fetch_ohlcv(
                    symbol,
                    timeframe=timeframe,
                    since=since,
                    limit=min(max_per_request, total_needed - len(all_data) + 100)
                )

                if not ohlcv or len(ohlcv) == 0:
                    consecutive_empty += 1
                    if consecutive_empty >= 3:
                        print(f"    ⚠️  No more data available")
                        break
                    time.sleep(0.5)
                    since += (timeframe_minutes * 60 * 1000 * 100)  # Skip ahead
                    continue

                consecutive_empty = 0
                all_data.extend(ohlcv)
                requests += 1

                # Update since to last timestamp + 1 period
                since = ohlcv[-1][0] + (timeframe_minutes * 60 * 1000)

                # Progress
                if requests % 5 == 0 or len(all_data) >= total_needed:
                    print(f"    📥 Downloaded {len(all_data):,}/{total_needed:,} candles ({requests} requests)")

                # Rate limiting
                time.sleep(0.1)

            except Exception as e:
                print(f"    ⚠️  Error in request #{requests}: {str(e)[:100]}")
                if requests == 0:
                    return None
                break

        if len(all_data) == 0:
            return None

        # Convert to DataFrame
        df = pd.DataFrame(
            all_data,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        # Remove duplicates and sort
        df = df[~df.index.duplicated(keep='first')].sort_index()

        # Trim to exact period requested
        cutoff_date = datetime.now() - timedelta(days=days)
        df = df[df.index >= cutoff_date]

        print(f"    ✅ Binance: {len(df):,} candles in {requests} requests")
        print(f"    📅 Period: {df.index[0]} to {df.index[-1]}")

        return df

    except ImportError:
        print("    ⚠️  ccxt not installed (pip install ccxt)")
        return None
    except Exception as e:
        print(f"    ❌ Binance failed: {str(e)[:100]}")
        return None


def download_from_bybit(symbol: str, days: int, timeframe: str) -> Optional[pd.DataFrame]:
    """Download de Bybit Linear Perpetual"""
    try:
        import ccxt

        exchange = ccxt.bybit({
            'enableRateLimit': True,
            'options': {'defaultType': 'linear'}
        })

        timeframe_minutes = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '4h': 240, '1d': 1440
        }.get(timeframe, 5)

        candles_per_day = (24 * 60) // timeframe_minutes
        total_needed = days * candles_per_day
        max_per_request = 1000

        start_date = datetime.now() - timedelta(days=days + 1)
        since = int(start_date.timestamp() * 1000)

        all_data = []
        requests = 0
        max_requests = (total_needed // max_per_request) + 5

        print(f"    📊 Need ~{total_needed} candles...")

        consecutive_empty = 0

        while len(all_data) < total_needed and requests < max_requests:
            try:
                ohlcv = exchange.fetch_ohlcv(
                    symbol,
                    timeframe=timeframe,
                    since=since,
                    limit=min(max_per_request, total_needed - len(all_data) + 100)
                )

                if not ohlcv or len(ohlcv) == 0:
                    consecutive_empty += 1
                    if consecutive_empty >= 3:
                        break
                    time.sleep(0.5)
                    since += (timeframe_minutes * 60 * 1000 * 100)
                    continue

                consecutive_empty = 0
                all_data.extend(ohlcv)
                requests += 1
                since = ohlcv[-1][0] + (timeframe_minutes * 60 * 1000)

                if requests % 5 == 0 or len(all_data) >= total_needed:
                    print(f"    📥 Downloaded {len(all_data):,}/{total_needed:,} candles ({requests} requests)")

                time.sleep(0.1)

            except Exception as e:
                print(f"    ⚠️  Error in request #{requests}: {str(e)[:100]}")
                if requests == 0:
                    return None
                break

        if len(all_data) == 0:
            return None

        df = pd.DataFrame(
            all_data,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df[~df.index.duplicated(keep='first')].sort_index()

        cutoff_date = datetime.now() - timedelta(days=days)
        df = df[df.index >= cutoff_date]

        print(f"    ✅ Bybit: {len(df):,} candles in {requests} requests")
        print(f"    📅 Period: {df.index[0]} to {df.index[-1]}")

        return df

    except ImportError:
        print("    ⚠️  ccxt not installed")
        return None
    except Exception as e:
        print(f"    ❌ Bybit failed: {str(e)[:100]}")
        return None


if __name__ == "__main__":
    # Teste
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', default='BTCUSDT')
    parser.add_argument('--days', type=int, default=90)
    parser.add_argument('--timeframe', default='5m')
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--force', action='store_true')

    args = parser.parse_args()

    data = download_market_data(
        symbol=args.symbol,
        days=args.days,
        timeframe=args.timeframe,
        force_download=args.force,
        demo=args.demo
    )

    print(f"\n📊 SUMMARY:")
    print(f"   Total candles: {len(data):,}")
    print(f"   Period: {data.index[0]} to {data.index[-1]}")
    print(f"   Days: {(data.index[-1] - data.index[0]).days}")
    print(f"\n   First 5 rows:")
    print(data.head())
    print(f"\n   Last 5 rows:")
    print(data.tail())
    print(f"\n   Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    print(f"   Current price: ${data['close'].iloc[-1]:.2f}")
