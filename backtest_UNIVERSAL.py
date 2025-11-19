#!/usr/bin/env python3
"""
🚀 BACKTEST UNIVERSAL - FLEXÍVEL E PODEROSO

Aceita qualquer modelo .pkl e permite ajustar:
- Threshold (long/short)
- Período de teste
- Capital inicial
- Parâmetros de risco

Suporta:
- Binance (via requests, sem precisar módulo binance)
- Bybit (se pybit instalado)
- CSV (carrega dados locais)

USO:
    python backtest_UNIVERSAL.py
    python backtest_UNIVERSAL.py --model storage/models/model_V6_365d.pkl
    python backtest_UNIVERSAL.py --long-threshold 0.50 --short-threshold 0.50
    python backtest_UNIVERSAL.py --days 180 --capital 10000
    python backtest_UNIVERSAL.py --exchange bybit
    python backtest_UNIVERSAL.py --csv data/btcusdt_15m.csv
"""

import sys
import os
import json
import argparse
import pickle
import warnings
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings('ignore')


class ModelWrapper:
    """Wrapper universal para qualquer combinação de modelos."""

    def __init__(self, models_list, model_weights, model_names, scaler,
                 feature_columns, has_dl=False, long_threshold=0.50,
                 short_threshold=0.50, lookback=10):
        self.models_list = models_list
        self.model_weights = model_weights
        self.model_names = model_names
        self.scaler = scaler
        self.feature_columns = feature_columns
        self.has_dl = has_dl
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        self.lookback = lookback

    def predict(self, X):
        """Predict with weighted ensemble."""
        X_scaled = self.scaler.transform(X[self.feature_columns])

        # Get predictions from all models
        predictions = []
        for model, weight in zip(self.models_list, self.model_weights):
            try:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X_scaled)[:, 1]
                else:
                    # For DL models that might need sequences
                    proba = model.predict(X_scaled).flatten()
                predictions.append(proba * weight)
            except Exception as e:
                print(f"   ⚠️  Warning: Model failed, skipping: {e}")
                continue

        if not predictions:
            raise ValueError("All models failed to predict!")

        # Weighted average
        proba = np.sum(predictions, axis=0)

        # Apply thresholds
        final_predictions = np.zeros(len(proba), dtype=int)
        final_predictions[proba >= self.long_threshold] = 1
        final_predictions[proba <= (1 - self.short_threshold)] = 0

        return final_predictions

    def predict_proba(self, X):
        """Get probability scores."""
        X_scaled = self.scaler.transform(X[self.feature_columns])

        predictions = []
        for model, weight in zip(self.models_list, self.model_weights):
            try:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X_scaled)[:, 1]
                else:
                    proba = model.predict(X_scaled).flatten()
                predictions.append(proba * weight)
            except:
                continue

        if not predictions:
            raise ValueError("All models failed to predict!")

        proba = np.sum(predictions, axis=0)

        # Return as (n_samples, 2) array
        result = np.zeros((len(proba), 2))
        result[:, 0] = 1 - proba
        result[:, 1] = proba

        return result


def load_model(model_path):
    """Load model from pickle file."""
    print(f"\n📦 Carregando modelo: {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

    with open(model_path, 'rb') as f:
        wrapper = pickle.load(f)

    print(f"   ✅ Modelo carregado!")
    print(f"   Modelos: {len(wrapper.models_list)}")
    print(f"   Features: {len(wrapper.feature_columns)}")
    print(f"   Long threshold: {wrapper.long_threshold}")
    print(f"   Short threshold: {wrapper.short_threshold}")
    print(f"   Has DL: {wrapper.has_dl}")

    return wrapper


def fetch_binance_data(symbol, interval, days):
    """Fetch data from Binance using requests (no binance module needed)."""
    print(f"\n📥 Baixando dados da Binance...")
    print(f"   Symbol: {symbol}")
    print(f"   Interval: {interval}")
    print(f"   Days: {days}")

    all_data = []
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

    url = "https://api.binance.com/api/v3/klines"
    current_time = start_time

    while current_time < end_time:
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': current_time,
            'limit': 1000
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            klines = response.json()

            if not klines:
                break

            all_data.extend(klines)
            current_time = klines[-1][0] + 1

            if len(klines) < 1000:
                break

        except Exception as e:
            print(f"   ⚠️  Erro: {e}")
            break

    if not all_data:
        raise ValueError("Nenhum dado baixado!")

    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])

    # Convert types
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

    df['taker_buy_volume'] = df['taker_buy_base'].astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('timestamp')

    print(f"   ✅ {len(df)} candles baixados")
    print(f"   Período: {df.index[0]} a {df.index[-1]}")

    return df


def fetch_bybit_data(symbol, interval, days):
    """Fetch data from Bybit using pybit."""
    print(f"\n📥 Baixando dados da Bybit...")
    print(f"   Symbol: {symbol}")
    print(f"   Interval: {interval}")
    print(f"   Days: {days}")

    try:
        from pybit.unified_trading import HTTP
    except ImportError:
        raise ImportError("pybit não instalado. Execute: pip install pybit")

    session = HTTP(testnet=False)

    # Bybit intervals
    interval_map = {
        '1m': '1', '3m': '3', '5m': '5', '15m': '15',
        '30m': '30', '1h': '60', '2h': '120', '4h': '240',
        '6h': '360', '12h': '720', '1d': 'D', '1w': 'W'
    }

    bybit_interval = interval_map.get(interval, '15')

    all_data = []
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    current_time = start_time

    while current_time < end_time:
        try:
            response = session.get_kline(
                category="linear",
                symbol=symbol,
                interval=bybit_interval,
                start=current_time,
                limit=200
            )

            if response['retCode'] != 0:
                print(f"   ⚠️  Erro Bybit: {response['retMsg']}")
                break

            klines = response['result']['list']

            if not klines:
                break

            all_data.extend(klines)

            # Bybit returns newest first, so we need oldest
            current_time = int(klines[-1][0]) + 1

            if len(klines) < 200:
                break

        except Exception as e:
            print(f"   ⚠️  Erro: {e}")
            break

    if not all_data:
        raise ValueError("Nenhum dado baixado!")

    # Bybit format: [timestamp, open, high, low, close, volume, turnover]
    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
    ])

    # Convert types
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
    df = df.set_index('timestamp')
    df = df.sort_index()  # Bybit returns newest first

    # Calculate taker_buy_volume (estimate as 50% for Bybit)
    df['taker_buy_volume'] = df['volume'] * 0.5

    print(f"   ✅ {len(df)} candles baixados")
    print(f"   Período: {df.index[0]} a {df.index[-1]}")

    return df


def load_csv_data(csv_path):
    """Load data from CSV file."""
    print(f"\n📂 Carregando dados do CSV: {csv_path}")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV não encontrado: {csv_path}")

    df = pd.read_csv(csv_path)

    # Try to parse timestamp column
    timestamp_cols = ['timestamp', 'time', 'date', 'datetime']
    timestamp_col = None

    for col in timestamp_cols:
        if col in df.columns:
            timestamp_col = col
            break

    if timestamp_col:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df = df.set_index(timestamp_col)
    elif 'Unnamed: 0' in df.columns:
        df = df.rename(columns={'Unnamed: 0': 'timestamp'})
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')

    # Ensure required columns exist
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Coluna '{col}' não encontrada no CSV!")

    # Add taker_buy_volume if not present (estimate as 50%)
    if 'taker_buy_volume' not in df.columns:
        df['taker_buy_volume'] = df['volume'] * 0.5
        print("   ℹ️  taker_buy_volume não encontrado, usando 50% do volume")

    print(f"   ✅ {len(df)} candles carregados")
    print(f"   Período: {df.index[0]} a {df.index[-1]}")

    return df


def create_features(df):
    """Create ALL 87 extraordinary features that the model expects."""

    # ========================================================================
    # PRICE ACTION AVANÇADO
    # ========================================================================
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

    # Body/Wick analysis (importante para reversão)
    df['body_size'] = np.abs(df['close'] - df['open']) / df['open']
    df['upper_wick'] = (df['high'] - df[['close', 'open']].max(axis=1)) / df['open']
    df['lower_wick'] = (df[['close', 'open']].min(axis=1) - df['low']) / df['open']
    df['total_wick'] = df['upper_wick'] + df['lower_wick']
    df['wick_body_ratio'] = df['total_wick'] / (df['body_size'] + 1e-8)

    # Candle patterns
    df['is_green'] = (df['close'] > df['open']).astype(int)
    df['green_streak'] = df['is_green'].rolling(3).sum()
    df['red_streak'] = (1 - df['is_green']).rolling(3).sum()

    # ========================================================================
    # ORDER FLOW - BUY/SELL PRESSURE
    # ========================================================================
    # Taker buy ratio (muito importante para scalping!)
    df['taker_buy_ratio'] = df['taker_buy_volume'] / (df['volume'] + 1e-8)
    df['taker_sell_ratio'] = 1 - df['taker_buy_ratio']

    # Buy/Sell pressure momentum
    df['buy_pressure_ma'] = df['taker_buy_ratio'].rolling(7).mean()
    df['sell_pressure_ma'] = df['taker_sell_ratio'].rolling(7).mean()
    df['pressure_delta'] = df['buy_pressure_ma'] - df['sell_pressure_ma']
    df['pressure_momentum'] = df['pressure_delta'].diff(3)

    # Order flow imbalance
    df['order_imbalance'] = (df['taker_buy_volume'] - (df['volume'] - df['taker_buy_volume'])) / (df['volume'] + 1e-8)
    df['imbalance_ma'] = df['order_imbalance'].rolling(5).mean()

    # ========================================================================
    # MOVING AVERAGES + CROSSOVERS
    # ========================================================================
    for period in [7, 14, 21, 50]:
        df[f'sma_{period}'] = df['close'].rolling(period).mean()
        df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        df[f'price_sma_{period}_ratio'] = df['close'] / df[f'sma_{period}']

    # Crossover signals
    df['ema7_above_ema14'] = (df['ema_7'] > df['ema_14']).astype(int)
    df['ema14_above_ema21'] = (df['ema_14'] > df['ema_21']).astype(int)
    df['ema21_above_ema50'] = (df['ema_21'] > df['ema_50']).astype(int)

    # Golden/Death cross
    df['golden_cross'] = df['ema7_above_ema14'] & df['ema14_above_ema21']
    df['death_cross'] = (1 - df['ema7_above_ema14']) & (1 - df['ema14_above_ema21'])

    # ========================================================================
    # VOLATILITY (ATR é KEY para SL/TP)
    # ========================================================================
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['atr_14'] = true_range.rolling(14).mean()
    df['atr_pct'] = df['atr_14'] / df['close']

    # Volatility regimes
    df['volatility_7'] = df['returns'].rolling(7).std()
    df['volatility_21'] = df['returns'].rolling(21).std()
    df['volatility_ratio'] = df['volatility_7'] / (df['volatility_21'] + 1e-8)

    # Volatility spike
    df['high_volatility'] = (df['volatility_ratio'] > 1.3).astype(int)
    df['low_volatility'] = (df['volatility_ratio'] < 0.7).astype(int)

    # ========================================================================
    # RSI AVANÇADO
    # ========================================================================
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # RSI zones
    df['rsi_extreme_oversold'] = (df['rsi_14'] < 25).astype(int)
    df['rsi_extreme_overbought'] = (df['rsi_14'] > 75).astype(int)
    df['rsi_mid'] = ((df['rsi_14'] >= 45) & (df['rsi_14'] <= 55)).astype(int)

    # RSI divergence
    df['price_slope'] = df['close'].diff(5)
    df['rsi_slope'] = df['rsi_14'].diff(5)
    df['bullish_divergence'] = ((df['rsi_slope'] > 0) & (df['price_slope'] < 0)).astype(int)
    df['bearish_divergence'] = ((df['rsi_slope'] < 0) & (df['price_slope'] > 0)).astype(int)

    # ========================================================================
    # MACD
    # ========================================================================
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # MACD signals
    df['macd_positive'] = (df['macd'] > 0).astype(int)
    df['macd_hist_increasing'] = (df['macd_hist'] > df['macd_hist'].shift(1)).astype(int)

    # ========================================================================
    # BOLLINGER BANDS
    # ========================================================================
    df['bb_middle'] = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)

    # BB breakout
    df['bb_upper_breakout'] = (df['close'] > df['bb_upper']).astype(int)
    df['bb_lower_breakout'] = (df['close'] < df['bb_lower']).astype(int)

    # ========================================================================
    # VOLUME (confirmação crucial)
    # ========================================================================
    df['volume_sma_20'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_sma_20'] + 1e-8)
    df['high_volume'] = (df['volume_ratio'] > 1.5).astype(int)

    # Volume trend
    df['volume_slope'] = df['volume'].diff(3)
    df['volume_increasing_trend'] = (df['volume_slope'] > 0).astype(int).rolling(3).sum()

    # ========================================================================
    # MOMENTUM
    # ========================================================================
    df['momentum_3'] = df['close'] / df['close'].shift(3) - 1
    df['momentum_7'] = df['close'] / df['close'].shift(7) - 1
    df['momentum_14'] = df['close'] / df['close'].shift(14) - 1

    # Momentum acceleration
    df['momentum_accel'] = df['momentum_7'].diff(3)

    # ========================================================================
    # PRICE POSITION
    # ========================================================================
    df['price_position_14'] = (df['close'] - df['low'].rolling(14).min()) / \
                               (df['high'].rolling(14).max() - df['low'].rolling(14).min() + 1e-8)

    df['price_position_50'] = (df['close'] - df['low'].rolling(50).min()) / \
                               (df['high'].rolling(50).max() - df['low'].rolling(50).min() + 1e-8)

    # ========================================================================
    # TREND STRENGTH
    # ========================================================================
    df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
    df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
    df['hh_count'] = df['higher_high'].rolling(5).sum()
    df['ll_count'] = df['lower_low'].rolling(5).sum()
    df['trend_strength'] = df['hh_count'] - df['ll_count']

    # ========================================================================
    # TIME-BASED FEATURES (importante para scalping!)
    # ========================================================================
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek

    # Trading session (UTC)
    df['asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
    df['london_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
    df['us_session'] = ((df['hour'] >= 13) & (df['hour'] < 22)).astype(int)

    # Weekend effect
    df['weekend'] = (df['day_of_week'] >= 5).astype(int)

    # ========================================================================
    # MICROSTRUCTURE
    # ========================================================================
    # Spread proxy (high-low range)
    df['spread_proxy'] = (df['high'] - df['low']) / df['close']
    df['spread_ma'] = df['spread_proxy'].rolling(10).mean()

    # Price impact (large candles)
    df['large_candle'] = (df['body_size'] > df['body_size'].rolling(20).mean() * 1.5).astype(int)

    # ========================================================================
    # CLEAN
    # ========================================================================
    df = df.dropna()

    return df


def run_backtest(df, wrapper, config):
    """Run backtest simulation."""

    print("\n" + "="*80)
    print("🚀 EXECUTANDO BACKTEST")
    print("="*80)
    print(f"\n📊 Período: {df.index[0]} a {df.index[-1]}")
    print(f"   Candles: {len(df):,}")

    # Generate predictions
    print(f"\n💹 Gerando predições...")

    try:
        predictions = wrapper.predict(df)

        long_pct = (predictions == 1).sum() / len(predictions) * 100
        short_pct = (predictions == 0).sum() / len(predictions) * 100

        print(f"\n   ✅ {len(predictions)} predições geradas")
        print(f"   Longs preditos: {(predictions == 1).sum()} ({long_pct:.2f}%)")
        print(f"   Shorts preditos: {(predictions == 0).sum()} ({short_pct:.2f}%)")

        if long_pct > 80 or short_pct > 80:
            print(f"\n   ⚠️  AVISO: Predições muito desbalanceadas!")
            print(f"   Considere ajustar thresholds:")
            print(f"   --long-threshold {wrapper.long_threshold} --short-threshold {wrapper.short_threshold}")
            print()

    except Exception as e:
        print(f"   ❌ Erro ao gerar predições: {e}")
        import traceback
        traceback.print_exc()
        return

    # Simulate trades
    print(f"\n💰 Simulando trades...")

    initial_capital = config.get('initial_capital', 10000)
    risk_per_trade = config.get('risk_per_trade', 2.0)
    sl_mult = config.get('sl_atr_mult', 1.5)
    tp_mult = config.get('tp_atr_mult', 2.0)

    balance = initial_capital
    capital_per_trade = balance * (risk_per_trade / 100)

    trades = []

    for i in range(len(df) - 1):
        signal = predictions[i]

        if signal == 1:  # Long
            entry_price = df.iloc[i]['close']
            atr = df.iloc[i]['atr_14']
            sl_price = entry_price - (atr * sl_mult)
            tp_price = entry_price + (atr * tp_mult)

            # Check exit on next candles
            for j in range(i + 1, min(i + 50, len(df))):
                low = df.iloc[j]['low']
                high = df.iloc[j]['high']

                # Stop loss hit
                if low <= sl_price:
                    exit_price = sl_price
                    pnl = ((exit_price - entry_price) / entry_price) * capital_per_trade
                    trades.append({
                        'entry_time': df.index[i],
                        'exit_time': df.index[j],
                        'type': 'LONG',
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'result': 'LOSS'
                    })
                    break

                # Take profit hit
                elif high >= tp_price:
                    exit_price = tp_price
                    pnl = ((exit_price - entry_price) / entry_price) * capital_per_trade
                    trades.append({
                        'entry_time': df.index[i],
                        'exit_time': df.index[j],
                        'type': 'LONG',
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'result': 'WIN'
                    })
                    break

        elif signal == 0:  # Short
            entry_price = df.iloc[i]['close']
            atr = df.iloc[i]['atr_14']
            sl_price = entry_price + (atr * sl_mult)
            tp_price = entry_price - (atr * tp_mult)

            # Check exit on next candles
            for j in range(i + 1, min(i + 50, len(df))):
                low = df.iloc[j]['low']
                high = df.iloc[j]['high']

                # Stop loss hit
                if high >= sl_price:
                    exit_price = sl_price
                    pnl = ((entry_price - exit_price) / entry_price) * capital_per_trade
                    trades.append({
                        'entry_time': df.index[i],
                        'exit_time': df.index[j],
                        'type': 'SHORT',
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'result': 'LOSS'
                    })
                    break

                # Take profit hit
                elif low <= tp_price:
                    exit_price = tp_price
                    pnl = ((entry_price - exit_price) / entry_price) * capital_per_trade
                    trades.append({
                        'entry_time': df.index[i],
                        'exit_time': df.index[j],
                        'type': 'SHORT',
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'result': 'WIN'
                    })
                    break

    # Calculate metrics
    if not trades:
        print("\n   ❌ Nenhum trade executado!")
        return

    trades_df = pd.DataFrame(trades)

    total_trades = len(trades_df)
    wins = len(trades_df[trades_df['result'] == 'WIN'])
    losses = len(trades_df[trades_df['result'] == 'LOSS'])
    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0

    total_pnl = trades_df['pnl'].sum()
    roi = (total_pnl / initial_capital) * 100

    avg_win = trades_df[trades_df['result'] == 'WIN']['pnl'].mean() if wins > 0 else 0
    avg_loss = trades_df[trades_df['result'] == 'LOSS']['pnl'].mean() if losses > 0 else 0
    risk_reward = abs(avg_win / avg_loss) if avg_loss != 0 else 0

    # By type
    longs = trades_df[trades_df['type'] == 'LONG']
    shorts = trades_df[trades_df['type'] == 'SHORT']

    long_wr = (len(longs[longs['result'] == 'WIN']) / len(longs)) * 100 if len(longs) > 0 else 0
    short_wr = (len(shorts[shorts['result'] == 'WIN']) / len(shorts)) * 100 if len(shorts) > 0 else 0

    # Print results
    print("\n" + "="*80)
    print("📊 RESULTADOS DO BACKTEST")
    print("="*80)

    print(f"\n💼 Trades Executados:")
    print(f"   Total: {total_trades}")
    print(f"   Longs: {len(longs)} ({len(longs)/total_trades*100:.1f}%)")
    print(f"   Shorts: {len(shorts)} ({len(shorts)/total_trades*100:.1f}%)")

    print(f"\n📈 Performance:")
    print(f"   Win Rate: {win_rate:.2f}% ({wins}W / {losses}L)")
    print(f"   Long WR: {long_wr:.2f}%")
    print(f"   Short WR: {short_wr:.2f}%")

    print(f"\n💰 Financeiro:")
    print(f"   Capital Inicial: ${initial_capital:,.2f}")
    print(f"   PnL Total: ${total_pnl:,.2f}")
    print(f"   ROI: {roi:+.2f}%")
    print(f"   Capital Final: ${initial_capital + total_pnl:,.2f}")

    print(f"\n⚖️  Risco/Retorno:")
    print(f"   Ganho Médio: ${avg_win:.2f}")
    print(f"   Perda Média: ${avg_loss:.2f}")
    print(f"   Risk/Reward: {risk_reward:.2f}")

    print(f"\n🎯 Parâmetros Usados:")
    print(f"   Long Threshold: {wrapper.long_threshold}")
    print(f"   Short Threshold: {wrapper.short_threshold}")
    print(f"   SL ATR Mult: {sl_mult}")
    print(f"   TP ATR Mult: {tp_mult}")
    print(f"   Risk per Trade: {risk_per_trade}%")

    # Approval criteria
    print("\n" + "="*80)
    print("✅ CRITÉRIOS DE APROVAÇÃO")
    print("="*80)

    criteria_met = 0
    total_criteria = 7

    checks = [
        ("Win Rate ≥ 48%", win_rate >= 48),
        ("ROI > 0%", roi > 0),
        ("Longs 40-60%", 40 <= (len(longs)/total_trades*100) <= 60),
        ("Shorts 40-60%", 40 <= (len(shorts)/total_trades*100) <= 60),
        ("Long WR 45-55%", 45 <= long_wr <= 55),
        ("Short WR 45-55%", 45 <= short_wr <= 55),
        ("Risk/Reward > 1.0", risk_reward > 1.0),
    ]

    for criteria, passed in checks:
        status = "✅" if passed else "❌"
        print(f"   {status} {criteria}")
        if passed:
            criteria_met += 1

    print(f"\n   Aprovação: {criteria_met}/{total_criteria} critérios atendidos")

    if criteria_met >= 5:
        print("\n   🎉 MODELO APROVADO! Pode usar em paper trading.")
    else:
        print("\n   ⚠️  MODELO REPROVADO. Considere ajustar thresholds ou retreinar.")


def main():
    """Main entry point."""

    parser = argparse.ArgumentParser(
        description='Backtest Universal - Teste qualquer modelo ML com flexibilidade total',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  %(prog)s
  %(prog)s --model storage/models/model_V6_365d.pkl
  %(prog)s --long-threshold 0.50 --short-threshold 0.50
  %(prog)s --exchange bybit
  %(prog)s --csv data/btcusdt_15m.csv
  %(prog)s --model my_model.pkl --long-threshold 0.45 --short-threshold 0.55 --days 90
        """
    )

    parser.add_argument('--model', type=str,
                        default='storage/models/model_DEFINITIVO_4ML_365d.pkl',
                        help='Caminho do modelo .pkl (default: model_DEFINITIVO_4ML_365d.pkl)')

    parser.add_argument('--long-threshold', type=float, default=None,
                        help='Threshold para long (0.0-1.0). Se não especificado, usa o do modelo.')

    parser.add_argument('--short-threshold', type=float, default=None,
                        help='Threshold para short (0.0-1.0). Se não especificado, usa o do modelo.')

    parser.add_argument('--exchange', type=str, default='binance', choices=['binance', 'bybit'],
                        help='Exchange para baixar dados (default: binance)')

    parser.add_argument('--csv', type=str, default=None,
                        help='Carregar dados de CSV ao invés de baixar da exchange')

    parser.add_argument('--days', type=int, default=52,
                        help='Dias de dados para backtest (default: 52 = ~2 meses)')

    parser.add_argument('--capital', type=float, default=10000,
                        help='Capital inicial (default: 10000)')

    parser.add_argument('--risk-per-trade', type=float, default=2.0,
                        help='Risco por trade em %% (default: 2.0)')

    parser.add_argument('--sl-atr-mult', type=float, default=1.5,
                        help='Multiplicador ATR para stop loss (default: 1.5)')

    parser.add_argument('--tp-atr-mult', type=float, default=2.0,
                        help='Multiplicador ATR para take profit (default: 2.0)')

    parser.add_argument('--symbol', type=str, default='BTCUSDT',
                        help='Symbol para trade (default: BTCUSDT)')

    parser.add_argument('--interval', type=str, default='15m',
                        help='Intervalo de candles (default: 15m)')

    args = parser.parse_args()

    print("="*80)
    print("🚀 BACKTEST UNIVERSAL - FLEXÍVEL E PODEROSO")
    print("="*80)

    # Load model
    wrapper = load_model(args.model)

    # Override thresholds if specified
    if args.long_threshold is not None:
        print(f"\n🔧 Ajustando long threshold: {wrapper.long_threshold} → {args.long_threshold}")
        wrapper.long_threshold = args.long_threshold

    if args.short_threshold is not None:
        print(f"🔧 Ajustando short threshold: {wrapper.short_threshold} → {args.short_threshold}")
        wrapper.short_threshold = args.short_threshold

    # Fetch data
    if args.csv:
        df = load_csv_data(args.csv)
    elif args.exchange == 'binance':
        df = fetch_binance_data(args.symbol, args.interval, args.days)
    elif args.exchange == 'bybit':
        df = fetch_bybit_data(args.symbol, args.interval, args.days)
    else:
        raise ValueError(f"Exchange não suportada: {args.exchange}")

    # Create features
    print(f"\n🔧 Criando features...")
    df = create_features(df)
    print(f"   ✅ Features criadas: {len(df.columns)}")

    # Limit to last 5000 candles for speed
    if len(df) > 5000:
        df = df.tail(5000)
        print(f"   ℹ️  Limitado a últimos 5000 candles para velocidade")

    # Config
    config = {
        'initial_capital': args.capital,
        'risk_per_trade': args.risk_per_trade,
        'sl_atr_mult': args.sl_atr_mult,
        'tp_atr_mult': args.tp_atr_mult,
    }

    # Run backtest
    run_backtest(df, wrapper, config)


if __name__ == '__main__':
    main()
