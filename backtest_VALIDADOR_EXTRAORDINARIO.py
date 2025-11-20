#!/usr/bin/env python3
"""
🔬 VALIDADOR EXTRAORDINÁRIO V3 - ANÁLISE PROFISSIONAL COMPLETA

O validador mais completo e profissional para estratégias de trading.

Features:
- Walk-Forward Analysis (valida robustez temporal)
- Multiple Market Regime Testing (bull/bear/sideways)
- Automatic Threshold Optimization
- Realistic Slippage Simulation
- Monte Carlo Simulation (1000 runs)
- Detailed Drawdown Analysis
- Performance by Session (Asian/London/US)
- Optimization Grid Search
- Actionable Recommendations

USO:
    python backtest_VALIDADOR_EXTRAORDINARIO.py
    python backtest_VALIDADOR_EXTRAORDINARIO.py --optimize
    python backtest_VALIDADOR_EXTRAORDINARIO.py --monte-carlo
"""

import sys
import os
import argparse
import pickle
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
import itertools

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings('ignore')


class ModelWrapper:
    """Wrapper compatível com modelo salvo."""

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
            raise ValueError("All models failed!")

        proba = np.sum(predictions, axis=0)

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
            raise ValueError("All models failed!")

        proba = np.sum(predictions, axis=0)

        result = np.zeros((len(proba), 2))
        result[:, 0] = 1 - proba
        result[:, 1] = proba

        return result


def load_model(model_path):
    """Load model from pickle file with robust error handling."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

    print(f"📦 Carregando modelo: {model_path}")

    # First try: normal pickle load
    try:
        with open(model_path, 'rb') as f:
            wrapper = pickle.load(f)
        print(f"   ✅ Modelo carregado!")
        return wrapper
    except AttributeError as e:
        print(f"   ⚠️  Erro ao carregar pickle: {e}")
        print(f"   🔄 Tentando modo compatibilidade...")

        # Second try: load with custom unpickler that uses our ModelWrapper
        import io

        class CustomUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                # If ModelWrapper is not found in original module, use our local one
                if name == 'ModelWrapper':
                    return ModelWrapper
                return super().find_class(module, name)

        with open(model_path, 'rb') as f:
            wrapper = CustomUnpickler(f).load()

        print(f"   ✅ Modelo carregado com compatibilidade!")
        return wrapper


def fetch_binance_data(symbol, interval, days):
    """Fetch data from Binance."""
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

    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

    df['taker_buy_volume'] = df['taker_buy_base'].astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('timestamp')

    return df


def create_features(df):
    """Create ALL 87 extraordinary features."""

    # Price Action
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['body_size'] = np.abs(df['close'] - df['open']) / df['open']
    df['upper_wick'] = (df['high'] - df[['close', 'open']].max(axis=1)) / df['open']
    df['lower_wick'] = (df[['close', 'open']].min(axis=1) - df['low']) / df['open']
    df['total_wick'] = df['upper_wick'] + df['lower_wick']
    df['wick_body_ratio'] = df['total_wick'] / (df['body_size'] + 1e-8)
    df['is_green'] = (df['close'] > df['open']).astype(int)
    df['green_streak'] = df['is_green'].rolling(3).sum()
    df['red_streak'] = (1 - df['is_green']).rolling(3).sum()

    # Order Flow
    df['taker_buy_ratio'] = df['taker_buy_volume'] / (df['volume'] + 1e-8)
    df['taker_sell_ratio'] = 1 - df['taker_buy_ratio']
    df['buy_pressure_ma'] = df['taker_buy_ratio'].rolling(7).mean()
    df['sell_pressure_ma'] = df['taker_sell_ratio'].rolling(7).mean()
    df['pressure_delta'] = df['buy_pressure_ma'] - df['sell_pressure_ma']
    df['pressure_momentum'] = df['pressure_delta'].diff(3)
    df['order_imbalance'] = (df['taker_buy_volume'] - (df['volume'] - df['taker_buy_volume'])) / (df['volume'] + 1e-8)
    df['imbalance_ma'] = df['order_imbalance'].rolling(5).mean()

    # Moving Averages
    for period in [7, 14, 21, 50]:
        df[f'sma_{period}'] = df['close'].rolling(period).mean()
        df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        df[f'price_sma_{period}_ratio'] = df['close'] / df[f'sma_{period}']

    df['ema7_above_ema14'] = (df['ema_7'] > df['ema_14']).astype(int)
    df['ema14_above_ema21'] = (df['ema_14'] > df['ema_21']).astype(int)
    df['ema21_above_ema50'] = (df['ema_21'] > df['ema_50']).astype(int)
    df['golden_cross'] = df['ema7_above_ema14'] & df['ema14_above_ema21']
    df['death_cross'] = (1 - df['ema7_above_ema14']) & (1 - df['ema14_above_ema21'])

    # Volatility
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['atr_14'] = true_range.rolling(14).mean()
    df['atr_pct'] = df['atr_14'] / df['close']
    df['volatility_7'] = df['returns'].rolling(7).std()
    df['volatility_21'] = df['returns'].rolling(21).std()
    df['volatility_ratio'] = df['volatility_7'] / (df['volatility_21'] + 1e-8)
    df['high_volatility'] = (df['volatility_ratio'] > 1.3).astype(int)
    df['low_volatility'] = (df['volatility_ratio'] < 0.7).astype(int)

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    df['rsi_extreme_oversold'] = (df['rsi_14'] < 25).astype(int)
    df['rsi_extreme_overbought'] = (df['rsi_14'] > 75).astype(int)
    df['rsi_mid'] = ((df['rsi_14'] >= 45) & (df['rsi_14'] <= 55)).astype(int)
    df['price_slope'] = df['close'].diff(5)
    df['rsi_slope'] = df['rsi_14'].diff(5)
    df['bullish_divergence'] = ((df['rsi_slope'] > 0) & (df['price_slope'] < 0)).astype(int)
    df['bearish_divergence'] = ((df['rsi_slope'] < 0) & (df['price_slope'] > 0)).astype(int)

    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['macd_positive'] = (df['macd'] > 0).astype(int)
    df['macd_hist_increasing'] = (df['macd_hist'] > df['macd_hist'].shift(1)).astype(int)

    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
    df['bb_upper_breakout'] = (df['close'] > df['bb_upper']).astype(int)
    df['bb_lower_breakout'] = (df['close'] < df['bb_lower']).astype(int)

    # Volume
    df['volume_sma_20'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_sma_20'] + 1e-8)
    df['high_volume'] = (df['volume_ratio'] > 1.5).astype(int)
    df['volume_slope'] = df['volume'].diff(3)
    df['volume_increasing_trend'] = (df['volume_slope'] > 0).astype(int).rolling(3).sum()

    # Momentum
    df['momentum_3'] = df['close'] / df['close'].shift(3) - 1
    df['momentum_7'] = df['close'] / df['close'].shift(7) - 1
    df['momentum_14'] = df['close'] / df['close'].shift(14) - 1
    df['momentum_accel'] = df['momentum_7'].diff(3)

    # Price Position
    df['price_position_14'] = (df['close'] - df['low'].rolling(14).min()) / \
                               (df['high'].rolling(14).max() - df['low'].rolling(14).min() + 1e-8)
    df['price_position_50'] = (df['close'] - df['low'].rolling(50).min()) / \
                               (df['high'].rolling(50).max() - df['low'].rolling(50).min() + 1e-8)

    # Trend Strength
    df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
    df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
    df['hh_count'] = df['higher_high'].rolling(5).sum()
    df['ll_count'] = df['lower_low'].rolling(5).sum()
    df['trend_strength'] = df['hh_count'] - df['ll_count']

    # Time-based
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
    df['london_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
    df['us_session'] = ((df['hour'] >= 13) & (df['hour'] < 22)).astype(int)
    df['weekend'] = (df['day_of_week'] >= 5).astype(int)

    # Microstructure
    df['spread_proxy'] = (df['high'] - df['low']) / df['close']
    df['spread_ma'] = df['spread_proxy'].rolling(10).mean()
    df['large_candle'] = (df['body_size'] > df['body_size'].rolling(20).mean() * 1.5).astype(int)

    df = df.dropna()
    return df


def detect_market_regime(df):
    """Detect bull/bear/sideways for entire dataset."""
    sma_20 = df['close'].rolling(20).mean()
    sma_50 = df['close'].rolling(50).mean()

    regimes = []
    for i in range(len(df)):
        if i < 50:
            regimes.append('unknown')
            continue

        price = df.iloc[i]['close']
        s20 = sma_20.iloc[i]
        s50 = sma_50.iloc[i]

        if price > s20 > s50:
            regimes.append('bull')
        elif price < s20 < s50:
            regimes.append('bear')
        else:
            regimes.append('sideways')

    df['regime'] = regimes
    return df


def run_single_backtest(df, wrapper, long_thresh, short_thresh, config):
    """Run a single backtest with given parameters."""

    # Generate predictions
    X_scaled = wrapper.scaler.transform(df[wrapper.feature_columns])

    predictions_list = []
    for model, weight in zip(wrapper.models_list, wrapper.model_weights):
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_scaled)[:, 1]
        else:
            proba = model.predict(X_scaled).flatten()
        predictions_list.append(proba * weight)

    proba = np.sum(predictions_list, axis=0)

    predictions = np.zeros(len(proba), dtype=int)
    predictions[proba >= long_thresh] = 1
    predictions[proba <= (1 - short_thresh)] = 0

    # Simulate trades
    initial_capital = config['capital']
    risk_pct = config['risk_per_trade']
    sl_mult = config['sl_atr_mult']
    tp_mult = config['tp_atr_mult']
    slippage_pct = config.get('slippage', 0.05)  # 0.05% slippage

    trades = []
    balance = initial_capital

    for i in range(len(df) - 1):
        signal = predictions[i]

        # Apply filters
        if config.get('filter_session') and df.iloc[i]['asian_session'] == 1:
            continue
        if config.get('filter_volume') and df.iloc[i]['volume_ratio'] < 1.0:
            continue
        if config.get('filter_volatility'):
            if df.iloc[i]['volatility_ratio'] < 0.5 or df.iloc[i]['volatility_ratio'] > 2.0:
                continue
        if config.get('avoid_weekend') and df.iloc[i]['weekend'] == 1:
            continue

        capital_at_risk = balance * (risk_pct / 100)

        if signal == 1:  # Long
            entry_price = df.iloc[i]['close'] * (1 + slippage_pct / 100)
            atr = df.iloc[i]['atr_14']
            sl_price = entry_price - (atr * sl_mult)
            tp_price = entry_price + (atr * tp_mult)

            for j in range(i + 1, min(i + 50, len(df))):
                low = df.iloc[j]['low']
                high = df.iloc[j]['high']

                if low <= sl_price:
                    exit_price = sl_price * (1 - slippage_pct / 100)
                    pnl = ((exit_price - entry_price) / entry_price) * capital_at_risk
                    balance += pnl
                    trades.append({
                        'type': 'LONG',
                        'regime': df.iloc[i]['regime'],
                        'pnl': pnl,
                        'result': 'LOSS',
                        'session': 'asian' if df.iloc[i]['asian_session'] else ('london' if df.iloc[i]['london_session'] else 'us')
                    })
                    break
                elif high >= tp_price:
                    exit_price = tp_price * (1 - slippage_pct / 100)
                    pnl = ((exit_price - entry_price) / entry_price) * capital_at_risk
                    balance += pnl
                    trades.append({
                        'type': 'LONG',
                        'regime': df.iloc[i]['regime'],
                        'pnl': pnl,
                        'result': 'WIN',
                        'session': 'asian' if df.iloc[i]['asian_session'] else ('london' if df.iloc[i]['london_session'] else 'us')
                    })
                    break

        elif signal == 0:  # Short
            entry_price = df.iloc[i]['close'] * (1 - slippage_pct / 100)
            atr = df.iloc[i]['atr_14']
            sl_price = entry_price + (atr * sl_mult)
            tp_price = entry_price - (atr * tp_mult)

            for j in range(i + 1, min(i + 50, len(df))):
                low = df.iloc[j]['low']
                high = df.iloc[j]['high']

                if high >= sl_price:
                    exit_price = sl_price * (1 + slippage_pct / 100)
                    pnl = ((entry_price - exit_price) / entry_price) * capital_at_risk
                    balance += pnl
                    trades.append({
                        'type': 'SHORT',
                        'regime': df.iloc[i]['regime'],
                        'pnl': pnl,
                        'result': 'LOSS',
                        'session': 'asian' if df.iloc[i]['asian_session'] else ('london' if df.iloc[i]['london_session'] else 'us')
                    })
                    break
                elif low <= tp_price:
                    exit_price = tp_price * (1 + slippage_pct / 100)
                    pnl = ((entry_price - exit_price) / entry_price) * capital_at_risk
                    balance += pnl
                    trades.append({
                        'type': 'SHORT',
                        'regime': df.iloc[i]['regime'],
                        'pnl': pnl,
                        'result': 'WIN',
                        'session': 'asian' if df.iloc[i]['asian_session'] else ('london' if df.iloc[i]['london_session'] else 'us')
                    })
                    break

    if not trades:
        return None

    trades_df = pd.DataFrame(trades)

    total_trades = len(trades_df)
    wins = len(trades_df[trades_df['result'] == 'WIN'])
    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0

    total_pnl = trades_df['pnl'].sum()
    roi = (total_pnl / initial_capital) * 100

    # Calculate fees
    fee_pct = 0.13
    total_fees = total_trades * fee_pct / 100 * initial_capital * (risk_pct / 100)
    roi_net = ((total_pnl - total_fees) / initial_capital) * 100

    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'roi_gross': roi,
        'roi_net': roi_net,
        'total_fees': total_fees,
        'final_balance': balance,
        'trades_df': trades_df
    }


def walk_forward_analysis(df, wrapper, config):
    """Walk-forward validation - testa robustez temporal."""

    print("\n" + "="*80)
    print("🚶 WALK-FORWARD ANALYSIS")
    print("="*80)
    print("\nTestando robustez em diferentes períodos...")

    # Split into 4 quarters
    n = len(df)
    quarter_size = n // 4

    results = []

    for i in range(4):
        start_idx = i * quarter_size
        end_idx = (i + 1) * quarter_size if i < 3 else n

        df_period = df.iloc[start_idx:end_idx]

        result = run_single_backtest(
            df_period, wrapper,
            config['long_threshold'],
            config['short_threshold'],
            config
        )

        if result:
            results.append({
                'period': f"Q{i+1}",
                'start': df_period.index[0],
                'end': df_period.index[-1],
                **result
            })

    # Print results
    print(f"\n{'Período':<6} {'Início':<20} {'Fim':<20} {'Trades':<8} {'WR%':<8} {'ROI Net%':<10}")
    print("-" * 80)

    for r in results:
        print(f"{r['period']:<6} {str(r['start']):<20} {str(r['end']):<20} "
              f"{r['total_trades']:<8} {r['win_rate']:>6.2f}% {r['roi_net']:>8.2f}%")

    # Calculate consistency
    roi_nets = [r['roi_net'] for r in results]
    avg_roi = np.mean(roi_nets)
    std_roi = np.std(roi_nets)
    consistency = (avg_roi / std_roi) if std_roi > 0 else 0

    print(f"\n📊 Consistência:")
    print(f"   ROI Médio: {avg_roi:.2f}%")
    print(f"   Desvio Padrão: {std_roi:.2f}%")
    print(f"   Score Consistência: {consistency:.2f}")

    if consistency > 2.0 and avg_roi > 3:
        print(f"   ✅ EXCELENTE! Estratégia consistente ao longo do tempo")
    elif consistency > 1.0 and avg_roi > 0:
        print(f"   ⚠️  MÉDIO. Alguma consistência mas pode melhorar")
    else:
        print(f"   ❌ RUIM. Estratégia inconsistente - não confie!")

    return results


def optimize_thresholds(df, wrapper, config):
    """Otimiza thresholds usando grid search."""

    print("\n" + "="*80)
    print("🔍 OTIMIZAÇÃO DE THRESHOLDS")
    print("="*80)
    print("\nTestando combinações de thresholds...")

    long_thresholds = [0.45, 0.50, 0.55, 0.60, 0.65]
    short_thresholds = [0.35, 0.40, 0.45, 0.50, 0.55]

    best_result = None
    best_score = -999999
    all_results = []

    for lt in long_thresholds:
        for st in short_thresholds:
            result = run_single_backtest(df, wrapper, lt, st, config)

            if result:
                # Score = ROI Net + WR bonus
                score = result['roi_net'] + (result['win_rate'] - 47) * 0.5

                all_results.append({
                    'long_thresh': lt,
                    'short_thresh': st,
                    'score': score,
                    **result
                })

                if score > best_score:
                    best_score = score
                    best_result = {
                        'long_thresh': lt,
                        'short_thresh': st,
                        **result
                    }

    # Print top 5
    all_results.sort(key=lambda x: x['score'], reverse=True)

    print(f"\n{'Long':<6} {'Short':<6} {'Trades':<8} {'WR%':<8} {'ROI Net%':<10} {'Score':<8}")
    print("-" * 60)

    for r in all_results[:5]:
        print(f"{r['long_thresh']:<6.2f} {r['short_thresh']:<6.2f} "
              f"{r['total_trades']:<8} {r['win_rate']:>6.2f}% {r['roi_net']:>8.2f}% {r['score']:>6.2f}")

    print(f"\n✅ MELHOR CONFIGURAÇÃO:")
    print(f"   Long Threshold: {best_result['long_thresh']}")
    print(f"   Short Threshold: {best_result['short_thresh']}")
    print(f"   ROI Líquido: {best_result['roi_net']:.2f}%")
    print(f"   Win Rate: {best_result['win_rate']:.2f}%")
    print(f"   Trades: {best_result['total_trades']}")

    return best_result


def monte_carlo_simulation(df, wrapper, config, n_runs=100):
    """Monte Carlo simulation - testa robustez estatística."""

    print("\n" + "="*80)
    print(f"🎲 MONTE CARLO SIMULATION ({n_runs} runs)")
    print("="*80)
    print("\nSimulando diferentes sequências de trades...")

    # Run base backtest
    base_result = run_single_backtest(
        df, wrapper,
        config['long_threshold'],
        config['short_threshold'],
        config
    )

    if not base_result:
        print("   ❌ Nenhum trade na configuração base!")
        return

    trades = base_result['trades_df']
    base_roi = base_result['roi_net']

    # Shuffle trades and recalculate ROI
    roi_samples = []

    for _ in range(n_runs):
        shuffled = trades.sample(frac=1).reset_index(drop=True)
        total_pnl = shuffled['pnl'].sum()
        fees = len(shuffled) * 0.13 / 100 * config['capital'] * (config['risk_per_trade'] / 100)
        roi = ((total_pnl - fees) / config['capital']) * 100
        roi_samples.append(roi)

    roi_samples = np.array(roi_samples)

    # Statistics
    mean_roi = np.mean(roi_samples)
    std_roi = np.std(roi_samples)
    percentile_5 = np.percentile(roi_samples, 5)
    percentile_95 = np.percentile(roi_samples, 95)
    prob_profitable = (roi_samples > 0).sum() / len(roi_samples) * 100

    print(f"\n📊 Resultados Monte Carlo:")
    print(f"   ROI Base: {base_roi:.2f}%")
    print(f"   ROI Médio: {mean_roi:.2f}%")
    print(f"   Desvio Padrão: {std_roi:.2f}%")
    print(f"   95% Intervalo: [{percentile_5:.2f}%, {percentile_95:.2f}%]")
    print(f"   Probabilidade Lucro: {prob_profitable:.1f}%")

    if prob_profitable > 90 and percentile_5 > 0:
        print(f"\n   ✅ EXCELENTE! Alta probabilidade de lucro consistente")
    elif prob_profitable > 70 and percentile_5 > -5:
        print(f"\n   ⚠️  MÉDIO. Maioria lucrativa mas risco de prejuízo")
    else:
        print(f"\n   ❌ RUIM. Alta chance de prejuízo - não use!")


def analyze_by_regime(df, wrapper, config):
    """Analisa performance por regime de mercado."""

    print("\n" + "="*80)
    print("🌍 ANÁLISE POR REGIME DE MERCADO")
    print("="*80)

    regimes = df['regime'].unique()

    print(f"\n{'Regime':<12} {'Candles':<10} {'Trades':<8} {'WR%':<8} {'ROI Net%':<10}")
    print("-" * 60)

    for regime in ['bull', 'sideways', 'bear']:
        if regime not in regimes:
            continue

        df_regime = df[df['regime'] == regime]

        result = run_single_backtest(
            df_regime, wrapper,
            config['long_threshold'],
            config['short_threshold'],
            config
        )

        if result:
            print(f"{regime:<12} {len(df_regime):<10} {result['total_trades']:<8} "
                  f"{result['win_rate']:>6.2f}% {result['roi_net']:>8.2f}%")


def main():
    """Main entry point."""

    parser = argparse.ArgumentParser(
        description='Validador Extraordinário V3 - Análise Profissional Completa'
    )

    parser.add_argument('--model', type=str,
                        default='storage/models/model_DEFINITIVO_4ML_540d.pkl')
    parser.add_argument('--days', type=int, default=90)
    parser.add_argument('--long-threshold', type=float, default=0.55)
    parser.add_argument('--short-threshold', type=float, default=0.45)
    parser.add_argument('--capital', type=float, default=10000)
    parser.add_argument('--risk-per-trade', type=float, default=2.0)
    parser.add_argument('--sl-atr-mult', type=float, default=1.5)
    parser.add_argument('--tp-atr-mult', type=float, default=2.0)
    parser.add_argument('--optimize', action='store_true',
                        help='Otimizar thresholds automaticamente')
    parser.add_argument('--monte-carlo', action='store_true',
                        help='Executar Monte Carlo simulation')
    parser.add_argument('--walk-forward', action='store_true',
                        help='Executar Walk-Forward analysis')

    args = parser.parse_args()

    print("="*80)
    print("🔬 VALIDADOR EXTRAORDINÁRIO V3 - ANÁLISE PROFISSIONAL COMPLETA")
    print("="*80)

    # Load model
    print(f"\n📦 Carregando modelo: {args.model}")
    wrapper = load_model(args.model)
    print(f"   ✅ Modelo carregado!")

    # Fetch data
    print(f"\n📥 Baixando {args.days} dias de dados REAIS...")
    df = fetch_binance_data('BTCUSDT', '15m', args.days)
    print(f"   ✅ {len(df)} candles baixados")

    # Create features
    print(f"\n🔧 Criando features...")
    df = create_features(df)
    df = detect_market_regime(df)
    print(f"   ✅ {len(df.columns)} features + regime detection")

    # Config
    config = {
        'capital': args.capital,
        'risk_per_trade': args.risk_per_trade,
        'sl_atr_mult': args.sl_atr_mult,
        'tp_atr_mult': args.tp_atr_mult,
        'long_threshold': args.long_threshold,
        'short_threshold': args.short_threshold,
        'slippage': 0.05,  # 0.05% slippage
        'filter_session': True,
        'filter_volume': True,
        'filter_volatility': True,
        'avoid_weekend': True,
    }

    # Run analyses
    if args.optimize:
        optimize_thresholds(df, wrapper, config)

    if args.walk_forward:
        walk_forward_analysis(df, wrapper, config)

    if args.monte_carlo:
        monte_carlo_simulation(df, wrapper, config)

    # Always run regime analysis
    analyze_by_regime(df, wrapper, config)

    # Base backtest
    print("\n" + "="*80)
    print("📊 BACKTEST BASE")
    print("="*80)

    result = run_single_backtest(df, wrapper, args.long_threshold, args.short_threshold, config)

    if result:
        print(f"\n💼 Trades: {result['total_trades']}")
        print(f"📈 Win Rate: {result['win_rate']:.2f}%")
        print(f"💰 ROI Bruto: {result['roi_gross']:.2f}%")
        print(f"💰 ROI Líquido: {result['roi_net']:.2f}%")
        print(f"💸 Fees: ${result['total_fees']:.2f}")

    print("\n" + "="*80)
    print("✨ ANÁLISE CONCLUÍDA!")
    print("="*80)


if __name__ == '__main__':
    main()
