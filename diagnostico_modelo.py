"""
🔍 DIAGNÓSTICO COMPLETO DO MODELO - REVELA O PROBLEMA REAL
================================================================================
Antes de fazer backtest, vamos ENTENDER o que o modelo está fazendo:
- Distribuição de probabilidades
- Quantos sinais por threshold
- Calibração do modelo
- Look-ahead bias check
- Feature importance
================================================================================
"""

import os
import sys
import pickle
import warnings
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


class ModelWrapper:
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

    def predict_proba(self, X):
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
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
    print(f"📦 Carregando modelo: {model_path}")
    try:
        with open(model_path, 'rb') as f:
            wrapper = pickle.load(f)
        print(f"   ✅ Modelo carregado!")
        return wrapper
    except AttributeError:
        class CustomUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if name == 'ModelWrapper':
                    return ModelWrapper
                return super().find_class(module, name)
        with open(model_path, 'rb') as f:
            wrapper = CustomUnpickler(f).load()
        print(f"   ✅ Modelo carregado!")
        return wrapper


def fetch_binance_data(symbol, interval, days):
    all_data = []
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    url = "https://api.binance.com/api/v3/klines"
    current_time = start_time

    while current_time < end_time:
        params = {'symbol': symbol, 'interval': interval, 'startTime': current_time, 'limit': 1000}
        response = requests.get(url, params=params)
        data = response.json()
        if not data:
            break
        all_data.extend(data)
        current_time = data[-1][6] + 1

    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_volume',
        'taker_buy_quote_volume', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    for col in ['open', 'high', 'low', 'close', 'volume', 'taker_buy_volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def create_features(df):
    """Create all features - SAME as training."""
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

    df['taker_buy_ratio'] = df['taker_buy_volume'] / (df['volume'] + 1e-8)
    df['taker_sell_ratio'] = 1 - df['taker_buy_ratio']
    df['buy_pressure_ma'] = df['taker_buy_ratio'].rolling(7).mean()
    df['sell_pressure_ma'] = df['taker_sell_ratio'].rolling(7).mean()
    df['pressure_delta'] = df['buy_pressure_ma'] - df['sell_pressure_ma']
    df['pressure_momentum'] = df['pressure_delta'].diff(3)
    df['order_imbalance'] = (df['taker_buy_volume'] - (df['volume'] - df['taker_buy_volume'])) / (df['volume'] + 1e-8)
    df['imbalance_ma'] = df['order_imbalance'].rolling(5).mean()

    for period in [7, 14, 21, 50]:
        df[f'sma_{period}'] = df['close'].rolling(period).mean()
        df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        df[f'price_sma_{period}_ratio'] = df['close'] / df[f'sma_{period}']

    df['ema7_above_ema14'] = (df['ema_7'] > df['ema_14']).astype(int)
    df['ema14_above_ema21'] = (df['ema_14'] > df['ema_21']).astype(int)
    df['ema21_above_ema50'] = (df['ema_21'] > df['ema_50']).astype(int)
    df['golden_cross'] = df['ema7_above_ema14'] & df['ema14_above_ema21']
    df['death_cross'] = (1 - df['ema7_above_ema14']) & (1 - df['ema14_above_ema21'])

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

    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['macd_positive'] = (df['macd'] > 0).astype(int)
    df['macd_hist_increasing'] = (df['macd_hist'] > df['macd_hist'].shift(1)).astype(int)

    df['bb_middle'] = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
    df['bb_upper_breakout'] = (df['close'] > df['bb_upper']).astype(int)
    df['bb_lower_breakout'] = (df['close'] < df['bb_lower']).astype(int)

    df['volume_sma_20'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_sma_20'] + 1e-8)
    df['high_volume'] = (df['volume_ratio'] > 1.5).astype(int)
    df['volume_slope'] = df['volume'].diff(3)
    df['volume_increasing_trend'] = (df['volume_slope'] > 0).astype(int).rolling(3).sum()

    df['momentum_3'] = df['close'] / df['close'].shift(3) - 1
    df['momentum_7'] = df['close'] / df['close'].shift(7) - 1
    df['momentum_14'] = df['close'] / df['close'].shift(14) - 1
    df['momentum_accel'] = df['momentum_7'].diff(3)

    df['price_position_14'] = (df['close'] - df['low'].rolling(14).min()) / (df['high'].rolling(14).max() - df['low'].rolling(14).min() + 1e-8)
    df['price_position_50'] = (df['close'] - df['low'].rolling(50).min()) / (df['high'].rolling(50).max() - df['low'].rolling(50).min() + 1e-8)

    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
    df['london_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
    df['us_session'] = ((df['hour'] >= 13) & (df['hour'] < 22)).astype(int)

    df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
    df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
    df['hh_count'] = df['higher_high'].rolling(5).sum()
    df['ll_count'] = df['lower_low'].rolling(5).sum()
    df['trend_strength'] = df['hh_count'] - df['ll_count']

    df['spread_proxy'] = (df['high'] - df['low']) / df['close']
    df['spread_ma'] = df['spread_proxy'].rolling(10).mean()
    df['large_candle'] = (df['body_size'] > df['body_size'].rolling(20).mean() * 1.5).astype(int)

    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['regime'] = 'sideways'
    df.loc[(df['close'] > df['sma_20']) & (df['sma_20'] > df['sma_50']), 'regime'] = 'bull'
    df.loc[(df['close'] < df['sma_20']) & (df['sma_20'] < df['sma_50']), 'regime'] = 'bear'

    df = df.dropna()
    return df


def diagnostico_completo(df, wrapper):
    """Diagnóstico completo do modelo."""

    print("\n" + "="*80)
    print("🔍 DIAGNÓSTICO COMPLETO DO MODELO V3")
    print("="*80)

    # Get probabilities
    print("\n1️⃣ Obtendo probabilidades do modelo...")
    probas = wrapper.predict_proba(df)[:, 1]
    print(f"   ✅ {len(probas)} probabilidades geradas")

    # Analyze distribution
    print("\n2️⃣ DISTRIBUIÇÃO DE PROBABILIDADES:")
    print(f"   Mínimo: {probas.min():.4f}")
    print(f"   Máximo: {probas.max():.4f}")
    print(f"   Média: {probas.mean():.4f}")
    print(f"   Mediana: {np.median(probas):.4f}")
    print(f"   Desvio Padrão: {probas.std():.4f}")

    # Percentiles
    print("\n3️⃣ PERCENTIS:")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"   P{p}: {np.percentile(probas, p):.4f}")

    # Test different thresholds
    print("\n4️⃣ ANÁLISE DE THRESHOLDS:")
    print(f"\n   {'Long Th':<10} {'Short Th':<10} {'Longs':<10} {'Shorts':<10} {'Holds':<10} {'% Sinais':<10}")
    print("   " + "-"*70)

    threshold_configs = [
        (0.50, 0.50),
        (0.55, 0.45),
        (0.60, 0.40),
        (0.65, 0.35),
        (0.70, 0.30),
        (0.75, 0.25),
        (0.80, 0.20),
    ]

    for long_t, short_t in threshold_configs:
        preds = np.full(len(probas), -1, dtype=int)
        preds[probas >= long_t] = 1
        preds[probas <= (1 - short_t)] = 0

        n_longs = np.sum(preds == 1)
        n_shorts = np.sum(preds == 0)
        n_holds = np.sum(preds == -1)
        pct_signals = ((n_longs + n_shorts) / len(preds)) * 100

        print(f"   {long_t:<10.2f} {short_t:<10.2f} {n_longs:<10} {n_shorts:<10} {n_holds:<10} {pct_signals:<10.1f}%")

    # Consecutive signals check
    print("\n5️⃣ ANÁLISE DE SINAIS CONSECUTIVOS (Threshold 0.65/0.50):")
    preds = np.full(len(probas), -1, dtype=int)
    preds[probas >= 0.65] = 1
    preds[probas <= 0.50] = 0

    signal_runs = []
    current_run = 1
    for i in range(1, len(preds)):
        if preds[i] == preds[i-1] and preds[i] != -1:
            current_run += 1
        else:
            if preds[i-1] != -1:
                signal_runs.append(current_run)
            current_run = 1

    if signal_runs:
        print(f"   Máx sinais consecutivos: {max(signal_runs)}")
        print(f"   Média sinais consecutivos: {np.mean(signal_runs):.1f}")
        print(f"   ⚠️  Se >5, modelo está dando MUITOS sinais seguidos!")

    # Probability drift check
    print("\n6️⃣ ANÁLISE DE DRIFT (mudança ao longo do tempo):")
    n_segments = 4
    segment_size = len(probas) // n_segments

    for i in range(n_segments):
        start = i * segment_size
        end = (i + 1) * segment_size if i < n_segments - 1 else len(probas)
        segment_probas = probas[start:end]

        print(f"   Segmento {i+1}: Média={segment_probas.mean():.4f}, Std={segment_probas.std():.4f}")

    # Recommendations
    print("\n7️⃣ RECOMENDAÇÕES:")

    # Check if model is too confident
    extreme_probas = np.sum((probas < 0.1) | (probas > 0.9)) / len(probas) * 100
    print(f"\n   Probabilidades extremas (<0.1 ou >0.9): {extreme_probas:.1f}%")
    if extreme_probas > 30:
        print(f"   ⚠️  PROBLEMA: Modelo muito confiante! Needs recalibration!")

    # Check signal frequency
    preds_065_050 = np.full(len(probas), -1, dtype=int)
    preds_065_050[probas >= 0.65] = 1
    preds_065_050[probas <= 0.50] = 0
    signal_rate = ((np.sum(preds_065_050 != -1) / len(preds_065_050)) * 100)

    print(f"\n   Taxa de sinais (0.65/0.50): {signal_rate:.1f}%")
    if signal_rate > 50:
        print(f"   ⚠️  OVERTRADING! Use thresholds MUITO mais altos!")
        print(f"   💡 Recomendação: Threshold 0.75/0.25 ou 0.80/0.20")
    elif signal_rate > 30:
        print(f"   ⚠️  Muitos sinais! Use threshold 0.70/0.30")
    elif signal_rate < 10:
        print(f"   ⚠️  Poucos sinais! Use threshold 0.60/0.40")
    else:
        print(f"   ✅ Taxa de sinais OK!")

    # Probability calibration
    mean_proba = probas.mean()
    if mean_proba < 0.45 or mean_proba > 0.55:
        print(f"\n   ⚠️  Modelo desbalanceado! Média deveria estar perto de 0.50")
        print(f"   💡 Considere ajustar thresholds para compensar")

    print("\n" + "="*80)
    print("✨ DIAGNÓSTICO CONCLUÍDO!")
    print("="*80)

    # Save probabilities for analysis
    prob_df = pd.DataFrame({
        'probability': probas,
        'timestamp': df.index
    })
    prob_df.to_csv('model_probabilities.csv', index=False)
    print(f"\n💾 Probabilidades salvas em: model_probabilities.csv")

    return probas


def main():
    print("="*80)
    print("🔍 DIAGNÓSTICO PROFUNDO DO MODELO V3")
    print("="*80)
    print()

    # Load model
    wrapper = load_model('storage/models/model_DEFINITIVO_4ML_540d.pkl')
    print()

    # Fetch data
    print(f"📥 Baixando 90 dias de dados...")
    df = fetch_binance_data('BTCUSDT', '15m', 90)
    print(f"   ✅ {len(df)} candles\n")

    # Create features
    print("🔧 Criando features...")
    df = create_features(df)
    print(f"   ✅ Features criadas\n")

    # Check compatibility
    print("🔍 Verificando compatibilidade...")
    missing = [f for f in wrapper.feature_columns if f not in df.columns]
    if missing:
        print(f"   ❌ {len(missing)} features faltando!")
        sys.exit(1)
    print(f"   ✅ OK!\n")

    # Run diagnosis
    probas = diagnostico_completo(df, wrapper)


if __name__ == '__main__':
    main()
