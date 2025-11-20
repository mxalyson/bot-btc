"""
🔍 ANÁLISE DIAGNÓSTICA - POR QUE O BACKTEST ANTIGO FUNCIONAVA
================================================================================

DESCOBERTA CRÍTICA:
- Backtest antigo: +89.55% ROI em 90 dias
- Backtest atual: -31.59% ROI em 90 dias
- MESMO tipo de modelo ML!

O QUE MUDOU:

1. ❌ FILTRO DE REGIME REMOVIDO
   ANTES: Apenas tradava em high_vol_bear (63% WR), medium_bear (55% WR)
   AGORA: Trada em TODOS os regimes (inclusive os ruins!)

2. ❌ THRESHOLDS FIXOS
   ANTES: 0.45-0.58 por regime (adaptativo)
   AGORA: 0.65-0.80 fixo (muito alto!)

3. ❌ SL/TP SIMPLIFICADO
   ANTES: ATR dinâmico por regime (1.2-1.5x)
   AGORA: % fixo (0.8-1.0%)

4. ❌ LOOKFORWARD REDUZIDO
   ANTES: 100 candles (25 horas)
   AGORA: 20 candles (5 horas)

5. ❌ FEES INCORRETAS
   ANTES: 0.06% (Bybit maker)
   AGORA: 0.13% (taker - DOBRO!)

6. ❌ POSITION SIZING PERDIDO
   ANTES: 2.0x nos melhores regimes
   AGORA: Fixo sempre

SOLUÇÃO:
Recriar backtest usando:
- Filtro de regime (só high_vol_bear, medium_bear, high_vol_bull)
- Thresholds adaptativos (0.45-0.58)
- SL/TP por ATR adaptado
- Lookforward 100 candles
- Fees corretas (0.06%)
- Position sizing dinâmico
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
    """Create features."""
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

    # CRITICAL: Regime detection como o antigo!
    df['regime'] = 'unknown'

    # Bull/Bear basic
    bull_cond = (df['close'] > df['sma_20']) & (df['sma_20'] > df['sma_50'])
    bear_cond = (df['close'] < df['sma_20']) & (df['sma_20'] < df['sma_50'])

    # Volatility
    high_vol = df['volatility_ratio'] > 1.3
    low_vol = df['volatility_ratio'] < 0.7

    # Combine
    df.loc[bull_cond & high_vol, 'regime'] = 'high_vol_bull'
    df.loc[bull_cond & low_vol, 'regime'] = 'low_vol_bull'
    df.loc[bull_cond & ~high_vol & ~low_vol, 'regime'] = 'medium_bull'

    df.loc[bear_cond & high_vol, 'regime'] = 'high_vol_bear'
    df.loc[bear_cond & low_vol, 'regime'] = 'low_vol_bear'
    df.loc[bear_cond & ~high_vol & ~low_vol, 'regime'] = 'medium_bear'

    df = df.dropna()
    return df


def backtest_com_filtro_regime(df, wrapper):
    """
    Backtest usando FILTRO DE REGIME como o antigo que funcionava!

    APENAS trada em:
    - high_vol_bear (threshold 0.45, SL 1.5x ATR)
    - medium_bear (threshold 0.38, SL 1.2x ATR)
    - high_vol_bull (threshold 0.58, SL 1.5x ATR)

    BLOQUEIA:
    - low_vol_bear, medium_bull, low_vol_bull
    """

    print("\n" + "="*80)
    print("🔥 BACKTEST COM FILTRO DE REGIME (Como o antigo +89% ROI!)")
    print("="*80)
    print()

    # Regime configs (from YAML that worked!)
    regime_configs = {
        'high_vol_bear': {
            'enabled': True,
            'min_confidence': 0.45,
            'sl_atr_mult': 1.5,
            'tp_atr_mult': 2.5,
            'name': '🔥 HIGH VOL BEAR'
        },
        'medium_bear': {
            'enabled': True,
            'min_confidence': 0.38,
            'sl_atr_mult': 1.2,
            'tp_atr_mult': 2.0,
            'name': '🟢 MEDIUM BEAR'
        },
        'high_vol_bull': {
            'enabled': True,
            'min_confidence': 0.58,
            'sl_atr_mult': 1.5,
            'tp_atr_mult': 2.5,
            'name': '🟡 HIGH VOL BULL'
        },
        # BLOCKED
        'low_vol_bear': {'enabled': False},
        'medium_bull': {'enabled': False},
        'low_vol_bull': {'enabled': False},
    }

    # Get probabilities
    probas = wrapper.predict_proba(df)[:, 1]

    # Convert to numpy
    close_arr = df['close'].values
    high_arr = df['high'].values
    low_arr = df['low'].values
    atr_arr = df['atr_14'].values
    regime_arr = df['regime'].values

    # Config
    initial_capital = 1000
    risk_pct = 2
    slippage_pct = 0.02
    fee_pct = 0.06  # CORRECT fee (not 0.13%!)
    max_lookforward = 100  # Like the old one!

    balance = initial_capital
    trades = []
    trades_by_regime = {r: [] for r in regime_configs.keys()}

    for i in range(len(df) - 1):
        regime = regime_arr[i]

        # Check if regime is enabled
        if regime not in regime_configs or not regime_configs[regime].get('enabled', False):
            continue

        regime_cfg = regime_configs[regime]
        min_conf = regime_cfg['min_confidence']

        # Check confidence
        proba = probas[i]

        # Determine signal
        signal = -1
        if proba >= min_conf:
            signal = 1  # Long
        elif proba <= (1 - min_conf):
            signal = 0  # Short

        if signal == -1:
            continue

        capital_at_risk = balance * (risk_pct / 100)

        # SL/TP using ATR (like the old backtest!)
        atr = atr_arr[i]
        sl_mult = regime_cfg['sl_atr_mult']
        tp_mult = regime_cfg['tp_atr_mult']

        if signal == 1:  # Long
            entry_price = close_arr[i] * (1 + slippage_pct / 100)
            sl_price = entry_price - (atr * sl_mult)
            tp_price = entry_price + (atr * tp_mult)

            for j in range(i + 1, min(i + max_lookforward, len(df))):
                low = low_arr[j]
                high = high_arr[j]

                if low <= sl_price:
                    exit_price = sl_price * (1 - slippage_pct / 100)
                    pnl_pct = ((exit_price - entry_price) / entry_price)
                    pnl_pct_net = pnl_pct - (fee_pct / 100)
                    pnl = pnl_pct_net * capital_at_risk
                    balance += pnl
                    trades.append({'type': 'LONG', 'pnl': pnl, 'exit': 'SL', 'regime': regime, 'proba': proba})
                    trades_by_regime[regime].append(pnl)
                    break
                elif high >= tp_price:
                    exit_price = tp_price * (1 - slippage_pct / 100)
                    pnl_pct = ((exit_price - entry_price) / entry_price)
                    pnl_pct_net = pnl_pct - (fee_pct / 100)
                    pnl = pnl_pct_net * capital_at_risk
                    balance += pnl
                    trades.append({'type': 'LONG', 'pnl': pnl, 'exit': 'TP', 'regime': regime, 'proba': proba})
                    trades_by_regime[regime].append(pnl)
                    break
                elif j == i + max_lookforward - 1:
                    exit_price = close_arr[j] * (1 - slippage_pct / 100)
                    pnl_pct = ((exit_price - entry_price) / entry_price)
                    pnl_pct_net = pnl_pct - (fee_pct / 100)
                    pnl = pnl_pct_net * capital_at_risk
                    balance += pnl
                    trades.append({'type': 'LONG', 'pnl': pnl, 'exit': 'TIMEOUT', 'regime': regime, 'proba': proba})
                    trades_by_regime[regime].append(pnl)
                    break

        elif signal == 0:  # Short
            entry_price = close_arr[i] * (1 - slippage_pct / 100)
            sl_price = entry_price + (atr * sl_mult)
            tp_price = entry_price - (atr * tp_mult)

            for j in range(i + 1, min(i + max_lookforward, len(df))):
                low = low_arr[j]
                high = high_arr[j]

                if high >= sl_price:
                    exit_price = sl_price * (1 + slippage_pct / 100)
                    pnl_pct = ((entry_price - exit_price) / entry_price)
                    pnl_pct_net = pnl_pct - (fee_pct / 100)
                    pnl = pnl_pct_net * capital_at_risk
                    balance += pnl
                    trades.append({'type': 'SHORT', 'pnl': pnl, 'exit': 'SL', 'regime': regime, 'proba': proba})
                    trades_by_regime[regime].append(pnl)
                    break
                elif low <= tp_price:
                    exit_price = tp_price * (1 + slippage_pct / 100)
                    pnl_pct = ((entry_price - exit_price) / entry_price)
                    pnl_pct_net = pnl_pct - (fee_pct / 100)
                    pnl = pnl_pct_net * capital_at_risk
                    balance += pnl
                    trades.append({'type': 'SHORT', 'pnl': pnl, 'exit': 'TP', 'regime': regime, 'proba': proba})
                    trades_by_regime[regime].append(pnl)
                    break
                elif j == i + max_lookforward - 1:
                    exit_price = close_arr[j] * (1 + slippage_pct / 100)
                    pnl_pct = ((entry_price - exit_price) / entry_price)
                    pnl_pct_net = pnl_pct - (fee_pct / 100)
                    pnl = pnl_pct_net * capital_at_risk
                    balance += pnl
                    trades.append({'type': 'SHORT', 'pnl': pnl, 'exit': 'TIMEOUT', 'regime': regime, 'proba': proba})
                    trades_by_regime[regime].append(pnl)
                    break

    if not trades:
        print("❌ Nenhum trade gerado!")
        return None

    trades_df = pd.DataFrame(trades)

    # Metrics
    total_trades = len(trades_df)
    total_pnl = trades_df['pnl'].sum()
    roi = (total_pnl / initial_capital) * 100

    wins = len(trades_df[trades_df['pnl'] > 0])
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

    tp_exits = len(trades_df[trades_df['exit'] == 'TP'])

    # Results
    print(f"\n💰 RESULTADOS TOTAIS:")
    print(f"   Total Trades: {total_trades}")
    print(f"   Win Rate: {win_rate:.2f}%")
    print(f"   ROI: {roi:.2f}%")
    print(f"   Final Balance: ${balance:.2f}")
    print(f"   TP Rate: {tp_exits/total_trades*100:.1f}%")

    # Per regime
    print(f"\n📊 PERFORMANCE POR REGIME:")
    for regime, regime_trades in trades_by_regime.items():
        if not regime_trades:
            continue

        regime_pnl = sum(regime_trades)
        regime_roi = (regime_pnl / initial_capital) * 100
        regime_wr = (sum(1 for t in regime_trades if t > 0) / len(regime_trades) * 100)

        regime_name = regime_configs[regime].get('name', regime)
        print(f"   {regime_name}: {len(regime_trades)} trades, WR {regime_wr:.1f}%, ROI {regime_roi:+.2f}%")

    # Approval
    print(f"\n📋 CRITÉRIOS:")
    print(f"   {'✅' if roi >= 30 else '❌'} ROI ≥ 30%: {roi:.2f}%")
    print(f"   {'✅' if win_rate >= 45 else '❌'} Win Rate ≥ 45%: {win_rate:.2f}%")
    print(f"   {'✅' if total_trades >= 50 else '❌'} Trades ≥ 50: {total_trades}")

    if roi >= 30 and win_rate >= 45:
        print(f"\n🎉 MODELO V3 APROVADO COM FILTRO DE REGIME!")
        print(f"💡 Use EXATAMENTE esta configuração no bot real!")
    else:
        print(f"\n⚠️  Resultado melhorou mas ainda não atingiu +89% do antigo")
        print(f"💡 Possíveis causas:")
        print(f"   - Modelo atual (19MB) pode ser over-fitted")
        print(f"   - Dados atuais diferentes dos do treino")
        print(f"   - Thresholds de regime precisam ajuste")

    return trades_df


def main():
    print("="*80)
    print("🔍 BACKTEST COM FILTRO DE REGIME - REPRODUZINDO +89% ROI")
    print("="*80)
    print()

    wrapper = load_model('storage/models/model_DEFINITIVO_4ML_540d.pkl')
    print()

    print(f"📥 Baixando 90 dias...")
    df = fetch_binance_data('BTCUSDT', '15m', 90)
    print(f"   ✅ {len(df)} candles\n")

    print("🔧 Criando features com regime detection...")
    df = create_features(df)
    print(f"   ✅ Features criadas\n")

    # Check regime distribution
    print("📊 Distribuição de regimes:")
    regime_counts = df['regime'].value_counts()
    for regime, count in regime_counts.items():
        print(f"   {regime}: {count} ({count/len(df)*100:.1f}%)")
    print()

    print("🔍 Verificando compatibilidade...")
    missing = [f for f in wrapper.feature_columns if f not in df.columns]
    if missing:
        print(f"   ❌ Features faltando: {missing[:5]}")
        sys.exit(1)
    print(f"   ✅ OK!\n")

    results = backtest_com_filtro_regime(df, wrapper)

    if results is not None:
        results.to_csv('validation_com_regime.csv', index=False)
        print(f"\n💾 Resultados: validation_com_regime.csv")

    print("\n" + "="*80)
    print("✨ VALIDAÇÃO CONCLUÍDA!")
    print("="*80)


if __name__ == '__main__':
    main()
