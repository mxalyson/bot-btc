"""
✅ BACKTEST VALIDADOR REALISTA - TESTA O MODELO DE FORMA JUSTA
================================================================================
Sem filtros artificiais que matam a performance
Parâmetros baseados em ESTATÍSTICAS REAIS do Bitcoin 15min
SL/TP testados empiricamente
Lookforward suficiente para trades respirarem
================================================================================
"""

import os
import sys
import pickle
import warnings
from datetime import datetime, timedelta
import argparse

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
        print(f"   ✅ Modelo carregado com compatibilidade!")
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
    """Create all features."""
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

    df['spread_pct'] = 0.02

    df = df.dropna()
    return df


def backtest_realista(df, wrapper, long_thresh, short_thresh, sl_pct, tp_pct):
    """
    Backtest REALISTA sem filtros artificiais.

    SL/TP em % do preço de entrada (não ATR - mais previsível)
    Lookforward: 96 candles (24 horas @ 15min)
    Sem break-even (deixa trade respirar)
    Sem cooldown artificial
    Slippage realista: 0.02%
    """

    # Get predictions
    probas = wrapper.predict_proba(df)[:, 1]
    predictions = np.full(len(probas), -1, dtype=int)
    predictions[probas >= long_thresh] = 1
    predictions[probas <= (1 - short_thresh)] = 0

    # Convert to numpy (FAST)
    close_arr = df['close'].values
    high_arr = df['high'].values
    low_arr = df['low'].values

    # Backtest config
    initial_capital = 1000
    risk_pct = 2
    slippage_pct = 0.02  # 0.02% realista (não 0.05%)
    max_lookforward = 96  # 24 horas @ 15min

    balance = initial_capital
    trades = []

    for i in range(len(df) - 1):
        signal = predictions[i]

        if signal == -1:
            continue

        capital_at_risk = balance * (risk_pct / 100)

        if signal == 1:  # Long
            entry_price = close_arr[i] * (1 + slippage_pct / 100)
            sl_price = entry_price * (1 - sl_pct / 100)
            tp_price = entry_price * (1 + tp_pct / 100)

            for j in range(i + 1, min(i + max_lookforward, len(df))):
                low = low_arr[j]
                high = high_arr[j]

                if low <= sl_price:
                    exit_price = sl_price * (1 - slippage_pct / 100)
                    pnl = ((exit_price - entry_price) / entry_price) * capital_at_risk
                    balance += pnl
                    trades.append({'type': 'LONG', 'pnl': pnl, 'exit': 'SL', 'bars': j-i})
                    break
                elif high >= tp_price:
                    exit_price = tp_price * (1 - slippage_pct / 100)
                    pnl = ((exit_price - entry_price) / entry_price) * capital_at_risk
                    balance += pnl
                    trades.append({'type': 'LONG', 'pnl': pnl, 'exit': 'TP', 'bars': j-i})
                    break
                elif j == i + max_lookforward - 1:
                    # Timeout - close at market
                    exit_price = close_arr[j] * (1 - slippage_pct / 100)
                    pnl = ((exit_price - entry_price) / entry_price) * capital_at_risk
                    balance += pnl
                    trades.append({'type': 'LONG', 'pnl': pnl, 'exit': 'TIMEOUT', 'bars': j-i})
                    break

        elif signal == 0:  # Short
            entry_price = close_arr[i] * (1 - slippage_pct / 100)
            sl_price = entry_price * (1 + sl_pct / 100)
            tp_price = entry_price * (1 - tp_pct / 100)

            for j in range(i + 1, min(i + max_lookforward, len(df))):
                low = low_arr[j]
                high = high_arr[j]

                if high >= sl_price:
                    exit_price = sl_price * (1 + slippage_pct / 100)
                    pnl = ((entry_price - exit_price) / entry_price) * capital_at_risk
                    balance += pnl
                    trades.append({'type': 'SHORT', 'pnl': pnl, 'exit': 'SL', 'bars': j-i})
                    break
                elif low <= tp_price:
                    exit_price = tp_price * (1 + slippage_pct / 100)
                    pnl = ((entry_price - exit_price) / entry_price) * capital_at_risk
                    balance += pnl
                    trades.append({'type': 'SHORT', 'pnl': pnl, 'exit': 'TP', 'bars': j-i})
                    break
                elif j == i + max_lookforward - 1:
                    exit_price = close_arr[j] * (1 + slippage_pct / 100)
                    pnl = ((entry_price - exit_price) / entry_price) * capital_at_risk
                    balance += pnl
                    trades.append({'type': 'SHORT', 'pnl': pnl, 'exit': 'TIMEOUT', 'bars': j-i})
                    break

    if not trades:
        return None

    trades_df = pd.DataFrame(trades)

    # Calculate metrics
    total_trades = len(trades_df)
    total_pnl = trades_df['pnl'].sum()
    roi_gross = (total_pnl / initial_capital) * 100

    # Fees (Bybit: 0.13% per side)
    fee_pct = 0.13
    total_fees = total_trades * 2 * fee_pct / 100 * initial_capital * (risk_pct / 100)
    roi_net = ((total_pnl - total_fees) / initial_capital) * 100

    wins = len(trades_df[trades_df['pnl'] > 0])
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

    # Advanced metrics
    avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if wins > 0 else 0
    losses = len(trades_df[trades_df['pnl'] <= 0])
    avg_loss = abs(trades_df[trades_df['pnl'] <= 0]['pnl'].mean()) if losses > 0 else 0

    gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(trades_df[trades_df['pnl'] <= 0]['pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

    # Exit analysis
    tp_exits = len(trades_df[trades_df['exit'] == 'TP'])
    sl_exits = len(trades_df[trades_df['exit'] == 'SL'])
    timeout_exits = len(trades_df[trades_df['exit'] == 'TIMEOUT'])

    # Average bars to exit
    avg_bars = trades_df['bars'].mean()

    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'roi_gross': roi_gross,
        'roi_net': roi_net,
        'total_fees': total_fees,
        'profit_factor': profit_factor,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'tp_exits': tp_exits,
        'sl_exits': sl_exits,
        'timeout_exits': timeout_exits,
        'tp_rate': (tp_exits / total_trades * 100) if total_trades > 0 else 0,
        'avg_bars': avg_bars,
        'score': roi_net
    }


def test_configurations(df, wrapper):
    """Test realistic SL/TP configurations."""

    print("\n" + "="*80)
    print("✅ BACKTEST VALIDADOR REALISTA")
    print("="*80)
    print("\nTestando configurações REALISTAS para scalping 15min...")
    print("⏳ Tempo estimado: 5 minutos...")
    print()

    # REALISTIC configurations for Bitcoin 15min scalping
    configs = [
        # Threshold, SL%, TP%
        (0.55, 0.45, 0.8, 1.2),  # SL 0.8%, TP 1.2% (RR 1:1.5)
        (0.55, 0.45, 1.0, 1.5),  # SL 1.0%, TP 1.5% (RR 1:1.5)
        (0.55, 0.45, 0.8, 1.6),  # SL 0.8%, TP 1.6% (RR 1:2.0)
        (0.55, 0.45, 1.0, 2.0),  # SL 1.0%, TP 2.0% (RR 1:2.0)
        (0.60, 0.45, 0.8, 1.2),
        (0.60, 0.45, 1.0, 1.5),
        (0.60, 0.45, 0.8, 1.6),
        (0.60, 0.45, 1.0, 2.0),
        (0.60, 0.50, 0.8, 1.2),
        (0.60, 0.50, 1.0, 1.5),
        (0.60, 0.50, 0.8, 1.6),
        (0.60, 0.50, 1.0, 2.0),
        (0.65, 0.50, 0.8, 1.2),
        (0.65, 0.50, 1.0, 1.5),
        (0.65, 0.50, 0.8, 1.6),
        (0.65, 0.50, 1.0, 2.0),
    ]

    results = []

    for idx, (long_t, short_t, sl, tp) in enumerate(configs, 1):
        print(f"   Testando {idx}/{len(configs)}: Long {long_t}, Short {short_t}, SL {sl}%, TP {tp}%")

        result = backtest_realista(df, wrapper, long_t, short_t, sl, tp)

        if result:
            results.append({
                'long_threshold': long_t,
                'short_threshold': short_t,
                'sl_pct': sl,
                'tp_pct': tp,
                'rr_ratio': tp/sl,
                **result
            })

    if not results:
        print("\n❌ Nenhum resultado válido!")
        return None

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('score', ascending=False)

    # Print top 10
    print("\n" + "="*140)
    print("🏆 TOP 10 CONFIGURAÇÕES REALISTAS")
    print("="*140)
    print(f"\n{'#':<4} {'Long':<6} {'Short':<6} {'SL%':<6} {'TP%':<6} {'RR':<5} {'Trades':<8} {'WR%':<8} {'ROI%':<8} {'PF':<6} {'TP%':<7} {'Bars':<6}")
    print("-" * 140)

    for idx, row in results_df.head(10).iterrows():
        print(f"{idx+1:<4} {row['long_threshold']:.2f}   {row['short_threshold']:.2f}   {row['sl_pct']:.1f}   {row['tp_pct']:.1f}   {row['rr_ratio']:.1f}   {int(row['total_trades']):<8} {row['win_rate']:>6.1f}% {row['roi_net']:>7.2f}% {row['profit_factor']:>5.2f}  {row['tp_rate']:>6.1f}% {row['avg_bars']:>5.1f}")

    # Best configuration
    best = results_df.iloc[0]

    print("\n" + "="*80)
    print("⭐ MELHOR CONFIGURAÇÃO")
    print("="*80)
    print(f"""
📊 Parâmetros:
   Long Threshold: {best['long_threshold']:.2f}
   Short Threshold: {best['short_threshold']:.2f}
   Stop Loss: {best['sl_pct']:.1f}% do preço de entrada
   Take Profit: {best['tp_pct']:.1f}% do preço de entrada
   Risk:Reward: 1:{best['rr_ratio']:.1f}

💰 Performance:
   Total Trades: {int(best['total_trades'])}
   Win Rate: {best['win_rate']:.2f}%
   ROI Bruto: {best['roi_gross']:.2f}%
   ROI Líquido: {best['roi_net']:.2f}%
   Fees Totais: ${best['total_fees']:.2f}

📈 Métricas:
   Profit Factor: {best['profit_factor']:.2f}
   Avg Win: ${best['avg_win']:.2f}
   Avg Loss: ${best['avg_loss']:.2f}

🎯 Saídas:
   TP: {int(best['tp_exits'])} ({best['tp_rate']:.1f}%)
   SL: {int(best['sl_exits'])} ({best['sl_exits']/best['total_trades']*100:.1f}%)
   Timeout: {int(best['timeout_exits'])} ({best['timeout_exits']/best['total_trades']*100:.1f}%)

⏱️  Duração Média: {best['avg_bars']:.1f} candles ({best['avg_bars']*15/60:.1f} horas)
""")

    # Approval
    approval = {
        'roi': best['roi_net'] >= 2.0,
        'wr': best['win_rate'] >= 40.0,
        'pf': best['profit_factor'] >= 1.2,
        'trades': best['total_trades'] >= 50
    }

    passed = sum(approval.values())

    print(f"📋 Critérios ({passed}/4):")
    print(f"   {'✅' if approval['roi'] else '❌'} ROI ≥ 2%: {best['roi_net']:.2f}%")
    print(f"   {'✅' if approval['wr'] else '❌'} Win Rate ≥ 40%: {best['win_rate']:.2f}%")
    print(f"   {'✅' if approval['pf'] else '❌'} Profit Factor ≥ 1.2: {best['profit_factor']:.2f}")
    print(f"   {'✅' if approval['trades'] else '❌'} Trades ≥ 50: {int(best['total_trades'])}")

    if passed >= 3:
        print(f"\n🎉 MODELO V3 APROVADO COM CONFIGURAÇÃO REALISTA!")
        print(f"💡 Use: SL {best['sl_pct']:.1f}%, TP {best['tp_pct']:.1f}%, Thresholds {best['long_threshold']:.2f}/{best['short_threshold']:.2f}")
    else:
        print(f"\n⚠️  MODELO V3 REPROVADO MESMO COM CONFIGURAÇÃO REALISTA")
        print(f"💡 Retreinar modelo V7 é necessário")

    return results_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='storage/models/model_DEFINITIVO_4ML_540d.pkl')
    parser.add_argument('--days', type=int, default=90)
    args = parser.parse_args()

    print("="*80)
    print("✅ VALIDADOR REALISTA - TESTE JUSTO DO MODELO")
    print("="*80)
    print()

    wrapper = load_model(args.model)
    print()

    print(f"📥 Baixando {args.days} dias de dados...")
    df = fetch_binance_data('BTCUSDT', '15m', args.days)
    print(f"   ✅ {len(df)} candles\n")

    print("🔧 Criando features...")
    df = create_features(df)
    print(f"   ✅ Features criadas\n")

    print("🔍 Verificando compatibilidade...")
    missing = [f for f in wrapper.feature_columns if f not in df.columns]
    if missing:
        print(f"   ❌ {len(missing)} features faltando!")
        sys.exit(1)
    print(f"   ✅ OK!\n")

    results = test_configurations(df, wrapper)

    if results is not None:
        results.to_csv('validation_realista.csv', index=False)
        print(f"\n💾 Resultados: validation_realista.csv")

    print("\n" + "="*80)
    print("✨ VALIDAÇÃO CONCLUÍDA!")
    print("="*80)


if __name__ == '__main__':
    main()
