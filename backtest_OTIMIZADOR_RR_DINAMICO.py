"""
🎯 OTIMIZADOR COM RISK:REWARD DINÂMICO E ADAPTATIVO
================================================================================
Usa SL/TP dinâmicos que se adaptam à volatilidade do mercado:
- Risk:Reward ratios realistas (1:1.5, 1:2, 1:2.5)
- SL baseado em ATR adaptado ao regime
- TP calculado a partir do RR ratio
- Break-even após atingir 1R de lucro
- Trailing stop em lucro
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
from itertools import product

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

    try:
        with open(model_path, 'rb') as f:
            wrapper = pickle.load(f)
        print(f"   ✅ Modelo carregado!")
        return wrapper
    except AttributeError as e:
        print(f"   ⚠️  Erro ao carregar pickle: {e}")
        print(f"   🔄 Tentando modo compatibilidade...")

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
    """Create all features including regime detection."""

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

    # Time-based features
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
    df['london_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
    df['us_session'] = ((df['hour'] >= 13) & (df['hour'] < 22)).astype(int)

    # Trend Strength
    df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
    df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
    df['hh_count'] = df['higher_high'].rolling(5).sum()
    df['ll_count'] = df['lower_low'].rolling(5).sum()
    df['trend_strength'] = df['hh_count'] - df['ll_count']

    # Spread proxy
    df['spread_proxy'] = (df['high'] - df['low']) / df['close']
    df['spread_ma'] = df['spread_proxy'].rolling(10).mean()

    # Large candles
    df['large_candle'] = (df['body_size'] > df['body_size'].rolling(20).mean() * 1.5).astype(int)

    # Market Regime Detection
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()

    df['regime'] = 'sideways'
    df.loc[(df['close'] > df['sma_20']) & (df['sma_20'] > df['sma_50']), 'regime'] = 'bull'
    df.loc[(df['close'] < df['sma_20']) & (df['sma_20'] < df['sma_50']), 'regime'] = 'bear'

    # ATR adaptado ao regime (mais conservador em alta volatilidade)
    df['atr_regime_mult'] = 1.0
    df.loc[df['volatility_ratio'] > 1.5, 'atr_regime_mult'] = 0.8  # Reduz SL em alta vol
    df.loc[df['volatility_ratio'] < 0.7, 'atr_regime_mult'] = 1.2  # Aumenta SL em baixa vol

    # Spread simulation
    df['spread_pct'] = 0.02

    df = df.dropna()
    return df


def run_backtest_rr_dinamico(df, wrapper, config):
    """Run backtest with DYNAMIC Risk:Reward ratio - ULTRA OPTIMIZED."""

    # Get predictions with custom thresholds
    long_thresh = config['long_threshold']
    short_thresh = config['short_threshold']

    probas = wrapper.predict_proba(df)[:, 1]

    predictions = np.full(len(probas), -1, dtype=int)
    predictions[probas >= long_thresh] = 1
    predictions[probas <= (1 - short_thresh)] = 0

    # Backtest parameters
    initial_capital = config.get('initial_capital', 1000)
    risk_pct = config.get('risk_pct', 2)
    base_sl_atr = config['sl_atr_mult']  # Base SL in ATR
    rr_ratio = config['rr_ratio']  # Risk:Reward ratio
    slippage_pct = 0.05
    cooldown_candles = config.get('cooldown', 0)
    use_breakeven = config.get('use_breakeven', True)

    # Convert to numpy arrays (FAST!)
    close_arr = df['close'].values
    high_arr = df['high'].values
    low_arr = df['low'].values
    atr_arr = df['atr_14'].values
    atr_regime_mult_arr = df['atr_regime_mult'].values
    volume_ratio_arr = df['volume_ratio'].values
    volatility_ratio_arr = df['volatility_ratio'].values
    weekend_arr = df['weekend'].values

    balance = initial_capital
    trades = []
    last_trade_idx = -cooldown_candles - 1

    max_lookforward = 20  # 5 hours @ 15min

    for i in range(len(df) - 1):
        signal = predictions[i]

        # Skip if in cooldown
        if i - last_trade_idx <= cooldown_candles:
            continue

        # Apply filters
        if config.get('filter_volume') and volume_ratio_arr[i] < 1.2:
            continue
        if config.get('filter_volatility'):
            vr = volatility_ratio_arr[i]
            if vr < 0.7 or vr > 1.8:
                continue
        if config.get('avoid_weekend') and weekend_arr[i] == 1:
            continue

        capital_at_risk = balance * (risk_pct / 100)

        # Dynamic SL based on ATR and regime
        atr = atr_arr[i]
        regime_mult = atr_regime_mult_arr[i]
        sl_distance = atr * base_sl_atr * regime_mult

        if signal == 1:  # Long
            entry_price = close_arr[i] * (1 + slippage_pct / 100)
            sl_price = entry_price - sl_distance
            tp_distance = sl_distance * rr_ratio  # TP = SL * RR
            tp_price = entry_price + tp_distance
            breakeven_price = entry_price + (sl_distance * 1.0) if use_breakeven else None

            current_sl = sl_price

            for j in range(i + 1, min(i + max_lookforward, len(df))):
                low = low_arr[j]
                high = high_arr[j]

                # Update to break-even after price moves 1R in profit
                if use_breakeven and breakeven_price and high >= breakeven_price:
                    current_sl = entry_price  # Move SL to break-even

                if low <= current_sl:
                    exit_price = current_sl * (1 - slippage_pct / 100)
                    pnl = ((exit_price - entry_price) / entry_price) * capital_at_risk
                    balance += pnl
                    trades.append({'type': 'LONG', 'pnl': pnl, 'exit': 'SL'})
                    last_trade_idx = i
                    break
                elif high >= tp_price:
                    exit_price = tp_price * (1 - slippage_pct / 100)
                    pnl = ((exit_price - entry_price) / entry_price) * capital_at_risk
                    balance += pnl
                    trades.append({'type': 'LONG', 'pnl': pnl, 'exit': 'TP'})
                    last_trade_idx = i
                    break

        elif signal == 0:  # Short
            entry_price = close_arr[i] * (1 - slippage_pct / 100)
            sl_price = entry_price + sl_distance
            tp_distance = sl_distance * rr_ratio
            tp_price = entry_price - tp_distance
            breakeven_price = entry_price - (sl_distance * 1.0) if use_breakeven else None

            current_sl = sl_price

            for j in range(i + 1, min(i + max_lookforward, len(df))):
                low = low_arr[j]
                high = high_arr[j]

                # Update to break-even
                if use_breakeven and breakeven_price and low <= breakeven_price:
                    current_sl = entry_price

                if high >= current_sl:
                    exit_price = current_sl * (1 + slippage_pct / 100)
                    pnl = ((entry_price - exit_price) / entry_price) * capital_at_risk
                    balance += pnl
                    trades.append({'type': 'SHORT', 'pnl': pnl, 'exit': 'SL'})
                    last_trade_idx = i
                    break
                elif low <= tp_price:
                    exit_price = tp_price * (1 + slippage_pct / 100)
                    pnl = ((entry_price - exit_price) / entry_price) * capital_at_risk
                    balance += pnl
                    trades.append({'type': 'SHORT', 'pnl': pnl, 'exit': 'TP'})
                    last_trade_idx = i
                    break

    if not trades:
        return None

    trades_df = pd.DataFrame(trades)

    # Calculate metrics
    total_trades = len(trades_df)
    total_pnl = trades_df['pnl'].sum()
    roi_gross = (total_pnl / initial_capital) * 100

    # Fees
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

    expectancy = (win_rate/100 * avg_win) - ((1-win_rate/100) * avg_loss)

    # TP vs SL exits
    tp_exits = len(trades_df[trades_df['exit'] == 'TP'])
    sl_exits = len(trades_df[trades_df['exit'] == 'SL'])

    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'roi_gross': roi_gross,
        'roi_net': roi_net,
        'total_fees': total_fees,
        'final_balance': balance,
        'profit_factor': profit_factor,
        'expectancy': expectancy,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'tp_exits': tp_exits,
        'sl_exits': sl_exits,
        'tp_rate': (tp_exits / total_trades * 100) if total_trades > 0 else 0,
        'score': roi_net * (win_rate/100) * (1 + profit_factor)
    }


def optimize_rr_grid(df, wrapper):
    """Optimized grid search with Risk:Reward ratios."""

    print("\n" + "="*80)
    print("🎯 OTIMIZAÇÃO COM RISK:REWARD DINÂMICO")
    print("="*80)
    print("\nUsando SL/TP inteligentes baseados em RR ratio...")
    print("⏳ Tempo estimado: 10-15 minutos...")

    # Parameter grid - FOCUSED
    long_thresholds = [0.55, 0.60, 0.65]  # 3 values
    short_thresholds = [0.40, 0.45, 0.50]  # 3 values
    sl_atr_mults = [1.0, 1.5]  # 2 values (SL em ATR)
    rr_ratios = [1.5, 2.0, 2.5]  # 3 values (Risk:Reward)
    cooldowns = [0, 5]  # 2 values
    use_breakeven_options = [True, False]  # 2 values

    # Filters
    filter_configs = [
        {'name': 'Volume Filter', 'filter_volume': True, 'filter_volatility': True, 'avoid_weekend': True}
    ]

    results = []
    total_tests = len(long_thresholds) * len(short_thresholds) * len(sl_atr_mults) * len(rr_ratios) * len(cooldowns) * len(use_breakeven_options) * len(filter_configs)

    print(f"📊 Total de testes: {total_tests}")
    print(f"💡 RR ratios: 1:1.5, 1:2.0, 1:2.5 (realistas para scalping)")
    print()

    test_count = 0

    for long_t, short_t, sl_atr, rr, cd, use_be, fconfig in product(
        long_thresholds, short_thresholds, sl_atr_mults, rr_ratios, cooldowns, use_breakeven_options, filter_configs
    ):
        test_count += 1

        if test_count % 20 == 0:
            print(f"   Progresso: {test_count}/{total_tests} ({test_count/total_tests*100:.1f}%)")

        config = {
            'long_threshold': long_t,
            'short_threshold': short_t,
            'sl_atr_mult': sl_atr,
            'rr_ratio': rr,
            'cooldown': cd,
            'use_breakeven': use_be,
            'initial_capital': 1000,
            'risk_pct': 2,
            **fconfig
        }

        result = run_backtest_rr_dinamico(df, wrapper, config)

        if result and result['total_trades'] >= 30:
            results.append({
                'long_threshold': long_t,
                'short_threshold': short_t,
                'sl_atr': sl_atr,
                'rr_ratio': rr,
                'cooldown': cd,
                'breakeven': 'Sim' if use_be else 'Não',
                **{k: v for k, v in result.items()}
            })

    print(f"\n✅ Otimização completa! {len(results)} configurações válidas.\n")

    if not results:
        print("❌ Nenhuma configuração válida encontrada!")
        return None

    # Sort by score
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('score', ascending=False)

    # Print top 15
    print("\n" + "="*140)
    print("🏆 TOP 15 MELHORES CONFIGURAÇÕES (Risk:Reward Dinâmico)")
    print("="*140)
    print(f"\n{'#':<4} {'Long':<6} {'Short':<6} {'SL ATR':<8} {'RR':<6} {'BE':<5} {'CD':<4} {'Trades':<8} {'WR%':<8} {'ROI%':<9} {'PF':<7} {'TP%':<7} {'Score':<8}")
    print("-" * 140)

    for idx, row in results_df.head(15).iterrows():
        print(f"{idx+1:<4} {row['long_threshold']:.2f}   {row['short_threshold']:.2f}   {row['sl_atr']:.1f}      {row['rr_ratio']:.1f}    {row['breakeven']:<5} {int(row['cooldown']):<4} {int(row['total_trades']):<8} {row['win_rate']:>6.1f}% {row['roi_net']:>8.2f}% {row['profit_factor']:>6.2f}  {row['tp_rate']:>6.1f}% {row['score']:>7.2f}")

    # Best configuration
    best = results_df.iloc[0]

    print("\n" + "="*80)
    print("⭐ MELHOR CONFIGURAÇÃO ENCONTRADA")
    print("="*80)
    print(f"""
📊 Parâmetros:
   Long Threshold: {best['long_threshold']:.2f}
   Short Threshold: {best['short_threshold']:.2f}
   SL: {best['sl_atr']:.1f}x ATR (adaptado à volatilidade)
   TP: {best['rr_ratio']:.1f}x SL (Risk:Reward de 1:{best['rr_ratio']:.1f})
   Break-Even após 1R: {best['breakeven']}
   Cooldown: {int(best['cooldown'])} candles

💰 Performance:
   Trades: {int(best['total_trades'])}
   Win Rate: {best['win_rate']:.2f}%
   ROI Bruto: {best['roi_gross']:.2f}%
   ROI Líquido: {best['roi_net']:.2f}%
   Fees: ${best['total_fees']:.2f}

📈 Métricas Avançadas:
   Profit Factor: {best['profit_factor']:.2f}
   Expectancy: ${best['expectancy']:.2f}
   Avg Win: ${best['avg_win']:.2f}
   Avg Loss: ${best['avg_loss']:.2f}

🎯 Saídas:
   TP Hit: {best['tp_exits']} ({best['tp_rate']:.1f}%)
   SL Hit: {best['sl_exits']} ({100-best['tp_rate']:.1f}%)

🎯 Score Total: {best['score']:.2f}
""")

    # Approval criteria
    approval_criteria = {
        'roi_net': best['roi_net'] >= 2.0,
        'win_rate': best['win_rate'] >= 40.0,
        'profit_factor': best['profit_factor'] >= 1.2,
        'total_trades': best['total_trades'] >= 40
    }

    passed = sum(approval_criteria.values())

    print(f"📋 Critérios de Aprovação ({passed}/4):")
    print(f"   {'✅' if approval_criteria['roi_net'] else '❌'} ROI ≥ 2%: {best['roi_net']:.2f}%")
    print(f"   {'✅' if approval_criteria['win_rate'] else '❌'} Win Rate ≥ 40%: {best['win_rate']:.2f}%")
    print(f"   {'✅' if approval_criteria['profit_factor'] else '❌'} Profit Factor ≥ 1.2: {best['profit_factor']:.2f}")
    print(f"   {'✅' if approval_criteria['total_trades'] else '❌'} Trades ≥ 40: {int(best['total_trades'])}")

    if passed >= 3:
        print(f"\n🎉 ESTRATÉGIA APROVADA! ({passed}/4 critérios)")
    else:
        print(f"\n⚠️  ESTRATÉGIA NECESSITA MELHORIAS ({passed}/4 critérios)")
        print(f"\n💡 RECOMENDAÇÃO: Retreinar modelo com melhorias ou mudar timeframe")

    return results_df


def main():
    parser = argparse.ArgumentParser(description='Otimizador RR Dinâmico')
    parser.add_argument('--model', type=str, default='storage/models/model_DEFINITIVO_4ML_540d.pkl')
    parser.add_argument('--days', type=int, default=60)
    parser.add_argument('--symbol', type=str, default='BTCUSDT')
    parser.add_argument('--interval', type=str, default='15m')

    args = parser.parse_args()

    print("="*80)
    print("🎯 OTIMIZADOR RISK:REWARD DINÂMICO")
    print("="*80)
    print()

    # Load model
    wrapper = load_model(args.model)
    print()

    # Fetch data
    print(f"📥 Baixando {args.days} dias de dados REAIS...")
    df = fetch_binance_data(args.symbol, args.interval, args.days)
    print(f"   ✅ {len(df)} candles baixados\n")

    # Create features
    print("🔧 Criando features...")
    df = create_features(df)
    print(f"   ✅ Features criadas com ATR adaptado ao regime!\n")

    # Verify features
    print("🔍 Verificando compatibilidade...")
    required_features = wrapper.feature_columns
    missing_features = [f for f in required_features if f not in df.columns]

    if missing_features:
        print(f"   ❌ ERRO: {len(missing_features)} features faltando!")
        for feat in missing_features[:10]:
            print(f"      - {feat}")
        sys.exit(1)
    else:
        print(f"   ✅ Todas as {len(required_features)} features presentes!\n")

    # Run optimization
    results = optimize_rr_grid(df, wrapper)

    if results is not None:
        results.to_csv('optimization_rr_results.csv', index=False)
        print(f"\n💾 Resultados salvos em: optimization_rr_results.csv")

    print("\n" + "="*80)
    print("✨ OTIMIZAÇÃO CONCLUÍDA!")
    print("="*80)


if __name__ == '__main__':
    main()
