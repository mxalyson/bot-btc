"""
🚀 MEGA OTIMIZADOR - VALIDAÇÃO EXTRAORDINÁRIA COM OTIMIZAÇÃO TOTAL
================================================================================
Testa TODAS as configurações possíveis para encontrar a melhor combinação:
- 100+ combinações de thresholds (10x10 grid)
- Múltiplos SL/TP (ATR multiplicadores)
- Filtros de qualidade (volume, volatilidade, spread, sessão)
- Cooldown anti-overtrading
- Métricas avançadas (Sharpe, Profit Factor, Max Drawdown)
- Otimização por regime de mercado
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

    # Trend Strength (features que faltavam!)
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

    # Spread simulation (for Bybit-like conditions)
    df['spread_pct'] = 0.02  # 0.02% spread típico

    df = df.dropna()
    return df


def calculate_advanced_metrics(trades_df, initial_capital):
    """Calculate advanced performance metrics."""
    if trades_df.empty:
        return {}

    # Basic metrics
    total_trades = len(trades_df)
    wins = len(trades_df[trades_df['pnl'] > 0])
    losses = len(trades_df[trades_df['pnl'] <= 0])
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

    # PnL metrics
    total_pnl = trades_df['pnl'].sum()
    avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if wins > 0 else 0
    avg_loss = abs(trades_df[trades_df['pnl'] <= 0]['pnl'].mean()) if losses > 0 else 0

    # Profit Factor
    gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(trades_df[trades_df['pnl'] <= 0]['pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

    # Expectancy
    expectancy = (win_rate/100 * avg_win) - ((1-win_rate/100) * avg_loss)

    # Drawdown
    cumulative = trades_df['pnl'].cumsum()
    running_max = cumulative.cummax()
    drawdown = running_max - cumulative
    max_drawdown = drawdown.max()
    max_drawdown_pct = (max_drawdown / initial_capital * 100) if initial_capital > 0 else 0

    # Recovery Factor
    recovery_factor = total_pnl / max_drawdown if max_drawdown > 0 else 0

    # Sharpe Ratio (simplified - assumes daily returns)
    returns = trades_df['pnl'] / initial_capital
    sharpe_ratio = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0

    # Win/Loss Ratio
    win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0

    return {
        'profit_factor': profit_factor,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown_pct': max_drawdown_pct,
        'recovery_factor': recovery_factor,
        'expectancy': expectancy,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'win_loss_ratio': win_loss_ratio
    }


def run_backtest_with_config(df, wrapper, config):
    """Run backtest with specific configuration."""

    # Override thresholds
    wrapper_copy = wrapper
    long_thresh = config['long_threshold']
    short_thresh = config['short_threshold']

    # Get predictions with custom thresholds
    probas = wrapper.predict_proba(df)[:, 1]

    predictions = np.full(len(probas), -1, dtype=int)
    predictions[probas >= long_thresh] = 1
    predictions[probas <= (1 - short_thresh)] = 0

    # Backtest parameters
    initial_capital = config.get('initial_capital', 1000)
    risk_pct = config.get('risk_pct', 2)
    sl_mult = config['sl_mult']
    tp_mult = config['tp_mult']
    slippage_pct = 0.05
    cooldown_candles = config.get('cooldown', 0)

    balance = initial_capital
    trades = []
    last_trade_idx = -cooldown_candles - 1

    for i in range(len(df) - 1):
        signal = predictions[i]

        # Skip if in cooldown
        if i - last_trade_idx <= cooldown_candles:
            continue

        # Apply filters
        if config.get('filter_volume') and df.iloc[i]['volume_ratio'] < config.get('min_volume_ratio', 1.0):
            continue
        if config.get('filter_volatility'):
            vr = df.iloc[i]['volatility_ratio']
            if vr < config.get('min_vol_ratio', 0.5) or vr > config.get('max_vol_ratio', 2.0):
                continue
        if config.get('filter_spread') and df.iloc[i]['spread_pct'] > config.get('max_spread', 0.05):
            continue
        if config.get('avoid_weekend') and df.iloc[i]['weekend'] == 1:
            continue
        if config.get('filter_session'):
            session_ok = False
            if config.get('allow_london') and df.iloc[i]['london_session'] == 1:
                session_ok = True
            if config.get('allow_us') and df.iloc[i]['us_session'] == 1:
                session_ok = True
            if not session_ok:
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
                    trades.append({'type': 'LONG', 'pnl': pnl, 'regime': df.iloc[i]['regime']})
                    last_trade_idx = i
                    break
                elif high >= tp_price:
                    exit_price = tp_price * (1 - slippage_pct / 100)
                    pnl = ((exit_price - entry_price) / entry_price) * capital_at_risk
                    balance += pnl
                    trades.append({'type': 'LONG', 'pnl': pnl, 'regime': df.iloc[i]['regime']})
                    last_trade_idx = i
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
                    trades.append({'type': 'SHORT', 'pnl': pnl, 'regime': df.iloc[i]['regime']})
                    last_trade_idx = i
                    break
                elif low <= tp_price:
                    exit_price = tp_price * (1 + slippage_pct / 100)
                    pnl = ((entry_price - exit_price) / entry_price) * capital_at_risk
                    balance += pnl
                    trades.append({'type': 'SHORT', 'pnl': pnl, 'regime': df.iloc[i]['regime']})
                    last_trade_idx = i
                    break

    if not trades:
        return None

    trades_df = pd.DataFrame(trades)

    # Calculate metrics
    total_trades = len(trades_df)
    total_pnl = trades_df['pnl'].sum()
    roi_gross = (total_pnl / initial_capital) * 100

    # Fees (Bybit: 0.13%)
    fee_pct = 0.13
    total_fees = total_trades * 2 * fee_pct / 100 * initial_capital * (risk_pct / 100)  # 2x for entry+exit
    roi_net = ((total_pnl - total_fees) / initial_capital) * 100

    wins = len(trades_df[trades_df['pnl'] > 0])
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

    # Advanced metrics
    advanced = calculate_advanced_metrics(trades_df, initial_capital)

    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'roi_gross': roi_gross,
        'roi_net': roi_net,
        'total_fees': total_fees,
        'final_balance': balance,
        **advanced,
        'trades_df': trades_df,
        'score': roi_net * (win_rate/100) * (1 + advanced.get('profit_factor', 0)) - advanced.get('max_drawdown_pct', 0)
    }


def mega_grid_search(df, wrapper):
    """Massive grid search across all parameters."""

    print("\n" + "="*80)
    print("🔬 MEGA GRID SEARCH - OTIMIZAÇÃO TOTAL")
    print("="*80)
    print("\nTestando TODAS as combinações possíveis...")
    print("⏳ Isso pode levar alguns minutos...")

    # Parameter grid
    long_thresholds = np.arange(0.45, 0.75, 0.03)  # ~10 values
    short_thresholds = np.arange(0.35, 0.65, 0.03)  # ~10 values
    sl_mults = [1.5, 2.0, 2.5, 3.0]
    tp_mults = [2.0, 2.5, 3.0, 3.5]
    cooldowns = [0, 3, 5]  # candles between trades

    # Filter combinations
    filter_configs = [
        {'name': 'No Filters', 'filter_volume': False, 'filter_volatility': False, 'avoid_weekend': False},
        {'name': 'Volume Only', 'filter_volume': True, 'min_volume_ratio': 1.2, 'filter_volatility': False, 'avoid_weekend': False},
        {'name': 'Volume + Volatility', 'filter_volume': True, 'min_volume_ratio': 1.2, 'filter_volatility': True, 'min_vol_ratio': 0.7, 'max_vol_ratio': 1.8, 'avoid_weekend': False},
        {'name': 'All Filters', 'filter_volume': True, 'min_volume_ratio': 1.2, 'filter_volatility': True, 'min_vol_ratio': 0.7, 'max_vol_ratio': 1.8, 'avoid_weekend': True, 'filter_session': True, 'allow_london': True, 'allow_us': True}
    ]

    results = []
    total_tests = len(long_thresholds) * len(short_thresholds) * len(sl_mults) * len(tp_mults) * len(cooldowns) * len(filter_configs)

    print(f"📊 Total de testes: {total_tests}")
    print()

    test_count = 0

    for long_t, short_t, sl, tp, cd, fconfig in product(
        long_thresholds, short_thresholds, sl_mults, tp_mults, cooldowns, filter_configs
    ):
        test_count += 1

        if test_count % 100 == 0:
            print(f"   Progresso: {test_count}/{total_tests} ({test_count/total_tests*100:.1f}%)")

        config = {
            'long_threshold': long_t,
            'short_threshold': short_t,
            'sl_mult': sl,
            'tp_mult': tp,
            'cooldown': cd,
            'initial_capital': 1000,
            'risk_pct': 2,
            **fconfig
        }

        result = run_backtest_with_config(df, wrapper, config)

        if result and result['total_trades'] >= 30:  # Minimum trades for statistical significance
            results.append({
                'long_threshold': long_t,
                'short_threshold': short_t,
                'sl_mult': sl,
                'tp_mult': tp,
                'cooldown': cd,
                'filters': fconfig['name'],
                **{k: v for k, v in result.items() if k != 'trades_df'}
            })

    print(f"\n✅ Grid search completo! {len(results)} configurações válidas encontradas.\n")

    if not results:
        print("❌ Nenhuma configuração válida encontrada!")
        return None

    # Sort by score
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('score', ascending=False)

    # Print top 20
    print("\n" + "="*120)
    print("🏆 TOP 20 MELHORES CONFIGURAÇÕES")
    print("="*120)
    print(f"\n{'#':<4} {'Long':<6} {'Short':<6} {'SL':<5} {'TP':<5} {'CD':<4} {'Filters':<20} {'Trades':<7} {'WR%':<7} {'ROI%':<8} {'PF':<6} {'Sharpe':<7} {'Score':<8}")
    print("-" * 120)

    for idx, row in results_df.head(20).iterrows():
        print(f"{idx+1:<4} {row['long_threshold']:.2f}   {row['short_threshold']:.2f}   {row['sl_mult']:.1f}   {row['tp_mult']:.1f}   {int(row['cooldown']):<4} {row['filters']:<20} {int(row['total_trades']):<7} {row['win_rate']:>6.1f}% {row['roi_net']:>7.2f}% {row['profit_factor']:>5.2f}  {row['sharpe_ratio']:>6.2f}  {row['score']:>7.2f}")

    # Best configuration
    best = results_df.iloc[0]

    print("\n" + "="*80)
    print("⭐ MELHOR CONFIGURAÇÃO ENCONTRADA")
    print("="*80)
    print(f"""
📊 Parâmetros:
   Long Threshold: {best['long_threshold']:.2f}
   Short Threshold: {best['short_threshold']:.2f}
   SL Multiplier: {best['sl_mult']:.1f}x ATR
   TP Multiplier: {best['tp_mult']:.1f}x ATR
   Cooldown: {int(best['cooldown'])} candles
   Filtros: {best['filters']}

💰 Performance:
   Trades: {int(best['total_trades'])}
   Win Rate: {best['win_rate']:.2f}%
   ROI Bruto: {best['roi_gross']:.2f}%
   ROI Líquido: {best['roi_net']:.2f}%
   Fees: ${best['total_fees']:.2f}

📈 Métricas Avançadas:
   Profit Factor: {best['profit_factor']:.2f}
   Sharpe Ratio: {best['sharpe_ratio']:.2f}
   Max Drawdown: {best['max_drawdown_pct']:.2f}%
   Recovery Factor: {best['recovery_factor']:.2f}
   Expectancy: ${best['expectancy']:.2f}
   Win/Loss Ratio: {best['win_loss_ratio']:.2f}

🎯 Score Total: {best['score']:.2f}
""")

    # Determine if APPROVED
    approval_criteria = {
        'roi_net': best['roi_net'] >= 3.0,
        'win_rate': best['win_rate'] >= 45.0,
        'profit_factor': best['profit_factor'] >= 1.3,
        'max_drawdown': best['max_drawdown_pct'] <= 15.0,
        'total_trades': best['total_trades'] >= 50
    }

    passed = sum(approval_criteria.values())

    print(f"📋 Critérios de Aprovação ({passed}/5):")
    print(f"   {'✅' if approval_criteria['roi_net'] else '❌'} ROI ≥ 3%: {best['roi_net']:.2f}%")
    print(f"   {'✅' if approval_criteria['win_rate'] else '❌'} Win Rate ≥ 45%: {best['win_rate']:.2f}%")
    print(f"   {'✅' if approval_criteria['profit_factor'] else '❌'} Profit Factor ≥ 1.3: {best['profit_factor']:.2f}")
    print(f"   {'✅' if approval_criteria['max_drawdown'] else '❌'} Max DD ≤ 15%: {best['max_drawdown_pct']:.2f}%")
    print(f"   {'✅' if approval_criteria['total_trades'] else '❌'} Trades ≥ 50: {int(best['total_trades'])}")

    if passed >= 4:
        print(f"\n🎉 ESTRATÉGIA APROVADA! ({passed}/5 critérios)")
    else:
        print(f"\n⚠️  ESTRATÉGIA NECESSITA MELHORIAS ({passed}/5 critérios)")

    return results_df


def main():
    parser = argparse.ArgumentParser(description='Mega Otimizador V3 Scalper')
    parser.add_argument('--model', type=str, default='storage/models/model_DEFINITIVO_4ML_540d.pkl')
    parser.add_argument('--days', type=int, default=90)
    parser.add_argument('--symbol', type=str, default='BTCUSDT')
    parser.add_argument('--interval', type=str, default='15m')

    args = parser.parse_args()

    print("="*80)
    print("🚀 MEGA OTIMIZADOR V3 - VALIDAÇÃO CIENTÍFICA TOTAL")
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
    print(f"   ✅ {len([c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'volume']])} features criadas\n")

    # DEBUG: Check if required features exist
    print("🔍 Verificando compatibilidade de features...")
    required_features = wrapper.feature_columns
    missing_features = [f for f in required_features if f not in df.columns]

    if missing_features:
        print(f"   ❌ ERRO: {len(missing_features)} features faltando!")
        print(f"   Features necessárias mas não encontradas:")
        for feat in missing_features[:10]:  # Show first 10
            print(f"      - {feat}")
        if len(missing_features) > 10:
            print(f"      ... e mais {len(missing_features) - 10}")
        print(f"\n   💡 SOLUÇÃO: Execute 'git pull' para atualizar o código!")
        print(f"   💡 Ou verifique se create_features() está completa.\n")
        sys.exit(1)
    else:
        print(f"   ✅ Todas as {len(required_features)} features necessárias estão presentes!\n")

    # Run mega grid search
    results = mega_grid_search(df, wrapper)

    if results is not None:
        # Save results
        results.to_csv('optimization_results.csv', index=False)
        print(f"\n💾 Resultados salvos em: optimization_results.csv")

    print("\n" + "="*80)
    print("✨ OTIMIZAÇÃO CONCLUÍDA!")
    print("="*80)


if __name__ == '__main__':
    main()
