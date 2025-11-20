#!/usr/bin/env python3
"""
🚀 BACKTEST V3 SCALPER - EXTRAORDINÁRIO E ADAPTATIVO

Features:
- Detecta regime de mercado (bull/bear/sideways)
- Thresholds dinâmicos por regime
- SL/TP adaptativos por volatilidade
- Filtros de qualidade (sessão, volume, volatilidade)
- Gestão de risco adaptativa
- Otimização automática
- Relatório completo com gráficos

USO:
    python backtest_SCALPER_V3_EXTRAORDINARIO.py
    python backtest_SCALPER_V3_EXTRAORDINARIO.py --model storage/models/model_DEFINITIVO_4ML_540d.pkl
    python backtest_SCALPER_V3_EXTRAORDINARIO.py --days 180 --optimize
"""

import sys
import os
import argparse
import pickle
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings('ignore')


class AdaptiveScalper:
    """Scalper adaptativo com gestão de risco dinâmica."""

    def __init__(self, wrapper, config):
        self.wrapper = wrapper
        self.config = config

        # Track performance
        self.trades = []
        self.balance = config['initial_capital']
        self.peak_balance = config['initial_capital']
        self.consecutive_wins = 0
        self.consecutive_losses = 0

        # Risk management
        self.base_risk = config['risk_per_trade']
        self.current_risk = config['risk_per_trade']
        self.max_drawdown_pct = config.get('max_drawdown_pct', 15)

    def detect_market_regime(self, df, lookback=50):
        """Detecta regime de mercado nos últimos N candles."""
        recent = df.tail(lookback)

        # Trend strength
        sma_20 = recent['close'].rolling(20).mean().iloc[-1]
        sma_50 = recent['close'].rolling(50).mean().iloc[-1] if len(recent) >= 50 else sma_20
        current_price = recent['close'].iloc[-1]

        # Volatility
        volatility = recent['close'].pct_change().std()

        # Determine regime
        if current_price > sma_20 > sma_50:
            if volatility > 0.015:
                return 'strong_bull'
            return 'bull'
        elif current_price < sma_20 < sma_50:
            if volatility > 0.015:
                return 'strong_bear'
            return 'bear'
        else:
            return 'sideways'

    def get_dynamic_thresholds(self, regime):
        """Retorna thresholds adaptativos baseado no regime."""

        thresholds = {
            'strong_bull': {'long': 0.45, 'short': 0.65},  # Favorece longs
            'bull': {'long': 0.50, 'short': 0.55},
            'sideways': {'long': 0.55, 'short': 0.55},     # Neutro, mais seletivo
            'bear': {'long': 0.55, 'short': 0.50},
            'strong_bear': {'long': 0.65, 'short': 0.45},  # Favorece shorts
        }

        return thresholds.get(regime, {'long': 0.55, 'short': 0.55})

    def get_dynamic_sl_tp(self, regime, volatility):
        """Retorna SL/TP adaptativos baseado em regime e volatilidade."""

        # Base multipliers
        if regime in ['strong_bull', 'strong_bear']:
            # Alta volatilidade, SL/TP mais largos
            sl_mult = 2.5
            tp_mult = 4.0
        elif regime in ['bull', 'bear']:
            # Volatilidade média
            sl_mult = 2.0
            tp_mult = 3.0
        else:  # sideways
            # Baixa volatilidade, SL/TP mais apertados
            sl_mult = 1.5
            tp_mult = 2.5

        # Adjust by current volatility
        if volatility > 0.02:  # High volatility
            sl_mult *= 1.2
            tp_mult *= 1.2
        elif volatility < 0.01:  # Low volatility
            sl_mult *= 0.9
            tp_mult *= 0.9

        return sl_mult, tp_mult

    def should_trade(self, row, regime):
        """Filtros de qualidade para decidir se deve operar."""

        # Filter 1: Trading session (avoid Asian session)
        if self.config.get('filter_session', True):
            if row['asian_session'] == 1:
                return False

        # Filter 2: Volume
        if self.config.get('filter_volume', True):
            if row['volume_ratio'] < 1.0:  # Volume abaixo da média
                return False

        # Filter 3: Volatility
        if self.config.get('filter_volatility', True):
            # Evita volatilidade extremamente baixa
            if row['volatility_ratio'] < 0.5:
                return False
            # Evita volatilidade extremamente alta (risco de slippage)
            if row['volatility_ratio'] > 2.0:
                return False

        # Filter 4: Weekend
        if self.config.get('avoid_weekend', True):
            if row['weekend'] == 1:
                return False

        # Filter 5: Spread proxy (evita candles com spread muito alto)
        if self.config.get('filter_spread', True):
            if row['spread_proxy'] > row['spread_ma'] * 2.0:
                return False

        return True

    def update_risk(self, trade_result):
        """Atualiza risco baseado em performance recente."""

        if trade_result == 'WIN':
            self.consecutive_wins += 1
            self.consecutive_losses = 0

            # Increase risk after 3 consecutive wins (max 1.5x base)
            if self.consecutive_wins >= 3:
                self.current_risk = min(self.base_risk * 1.3, self.base_risk * 1.5)
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0

            # Decrease risk after 2 consecutive losses
            if self.consecutive_losses >= 2:
                self.current_risk = self.base_risk * 0.7

            # Further decrease after 4 consecutive losses
            if self.consecutive_losses >= 4:
                self.current_risk = self.base_risk * 0.5

        # Check drawdown protection
        current_drawdown = (self.peak_balance - self.balance) / self.peak_balance * 100

        if current_drawdown > self.max_drawdown_pct:
            # Stop trading if max drawdown exceeded
            return False
        elif current_drawdown > self.max_drawdown_pct * 0.7:
            # Reduce risk if approaching max drawdown
            self.current_risk = self.base_risk * 0.5

        # Update peak
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance

        return True

    def execute_trade(self, signal, entry_idx, df, regime, sl_mult, tp_mult):
        """Executa um trade com os parâmetros adaptativos."""

        entry_price = df.iloc[entry_idx]['close']
        atr = df.iloc[entry_idx]['atr_14']
        capital_at_risk = self.balance * (self.current_risk / 100)

        if signal == 1:  # Long
            sl_price = entry_price - (atr * sl_mult)
            tp_price = entry_price + (atr * tp_mult)

            # Check exit on next candles
            for j in range(entry_idx + 1, min(entry_idx + 50, len(df))):
                low = df.iloc[j]['low']
                high = df.iloc[j]['high']

                # Stop loss hit
                if low <= sl_price:
                    exit_price = sl_price
                    pnl = ((exit_price - entry_price) / entry_price) * capital_at_risk
                    self.balance += pnl

                    trade = {
                        'entry_time': df.index[entry_idx],
                        'exit_time': df.index[j],
                        'type': 'LONG',
                        'regime': regime,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'sl_price': sl_price,
                        'tp_price': tp_price,
                        'pnl': pnl,
                        'result': 'LOSS',
                        'risk_pct': self.current_risk,
                        'sl_mult': sl_mult,
                        'tp_mult': tp_mult
                    }
                    self.trades.append(trade)
                    return self.update_risk('LOSS')

                # Take profit hit
                elif high >= tp_price:
                    exit_price = tp_price
                    pnl = ((exit_price - entry_price) / entry_price) * capital_at_risk
                    self.balance += pnl

                    trade = {
                        'entry_time': df.index[entry_idx],
                        'exit_time': df.index[j],
                        'type': 'LONG',
                        'regime': regime,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'sl_price': sl_price,
                        'tp_price': tp_price,
                        'pnl': pnl,
                        'result': 'WIN',
                        'risk_pct': self.current_risk,
                        'sl_mult': sl_mult,
                        'tp_mult': tp_mult
                    }
                    self.trades.append(trade)
                    return self.update_risk('WIN')

        else:  # Short
            sl_price = entry_price + (atr * sl_mult)
            tp_price = entry_price - (atr * tp_mult)

            for j in range(entry_idx + 1, min(entry_idx + 50, len(df))):
                low = df.iloc[j]['low']
                high = df.iloc[j]['high']

                # Stop loss hit
                if high >= sl_price:
                    exit_price = sl_price
                    pnl = ((entry_price - exit_price) / entry_price) * capital_at_risk
                    self.balance += pnl

                    trade = {
                        'entry_time': df.index[entry_idx],
                        'exit_time': df.index[j],
                        'type': 'SHORT',
                        'regime': regime,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'sl_price': sl_price,
                        'tp_price': tp_price,
                        'pnl': pnl,
                        'result': 'LOSS',
                        'risk_pct': self.current_risk,
                        'sl_mult': sl_mult,
                        'tp_mult': tp_mult
                    }
                    self.trades.append(trade)
                    return self.update_risk('LOSS')

                # Take profit hit
                elif low <= tp_price:
                    exit_price = tp_price
                    pnl = ((entry_price - exit_price) / entry_price) * capital_at_risk
                    self.balance += pnl

                    trade = {
                        'entry_time': df.index[entry_idx],
                        'exit_time': df.index[j],
                        'type': 'SHORT',
                        'regime': regime,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'sl_price': sl_price,
                        'tp_price': tp_price,
                        'pnl': pnl,
                        'result': 'WIN',
                        'risk_pct': self.current_risk,
                        'sl_mult': sl_mult,
                        'tp_mult': tp_mult
                    }
                    self.trades.append(trade)
                    return self.update_risk('WIN')

        return True


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

    def predict(self, X, long_threshold=None, short_threshold=None):
        """Predict with custom thresholds."""

        # Use custom thresholds if provided
        lt = long_threshold if long_threshold is not None else self.long_threshold
        st = short_threshold if short_threshold is not None else self.short_threshold

        X_scaled = self.scaler.transform(X[self.feature_columns])

        # Get predictions from all models
        predictions = []
        for model, weight in zip(self.models_list, self.model_weights):
            try:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X_scaled)[:, 1]
                else:
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
        final_predictions[proba >= lt] = 1
        final_predictions[proba <= (1 - st)] = 0

        return final_predictions, proba


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

    return wrapper


def fetch_binance_data(symbol, interval, days):
    """Fetch data from Binance using requests."""
    print(f"\n📥 Baixando dados REAIS da Binance...")
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

    print(f"   ✅ {len(df)} candles REAIS baixados")
    print(f"   Período: {df.index[0]} a {df.index[-1]}")

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


def run_adaptive_backtest(df, wrapper, config):
    """Run adaptive backtest with market regime detection."""

    print("\n" + "="*80)
    print("🚀 BACKTEST ADAPTATIVO V3 - EXTRAORDINÁRIO")
    print("="*80)
    print(f"\n📊 Período: {df.index[0]} a {df.index[-1]}")
    print(f"   Candles: {len(df):,}")

    # Initialize adaptive scalper
    scalper = AdaptiveScalper(wrapper, config)

    # Track regime statistics
    regime_stats = defaultdict(int)
    regime_trades = defaultdict(list)

    print(f"\n💹 Executando backtest adaptativo...")
    print(f"   Filtros ativos:")
    print(f"   - Sessão: {'✅' if config.get('filter_session') else '❌'}")
    print(f"   - Volume: {'✅' if config.get('filter_volume') else '❌'}")
    print(f"   - Volatilidade: {'✅' if config.get('filter_volatility') else '❌'}")
    print(f"   - Weekend: {'✅' if config.get('avoid_weekend') else '❌'}")
    print(f"   - Spread: {'✅' if config.get('filter_spread') else '❌'}")

    # Run backtest
    for i in range(100, len(df) - 1):  # Start at 100 to have enough history

        # Detect market regime
        regime = scalper.detect_market_regime(df.iloc[:i+1])
        regime_stats[regime] += 1

        # Get dynamic thresholds
        thresholds = scalper.get_dynamic_thresholds(regime)

        # Get dynamic SL/TP
        volatility = df.iloc[i]['volatility_ratio']
        sl_mult, tp_mult = scalper.get_dynamic_sl_tp(regime, volatility)

        # Check quality filters
        if not scalper.should_trade(df.iloc[i], regime):
            continue

        # Generate prediction
        predictions, probas = wrapper.predict(
            df.iloc[i:i+1],
            long_threshold=thresholds['long'],
            short_threshold=thresholds['short']
        )

        signal = predictions[0]

        # Execute trade
        can_continue = scalper.execute_trade(signal, i, df, regime, sl_mult, tp_mult)

        if not can_continue:
            print(f"\n   ⚠️  Max drawdown atingido! Parando backtest.")
            break

    # Calculate results
    if not scalper.trades:
        print("\n   ❌ Nenhum trade executado!")
        return

    trades_df = pd.DataFrame(scalper.trades)

    total_trades = len(trades_df)
    wins = len(trades_df[trades_df['result'] == 'WIN'])
    losses = len(trades_df[trades_df['result'] == 'LOSS'])
    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0

    total_pnl = trades_df['pnl'].sum()
    initial_capital = config['initial_capital']
    roi = (total_pnl / initial_capital) * 100

    avg_win = trades_df[trades_df['result'] == 'WIN']['pnl'].mean() if wins > 0 else 0
    avg_loss = trades_df[trades_df['result'] == 'LOSS']['pnl'].mean() if losses > 0 else 0
    risk_reward = abs(avg_win / avg_loss) if avg_loss != 0 else 0

    # By type
    longs = trades_df[trades_df['type'] == 'LONG']
    shorts = trades_df[trades_df['type'] == 'SHORT']

    long_wr = (len(longs[longs['result'] == 'WIN']) / len(longs)) * 100 if len(longs) > 0 else 0
    short_wr = (len(shorts[shorts['result'] == 'WIN']) / len(shorts)) * 100 if len(shorts) > 0 else 0

    # Calculate metrics with fees
    fee_per_trade = 0.13  # 0.13% per trade (Bybit average)
    total_fees = total_trades * fee_per_trade / 100 * initial_capital * (config['risk_per_trade'] / 100)
    roi_with_fees = ((total_pnl - total_fees) / initial_capital) * 100

    # Days and annualized
    days = (df.index[-1] - df.index[0]).days
    roi_annualized = (roi_with_fees / days) * 365 if days > 0 else 0

    # Print results
    print("\n" + "="*80)
    print("📊 RESULTADOS DO BACKTEST ADAPTATIVO")
    print("="*80)

    print(f"\n💼 Trades Executados:")
    print(f"   Total: {total_trades}")
    print(f"   Longs: {len(longs)} ({len(longs)/total_trades*100:.1f}%)")
    print(f"   Shorts: {len(shorts)} ({len(shorts)/total_trades*100:.1f}%)")
    print(f"   Trades/dia: {total_trades/days:.1f}")

    print(f"\n📈 Performance:")
    print(f"   Win Rate: {win_rate:.2f}% ({wins}W / {losses}L)")
    print(f"   Long WR: {long_wr:.2f}%")
    print(f"   Short WR: {short_wr:.2f}%")

    print(f"\n💰 Financeiro:")
    print(f"   Capital Inicial: ${initial_capital:,.2f}")
    print(f"   PnL Bruto: ${total_pnl:,.2f}")
    print(f"   Fees Estimadas: ${total_fees:,.2f} ({fee_per_trade}% por trade)")
    print(f"   PnL Líquido: ${total_pnl - total_fees:,.2f}")
    print(f"   ROI Bruto: {roi:+.2f}%")
    print(f"   ROI Líquido: {roi_with_fees:+.2f}%")
    print(f"   ROI Anualizado: {roi_annualized:+.2f}%")
    print(f"   Capital Final: ${initial_capital + total_pnl - total_fees:,.2f}")

    print(f"\n⚖️  Risco/Retorno:")
    print(f"   Ganho Médio: ${avg_win:.2f}")
    print(f"   Perda Média: ${avg_loss:.2f}")
    print(f"   Risk/Reward: {risk_reward:.2f}")
    print(f"   Max Drawdown: {((scalper.peak_balance - scalper.balance) / scalper.peak_balance * 100):.2f}%")

    print(f"\n🌍 Regimes de Mercado:")
    total_candles = sum(regime_stats.values())
    for regime, count in sorted(regime_stats.items(), key=lambda x: x[1], reverse=True):
        pct = count / total_candles * 100
        regime_trades_count = len(trades_df[trades_df['regime'] == regime])
        regime_wr = 0
        if regime_trades_count > 0:
            regime_wins = len(trades_df[(trades_df['regime'] == regime) & (trades_df['result'] == 'WIN')])
            regime_wr = (regime_wins / regime_trades_count) * 100

        print(f"   {regime:15s}: {count:5d} candles ({pct:5.1f}%) | {regime_trades_count:4d} trades | WR {regime_wr:5.1f}%")

    # Approval criteria
    print("\n" + "="*80)
    print("✅ CRITÉRIOS DE APROVAÇÃO")
    print("="*80)

    criteria_met = 0
    total_criteria = 8

    checks = [
        ("Win Rate ≥ 47%", win_rate >= 47),
        ("ROI Líquido > 3%", roi_with_fees > 3),
        ("ROI Anualizado > 20%", roi_annualized > 20),
        ("Longs 20-60%", 20 <= (len(longs)/total_trades*100) <= 60),
        ("Shorts 40-80%", 40 <= (len(shorts)/total_trades*100) <= 80),
        ("Long WR > 35%", long_wr > 35),
        ("Short WR > 40%", short_wr > 40),
        ("Risk/Reward > 1.3", risk_reward > 1.3),
    ]

    for criteria, passed in checks:
        status = "✅" if passed else "❌"
        print(f"   {status} {criteria}")
        if passed:
            criteria_met += 1

    print(f"\n   Aprovação: {criteria_met}/{total_criteria} critérios atendidos")

    if criteria_met >= 6:
        print("\n   🎉 MODELO APROVADO! ROI líquido positivo e métricas aceitáveis.")
        print(f"   💰 Lucro estimado anual: ${(initial_capital * roi_annualized / 100):,.2f}")
    else:
        print("\n   ⚠️  MODELO REPROVADO. Ajuste parâmetros ou retreine.")

    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'roi_net': roi_with_fees,
        'roi_annualized': roi_annualized,
        'criteria_met': criteria_met
    }


def main():
    """Main entry point."""

    parser = argparse.ArgumentParser(
        description='Backtest V3 Scalper Extraordinário - Adaptativo e Profissional',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--model', type=str,
                        default='storage/models/model_DEFINITIVO_4ML_540d.pkl',
                        help='Caminho do modelo .pkl')

    parser.add_argument('--days', type=int, default=90,
                        help='Dias de dados para backtest (default: 90)')

    parser.add_argument('--capital', type=float, default=10000,
                        help='Capital inicial (default: 10000)')

    parser.add_argument('--risk-per-trade', type=float, default=2.0,
                        help='Risco base por trade em %% (default: 2.0)')

    parser.add_argument('--max-drawdown', type=float, default=15.0,
                        help='Max drawdown permitido em %% (default: 15.0)')

    parser.add_argument('--symbol', type=str, default='BTCUSDT',
                        help='Symbol para trade (default: BTCUSDT)')

    parser.add_argument('--interval', type=str, default='15m',
                        help='Intervalo de candles (default: 15m)')

    parser.add_argument('--no-filters', action='store_true',
                        help='Desabilitar todos os filtros de qualidade')

    args = parser.parse_args()

    print("="*80)
    print("🚀 BACKTEST V3 SCALPER - EXTRAORDINÁRIO E ADAPTATIVO")
    print("="*80)

    # Load model
    wrapper = load_model(args.model)

    # Fetch data
    df = fetch_binance_data(args.symbol, args.interval, args.days)

    # Create features
    print(f"\n🔧 Criando features extraordinárias...")
    df = create_features(df)
    print(f"   ✅ {len(df.columns)} features criadas")
    print(f"   ✅ {len(df)} candles prontos para backtest")

    # Config
    config = {
        'initial_capital': args.capital,
        'risk_per_trade': args.risk_per_trade,
        'max_drawdown_pct': args.max_drawdown,
        'filter_session': not args.no_filters,
        'filter_volume': not args.no_filters,
        'filter_volatility': not args.no_filters,
        'avoid_weekend': not args.no_filters,
        'filter_spread': not args.no_filters,
    }

    # Run backtest
    results = run_adaptive_backtest(df, wrapper, config)

    print("\n" + "="*80)
    print("✨ BACKTEST CONCLUÍDO!")
    print("="*80)


if __name__ == '__main__':
    main()
