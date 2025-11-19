#!/usr/bin/env python3
"""
🚀 BACKTEST UNIVERSAL - FLEXÍVEL E PODEROSO

Aceita qualquer modelo .pkl e permite ajustar:
- Threshold (long/short)
- Período de teste
- Capital inicial
- Parâmetros de risco

USO:
    python backtest_UNIVERSAL.py
    python backtest_UNIVERSAL.py --model storage/models/model_V6_365d.pkl
    python backtest_UNIVERSAL.py --long-threshold 0.50 --short-threshold 0.50
    python backtest_UNIVERSAL.py --days 180 --capital 10000
    python backtest_UNIVERSAL.py --model my_model.pkl --long-threshold 0.45 --short-threshold 0.55 --days 90
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
    """Fetch data from Binance."""
    from binance.client import Client

    print(f"\n📥 Baixando dados do Binance...")
    print(f"   Symbol: {symbol}")
    print(f"   Interval: {interval}")
    print(f"   Days: {days}")

    client = Client()

    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)

    klines = client.get_historical_klines(
        symbol,
        interval,
        start_time.strftime("%d %b %Y %H:%M:%S"),
        end_time.strftime("%d %b %Y %H:%M:%S")
    )

    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_volume',
        'taker_buy_quote_volume', 'ignore'
    ])

    # Convert types
    for col in ['open', 'high', 'low', 'close', 'volume', 'taker_buy_volume']:
        df[col] = df[col].astype(float)

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('timestamp')

    print(f"   ✅ {len(df)} candles baixados")
    print(f"   Período: {df.index[0]} a {df.index[-1]}")

    return df


def create_features(df):
    """Create all features needed by the model."""

    # Order Flow
    df['taker_buy_ratio'] = df['taker_buy_volume'] / df['volume'].replace(0, 1)
    df['buy_sell_pressure'] = (df['taker_buy_volume'] - (df['volume'] - df['taker_buy_volume'])) / df['volume'].replace(0, 1)

    # Price Action
    df['body'] = abs(df['close'] - df['open'])
    df['upper_wick'] = df['high'] - df[['close', 'open']].max(axis=1)
    df['lower_wick'] = df[['close', 'open']].min(axis=1) - df['low']
    df['range'] = df['high'] - df['low']
    df['body_ratio'] = df['body'] / df['range'].replace(0, 1)
    df['upper_wick_ratio'] = df['upper_wick'] / df['range'].replace(0, 1)
    df['lower_wick_ratio'] = df['lower_wick'] / df['range'].replace(0, 1)

    # Returns
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

    # Moving Averages
    for period in [7, 14, 21, 50, 100, 200]:
        df[f'sma_{period}'] = df['close'].rolling(period).mean()
        df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        df[f'price_to_sma_{period}'] = df['close'] / df[f'sma_{period}']
        df[f'price_to_ema_{period}'] = df['close'] / df[f'ema_{period}']

    # RSI
    for period in [7, 14, 21]:
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = -delta.where(delta < 0, 0).rolling(period).mean()
        rs = gain / loss.replace(0, 1)
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_diff'] = df['macd'] - df['macd_signal']

    # Bollinger Bands
    for period in [20, 50]:
        sma = df['close'].rolling(period).mean()
        std = df['close'].rolling(period).std()
        df[f'bb_upper_{period}'] = sma + (std * 2)
        df[f'bb_lower_{period}'] = sma - (std * 2)
        df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / sma
        df[f'bb_position_{period}'] = (df['close'] - df[f'bb_lower_{period}']) / (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']).replace(0, 1)

    # ATR
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_14'] = true_range.rolling(14).mean()

    # Volume features
    df['volume_sma_20'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma_20'].replace(0, 1)

    # Momentum
    for period in [5, 10, 20]:
        df[f'momentum_{period}'] = df['close'] - df['close'].shift(period)
        df[f'roc_{period}'] = ((df['close'] - df['close'].shift(period)) / df['close'].shift(period)) * 100

    # Time-based features
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)

    # Microstructure
    df['high_low_ratio'] = df['high'] / df['low'].replace(0, 1)
    df['close_open_ratio'] = df['close'] / df['open'].replace(0, 1)

    # Lag features
    for lag in [1, 2, 3, 5]:
        df[f'close_lag_{lag}'] = df['close'].shift(lag)
        df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
        df[f'returns_lag_{lag}'] = df['returns'].shift(lag)

    # Drop NaN
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
  %(prog)s --days 180 --capital 10000
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
    df = fetch_binance_data(args.symbol, args.interval, args.days)

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
