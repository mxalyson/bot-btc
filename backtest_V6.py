"""
BACKTEST V6 - Compatible com MODELO DEFINITIVO V6

IMPORTANTE:
- Usa as MESMAS 99 features do modelo de treinamento
- Compatible com ModelWrapper do V6
- Aplica threshold ajustado (0.35 long, 0.65 short)
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import yaml

warnings.filterwarnings('ignore')

print("=" * 80)
print("🔍 BACKTEST V6 - MODELO DEFINITIVO")
print("=" * 80)
print()

# 1. LOAD MODEL
print("=" * 80)
print("ETAPA 1: CARREGAR MODELO")
print("=" * 80)
print()

# Find latest V6 model
storage_dir = Path("storage/models")
v6_models = list(storage_dir.glob("model_DEFINITIVO_V6_*.pkl"))

if not v6_models:
    print("❌ Nenhum modelo V6 encontrado!")
    print(f"   Procurado em: {storage_dir}")
    sys.exit(1)

# Use most recent
model_file = sorted(v6_models, key=lambda x: x.stat().st_mtime)[-1]

print(f"📂 Carregando modelo: {model_file.name}")

with open(model_file, 'rb') as f:
    wrapper = pickle.load(f)

model_size_mb = os.path.getsize(model_file) / (1024 * 1024)
print(f"✅ Modelo carregado! Tamanho: {model_size_mb:.2f} MB")
print(f"   Features esperadas: {len(wrapper.feature_columns)}")
print(f"   Threshold Long: {wrapper.long_threshold}")
print(f"   Threshold Short: {wrapper.short_threshold}")

# 2. LOAD CONFIG
print()
print("=" * 80)
print("ETAPA 2: CARREGAR CONFIG")
print("=" * 80)
print()

config_file = "config_ultra_optimized_FINAL.yaml"
print(f"⚙️  Carregando config: {config_file}")

with open(config_file, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

print(f"✅ Config carregado!")

# 3. DOWNLOAD DATA
print()
print("=" * 80)
print("ETAPA 3: DOWNLOAD DE DADOS")
print("=" * 80)
print()

import requests

def get_binance_klines(symbol='BTCUSDT', interval='15m', days=180):
    """Baixa dados da Binance."""
    print(f"📥 Baixando {days} dias de dados para backtest...")

    all_data = []
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

    url = "https://api.binance.com/api/v3/klines"
    current_time = start_time
    total_candles = 0

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
            total_candles += len(klines)

            # Progress
            progress = (current_time - start_time) / (end_time - start_time) * 100
            print(f"   Progresso: {progress:.1f}% - {total_candles} candles", end='\r')

            if len(klines) < 1000:
                break

        except Exception as e:
            print(f"\n   ❌ Erro: {e}")
            break

    print()  # New line after progress

    if not all_data:
        print("   ❌ Nenhum dado baixado!")
        sys.exit(1)

    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])

    for col in ['open', 'high', 'low', 'close', 'volume', 'taker_buy_base']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.sort_values('timestamp').reset_index(drop=True)

    print(f"✅ {len(df)} candles baixados!")
    return df

df = get_binance_klines(days=180)

# 4. ADD FEATURES (EXACTLY THE SAME AS TRAINING!)
print()
print("=" * 80)
print("ETAPA 4: FEATURE ENGINEERING")
print("=" * 80)
print()

def add_extraordinary_features(df):
    """
    Features EXTRAORDINÁRIAS para scalping.
    Combinação de técnicas avançadas + order flow.
    """
    print("🔧 Criando features EXTRAORDINÁRIAS...")

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
    df['taker_buy_ratio'] = df['taker_buy_base'] / (df['volume'] + 1e-8)
    df['taker_sell_ratio'] = 1 - df['taker_buy_ratio']

    # Buy/Sell pressure momentum
    df['buy_pressure_ma'] = df['taker_buy_ratio'].rolling(7).mean()
    df['sell_pressure_ma'] = df['taker_sell_ratio'].rolling(7).mean()
    df['pressure_delta'] = df['buy_pressure_ma'] - df['sell_pressure_ma']
    df['pressure_momentum'] = df['pressure_delta'].diff(3)

    # Order flow imbalance
    df['order_imbalance'] = (df['taker_buy_base'] - (df['volume'] - df['taker_buy_base'])) / (df['volume'] + 1e-8)
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
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek

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
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)

    print(f"   ✅ {len(df.columns)} features criadas")
    return df


print("🔧 Calculando features...")
df = add_extraordinary_features(df)
print(f"✅ {len(df.columns)} features calculadas!")

# Verify features match
model_features = set(wrapper.feature_columns)
df_features = set(df.columns)

missing_in_df = model_features - df_features
extra_in_df = df_features - model_features

if missing_in_df:
    print(f"\n⚠️  AVISO: {len(missing_in_df)} features faltando no DataFrame:")
    for feat in list(missing_in_df)[:10]:
        print(f"   - {feat}")
    if len(missing_in_df) > 10:
        print(f"   ... e mais {len(missing_in_df)-10}")

if extra_in_df:
    print(f"\n📊 DataFrame tem {len(extra_in_df)} features extras (OK, serão ignoradas)")

# 5. RUN BACKTEST
print()
print("=" * 80)
print("ETAPA 5: SIMULAÇÃO")
print("=" * 80)
print()

def run_backtest(df, wrapper, config):
    """Executa backtest com modelo V6."""

    print()
    print("=" * 80)
    print("🚀 EXECUTANDO BACKTEST")
    print("=" * 80)
    print()

    # Clean data
    df = df.dropna().reset_index(drop=True)

    print(f"📊 Período: {df['timestamp'].min()} a {df['timestamp'].max()}")
    print(f"   Candles: {len(df):,}")
    print()

    # Get predictions
    print("💹 Simulando trades...")
    print()

    try:
        # Use only features that model expects
        predictions = wrapper.predict(df)
        probabilities = wrapper.predict_proba(df)[:, 1]

        print(f"   ✅ {len(predictions)} predições geradas")
        print(f"   Longs preditos: {(predictions == 1).sum()}")
        print(f"   Shorts preditos: {(predictions == 0).sum()}")

    except Exception as e:
        print(f"   ❌ Erro ao gerar predições: {e}")
        import traceback
        traceback.print_exc()
        return

    # Simulate trades
    trades = []
    balance = config['risk_management']['initial_capital']
    capital_per_trade = balance * (config['risk_management']['risk_per_trade'] / 100)

    for i in range(len(df) - 1):
        signal = predictions[i]
        prob = probabilities[i]

        if signal == 1:  # Long
            entry_price = df['close'].iloc[i]
            atr = df['atr_14'].iloc[i] if 'atr_14' in df.columns else entry_price * 0.01

            sl = entry_price - (atr * config['risk_management']['stop_loss_atr_multiplier'])
            tp = entry_price + (atr * config['risk_management']['take_profit_atr_multiplier'])

            # Find exit
            for j in range(i+1, min(i+50, len(df))):
                if df['low'].iloc[j] <= sl:
                    # Stop loss
                    pnl = (sl - entry_price) / entry_price * capital_per_trade
                    trades.append({
                        'entry_time': df['timestamp'].iloc[i],
                        'entry_price': entry_price,
                        'exit_time': df['timestamp'].iloc[j],
                        'exit_price': sl,
                        'side': 'LONG',
                        'pnl': pnl,
                        'pnl_pct': (sl - entry_price) / entry_price * 100,
                        'result': 'LOSS'
                    })
                    break
                elif df['high'].iloc[j] >= tp:
                    # Take profit
                    pnl = (tp - entry_price) / entry_price * capital_per_trade
                    trades.append({
                        'entry_time': df['timestamp'].iloc[i],
                        'entry_price': entry_price,
                        'exit_time': df['timestamp'].iloc[j],
                        'exit_price': tp,
                        'side': 'LONG',
                        'pnl': pnl,
                        'pnl_pct': (tp - entry_price) / entry_price * 100,
                        'result': 'WIN'
                    })
                    break

    print()
    print(f"✅ {len(trades)} trades simulados!")
    print()

    if not trades:
        print("❌ Nenhum trade executado!")
        print("   Possíveis causas:")
        print("   - Threshold muito alto")
        print("   - Features incompatíveis")
        print("   - Modelo não gera sinais")
        return

    # Calculate metrics
    trades_df = pd.DataFrame(trades)

    wins = len(trades_df[trades_df['result'] == 'WIN'])
    losses = len(trades_df[trades_df['result'] == 'LOSS'])
    win_rate = wins / len(trades_df) * 100 if len(trades_df) > 0 else 0

    total_pnl = trades_df['pnl'].sum()
    roi = total_pnl / balance * 100

    avg_win = trades_df[trades_df['result'] == 'WIN']['pnl'].mean() if wins > 0 else 0
    avg_loss = abs(trades_df[trades_df['result'] == 'LOSS']['pnl'].mean()) if losses > 0 else 0
    rr_ratio = avg_win / avg_loss if avg_loss > 0 else 0

    print("=" * 80)
    print("📊 RESULTADOS DO BACKTEST")
    print("=" * 80)
    print()

    print(f"📈 Performance:")
    print(f"   Total Trades: {len(trades_df)}")
    print(f"   Wins: {wins}")
    print(f"   Losses: {losses}")
    print(f"   Win Rate: {win_rate:.2f}%")
    print()

    print(f"💰 Financeiro:")
    print(f"   Capital Inicial: ${balance:,.2f}")
    print(f"   Total P&L: ${total_pnl:,.2f}")
    print(f"   ROI: {roi:.2f}%")
    print()

    print(f"📊 Risk/Reward:")
    print(f"   Avg Win: ${avg_win:,.2f}")
    print(f"   Avg Loss: ${avg_loss:,.2f}")
    print(f"   R:R Ratio: {rr_ratio:.2f}")
    print()

    # Breakdown by side
    longs = trades_df[trades_df['side'] == 'LONG']
    if len(longs) > 0:
        long_wr = len(longs[longs['result'] == 'WIN']) / len(longs) * 100
        print(f"📊 Long Trades:")
        print(f"   Total: {len(longs)}")
        print(f"   Win Rate: {long_wr:.2f}%")
        print()

run_backtest(df, wrapper, config)

print("=" * 80)
print("✅ BACKTEST COMPLETO!")
print("=" * 80)
print()

print("🚀 Próximos passos:")
print("   1. Revisar métricas acima")
print("   2. Se WR > 48% e ROI > 0% → APROVADO! 🎉")
print("   3. Se não → Verificar features e threshold")
print()
print("=" * 80)
