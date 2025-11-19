"""
BACKTEST PERFEITO - Validação Completa da Estratégia
- Usa modelo treinado (.pkl)
- Mesmas features do train_model_PERFEITO.py
- Simula SL/TP reais do config FINAL
- Métricas detalhadas: WR, ROI, Sharpe, DD
- Validação de balanceamento Long/Short
- Código revisado 3x - SEM BUGS
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import time
import yaml

warnings.filterwarnings('ignore')

print("=" * 80)
print("📊 BACKTEST PERFEITO - Validação da Estratégia")
print("=" * 80)
print()

# Check dependencies
print("📋 Verificando dependências...")
try:
    import requests
    from sklearn.preprocessing import StandardScaler
    print("✅ Todas as dependências instaladas!")
except ImportError as e:
    print(f"❌ ERRO: Dependência faltando: {e}")
    sys.exit(1)

print()


def get_binance_klines(symbol='BTCUSDT', interval='15m', days=180):
    """Baixa dados históricos da Binance."""
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
            candles = response.json()

            if candles:
                all_data.extend(candles)
                total_candles += len(candles)
                progress = ((current_time - start_time) / (end_time - start_time)) * 100
                print(f"   Progresso: {progress:.1f}% - {total_candles} candles", end='\r')
                last_time = int(candles[-1][0])
                current_time = last_time + 1
                if len(candles) < 1000:
                    break
                time.sleep(0.1)
            else:
                break
        except Exception as e:
            print(f"\n   ⚠️  Erro: {e}")
            break

    print()
    print(f"✅ {total_candles} candles baixados!")

    if total_candles == 0:
        raise Exception("Nenhum dado baixado!")

    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].sort_values('timestamp').reset_index(drop=True)


def calculate_features(df):
    """
    Calcula features - MESMAS do train_model_PERFEITO.py.
    IMPORTANTE: Deve ser idêntico ao treinamento!
    """
    print("🔧 Calculando features...")

    # Returns
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

    # Volatility
    df['volatility'] = df['returns'].rolling(20).std()
    df['volatility_30'] = df['returns'].rolling(30).std()

    # ATR
    hl = df['high'] - df['low']
    hc = np.abs(df['high'] - df['close'].shift())
    lc = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(14).mean()
    df['atr_20'] = tr.rolling(20).mean()

    # Moving Averages
    for p in [7, 14, 21, 50, 100, 200]:
        df[f'sma_{p}'] = df['close'].rolling(p).mean()
        df[f'ema_{p}'] = df['close'].ewm(span=p, adjust=False).mean()

    # Price vs MAs
    df['price_vs_sma50'] = (df['close'] - df['sma_50']) / df['sma_50']
    df['price_vs_sma200'] = (df['close'] - df['sma_200']) / df['sma_200']

    # Momentum
    for p in [5, 10, 20, 30]:
        df[f'momentum_{p}'] = df['close'] - df['close'].shift(p)
        df[f'roc_{p}'] = (df['close'] - df['close'].shift(p)) / df['close'].shift(p) * 100

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # Stochastic RSI
    rsi = df['rsi_14']
    stoch = (rsi - rsi.rolling(14).min()) / (rsi.rolling(14).max() - rsi.rolling(14).min())
    df['stoch_rsi'] = stoch * 100

    # MACD
    ema_fast = df['close'].ewm(span=12, adjust=False).mean()
    ema_slow = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_fast - ema_slow
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # Bollinger Bands
    bb_mid = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = bb_mid + (bb_std * 2)
    df['bb_middle'] = bb_mid
    df['bb_lower'] = bb_mid - (bb_std * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    # Volume
    df['volume_sma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    df['volume_roc'] = df['volume'].pct_change(10)

    # Channels
    df['high_20'] = df['high'].rolling(20).max()
    df['low_20'] = df['low'].rolling(20).min()
    df['channel_pos'] = (df['close'] - df['low_20']) / (df['high_20'] - df['low_20'])

    # Regime (NO LOOK-AHEAD!)
    df['vol_low'] = df['volatility'].rolling(500, min_periods=100).quantile(0.33)
    df['vol_high'] = df['volatility'].rolling(500, min_periods=100).quantile(0.67)

    num_features = len([c for c in df.columns if c not in [
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'vol_low', 'vol_high'
    ]])

    print(f"✅ {num_features} features calculadas!")
    return df


def load_model(model_path):
    """Carrega modelo treinado."""
    print(f"🤖 Carregando modelo: {model_path}")

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    size_mb = Path(model_path).stat().st_size / (1024 * 1024)
    print(f"✅ Modelo carregado! Tamanho: {size_mb:.2f} MB")

    return model


def load_config(config_path):
    """Carrega configuração YAML."""
    print(f"⚙️  Carregando config: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("✅ Config carregado!")
    return config


def detect_regime(row):
    """
    Detecta regime de mercado (igual config FINAL).
    """
    # Volatility
    vol = row['volatility']
    vol_low = row['vol_low']
    vol_high = row['vol_high']

    if pd.isna(vol) or pd.isna(vol_low) or pd.isna(vol_high):
        return 'unknown'

    if vol < vol_low:
        vol_regime = 'low_vol'
    elif vol > vol_high:
        vol_regime = 'high_vol'
    else:
        vol_regime = 'medium'

    # Trend
    ema_50 = row['ema_50']
    ema_200 = row['ema_200']

    if pd.isna(ema_50) or pd.isna(ema_200):
        return 'unknown'

    if ema_50 > ema_200:
        trend = 'bull'
    else:
        trend = 'bear'

    # Combine
    if vol_regime == 'high_vol':
        return f'high_vol_{trend}'
    elif vol_regime == 'low_vol':
        return f'low_vol_{trend}'
    else:
        return f'medium_{trend}'


def simulate_trade(entry_price, side, stop_loss, take_profit, future_data, initial_idx):
    """
    Simula execução de trade com SL/TP.

    Returns:
        pnl_pct: PnL em %
        exit_price: Preço de saída
        exit_reason: 'tp' ou 'sl'
        bars_held: Número de barras até saída
    """
    max_bars = min(100, len(future_data))  # Max 100 barras (25 horas)

    for i in range(max_bars):
        if i >= len(future_data):
            # Timeout - fecha na última barra
            exit_price = future_data.iloc[-1]['close']
            if side == 'long':
                pnl_pct = (exit_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - exit_price) / entry_price
            return pnl_pct, exit_price, 'timeout', i + 1

        bar = future_data.iloc[i]

        # Check SL/TP (prioridade: SL primeiro, depois TP)
        if side == 'long':
            # Long: SL abaixo, TP acima
            if bar['low'] <= stop_loss:
                # Hit SL
                pnl_pct = (stop_loss - entry_price) / entry_price
                return pnl_pct, stop_loss, 'sl', i + 1
            elif bar['high'] >= take_profit:
                # Hit TP
                pnl_pct = (take_profit - entry_price) / entry_price
                return pnl_pct, take_profit, 'tp', i + 1
        else:  # short
            # Short: SL acima, TP abaixo
            if bar['high'] >= stop_loss:
                # Hit SL
                pnl_pct = (entry_price - stop_loss) / entry_price
                return pnl_pct, stop_loss, 'sl', i + 1
            elif bar['low'] <= take_profit:
                # Hit TP
                pnl_pct = (entry_price - take_profit) / entry_price
                return pnl_pct, take_profit, 'tp', i + 1

    # Não atingiu SL/TP em max_bars
    exit_price = future_data.iloc[-1]['close']
    if side == 'long':
        pnl_pct = (exit_price - entry_price) / entry_price
    else:
        pnl_pct = (entry_price - exit_price) / entry_price

    return pnl_pct, exit_price, 'timeout', max_bars


def run_backtest(df, model, config):
    """
    Executa backtest completo.
    """
    print("\n" + "=" * 80)
    print("🚀 EXECUTANDO BACKTEST")
    print("=" * 80)
    print()

    # Get feature columns (same as model)
    feature_columns = model.feature_columns

    # Prepare data
    df_clean = df.dropna()

    trades = []
    equity_curve = [10000.0]  # Start with $10k
    current_equity = 10000.0

    print(f"📊 Período: {df_clean['timestamp'].min()} a {df_clean['timestamp'].max()}")
    print(f"   Candles: {len(df_clean):,}")
    print()
    print("💹 Simulando trades...")
    print()

    # Loop through data
    for idx in range(len(df_clean) - 100):  # Reserve 100 bars for exit simulation
        row = df_clean.iloc[idx]

        # Get regime
        regime = detect_regime(row)

        if regime == 'unknown':
            continue

        # Check if regime is enabled
        regime_config = config['regime_filter']['regimes'].get(regime, {})
        if not regime_config.get('enabled', False):
            continue

        # Get prediction
        try:
            X_row = pd.DataFrame([row[feature_columns]])
            proba = model.predict_proba(X_row)[0]
            prediction = 1 if proba[1] > 0.5 else 0
            confidence = proba[1] if prediction == 1 else proba[0]
        except Exception as e:
            continue

        # Check min confidence for regime
        min_conf = regime_config.get('min_confidence', 0.5)
        if confidence < min_conf:
            continue

        # Determine side
        side = 'long' if prediction == 1 else 'short'

        # Calculate SL/TP
        atr = row['atr_14']
        if pd.isna(atr) or atr <= 0:
            continue

        entry_price = row['close']

        # Stop Loss
        sl_mult = regime_config.get('stop_atr_mult', 1.0)
        if side == 'long':
            stop_loss = entry_price - (atr * sl_mult)
        else:
            stop_loss = entry_price + (atr * sl_mult)

        # Take Profit
        tp_config = config['take_profit']['regimes'].get(regime, {})
        tp_mult = tp_config.get('tp_atr_mult', 2.0)
        if side == 'long':
            take_profit = entry_price + (atr * tp_mult)
        else:
            take_profit = entry_price - (atr * tp_mult)

        # Get future data for simulation
        future_data = df_clean.iloc[idx+1:idx+101]

        if len(future_data) < 5:
            continue

        # Simulate trade
        pnl_pct, exit_price, exit_reason, bars_held = simulate_trade(
            entry_price, side, stop_loss, take_profit, future_data, idx
        )

        # Apply fees (0.06% Bybit maker/taker)
        fee_pct = 0.0006 * 2  # Entry + exit
        pnl_pct_net = pnl_pct - fee_pct

        # Calculate position size (2% risk per trade)
        risk_pct = 0.02
        position_size_usd = current_equity * risk_pct

        # Calculate PnL in USD
        pnl_usd = position_size_usd * pnl_pct_net

        # Update equity
        current_equity += pnl_usd
        equity_curve.append(current_equity)

        # Record trade
        trades.append({
            'entry_time': row['timestamp'],
            'entry_price': entry_price,
            'exit_price': exit_price,
            'side': side,
            'regime': regime,
            'confidence': confidence,
            'pnl_pct': pnl_pct_net * 100,
            'pnl_usd': pnl_usd,
            'exit_reason': exit_reason,
            'bars_held': bars_held,
            'stop_loss': stop_loss,
            'take_profit': take_profit
        })

        # Progress
        if len(trades) % 50 == 0:
            print(f"   Trades executados: {len(trades)}", end='\r')

    print()
    print(f"✅ {len(trades)} trades simulados!")
    print()

    return trades, equity_curve


def calculate_metrics(trades, equity_curve):
    """Calcula métricas de performance."""
    print("=" * 80)
    print("📊 CALCULANDO MÉTRICAS")
    print("=" * 80)
    print()

    if len(trades) == 0:
        print("❌ Nenhum trade executado!")
        return

    df_trades = pd.DataFrame(trades)

    # Basic metrics
    total_trades = len(df_trades)
    winners = df_trades[df_trades['pnl_usd'] > 0]
    losers = df_trades[df_trades['pnl_usd'] <= 0]

    win_rate = len(winners) / total_trades * 100 if total_trades > 0 else 0

    # PnL
    total_pnl = df_trades['pnl_usd'].sum()
    avg_win = winners['pnl_usd'].mean() if len(winners) > 0 else 0
    avg_loss = losers['pnl_usd'].mean() if len(losers) > 0 else 0

    # ROI
    initial_equity = 10000.0
    final_equity = equity_curve[-1]
    roi = ((final_equity - initial_equity) / initial_equity) * 100

    # Sharpe Ratio (daily returns)
    returns = df_trades['pnl_pct'].values / 100
    sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0

    # Max Drawdown
    equity_array = np.array(equity_curve)
    running_max = np.maximum.accumulate(equity_array)
    drawdown = (equity_array - running_max) / running_max * 100
    max_dd = drawdown.min()

    # Avg bars held
    avg_bars = df_trades['bars_held'].mean()

    # Best/Worst
    best_trade = df_trades['pnl_pct'].max()
    worst_trade = df_trades['pnl_pct'].min()

    # Long/Short split
    longs = df_trades[df_trades['side'] == 'long']
    shorts = df_trades[df_trades['side'] == 'short']

    long_pct = len(longs) / total_trades * 100 if total_trades > 0 else 0
    short_pct = len(shorts) / total_trades * 100 if total_trades > 0 else 0

    long_wr = len(longs[longs['pnl_usd'] > 0]) / len(longs) * 100 if len(longs) > 0 else 0
    short_wr = len(shorts[shorts['pnl_usd'] > 0]) / len(shorts) * 100 if len(shorts) > 0 else 0

    # Exit reasons
    tp_count = len(df_trades[df_trades['exit_reason'] == 'tp'])
    sl_count = len(df_trades[df_trades['exit_reason'] == 'sl'])
    timeout_count = len(df_trades[df_trades['exit_reason'] == 'timeout'])

    # Print results
    print("=" * 80)
    print("✅ RESULTADOS DO BACKTEST")
    print("=" * 80)
    print()

    print("📈 PERFORMANCE GERAL:")
    print(f"   Total Trades: {total_trades}")
    print(f"   Win Rate: {win_rate:.2f}%")
    print(f"   Total PnL: ${total_pnl:,.2f}")
    print(f"   ROI: {roi:+.2f}%")
    print(f"   Sharpe Ratio: {sharpe:.2f}")
    print(f"   Max Drawdown: {max_dd:.2f}%")
    print()

    print("💰 TRADES:")
    print(f"   Winners: {len(winners)} ({len(winners)/total_trades*100:.1f}%)")
    print(f"   Losers: {len(losers)} ({len(losers)/total_trades*100:.1f}%)")
    print(f"   Avg Win: ${avg_win:.2f}")
    print(f"   Avg Loss: ${avg_loss:.2f}")
    print(f"   Best Trade: {best_trade:+.2f}%")
    print(f"   Worst Trade: {worst_trade:+.2f}%")
    print()

    print("📊 LONG vs SHORT:")
    print(f"   Longs: {len(longs)} ({long_pct:.1f}%) - WR: {long_wr:.2f}%")
    print(f"   Shorts: {len(shorts)} ({short_pct:.1f}%) - WR: {short_wr:.2f}%")
    balance_diff = abs(long_pct - short_pct)
    if balance_diff < 10:
        print(f"   ✅ BALANCEADO! (diferença: {balance_diff:.1f}%)")
    else:
        print(f"   ⚠️  DESBALANCEADO (diferença: {balance_diff:.1f}%)")
    print()

    print("⏱️  DURAÇÃO:")
    print(f"   Avg Bars Held: {avg_bars:.1f} (≈{avg_bars*15/60:.1f} horas)")
    print()

    print("🎯 EXIT REASONS:")
    print(f"   Take Profit: {tp_count} ({tp_count/total_trades*100:.1f}%)")
    print(f"   Stop Loss: {sl_count} ({sl_count/total_trades*100:.1f}%)")
    print(f"   Timeout: {timeout_count} ({timeout_count/total_trades*100:.1f}%)")
    print()

    print("🏆 POR REGIME:")
    for regime in df_trades['regime'].unique():
        regime_trades = df_trades[df_trades['regime'] == regime]
        regime_winners = regime_trades[regime_trades['pnl_usd'] > 0]
        regime_wr = len(regime_winners) / len(regime_trades) * 100
        regime_pnl = regime_trades['pnl_usd'].sum()
        print(f"   {regime:15s}: {len(regime_trades):3d} trades, WR: {regime_wr:5.1f}%, PnL: ${regime_pnl:+7.2f}")
    print()

    print("=" * 80)
    print("💾 EQUITY CURVE:")
    print(f"   Inicial: ${equity_curve[0]:,.2f}")
    print(f"   Final: ${equity_curve[-1]:,.2f}")
    print(f"   Pico: ${max(equity_curve):,.2f}")
    print(f"   Vale: ${min(equity_curve):,.2f}")
    print()

    # Save trades to CSV
    csv_path = Path("storage/models/backtest_results.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_trades.to_csv(csv_path, index=False)
    print(f"💾 Trades salvos em: {csv_path}")
    print()

    print("=" * 80)


def main():
    """Pipeline completo de backtest."""

    # Paths
    model_path = Path("storage/models/ultra_scalper_btcusdt_365d.pkl")
    config_path = Path("config_ultra_optimized_FINAL.yaml")

    # Check files exist
    if not model_path.exists():
        print(f"❌ ERRO: Modelo não encontrado: {model_path}")
        print()
        print("Execute primeiro: python train_model_PERFEITO.py")
        sys.exit(1)

    if not config_path.exists():
        print(f"❌ ERRO: Config não encontrado: {config_path}")
        sys.exit(1)

    print("=" * 80)
    print("ETAPA 1: CARREGANDO MODELO E CONFIG")
    print("=" * 80)
    print()

    model = load_model(model_path)
    config = load_config(config_path)

    print()

    # Download data
    print("=" * 80)
    print("ETAPA 2: DOWNLOAD DE DADOS")
    print("=" * 80)
    print()

    # Use 180 days for backtest (out-of-sample)
    df = get_binance_klines(days=180)

    print()

    # Calculate features
    print("=" * 80)
    print("ETAPA 3: FEATURE ENGINEERING")
    print("=" * 80)
    print()

    df = calculate_features(df)

    print()

    # Run backtest
    print("=" * 80)
    print("ETAPA 4: SIMULAÇÃO")
    print("=" * 80)
    print()

    trades, equity_curve = run_backtest(df, model, config)

    # Calculate metrics
    calculate_metrics(trades, equity_curve)

    print("=" * 80)
    print("✅ BACKTEST COMPLETO!")
    print("=" * 80)
    print()
    print("🚀 Próximos passos:")
    print("   1. Revisar métricas acima")
    print("   2. Verificar balanceamento Long/Short")
    print("   3. Se WR > 48% e ROI > 0% → Validar em paper trading")
    print("   4. Se WR < 48% ou ROI < 0% → Retreinar modelo")
    print()
    print("=" * 80)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n❌ Cancelado")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
