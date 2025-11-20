"""
BACKTEST COM THRESHOLDS REAIS DO MODELO
==========================================

PROBLEMA IDENTIFICADO:
- Modelo foi treinado com long_threshold=0.35, short_threshold=0.65
- TODOS os backtests anteriores usavam 0.65-0.80 para longs!
- Resultado: filtramos os sinais que o modelo foi desenhado para gerar

SOLUÇÃO:
- Usar thresholds EXATOS do treino (0.35 / 0.65)
- SL/TP simples baseado em ATR (como configs antigas)
- Sem filtros de regime (modelo não foi treinado com isso)
- Lookforward realista (100 candles = 25h)

EXPECTED:
- Win Rate 45-50% (vs terrible 35% com thresholds errados)
- ROI 2-5% em 90 dias
- Trades ~100-150 (vs 5340 ou 45 com thresholds errados)
"""

import pickle
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class ModelWrapper:
    """Compatível com pickle do modelo."""
    def __init__(self, models_list, model_weights, model_names, scaler,
                 feature_columns, has_dl=False, long_threshold=0.35,
                 short_threshold=0.65, lookback=10):
        self.models_list = models_list
        self.model_weights = model_weights
        self.model_names = model_names
        self.scaler = scaler
        self.feature_columns = feature_columns
        self.has_dl = has_dl
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        self.lookback = lookback


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'ModelWrapper':
            return ModelWrapper
        return super().find_class(module, name)


def load_model(model_path):
    """Carrega modelo pickle."""
    print(f"📦 Carregando: {model_path}")
    with open(model_path, 'rb') as f:
        wrapper = CustomUnpickler(f).load()

    print(f"   ✅ Modelo carregado!")
    print(f"   Features: {len(wrapper.feature_columns)}")
    print(f"   Long threshold: {wrapper.long_threshold}")
    print(f"   Short threshold: {wrapper.short_threshold}")
    print(f"   Models: {wrapper.model_names}")
    return wrapper


def get_binance_data(days=90):
    """Baixa dados da Binance."""
    print(f"\n📥 Baixando {days} dias...")

    all_data = []
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

    url = "https://api.binance.com/api/v3/klines"
    current_time = start_time

    while current_time < end_time:
        params = {
            'symbol': 'BTCUSDT',
            'interval': '15m',
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
            print(f"   ❌ Erro: {e}")
            break

    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])

    for col in ['open', 'high', 'low', 'close', 'volume', 'taker_buy_base', 'taker_buy_quote']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.sort_values('timestamp').reset_index(drop=True)

    print(f"   ✅ {len(df)} candles de {df['timestamp'].min()} a {df['timestamp'].max()}")
    return df


def create_features(df):
    """Cria features IDÊNTICAS ao treino."""
    print("\n🔧 Criando features...")

    # Price action
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

    # Body/Wick
    df['body_size'] = np.abs(df['close'] - df['open']) / df['open']
    df['upper_wick'] = (df['high'] - df[['close', 'open']].max(axis=1)) / df['open']
    df['lower_wick'] = (df[['close', 'open']].min(axis=1) - df['low']) / df['open']
    df['total_wick'] = df['upper_wick'] + df['lower_wick']
    df['wick_body_ratio'] = df['total_wick'] / (df['body_size'] + 1e-8)

    # Candle patterns
    df['is_green'] = (df['close'] > df['open']).astype(int)
    df['green_streak'] = df['is_green'].rolling(3).sum()
    df['red_streak'] = (1 - df['is_green']).rolling(3).sum()

    # Order flow
    df['taker_buy_ratio'] = df['taker_buy_base'] / (df['volume'] + 1e-8)
    df['taker_sell_ratio'] = 1 - df['taker_buy_ratio']
    df['buy_pressure_ma'] = df['taker_buy_ratio'].rolling(7).mean()
    df['sell_pressure_ma'] = df['taker_sell_ratio'].rolling(7).mean()
    df['pressure_delta'] = df['buy_pressure_ma'] - df['sell_pressure_ma']
    df['pressure_momentum'] = df['pressure_delta'].diff(3)
    df['order_imbalance'] = (df['taker_buy_base'] - (df['volume'] - df['taker_buy_base'])) / (df['volume'] + 1e-8)
    df['imbalance_ma'] = df['order_imbalance'].rolling(5).mean()

    # Moving averages
    for period in [7, 14, 21, 50]:
        df[f'sma_{period}'] = df['close'].rolling(period).mean()
        df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        df[f'price_sma_{period}_ratio'] = df['close'] / df[f'sma_{period}']

    # Crossovers
    df['ema7_above_ema14'] = (df['ema_7'] > df['ema_14']).astype(int)
    df['ema14_above_ema21'] = (df['ema_14'] > df['ema_21']).astype(int)
    df['ema21_above_ema50'] = (df['ema_21'] > df['ema_50']).astype(int)
    df['golden_cross'] = df['ema7_above_ema14'] & df['ema14_above_ema21']
    df['death_cross'] = (1 - df['ema7_above_ema14']) & (1 - df['ema14_above_ema21'])

    # Volatility (ATR)
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

    # Price position
    df['price_position_14'] = (df['close'] - df['low'].rolling(14).min()) / \
                               (df['high'].rolling(14).max() - df['low'].rolling(14).min() + 1e-8)
    df['price_position_50'] = (df['close'] - df['low'].rolling(50).min()) / \
                               (df['high'].rolling(50).max() - df['low'].rolling(50).min() + 1e-8)

    # Trend strength
    df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
    df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
    df['hh_count'] = df['higher_high'].rolling(5).sum()
    df['ll_count'] = df['lower_low'].rolling(5).sum()
    df['trend_strength'] = df['hh_count'] - df['ll_count']

    # Time features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
    df['london_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
    df['us_session'] = ((df['hour'] >= 13) & (df['hour'] < 22)).astype(int)
    df['weekend'] = (df['day_of_week'] >= 5).astype(int)

    # Microstructure
    df['spread_proxy'] = (df['high'] - df['low']) / df['close']
    df['spread_ma'] = df['spread_proxy'].rolling(10).mean()
    df['large_candle'] = (df['body_size'] > df['body_size'].rolling(20).mean() * 1.5).astype(int)

    # Clean
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)

    print(f"   ✅ {len(df.columns)} features criadas")
    return df


def backtest(df, wrapper):
    """
    Backtest SIMPLES com thresholds REAIS do modelo.
    """
    print("\n" + "="*80)
    print("📊 BACKTEST COM THRESHOLDS REAIS DO MODELO")
    print("="*80)

    # Get predictions usando THRESHOLDS DO MODELO
    X = df[wrapper.feature_columns].values
    proba = wrapper.predict_proba(X)[:, 1]

    # Usar thresholds DO MODELO (não thresholds customizados!)
    long_thresh = wrapper.long_threshold   # 0.35
    short_thresh = wrapper.short_threshold # 0.65

    print(f"\n🎯 Thresholds (do modelo):")
    print(f"   Long: {long_thresh}")
    print(f"   Short: {short_thresh}")

    # Convert to numpy for speed
    close_arr = df['close'].values
    high_arr = df['high'].values
    low_arr = df['low'].values
    atr_arr = df['atr_14'].values

    # Tracking
    trades = []
    balance = 10000
    position = None
    last_trade_idx = -999

    print(f"\n💰 Capital inicial: ${balance:,.2f}")
    print(f"🎲 Risco por trade: 2%")
    print("\n🔄 Executando backtest...")

    for i in range(100, len(df) - 100):  # 100 candles lookforward

        # Se temos posição, verificar saída
        if position:
            high = high_arr[i]
            low = low_arr[i]

            if position['side'] == 'long':
                # Check TP
                if high >= position['tp']:
                    pnl = (position['tp'] - position['entry']) * position['size']
                    balance += pnl
                    trades.append({
                        'entry_idx': position['entry_idx'],
                        'exit_idx': i,
                        'side': 'long',
                        'entry': position['entry'],
                        'exit': position['tp'],
                        'pnl': pnl,
                        'exit_reason': 'TP',
                        'bars_held': i - position['entry_idx']
                    })
                    position = None
                    last_trade_idx = i
                    continue

                # Check SL
                if low <= position['sl']:
                    pnl = (position['sl'] - position['entry']) * position['size']
                    balance += pnl
                    trades.append({
                        'entry_idx': position['entry_idx'],
                        'exit_idx': i,
                        'side': 'long',
                        'entry': position['entry'],
                        'exit': position['sl'],
                        'pnl': pnl,
                        'exit_reason': 'SL',
                        'bars_held': i - position['entry_idx']
                    })
                    position = None
                    last_trade_idx = i
                    continue

            else:  # short
                # Check TP
                if low <= position['tp']:
                    pnl = (position['entry'] - position['tp']) * position['size']
                    balance += pnl
                    trades.append({
                        'entry_idx': position['entry_idx'],
                        'exit_idx': i,
                        'side': 'short',
                        'entry': position['entry'],
                        'exit': position['tp'],
                        'pnl': pnl,
                        'exit_reason': 'TP',
                        'bars_held': i - position['entry_idx']
                    })
                    position = None
                    last_trade_idx = i
                    continue

                # Check SL
                if high >= position['sl']:
                    pnl = (position['entry'] - position['sl']) * position['size']
                    balance += pnl
                    trades.append({
                        'entry_idx': position['entry_idx'],
                        'exit_idx': i,
                        'side': 'short',
                        'entry': position['entry'],
                        'exit': position['sl'],
                        'pnl': pnl,
                        'exit_reason': 'SL',
                        'bars_held': i - position['entry_idx']
                    })
                    position = None
                    last_trade_idx = i
                    continue

        # Se não temos posição, procurar entrada
        if not position and (i - last_trade_idx) >= 5:  # Cooldown 5 candles

            prob = proba[i]
            close = close_arr[i]
            atr = atr_arr[i]

            # LONG signal
            if prob >= long_thresh:
                risk_amount = balance * 0.02
                sl_distance = atr * 1.5  # 1.5x ATR
                position_size = risk_amount / sl_distance

                position = {
                    'side': 'long',
                    'entry': close,
                    'sl': close - sl_distance,
                    'tp': close + (sl_distance * 2.0),  # RR 1:2
                    'size': position_size,
                    'entry_idx': i,
                    'prob': prob
                }

            # SHORT signal (apenas se prob < threshold de long, para evitar conflito)
            elif prob <= (1 - short_thresh):  # Inverso para short
                risk_amount = balance * 0.02
                sl_distance = atr * 1.5
                position_size = risk_amount / sl_distance

                position = {
                    'side': 'short',
                    'entry': close,
                    'sl': close + sl_distance,
                    'tp': close - (sl_distance * 2.0),  # RR 1:2
                    'size': position_size,
                    'entry_idx': i,
                    'prob': prob
                }

    # Close any open position
    if position:
        close = close_arr[-1]
        if position['side'] == 'long':
            pnl = (close - position['entry']) * position['size']
        else:
            pnl = (position['entry'] - close) * position['size']

        balance += pnl
        trades.append({
            'entry_idx': position['entry_idx'],
            'exit_idx': len(df) - 1,
            'side': position['side'],
            'entry': position['entry'],
            'exit': close,
            'pnl': pnl,
            'exit_reason': 'END',
            'bars_held': len(df) - 1 - position['entry_idx']
        })

    # Results
    if not trades:
        print("\n❌ Nenhuma trade executada!")
        return

    df_trades = pd.DataFrame(trades)

    total_trades = len(df_trades)
    wins = (df_trades['pnl'] > 0).sum()
    losses = (df_trades['pnl'] <= 0).sum()
    win_rate = wins / total_trades * 100

    total_pnl = df_trades['pnl'].sum()
    fees = total_pnl * 0.13  # 0.13% fees
    net_pnl = total_pnl - fees
    roi = (net_pnl / 10000) * 100

    avg_win = df_trades[df_trades['pnl'] > 0]['pnl'].mean() if wins > 0 else 0
    avg_loss = abs(df_trades[df_trades['pnl'] <= 0]['pnl'].mean()) if losses > 0 else 0

    # Saídas
    tp_exits = (df_trades['exit_reason'] == 'TP').sum()
    sl_exits = (df_trades['exit_reason'] == 'SL').sum()

    print(f"\n✅ Backtest concluído!")
    print("\n" + "="*80)
    print("📊 RESULTADOS")
    print("="*80)
    print(f"\n💹 Performance:")
    print(f"   Trades: {total_trades}")
    print(f"   Wins: {wins} ({win_rate:.2f}%)")
    print(f"   Losses: {losses} ({(losses/total_trades)*100:.2f}%)")
    print(f"   ROI Bruto: {(total_pnl/10000)*100:+.2f}%")
    print(f"   Fees: ${fees:.2f}")
    print(f"   ROI Líquido: {roi:+.2f}%")
    print(f"   Balance Final: ${balance:,.2f}")

    print(f"\n💰 Por Trade:")
    print(f"   Avg Win: ${avg_win:.2f}")
    print(f"   Avg Loss: ${avg_loss:.2f}")
    print(f"   Avg Bars Held: {df_trades['bars_held'].mean():.1f}")

    print(f"\n🎯 Saídas:")
    print(f"   TP Hit: {tp_exits} ({tp_exits/total_trades*100:.1f}%)")
    print(f"   SL Hit: {sl_exits} ({sl_exits/total_trades*100:.1f}%)")

    # Critérios
    print(f"\n📋 Critérios de Sucesso:")
    roi_ok = roi >= 2.0
    wr_ok = win_rate >= 40
    trades_ok = total_trades >= 40
    tp_rate_ok = (tp_exits / total_trades) >= 0.30

    print(f"   {'✅' if roi_ok else '❌'} ROI ≥ 2%: {roi:+.2f}%")
    print(f"   {'✅' if wr_ok else '❌'} Win Rate ≥ 40%: {win_rate:.2f}%")
    print(f"   {'✅' if trades_ok else '❌'} Trades ≥ 40: {total_trades}")
    print(f"   {'✅' if tp_rate_ok else '❌'} TP Rate ≥ 30%: {tp_exits/total_trades*100:.1f}%")

    approved = sum([roi_ok, wr_ok, trades_ok, tp_rate_ok])

    print(f"\n{'🎉' if approved >= 3 else '⚠️ '} APROVADO: {approved}/4 critérios")

    if approved >= 3:
        print("\n✅ ESTRATÉGIA APROVADA!")
        print("   Modelo V3 FUNCIONA com thresholds corretos!")
    else:
        print("\n❌ ESTRATÉGIA REPROVADA")
        print("   Mesmo com thresholds corretos, modelo não atinge critérios")
        print("   Recomendação: Retreinar modelo V7")

    print("\n" + "="*80)


if __name__ == "__main__":
    print("="*80)
    print("🔬 BACKTEST COM THRESHOLDS REAIS DO MODELO")
    print("="*80)
    print()
    print("🎯 TESTE DEFINITIVO:")
    print("   - Thresholds do treino (0.35 long / 0.65 short)")
    print("   - SL/TP simples ATR-based (1.5x ATR, RR 1:2)")
    print("   - Sem filtros de regime")
    print("   - Lookforward 100 candles")
    print()

    # Load model
    wrapper = load_model('storage/models/model_DEFINITIVO_4ML_540d.pkl')

    # Get data
    df = get_binance_data(days=90)

    # Features
    df = create_features(df)

    # Backtest
    backtest(df, wrapper)

    print("\n✅ Análise completa!")
