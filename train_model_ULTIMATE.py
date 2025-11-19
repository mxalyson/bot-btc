"""
ULTIMATE ENSEMBLE - Deep Learning + Gradient Boosting + Meta-Learning

Architecture:
Level 0 (Base Models):
- LightGBM (Gradient Boosting)
- XGBoost (Gradient Boosting)
- Random Forest (Bagging)
- LSTM (Deep Learning - Sequence)
- Transformer (Deep Learning - Attention)
- CNN 1D (Deep Learning - Patterns)

Level 1 (Meta-Learner):
- Neural Network (aprende a combinar as 6 previsões)

BEST OF BOTH WORLDS: ML + DL!
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

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Silence TensorFlow

print("=" * 80)
print("🚀 ULTIMATE ENSEMBLE - ML + DEEP LEARNING")
print("=" * 80)
print()

# Check dependencies
print("📋 Verificando dependências...")
try:
    import requests
    from sklearn.ensemble import RandomForestClassifier, StackingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.metrics import accuracy_score, classification_report
    import lightgbm as lgb
    import xgboost as xgb

    # Deep Learning
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks
    from tensorflow.keras.optimizers import Adam

    print("✅ Todas as dependências instaladas!")
    print(f"   TensorFlow: {tf.__version__}")
except ImportError as e:
    print(f"❌ ERRO: Dependência faltando: {e}")
    print()
    print("Para Deep Learning, instale:")
    print("pip install tensorflow")
    sys.exit(1)

print()

storage_dir = Path("storage/models")
storage_dir.mkdir(parents=True, exist_ok=True)


def get_binance_klines(symbol='BTCUSDT', interval='15m', days=365):
    """Baixa dados da Binance."""
    print("📥 Baixando dados do BTC via BINANCE...")
    print(f"   Símbolo: {symbol}")
    print(f"   Timeframe: {interval}")
    print(f"   Período: {days} dias")
    print()

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
    print()

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
    """Calcula 50+ features."""
    print("🔧 Calculando features...")

    # Returns & Volatility
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
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

    # Price Channels
    df['high_20'] = df['high'].rolling(20).max()
    df['low_20'] = df['low'].rolling(20).min()
    df['channel_pos'] = (df['close'] - df['low_20']) / (df['high_20'] - df['low_20'])

    # Regime
    df['vol_low'] = df['volatility'].rolling(500, min_periods=100).quantile(0.33)
    df['vol_high'] = df['volatility'].rolling(500, min_periods=100).quantile(0.67)

    num_features = len([c for c in df.columns if c not in ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'vol_low', 'vol_high']])
    print(f"✅ {num_features} features calculadas!")
    print()
    return df


def create_labels(df, future_periods=3, threshold=0.003):
    """Cria labels."""
    print("🏷️  Criando labels...")
    df['future_return'] = df['close'].shift(-future_periods) / df['close'] - 1
    df['label'] = (df['future_return'] > threshold).astype(int)
    longs = (df['label'] == 1).sum()
    shorts = (df['label'] == 0).sum()
    print(f"✅ Labels: Longs {longs} ({longs/len(df)*100:.1f}%), Shorts {shorts} ({shorts/len(df)*100:.1f}%)")
    print()
    return df


def create_sequences(X, y, seq_length=60):
    """Cria sequências para LSTM/Transformer."""
    X_seq, y_seq = [], []
    for i in range(seq_length, len(X)):
        X_seq.append(X[i-seq_length:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)


def build_lstm_model(input_shape):
    """LSTM para séries temporais."""
    model = models.Sequential([
        layers.LSTM(128, return_sequences=True, input_shape=input_shape),
        layers.Dropout(0.3),
        layers.LSTM(64, return_sequences=False),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model


def build_transformer_model(input_shape):
    """Transformer com multi-head attention."""
    inputs = layers.Input(shape=input_shape)

    # Multi-Head Attention
    attn_output = layers.MultiHeadAttention(
        num_heads=4, key_dim=32
    )(inputs, inputs)
    attn_output = layers.Dropout(0.3)(attn_output)
    attn_output = layers.LayerNormalization(epsilon=1e-6)(attn_output + inputs)

    # Feed Forward
    ff_output = layers.Dense(128, activation='relu')(attn_output)
    ff_output = layers.Dropout(0.3)(ff_output)
    ff_output = layers.Dense(input_shape[-1])(ff_output)
    ff_output = layers.LayerNormalization(epsilon=1e-6)(ff_output + attn_output)

    # Global Pooling
    gap = layers.GlobalAveragePooling1D()(ff_output)

    # Classification
    dense = layers.Dense(64, activation='relu')(gap)
    dense = layers.Dropout(0.2)(dense)
    outputs = layers.Dense(1, activation='sigmoid')(dense)

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model


def build_cnn_model(input_shape):
    """CNN 1D para detectar padrões."""
    model = models.Sequential([
        layers.Conv1D(64, 3, activation='relu', input_shape=input_shape),
        layers.MaxPooling1D(2),
        layers.Dropout(0.3),
        layers.Conv1D(128, 3, activation='relu'),
        layers.MaxPooling1D(2),
        layers.Dropout(0.3),
        layers.Conv1D(64, 3, activation='relu'),
        layers.GlobalAveragePooling1D(),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model


def build_meta_nn(input_dim):
    """Meta-Learner Neural Network."""
    model = models.Sequential([
        layers.Dense(32, activation='relu', input_dim=input_dim),
        layers.Dropout(0.3),
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model


def train_ultimate_ensemble(X_train, y_train, X_test, y_test, X_train_scaled, X_test_scaled):
    """Treina ULTIMATE ENSEMBLE: ML + DL."""
    print("🤖 Treinando ULTIMATE ENSEMBLE...")
    print()

    base_predictions_train = []
    base_predictions_test = []
    scores = {}

    # ========== GRADIENT BOOSTING MODELS ==========
    print("📊 Parte 1/2: Gradient Boosting Models")
    print()

    # LightGBM
    print("   1/6 - LightGBM...")
    lgb_model = lgb.LGBMClassifier(n_estimators=150, max_depth=8, learning_rate=0.05, verbose=-1, n_jobs=-1)
    lgb_model.fit(X_train_scaled, y_train)
    lgb_score = lgb_model.score(X_test_scaled, y_test)
    scores['LightGBM'] = lgb_score
    base_predictions_train.append(lgb_model.predict_proba(X_train_scaled)[:, 1])
    base_predictions_test.append(lgb_model.predict_proba(X_test_scaled)[:, 1])
    print(f"      ✅ LightGBM: {lgb_score:.2%}")

    # XGBoost
    print("   2/6 - XGBoost...")
    xgb_model = xgb.XGBClassifier(n_estimators=150, max_depth=8, learning_rate=0.05, verbosity=0, n_jobs=-1)
    xgb_model.fit(X_train_scaled, y_train)
    xgb_score = xgb_model.score(X_test_scaled, y_test)
    scores['XGBoost'] = xgb_score
    base_predictions_train.append(xgb_model.predict_proba(X_train_scaled)[:, 1])
    base_predictions_test.append(xgb_model.predict_proba(X_test_scaled)[:, 1])
    print(f"      ✅ XGBoost: {xgb_score:.2%}")

    # Random Forest
    print("   3/6 - Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_scaled, y_train)
    rf_score = rf_model.score(X_test_scaled, y_test)
    scores['RandomForest'] = rf_score
    base_predictions_train.append(rf_model.predict_proba(X_train_scaled)[:, 1])
    base_predictions_test.append(rf_model.predict_proba(X_test_scaled)[:, 1])
    print(f"      ✅ Random Forest: {rf_score:.2%}")

    print()
    print("📊 Parte 2/2: Deep Learning Models")
    print()

    # Prepare sequences for DL
    SEQ_LENGTH = 60
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train.values, SEQ_LENGTH)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test.values, SEQ_LENGTH)

    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # LSTM
    print("   4/6 - LSTM (Deep Learning)...")
    lstm_model = build_lstm_model((SEQ_LENGTH, X_train_scaled.shape[1]))
    lstm_model.fit(X_train_seq, y_train_seq, epochs=30, batch_size=32,
                   validation_split=0.2, callbacks=[early_stop], verbose=0)
    lstm_pred_test = lstm_model.predict(X_test_seq, verbose=0).flatten()
    lstm_score = accuracy_score(y_test_seq, (lstm_pred_test > 0.5).astype(int))
    scores['LSTM'] = lstm_score

    # Align predictions
    lstm_pred_train_full = np.zeros(len(y_train))
    lstm_pred_train_full[:] = 0.5
    lstm_pred_train = lstm_model.predict(X_train_seq, verbose=0).flatten()
    lstm_pred_train_full[SEQ_LENGTH:] = lstm_pred_train

    lstm_pred_test_full = np.zeros(len(y_test))
    lstm_pred_test_full[:] = 0.5
    lstm_pred_test_full[SEQ_LENGTH:] = lstm_pred_test

    base_predictions_train.append(lstm_pred_train_full)
    base_predictions_test.append(lstm_pred_test_full)
    print(f"      ✅ LSTM: {lstm_score:.2%}")

    # Transformer
    print("   5/6 - Transformer (Attention)...")
    trans_model = build_transformer_model((SEQ_LENGTH, X_train_scaled.shape[1]))
    trans_model.fit(X_train_seq, y_train_seq, epochs=30, batch_size=32,
                    validation_split=0.2, callbacks=[early_stop], verbose=0)
    trans_pred_test = trans_model.predict(X_test_seq, verbose=0).flatten()
    trans_score = accuracy_score(y_test_seq, (trans_pred_test > 0.5).astype(int))
    scores['Transformer'] = trans_score

    trans_pred_train_full = np.zeros(len(y_train))
    trans_pred_train_full[:] = 0.5
    trans_pred_train = trans_model.predict(X_train_seq, verbose=0).flatten()
    trans_pred_train_full[SEQ_LENGTH:] = trans_pred_train

    trans_pred_test_full = np.zeros(len(y_test))
    trans_pred_test_full[:] = 0.5
    trans_pred_test_full[SEQ_LENGTH:] = trans_pred_test

    base_predictions_train.append(trans_pred_train_full)
    base_predictions_test.append(trans_pred_test_full)
    print(f"      ✅ Transformer: {trans_score:.2%}")

    # CNN 1D
    print("   6/6 - CNN 1D (Patterns)...")
    cnn_model = build_cnn_model((SEQ_LENGTH, X_train_scaled.shape[1]))
    cnn_model.fit(X_train_seq, y_train_seq, epochs=30, batch_size=32,
                  validation_split=0.2, callbacks=[early_stop], verbose=0)
    cnn_pred_test = cnn_model.predict(X_test_seq, verbose=0).flatten()
    cnn_score = accuracy_score(y_test_seq, (cnn_pred_test > 0.5).astype(int))
    scores['CNN'] = cnn_score

    cnn_pred_train_full = np.zeros(len(y_train))
    cnn_pred_train_full[:] = 0.5
    cnn_pred_train = cnn_model.predict(X_train_seq, verbose=0).flatten()
    cnn_pred_train_full[SEQ_LENGTH:] = cnn_pred_train

    cnn_pred_test_full = np.zeros(len(y_test))
    cnn_pred_test_full[:] = 0.5
    cnn_pred_test_full[SEQ_LENGTH:] = cnn_pred_test

    base_predictions_train.append(cnn_pred_train_full)
    base_predictions_test.append(cnn_pred_test_full)
    print(f"      ✅ CNN 1D: {cnn_score:.2%}")

    print()
    print("🎯 Meta-Learner (Neural Network)...")

    # Stack predictions
    meta_X_train = np.column_stack(base_predictions_train)
    meta_X_test = np.column_stack(base_predictions_test)

    # Train meta-learner
    meta_model = build_meta_nn(6)  # 6 base models
    meta_model.fit(meta_X_train, y_train, epochs=50, batch_size=64,
                   validation_split=0.2, callbacks=[early_stop], verbose=0)

    meta_pred_test = meta_model.predict(meta_X_test, verbose=0).flatten()
    meta_score = accuracy_score(y_test, (meta_pred_test > 0.5).astype(int))
    scores['MetaNN'] = meta_score

    print(f"   ✅ Meta-NN: {meta_score:.2%}")
    print()

    avg_base = np.mean(list(scores.values())[:-1])
    improvement = ((meta_score - avg_base) / avg_base) * 100
    print(f"🚀 Melhoria: {improvement:+.1f}%")
    print()

    # Wrap models
    models_dict = {
        'lgb': lgb_model,
        'xgb': xgb_model,
        'rf': rf_model,
        'lstm': lstm_model,
        'transformer': trans_model,
        'cnn': cnn_model,
        'meta': meta_model,
        'seq_length': SEQ_LENGTH
    }

    return models_dict, scores


class UltimateEnsemble:
    """Wrapper para ULTIMATE ENSEMBLE."""
    def __init__(self, models, scaler, feature_columns):
        self.models = models
        self.scaler = scaler
        self.feature_columns = feature_columns
        self.seq_length = models['seq_length']

    def predict(self, X):
        X = X[self.feature_columns]
        X_scaled = self.scaler.transform(X)

        # ML predictions
        lgb_pred = self.models['lgb'].predict_proba(X_scaled)[:, 1]
        xgb_pred = self.models['xgb'].predict_proba(X_scaled)[:, 1]
        rf_pred = self.models['rf'].predict_proba(X_scaled)[:, 1]

        # DL predictions (need sequences)
        if len(X) >= self.seq_length:
            X_seq = []
            for i in range(self.seq_length - 1, len(X)):
                X_seq.append(X_scaled[i-self.seq_length+1:i+1])
            X_seq = np.array(X_seq)

            lstm_pred = self.models['lstm'].predict(X_seq, verbose=0).flatten()
            trans_pred = self.models['transformer'].predict(X_seq, verbose=0).flatten()
            cnn_pred = self.models['cnn'].predict(X_seq, verbose=0).flatten()

            # Pad beginning
            lstm_pred_full = np.concatenate([np.full(self.seq_length-1, 0.5), lstm_pred])
            trans_pred_full = np.concatenate([np.full(self.seq_length-1, 0.5), trans_pred])
            cnn_pred_full = np.concatenate([np.full(self.seq_length-1, 0.5), cnn_pred])
        else:
            lstm_pred_full = np.full(len(X), 0.5)
            trans_pred_full = np.full(len(X), 0.5)
            cnn_pred_full = np.full(len(X), 0.5)

        # Meta-prediction
        meta_X = np.column_stack([lgb_pred, xgb_pred, rf_pred, lstm_pred_full, trans_pred_full, cnn_pred_full])
        final_pred = self.models['meta'].predict(meta_X, verbose=0).flatten()

        return (final_pred > 0.5).astype(int)

    def predict_proba(self, X):
        X = X[self.feature_columns]
        X_scaled = self.scaler.transform(X)

        lgb_pred = self.models['lgb'].predict_proba(X_scaled)[:, 1]
        xgb_pred = self.models['xgb'].predict_proba(X_scaled)[:, 1]
        rf_pred = self.models['rf'].predict_proba(X_scaled)[:, 1]

        if len(X) >= self.seq_length:
            X_seq = []
            for i in range(self.seq_length - 1, len(X)):
                X_seq.append(X_scaled[i-self.seq_length+1:i+1])
            X_seq = np.array(X_seq)

            lstm_pred = self.models['lstm'].predict(X_seq, verbose=0).flatten()
            trans_pred = self.models['transformer'].predict(X_seq, verbose=0).flatten()
            cnn_pred = self.models['cnn'].predict(X_seq, verbose=0).flatten()

            lstm_pred_full = np.concatenate([np.full(self.seq_length-1, 0.5), lstm_pred])
            trans_pred_full = np.concatenate([np.full(self.seq_length-1, 0.5), trans_pred])
            cnn_pred_full = np.concatenate([np.full(self.seq_length-1, 0.5), cnn_pred])
        else:
            lstm_pred_full = np.full(len(X), 0.5)
            trans_pred_full = np.full(len(X), 0.5)
            cnn_pred_full = np.full(len(X), 0.5)

        meta_X = np.column_stack([lgb_pred, xgb_pred, rf_pred, lstm_pred_full, trans_pred_full, cnn_pred_full])
        final_pred = self.models['meta'].predict(meta_X, verbose=0).flatten()

        return np.column_stack([1 - final_pred, final_pred])


def main():
    """Pipeline completo."""

    # 1. Download
    df = get_binance_klines(days=365)

    # 2. Features
    df = calculate_features(df)

    # 3. Labels
    df = create_labels(df)

    # 4. Prepare
    df = df.dropna()
    feature_columns = [c for c in df.columns if c not in [
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'future_return', 'label', 'vol_low', 'vol_high'
    ]]

    X = df[feature_columns]
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. Train
    models, scores = train_ultimate_ensemble(X_train, y_train, X_test, y_test, X_train_scaled, X_test_scaled)

    # 6. Wrap
    ensemble = UltimateEnsemble(models, scaler, feature_columns)

    # 7. Save
    model_path = storage_dir / "ultra_scalper_btcusdt_365d.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(ensemble, f)

    size_mb = model_path.stat().st_size / (1024 * 1024)

    print("=" * 80)
    print("✅ ULTIMATE ENSEMBLE COMPLETO!")
    print("=" * 80)
    print()
    print("🎯 ARCHITECTURE:")
    print("   LightGBM + XGBoost + Random Forest + LSTM + Transformer + CNN + Meta-NN")
    print()
    print("📊 ACCURACIES:")
    for name, score in scores.items():
        star = " ⭐" if name == "MetaNN" else ""
        print(f"   {name:15s}: {score:.2%}{star}")
    print()
    print(f"💾 MODELO: {model_path}")
    print(f"   Tamanho: {size_mb:.2f} MB")
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
