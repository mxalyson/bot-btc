"""
Script MELHORADO para treinar modelo ML com ENSEMBLE COMPLETO
- API V5 Bybit + Fallback Binance
- Stacking/Blending: LightGBM + XGBoost + Random Forest + Meta-Learner
- 40+ features técnicas
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

print("=" * 80)
print("🤖 TREINAMENTO MODELO ML - BTC SCALPER V2 (ENSEMBLE STACKING)")
print("=" * 80)
print()

# Check dependencies
print("📋 Verificando dependências...")
try:
    import requests
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, classification_report
    import lightgbm as lgb
    import xgboost as xgb
    print("✅ Todas as dependências instaladas!")
except ImportError as e:
    print(f"❌ ERRO: Dependência faltando: {e}")
    print()
    print("Execute: pip install -r requirements.txt")
    sys.exit(1)

print()

# Create storage directory
storage_dir = Path("storage/models")
storage_dir.mkdir(parents=True, exist_ok=True)


def get_binance_klines(symbol='BTCUSDT', interval='15m', days=365):
    """
    Baixa dados da Binance (fallback confiável).
    Binance API é mais estável que Bybit.
    """
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
            'limit': 1000  # Binance permite 1000
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            candles = response.json()

            if candles:
                all_data.extend(candles)
                total_candles += len(candles)

                # Progress
                progress = ((current_time - start_time) / (end_time - start_time)) * 100
                print(f"   Progresso: {progress:.1f}% - {total_candles} candles", end='\r')

                # Next batch
                last_time = int(candles[-1][0])
                current_time = last_time + 1

                if len(candles) < 1000:
                    break

                # Rate limit
                time.sleep(0.1)
            else:
                break

        except Exception as e:
            print(f"\n   ⚠️  Erro: {e}")
            break

    print()
    print(f"✅ {total_candles} candles baixados!")
    print()

    if total_candles == 0:
        raise Exception("Nenhum dado foi baixado!")

    # Convert to DataFrame
    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)

    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].sort_values('timestamp').reset_index(drop=True)

    return df


def calculate_features(df):
    """Calcula 40+ features técnicas."""
    print("🔧 Calculando features técnicas...")

    # Returns
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

    # Volatility
    df['volatility'] = df['returns'].rolling(window=20).std()
    df['volatility_30'] = df['returns'].rolling(window=30).std()

    # ATR
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(window=14).mean()
    df['atr_20'] = tr.rolling(window=20).mean()

    # Moving averages
    for period in [7, 14, 21, 50, 100, 200]:
        df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

    # Price distance from MAs
    df['price_vs_sma50'] = (df['close'] - df['sma_50']) / df['sma_50']
    df['price_vs_sma200'] = (df['close'] - df['sma_200']) / df['sma_200']

    # Price momentum
    for period in [5, 10, 20, 30]:
        df[f'momentum_{period}'] = df['close'] - df['close'].shift(period)
        df[f'roc_{period}'] = (df['close'] - df['close'].shift(period)) / df['close'].shift(period) * 100

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # Stochastic RSI
    rsi = df['rsi_14']
    stoch_rsi = (rsi - rsi.rolling(14).min()) / (rsi.rolling(14).max() - rsi.rolling(14).min())
    df['stoch_rsi'] = stoch_rsi * 100

    # MACD
    ema_fast = df['close'].ewm(span=12, adjust=False).mean()
    ema_slow = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_fast - ema_slow
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # Bollinger Bands
    bb_middle = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = bb_middle + (bb_std * 2)
    df['bb_middle'] = bb_middle
    df['bb_lower'] = bb_middle - (bb_std * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    # Volume indicators
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    df['volume_roc'] = df['volume'].pct_change(periods=10)

    # Price channels
    df['high_20'] = df['high'].rolling(window=20).max()
    df['low_20'] = df['low'].rolling(window=20).min()
    df['channel_position'] = (df['close'] - df['low_20']) / (df['high_20'] - df['low_20'])

    # Regime detection (NO LOOK-AHEAD BIAS!)
    df['vol_low'] = df['volatility'].rolling(window=500, min_periods=100).quantile(0.33)
    df['vol_high'] = df['volatility'].rolling(window=500, min_periods=100).quantile(0.67)

    df['vol_regime'] = 'medium'
    df.loc[df['volatility'] < df['vol_low'], 'vol_regime'] = 'low'
    df.loc[df['volatility'] > df['vol_high'], 'vol_regime'] = 'high'

    df['trend'] = 'neutral'
    df.loc[df['ema_50'] > df['ema_200'], 'trend'] = 'bull'
    df.loc[df['ema_50'] < df['ema_200'], 'trend'] = 'bear'

    num_features = len([col for col in df.columns if col not in [
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'vol_regime', 'trend', 'vol_low', 'vol_high'
    ]])

    print(f"✅ {num_features} features calculadas!")
    print()

    return df


def create_labels(df, future_periods=3, threshold=0.003):
    """
    Cria labels para classificação.
    threshold=0.3% - precisa mover 0.3% para ser considerado sinal
    """
    print("🏷️  Criando labels...")

    # Future returns
    df['future_return'] = df['close'].shift(-future_periods) / df['close'] - 1

    # Label: 1 = Long (bullish), 0 = Short (bearish)
    df['label'] = (df['future_return'] > threshold).astype(int)

    longs = (df['label'] == 1).sum()
    shorts = (df['label'] == 0).sum()

    print(f"✅ Labels criados (threshold: {threshold*100:.1f}%):")
    print(f"   Longs: {longs} ({longs/len(df)*100:.1f}%)")
    print(f"   Shorts: {shorts} ({shorts/len(df)*100:.1f}%)")
    print()

    return df


def train_stacking_ensemble(X_train, y_train, X_test, y_test):
    """
    Treina ENSEMBLE COMPLETO com STACKING.

    Architecture:
    Level 0 (Base Models):
    - LightGBM
    - XGBoost
    - Random Forest

    Level 1 (Meta-Learner):
    - Logistic Regression (aprende a combinar as previsões)
    """
    print("🤖 Treinando ENSEMBLE com STACKING...")
    print()

    # Level 0: Base models
    print("📊 Level 0 - Base Models:")
    print()

    base_models = []

    # 1. LightGBM
    print("   1/3 - LightGBM...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=150,
        max_depth=8,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
        n_jobs=-1
    )
    lgb_model.fit(X_train, y_train)
    lgb_score = lgb_model.score(X_test, y_test)
    base_models.append(('lightgbm', lgb_model))
    print(f"      ✅ LightGBM - Accuracy: {lgb_score:.2%}")

    # 2. XGBoost
    print("   2/3 - XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    xgb_score = xgb_model.score(X_test, y_test)
    base_models.append(('xgboost', xgb_model))
    print(f"      ✅ XGBoost - Accuracy: {xgb_score:.2%}")

    # 3. Random Forest
    print("   3/3 - Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_score = rf_model.score(X_test, y_test)
    base_models.append(('random_forest', rf_model))
    print(f"      ✅ Random Forest - Accuracy: {rf_score:.2%}")

    print()
    avg_base = (lgb_score + xgb_score + rf_score) / 3
    print(f"📊 Base Models Média: {avg_base:.2%}")
    print()

    # Level 1: Meta-learner (Stacking)
    print("🎯 Level 1 - Meta-Learner (Stacking):")
    print()

    meta_learner = LogisticRegression(
        max_iter=1000,
        random_state=42,
        n_jobs=-1
    )

    print("   Treinando Stacking Ensemble...")
    stacking_model = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_learner,
        cv=5,
        n_jobs=-1
    )

    stacking_model.fit(X_train, y_train)
    stacking_score = stacking_model.score(X_test, y_test)

    print(f"   ✅ Stacking Ensemble - Accuracy: {stacking_score:.2%}")
    print()

    # Improvement
    improvement = ((stacking_score - avg_base) / avg_base) * 100
    print(f"🚀 Melhoria do Stacking vs Média: {improvement:+.1f}%")
    print()

    return stacking_model, {
        'lightgbm': lgb_score,
        'xgboost': xgb_score,
        'random_forest': rf_score,
        'stacking': stacking_score
    }


def create_model_wrapper(model, scaler, feature_columns):
    """Cria wrapper compatível com o bot."""

    class ModelWrapper:
        def __init__(self, model, scaler, feature_columns):
            self.model = model
            self.scaler = scaler
            self.feature_columns = feature_columns

        def predict(self, X):
            X = X[self.feature_columns]
            X_scaled = self.scaler.transform(X)
            return self.model.predict(X_scaled)

        def predict_proba(self, X):
            X = X[self.feature_columns]
            X_scaled = self.scaler.transform(X)
            return self.model.predict_proba(X_scaled)

    return ModelWrapper(model, scaler, feature_columns)


def main():
    """Pipeline completo de treinamento."""

    # 1. Download data
    print("=" * 80)
    print("ETAPA 1: DOWNLOAD DE DADOS")
    print("=" * 80)
    print()

    try:
        df = get_binance_klines(days=365)
    except Exception as e:
        print(f"❌ Erro ao baixar dados: {e}")
        sys.exit(1)

    if len(df) < 1000:
        print("❌ Dados insuficientes! Precisa de pelo menos 1000 candles.")
        sys.exit(1)

    print(f"📊 Dataset: {len(df):,} candles")
    print(f"   Período: {df['timestamp'].min()} a {df['timestamp'].max()}")
    print()

    # 2. Calculate features
    print("=" * 80)
    print("ETAPA 2: FEATURE ENGINEERING")
    print("=" * 80)
    print()

    df = calculate_features(df)

    # 3. Create labels
    print("=" * 80)
    print("ETAPA 3: LABEL GENERATION")
    print("=" * 80)
    print()

    df = create_labels(df, future_periods=3, threshold=0.003)

    # 4. Prepare data
    print("=" * 80)
    print("ETAPA 4: PREPARAÇÃO DOS DADOS")
    print("=" * 80)
    print()

    # Remove NaN
    df = df.dropna()

    # Select features
    feature_columns = [col for col in df.columns if col not in [
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'future_return', 'label', 'vol_regime', 'trend', 'vol_low', 'vol_high'
    ]]

    X = df[feature_columns]
    y = df['label']

    print(f"📊 Dataset Final:")
    print(f"   Samples: {len(X):,}")
    print(f"   Features: {len(feature_columns)}")
    print()

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )

    print(f"   Train: {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   Test: {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")
    print()

    # Scale
    print("📐 Normalizando features (StandardScaler)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("✅ Features normalizadas!")
    print()

    # 5. Train model
    print("=" * 80)
    print("ETAPA 5: TREINAMENTO DO ENSEMBLE")
    print("=" * 80)
    print()

    model, scores = train_stacking_ensemble(X_train_scaled, y_train, X_test_scaled, y_test)

    # 6. Detailed evaluation
    print("=" * 80)
    print("ETAPA 6: AVALIAÇÃO DETALHADA")
    print("=" * 80)
    print()

    y_pred = model.predict(X_test_scaled)

    print("📊 Classification Report:")
    print()
    print(classification_report(y_test, y_pred, target_names=['Short', 'Long'], digits=3))
    print()

    # 7. Create wrapper
    print("=" * 80)
    print("ETAPA 7: CRIANDO WRAPPER")
    print("=" * 80)
    print()

    wrapper = create_model_wrapper(model, scaler, feature_columns)
    print("✅ Wrapper criado!")
    print()

    # 8. Save model
    print("=" * 80)
    print("ETAPA 8: SALVANDO MODELO")
    print("=" * 80)
    print()

    model_path = storage_dir / "ultra_scalper_btcusdt_365d.pkl"

    print(f"💾 Salvando em: {model_path}")

    with open(model_path, 'wb') as f:
        pickle.dump(wrapper, f)

    size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"✅ Modelo salvo! Tamanho: {size_mb:.2f} MB")
    print()

    # Final summary
    print("=" * 80)
    print("✅ TREINAMENTO COMPLETO!")
    print("=" * 80)
    print()
    print(f"🎯 ENSEMBLE ARCHITECTURE:")
    print(f"   Base Models: LightGBM + XGBoost + Random Forest")
    print(f"   Meta-Learner: Logistic Regression (Stacking)")
    print()
    print(f"📊 ACCURACIES:")
    print(f"   LightGBM:     {scores['lightgbm']:.2%}")
    print(f"   XGBoost:      {scores['xgboost']:.2%}")
    print(f"   Random Forest: {scores['random_forest']:.2%}")
    print(f"   ---")
    print(f"   STACKING:     {scores['stacking']:.2%} ⭐")
    print()
    print(f"📈 DATASET:")
    print(f"   Candles: {len(df):,}")
    print(f"   Features: {len(feature_columns)}")
    print(f"   Período: {df['timestamp'].min().date()} a {df['timestamp'].max().date()}")
    print()
    print(f"💾 MODELO:")
    print(f"   Arquivo: {model_path}")
    print(f"   Tamanho: {size_mb:.2f} MB")
    print()
    print("🚀 Próximo passo: python setup.py")
    print()
    print("=" * 80)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print()
        print("❌ Treinamento cancelado pelo usuário")
        sys.exit(1)
    except Exception as e:
        print()
        print(f"❌ ERRO: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
