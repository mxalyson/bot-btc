"""
Script para treinar o modelo ML do zero
Execute este script para criar o modelo ultra_scalper_btcusdt_365d.pkl
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("🤖 TREINAMENTO DO MODELO ML - BTC SCALPER")
print("=" * 80)
print()

# Check dependencies
print("📋 Verificando dependências...")
try:
    import requests
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
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

print("📥 Baixando dados históricos do BTC...")
print()

def get_bybit_klines(symbol='BTCUSDT', interval='15', days=365):
    """Baixa dados históricos da Bybit."""
    print(f"   Símbolo: {symbol}")
    print(f"   Timeframe: {interval}min")
    print(f"   Período: {days} dias")
    print()

    all_data = []
    end_time = int(datetime.now().timestamp())
    start_time = int((datetime.now() - timedelta(days=days)).timestamp())

    url = "https://api.bybit.com/v2/public/kline/list"

    current_time = start_time
    total_candles = 0

    while current_time < end_time:
        params = {
            'symbol': symbol,
            'interval': interval,
            'from': current_time,
            'limit': 200
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()

            if data['ret_code'] == 0 and data['result']:
                candles = data['result']
                all_data.extend(candles)
                total_candles += len(candles)

                # Update progress
                progress = ((current_time - start_time) / (end_time - start_time)) * 100
                print(f"   Progresso: {progress:.1f}% - {total_candles} candles baixados", end='\r')

                # Next batch
                last_time = int(candles[-1]['open_time'])
                current_time = last_time + 1

                if len(candles) < 200:
                    break
            else:
                print(f"\n   ⚠️  API retornou erro: {data.get('ret_msg')}")
                break

        except Exception as e:
            print(f"\n   ❌ Erro ao baixar dados: {e}")
            break

    print()
    print(f"✅ {total_candles} candles baixados!")
    print()

    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    df['timestamp'] = pd.to_datetime(df['open_time'], unit='s')
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)

    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].sort_values('timestamp').reset_index(drop=True)

    return df


def calculate_features(df):
    """Calcula features técnicas."""
    print("🔧 Calculando features técnicas...")

    # Returns
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

    # Volatility
    df['volatility'] = df['returns'].rolling(window=20).std()

    # ATR
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean()

    # Moving averages
    for period in [7, 14, 21, 50, 100, 200]:
        df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

    # Price momentum
    for period in [5, 10, 20]:
        df[f'momentum_{period}'] = df['close'] - df['close'].shift(period)
        df[f'roc_{period}'] = (df['close'] - df['close'].shift(period)) / df['close'].shift(period) * 100

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))

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

    # Volume indicators
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']

    # Regime detection
    df['vol_low'] = df['volatility'].rolling(window=500, min_periods=100).quantile(0.33)
    df['vol_high'] = df['volatility'].rolling(window=500, min_periods=100).quantile(0.67)

    df['vol_regime'] = 'medium'
    df.loc[df['volatility'] < df['vol_low'], 'vol_regime'] = 'low'
    df.loc[df['volatility'] > df['vol_high'], 'vol_regime'] = 'high'

    df['trend'] = 'neutral'
    df.loc[df['ema_50'] > df['ema_200'], 'trend'] = 'bull'
    df.loc[df['ema_50'] < df['ema_200'], 'trend'] = 'bear'

    print(f"✅ {len(df.columns)} features calculadas!")
    print()

    return df


def create_labels(df, future_periods=3):
    """Cria labels para classificação (compra/venda)."""
    print("🏷️  Criando labels...")

    # Future returns
    df['future_return'] = df['close'].shift(-future_periods) / df['close'] - 1

    # Label: 1 = Long (price goes up), 0 = Short (price goes down)
    df['label'] = (df['future_return'] > 0.002).astype(int)  # 0.2% threshold

    longs = (df['label'] == 1).sum()
    shorts = (df['label'] == 0).sum()

    print(f"✅ Labels criados:")
    print(f"   Longs: {longs} ({longs/len(df)*100:.1f}%)")
    print(f"   Shorts: {shorts} ({shorts/len(df)*100:.1f}%)")
    print()

    return df


def train_ensemble_model(X_train, y_train, X_test, y_test):
    """Treina ensemble de modelos."""
    print("🤖 Treinando modelos ML...")
    print()

    models = {}

    # LightGBM
    print("   1/3 - Treinando LightGBM...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=7,
        learning_rate=0.05,
        random_state=42,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train)
    lgb_score = lgb_model.score(X_test, y_test)
    models['lightgbm'] = lgb_model
    print(f"      ✅ LightGBM - Accuracy: {lgb_score:.2%}")

    # XGBoost
    print("   2/3 - Treinando XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=7,
        learning_rate=0.05,
        random_state=42,
        verbosity=0
    )
    xgb_model.fit(X_train, y_train)
    xgb_score = xgb_model.score(X_test, y_test)
    models['xgboost'] = xgb_model
    print(f"      ✅ XGBoost - Accuracy: {xgb_score:.2%}")

    # Random Forest
    print("   3/3 - Treinando Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_score = rf_model.score(X_test, y_test)
    models['random_forest'] = rf_model
    print(f"      ✅ Random Forest - Accuracy: {rf_score:.2%}")

    print()
    print(f"✅ Ensemble treinado - Média: {(lgb_score + xgb_score + rf_score) / 3:.2%}")
    print()

    return models


def create_ensemble_wrapper(models, scaler, feature_columns):
    """Cria wrapper para ensemble."""

    class EnsembleModel:
        def __init__(self, models, scaler, feature_columns):
            self.models = models
            self.scaler = scaler
            self.feature_columns = feature_columns

        def predict(self, X):
            # Ensure correct features
            X = X[self.feature_columns]
            X_scaled = self.scaler.transform(X)

            # Get predictions from all models
            predictions = []
            for model in self.models.values():
                pred = model.predict(X_scaled)
                predictions.append(pred)

            # Voting (majority wins)
            predictions = np.array(predictions)
            final_pred = np.round(predictions.mean(axis=0)).astype(int)

            return final_pred

        def predict_proba(self, X):
            # Ensure correct features
            X = X[self.feature_columns]
            X_scaled = self.scaler.transform(X)

            # Get probabilities from all models
            probabilities = []
            for model in self.models.values():
                proba = model.predict_proba(X_scaled)
                probabilities.append(proba)

            # Average probabilities
            avg_proba = np.mean(probabilities, axis=0)

            return avg_proba

    return EnsembleModel(models, scaler, feature_columns)


def main():
    """Main training pipeline."""

    # 1. Download data
    df = get_bybit_klines(days=365)

    if len(df) < 1000:
        print("❌ Dados insuficientes! Precisa de pelo menos 1000 candles.")
        sys.exit(1)

    # 2. Calculate features
    df = calculate_features(df)

    # 3. Create labels
    df = create_labels(df)

    # 4. Prepare data
    print("📊 Preparando dados para treinamento...")

    # Remove NaN
    df = df.dropna()

    # Select features
    feature_columns = [col for col in df.columns if col not in [
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'future_return', 'label', 'vol_regime', 'trend', 'vol_low', 'vol_high'
    ]]

    X = df[feature_columns]
    y = df['label']

    print(f"   Samples: {len(X):,}")
    print(f"   Features: {len(feature_columns)}")
    print()

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )

    print(f"   Train: {len(X_train):,} samples")
    print(f"   Test: {len(X_test):,} samples")
    print()

    # Scale
    print("📐 Normalizando features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("✅ Features normalizadas!")
    print()

    # 5. Train models
    models = train_ensemble_model(X_train_scaled, y_train, X_test_scaled, y_test)

    # 6. Create ensemble
    print("🎯 Criando ensemble final...")
    ensemble = create_ensemble_wrapper(models, scaler, feature_columns)

    # Test ensemble
    test_preds = ensemble.predict(X_test)
    test_accuracy = (test_preds == y_test).mean()
    print(f"✅ Ensemble Accuracy: {test_accuracy:.2%}")
    print()

    # 7. Save model
    model_path = storage_dir / "ultra_scalper_btcusdt_365d.pkl"

    print(f"💾 Salvando modelo em: {model_path}")

    with open(model_path, 'wb') as f:
        pickle.dump(ensemble, f)

    # Check size
    size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"✅ Modelo salvo! Tamanho: {size_mb:.2f} MB")
    print()

    # Final summary
    print("=" * 80)
    print("✅ TREINAMENTO COMPLETO!")
    print("=" * 80)
    print()
    print(f"📊 Estatísticas:")
    print(f"   Modelo: Ensemble (LightGBM + XGBoost + Random Forest)")
    print(f"   Dados: {len(df):,} candles ({df['timestamp'].min()} a {df['timestamp'].max()})")
    print(f"   Features: {len(feature_columns)}")
    print(f"   Accuracy: {test_accuracy:.2%}")
    print(f"   Arquivo: {model_path}")
    print(f"   Tamanho: {size_mb:.2f} MB")
    print()
    print("🚀 Próximo passo: Execute python setup.py para validar tudo!")
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
