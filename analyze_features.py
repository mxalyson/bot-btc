"""
ANÁLISE DE FEATURES - Identifica features boas e ruins
Executa ANTES de treinar para saber o que manter/remover
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings('ignore')

print("=" * 80)
print("🔍 ANÁLISE DE FEATURES - Identificando features de qualidade")
print("=" * 80)
print()

# Check dependencies
try:
    import requests
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    import lightgbm as lgb
    print("✅ Dependências OK!")
except ImportError as e:
    print(f"❌ ERRO: {e}")
    sys.exit(1)

print()


def get_binance_klines(symbol='BTCUSDT', interval='15m', days=365):
    """Baixa dados da Binance."""
    print(f"📥 Baixando {days} dias de dados...")

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

            if len(klines) < 1000:
                break

        except Exception as e:
            print(f"   ❌ Erro: {e}")
            break

    if not all_data:
        print("   ❌ Nenhum dado baixado!")
        sys.exit(1)

    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])

    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.sort_values('timestamp').reset_index(drop=True)

    print(f"   ✅ {len(df)} candles de {df['timestamp'].min()} a {df['timestamp'].max()}")
    return df


def add_features(df):
    """Adiciona TODAS as features do V2."""
    print("🔧 Criando features...")

    # Price features
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['high_low_ratio'] = df['high'] / df['low']
    df['close_open_ratio'] = df['close'] / df['open']

    # Moving averages
    for period in [7, 14, 21, 50]:
        df[f'sma_{period}'] = df['close'].rolling(period).mean()
        df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

    # Volatility
    df['volatility_7'] = df['returns'].rolling(7).std()
    df['volatility_14'] = df['returns'].rolling(14).std()
    df['volatility_21'] = df['returns'].rolling(21).std()

    # ATR
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['atr_14'] = true_range.rolling(14).mean()

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    # Volume features
    df['volume_sma_20'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma_20']

    # Momentum
    df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
    df['momentum_10'] = df['close'] / df['close'].shift(10) - 1

    # Price position
    df['price_position'] = (df['close'] - df['low'].rolling(14).min()) / \
                           (df['high'].rolling(14).max() - df['low'].rolling(14).min())

    # Clean
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(method='ffill').fillna(0)

    print(f"   ✅ {len(df.columns)} features criadas")
    return df


def create_labels(df, threshold=0.002):
    """Cria labels simples baseado em threshold."""
    future_returns = df['close'].shift(-5) / df['close'] - 1
    labels = (future_returns > threshold).astype(int)
    return labels[:-5]


def analyze_features(df, labels):
    """Analisa importância de cada feature."""
    print()
    print("🔬 ANALISANDO IMPORTÂNCIA DAS FEATURES...")
    print()

    # Get feature columns
    feature_cols = [col for col in df.columns if col not in
                   ['timestamp', 'close_time', 'quote_volume', 'trades',
                    'taker_buy_base', 'taker_buy_quote', 'ignore']]

    X = df[feature_cols].iloc[:-5].values
    y = labels.values

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train LightGBM (rápido e bom para feature importance)
    print("   Treinando LightGBM para análise...")
    model = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
        verbose=-1
    )

    model.fit(X_train_scaled, y_train)

    # Get importance
    importances = model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances
    }).sort_values('importance', ascending=False)

    # Test accuracy
    train_acc = accuracy_score(y_train, model.predict(X_train_scaled))
    test_acc = accuracy_score(y_test, model.predict(X_test_scaled))

    print(f"   ✅ Train Accuracy: {train_acc*100:.2f}%")
    print(f"   ✅ Test Accuracy: {test_acc*100:.2f}%")
    print()

    # Show top features
    print("🏆 TOP 20 FEATURES (BOAS):")
    print("-" * 80)
    for idx, row in feature_importance.head(20).iterrows():
        print(f"   {row['feature']:30s} → {row['importance']:.4f}")

    print()
    print("💩 BOTTOM 20 FEATURES (RUINS - CANDIDATAS PARA REMOÇÃO):")
    print("-" * 80)
    for idx, row in feature_importance.tail(20).iterrows():
        print(f"   {row['feature']:30s} → {row['importance']:.4f}")

    print()

    # Features to remove (importance < 0.5% do total)
    total_importance = importances.sum()
    threshold_importance = total_importance * 0.005  # 0.5%

    bad_features = feature_importance[feature_importance['importance'] < threshold_importance]['feature'].tolist()

    print(f"📋 RECOMENDAÇÃO: Remover {len(bad_features)} features com importance < 0.5%:")
    print("-" * 80)
    for feat in bad_features:
        print(f"   ❌ {feat}")

    print()
    print(f"✅ MANTER: {len(feature_cols) - len(bad_features)} features de qualidade")

    # Save analysis
    output_file = "storage/feature_analysis.csv"
    feature_importance.to_csv(output_file, index=False)
    print()
    print(f"💾 Análise salva em: {output_file}")

    return feature_importance, bad_features


def test_feature_removal(df, labels, bad_features):
    """Testa performance SEM as features ruins."""
    print()
    print("🧪 TESTANDO REMOÇÃO DE FEATURES RUINS...")
    print()

    # All features
    all_feature_cols = [col for col in df.columns if col not in
                       ['timestamp', 'close_time', 'quote_volume', 'trades',
                        'taker_buy_base', 'taker_buy_quote', 'ignore']]

    # Good features only
    good_feature_cols = [col for col in all_feature_cols if col not in bad_features]

    X_all = df[all_feature_cols].iloc[:-5].values
    X_good = df[good_feature_cols].iloc[:-5].values
    y = labels.values

    # Split
    X_train_all, X_test_all, y_train, y_test = train_test_split(
        X_all, y, test_size=0.3, random_state=42, stratify=y
    )

    X_train_good, X_test_good, _, _ = train_test_split(
        X_good, y, test_size=0.3, random_state=42, stratify=y
    )

    # Scale
    scaler_all = StandardScaler()
    X_train_all = scaler_all.fit_transform(X_train_all)
    X_test_all = scaler_all.transform(X_test_all)

    scaler_good = StandardScaler()
    X_train_good = scaler_good.fit_transform(X_train_good)
    X_test_good = scaler_good.transform(X_test_good)

    # Test with ALL features
    print(f"   Testando com TODAS as {len(all_feature_cols)} features...")
    model_all = lgb.LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, verbose=-1)
    model_all.fit(X_train_all, y_train)
    acc_all = accuracy_score(y_test, model_all.predict(X_test_all))
    print(f"      Accuracy: {acc_all*100:.2f}%")

    # Test with GOOD features only
    print(f"   Testando com apenas {len(good_feature_cols)} features BOAS...")
    model_good = lgb.LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, verbose=-1)
    model_good.fit(X_train_good, y_train)
    acc_good = accuracy_score(y_test, model_good.predict(X_test_good))
    print(f"      Accuracy: {acc_good*100:.2f}%")

    print()
    diff = acc_good - acc_all
    if diff > 0:
        print(f"✅ MELHORIA: +{diff*100:.2f}% removendo features ruins!")
    elif diff < 0:
        print(f"⚠️  PIORA: {diff*100:.2f}% removendo features")
    else:
        print(f"➡️  SEM DIFERENÇA - features ruins não afetam")

    print()
    print("💡 CONCLUSÃO:")
    if diff >= 0:
        print(f"   → Use apenas as {len(good_feature_cols)} features BOAS no modelo final!")
    else:
        print(f"   → Mantenha todas as features (remoção piora performance)")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":

    # 1. Download data
    df = get_binance_klines(days=365)

    # 2. Add features
    df = add_features(df)

    # 3. Create labels
    labels = create_labels(df, threshold=0.002)

    # 4. Analyze features
    feature_importance, bad_features = analyze_features(df, labels)

    # 5. Test removal
    test_feature_removal(df, labels, bad_features)

    print()
    print("=" * 80)
    print("✅ ANÁLISE COMPLETA!")
    print("=" * 80)
    print()
    print("📋 PRÓXIMOS PASSOS:")
    print("   1. Revise storage/feature_analysis.csv")
    print("   2. Use apenas features BOAS no modelo final")
    print("   3. Adicione features AVANÇADAS de qualidade")
    print("   4. Execute hyperparameter tuning")
    print()
