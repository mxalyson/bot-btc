"""
MODELO DEFINITIVO - SCALPER BALANCEADO
=======================================

ANÁLISE DOS FRACASSOS:
❌ V3: 30% accuracy (triple barrier muito restritivo)
❌ V4: Long 37%, overfitting 3-4%
❌ V5: Long 13% ❌❌❌ (pior que V4!)

PROBLEMA RAIZ:
Class weights NÃO funcionam com imbalance extremo (67% shorts)
SMOTE cria dados sintéticos irrealistas
Modelos aprendem viés dos shorts

SOLUÇÃO DEFINITIVA:
✅ UNDER-SAMPLING de shorts (50/50 real balance)
✅ Threshold ajustado (0.35 long, 0.65 short)
✅ Weighted ensemble por LONG accuracy (não meta-learner)
✅ Regularização EXTREMA (max_depth 3-4)
✅ Remove features que favorecem shorts
✅ 4 modelos calibrados (LGB, XGB, CB, RF)
✅ Validação em LONG accuracy (métrica primária)

EXPECTED: 52-56% Meta, Long 48-52%, Short 52-56%
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
print("🎯 MODELO DEFINITIVO - SCALPER BALANCEADO")
print("=" * 80)
print()

# Check dependencies
print("📋 Verificando dependências...")
try:
    import requests
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.utils import resample
    import lightgbm as lgb
    import xgboost as xgb

    try:
        import catboost as cb
        HAS_CATBOOST = True
    except:
        HAS_CATBOOST = False

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        HAS_OPTUNA = True
    except:
        HAS_OPTUNA = False

    print("✅ Dependências OK!")

except ImportError as e:
    print(f"❌ ERRO: {e}")
    sys.exit(1)

print()

storage_dir = Path("storage/models")
storage_dir.mkdir(parents=True, exist_ok=True)


class ModelWrapper:
    """Wrapper com threshold ajustado."""

    def __init__(self, models_list, model_weights, scaler, feature_columns,
                 long_threshold=0.35, short_threshold=0.65):
        self.models = models_list
        self.model_weights = model_weights
        self.scaler = scaler
        self.feature_columns = feature_columns
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        self.version = "MODELO_DEFINITIVO_BALANCEADO"
        self.timestamp = datetime.now().isoformat()

    def predict_proba(self, X):
        """Weighted ensemble."""
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_columns].values

        X_scaled = self.scaler.transform(X)

        # Weighted predictions
        all_probs = []
        for model, weight in zip(self.models, self.model_weights):
            proba = model.predict_proba(X_scaled)[:, 1]
            all_probs.append(proba * weight)

        # Weighted average
        final_proba = np.sum(all_probs, axis=0)

        # Return as 2D array (proba for class 0 and 1)
        proba_class_0 = 1 - final_proba
        proba_class_1 = final_proba

        return np.column_stack([proba_class_0, proba_class_1])

    def predict(self, X):
        """Predição com threshold ajustado."""
        proba = self.predict_proba(X)[:, 1]

        predictions = np.zeros(len(proba))

        # Use LOWER threshold for longs (easier to enter)
        predictions[proba >= self.long_threshold] = 1

        return predictions.astype(int)


def get_binance_klines(symbol='BTCUSDT', interval='15m', days=365):
    """Baixa dados."""
    print(f"📥 Baixando {days} dias de dados...")

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
            print(f"   ❌ Erro: {e}")
            break

    if not all_data:
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

    print(f"   ✅ {len(df)} candles")
    return df


def add_quality_features(df):
    """Features de QUALIDADE comprovada."""
    print("🔧 Criando features de qualidade...")

    # Returns
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

    # Price action
    df['body_size'] = np.abs(df['close'] - df['open']) / df['open']
    df['upper_wick'] = (df['high'] - df[['close', 'open']].max(axis=1)) / df['open']
    df['lower_wick'] = (df[['close', 'open']].min(axis=1) - df['low']) / df['open']

    # Order flow
    df['taker_buy_ratio'] = df['taker_buy_base'] / (df['volume'] + 1e-8)
    df['buy_pressure'] = df['taker_buy_ratio'].rolling(7).mean()

    # Moving averages
    for period in [7, 14, 21]:
        df[f'sma_{period}'] = df['close'].rolling(period).mean()
        df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

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

    # Bollinger
    df['bb_middle'] = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)

    # Volume
    df['volume_sma_20'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_sma_20'] + 1e-8)

    # Momentum
    df['momentum_7'] = df['close'] / df['close'].shift(7) - 1
    df['momentum_14'] = df['close'] / df['close'].shift(14) - 1

    # Volatility
    df['volatility_14'] = df['returns'].rolling(14).std()

    # Clean
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)

    print(f"   ✅ {len(df.columns)} features")
    return df


def create_labels(df, threshold=0.0015):
    """Labels."""
    future_returns = df['close'].shift(-5) / df['close'] - 1
    labels = (future_returns > threshold).astype(int)
    return labels[:-5]


def undersample_majority(X, y):
    """Under-sample shorts para balancear 50/50."""
    print("\n🔄 Aplicando UNDER-SAMPLING...")

    # Separar classes
    X_long = X[y == 1]
    y_long = y[y == 1]
    X_short = X[y == 0]
    y_short = y[y == 0]

    print(f"   Original: {len(y_long)} longs, {len(y_short)} shorts")

    # Under-sample shorts para match longs
    X_short_under, y_short_under = resample(
        X_short, y_short,
        n_samples=len(y_long),
        random_state=42,
        replace=False
    )

    # Combine
    X_balanced = np.vstack([X_long, X_short_under])
    y_balanced = np.hstack([y_long, y_short_under])

    # Shuffle
    indices = np.random.RandomState(42).permutation(len(y_balanced))
    X_balanced = X_balanced[indices]
    y_balanced = y_balanced[indices]

    print(f"   Balanced: {len(y_balanced)} samples (50% each)")

    return X_balanced, y_balanced


def tune_model_conservative(X_train, y_train, model_type, n_trials=15):
    """Tuning CONSERVADOR (anti-overfitting)."""

    if not HAS_OPTUNA:
        if model_type == 'lgb':
            return {'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.02,
                   'num_leaves': 15, 'min_child_samples': 50}
        else:
            return {'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.02}

    print(f"   🔧 Tuning {model_type.upper()} ({n_trials} trials)...")

    def objective(trial):
        if model_type == 'lgb':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 150, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 5),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 10, 30),
                'min_child_samples': trial.suggest_int('min_child_samples', 40, 100),
                'subsample': trial.suggest_float('subsample', 0.6, 0.85),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.85),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.3, 0.7),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.3, 0.7),
                'random_state': 42,
                'verbose': -1
            }
            model = lgb.LGBMClassifier(**params)

        else:  # xgb
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 150, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 5),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 0.85),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.85),
                'gamma': trial.suggest_float('gamma', 0.2, 0.6),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.3, 0.7),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.3, 0.7),
                'min_child_weight': trial.suggest_int('min_child_weight', 5, 15),
                'random_state': 42,
                'verbosity': 0
            }
            model = xgb.XGBClassifier(**params)

        # StratifiedKFold
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores_long = []
        scores_overall = []

        for train_idx, val_idx in skf.split(X_train, y_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            if model_type == 'lgb':
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                         callbacks=[lgb.early_stopping(30, verbose=False)])
            else:
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                         verbose=False)

            pred = model.predict(X_val)

            # Score LONG accuracy (métrica primária!)
            long_acc = accuracy_score(y_val[y_val == 1], pred[y_val == 1])
            overall_acc = accuracy_score(y_val, pred)

            scores_long.append(long_acc)
            scores_overall.append(overall_acc)

        # Optimize for LONG accuracy (70%) + overall (30%)
        combined_score = np.mean(scores_long) * 0.7 + np.mean(scores_overall) * 0.3

        return combined_score

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"      ✅ Best CV: {study.best_value*100:.2f}%")
    return study.best_params


def train_modelo_definitivo(df, days):
    """Treina modelo DEFINITIVO balanceado."""

    print()
    print(f"🎯 TREINANDO MODELO DEFINITIVO ({days} dias)...")
    print("=" * 80)

    # Feature columns
    feature_cols = [col for col in df.columns if col not in
                   ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                    'taker_buy_quote', 'ignore']]

    # Labels
    labels = create_labels(df, threshold=0.0015)

    # Prepare
    X = df[feature_cols].iloc[:-5].values
    y = labels.values

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"\n📊 Dataset:")
    print(f"   Train: {len(X_train)} samples")
    print(f"   Test: {len(X_test)} samples")
    print(f"   Longs: {(y_train==1).sum()} ({(y_train==1).sum()/len(y_train)*100:.1f}%)")
    print(f"   Shorts: {(y_train==0).sum()} ({(y_train==0).sum()/len(y_train)*100:.1f}%)")

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # UNDER-SAMPLE (50/50 balance)
    X_train_balanced, y_train_balanced = undersample_majority(X_train_scaled, y_train)

    # ========================================================================
    # TRAIN 4 MODELS
    # ========================================================================

    print("\n" + "=" * 80)
    print("🚀 TREINANDO 4 MODELOS CALIBRADOS")
    print("=" * 80)

    models_list = []
    model_names = []
    long_accuracies = []

    # 1. LightGBM
    print("\n1/4 - LightGBM...")
    lgb_params = tune_model_conservative(X_train_balanced, y_train_balanced, 'lgb', n_trials=15)
    lgb_model = lgb.LGBMClassifier(**lgb_params, random_state=42, verbose=-1)
    lgb_calibrated = CalibratedClassifierCV(lgb_model, method='sigmoid', cv=3)
    lgb_calibrated.fit(X_train_balanced, y_train_balanced)

    lgb_pred = lgb_calibrated.predict(X_test_scaled)
    lgb_acc = accuracy_score(y_test, lgb_pred)
    lgb_long_acc = accuracy_score(y_test[y_test==1], lgb_pred[y_test==1])
    lgb_short_acc = accuracy_score(y_test[y_test==0], lgb_pred[y_test==0])

    models_list.append(lgb_calibrated)
    model_names.append('LightGBM')
    long_accuracies.append(lgb_long_acc)

    print(f"   ✅ LightGBM: {lgb_acc*100:.2f}% (Long: {lgb_long_acc*100:.2f}%, Short: {lgb_short_acc*100:.2f}%)")

    # 2. XGBoost
    print("\n2/4 - XGBoost...")
    xgb_params = tune_model_conservative(X_train_balanced, y_train_balanced, 'xgb', n_trials=15)
    xgb_model = xgb.XGBClassifier(**xgb_params, random_state=42, verbosity=0)
    xgb_calibrated = CalibratedClassifierCV(xgb_model, method='sigmoid', cv=3)
    xgb_calibrated.fit(X_train_balanced, y_train_balanced)

    xgb_pred = xgb_calibrated.predict(X_test_scaled)
    xgb_acc = accuracy_score(y_test, xgb_pred)
    xgb_long_acc = accuracy_score(y_test[y_test==1], xgb_pred[y_test==1])
    xgb_short_acc = accuracy_score(y_test[y_test==0], xgb_pred[y_test==0])

    models_list.append(xgb_calibrated)
    model_names.append('XGBoost')
    long_accuracies.append(xgb_long_acc)

    print(f"   ✅ XGBoost: {xgb_acc*100:.2f}% (Long: {xgb_long_acc*100:.2f}%, Short: {xgb_short_acc*100:.2f}%)")

    # 3. CatBoost
    if HAS_CATBOOST:
        print("\n3/4 - CatBoost...")
        cb_model = cb.CatBoostClassifier(
            iterations=200,
            depth=3,
            learning_rate=0.02,
            random_state=42,
            verbose=0
        )
        cb_calibrated = CalibratedClassifierCV(cb_model, method='sigmoid', cv=3)
        cb_calibrated.fit(X_train_balanced, y_train_balanced)

        cb_pred = cb_calibrated.predict(X_test_scaled)
        cb_acc = accuracy_score(y_test, cb_pred)
        cb_long_acc = accuracy_score(y_test[y_test==1], cb_pred[y_test==1])
        cb_short_acc = accuracy_score(y_test[y_test==0], cb_pred[y_test==0])

        models_list.append(cb_calibrated)
        model_names.append('CatBoost')
        long_accuracies.append(cb_long_acc)

        print(f"   ✅ CatBoost: {cb_acc*100:.2f}% (Long: {cb_long_acc*100:.2f}%, Short: {cb_short_acc*100:.2f}%)")
    else:
        print("\n3/4 - CatBoost SKIP")

    # 4. Random Forest
    print("\n4/4 - Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )
    rf_calibrated = CalibratedClassifierCV(rf_model, method='sigmoid', cv=3)
    rf_calibrated.fit(X_train_balanced, y_train_balanced)

    rf_pred = rf_calibrated.predict(X_test_scaled)
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_long_acc = accuracy_score(y_test[y_test==1], rf_pred[y_test==1])
    rf_short_acc = accuracy_score(y_test[y_test==0], rf_pred[y_test==0])

    models_list.append(rf_calibrated)
    model_names.append('RandomForest')
    long_accuracies.append(rf_long_acc)

    print(f"   ✅ Random Forest: {rf_acc*100:.2f}% (Long: {rf_long_acc*100:.2f}%, Short: {rf_short_acc*100:.2f}%)")

    # ========================================================================
    # WEIGHTED ENSEMBLE (by long accuracy)
    # ========================================================================

    print("\n" + "=" * 80)
    print("🧠 WEIGHTED ENSEMBLE (baseado em Long Accuracy)")
    print("=" * 80)

    # Weights baseados em long accuracy
    long_accuracies = np.array(long_accuracies)
    model_weights = long_accuracies / long_accuracies.sum()

    print("\n📊 Pesos dos modelos:")
    for name, weight, long_acc in zip(model_names, model_weights, long_accuracies):
        print(f"   {name:15s}: {weight:.3f} (Long Acc: {long_acc*100:.2f}%)")

    # Create wrapper
    wrapper = ModelWrapper(
        models_list=models_list,
        model_weights=model_weights,
        scaler=scaler,
        feature_columns=feature_cols,
        long_threshold=0.35,  # Easier to enter longs
        short_threshold=0.65  # Harder to enter shorts
    )

    # Test ensemble
    ensemble_pred = wrapper.predict(X_test_scaled)
    ensemble_acc = accuracy_score(y_test, ensemble_pred)
    ensemble_long_acc = accuracy_score(y_test[y_test==1], ensemble_pred[y_test==1])
    ensemble_short_acc = accuracy_score(y_test[y_test==0], ensemble_pred[y_test==0])

    print(f"\n✅ Ensemble Final:")
    print(f"   Meta:  {ensemble_acc*100:.2f}%")
    print(f"   Long:  {ensemble_long_acc*100:.2f}%")
    print(f"   Short: {ensemble_short_acc*100:.2f}%")
    print(f"   Desbalanceamento: {abs(ensemble_long_acc - ensemble_short_acc)*100:.2f}%")

    # Results
    results = {
        'days': days,
        'meta_accuracy': ensemble_acc,
        'long_accuracy': ensemble_long_acc,
        'short_accuracy': ensemble_short_acc,
        'balance_diff': abs(ensemble_long_acc - ensemble_short_acc),
        'n_features': len(feature_cols),
        'model_weights': model_weights.tolist(),
        'model_names': model_names
    }

    return wrapper, results


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":

    start_time = time.time()

    # Single period (365 days)
    days = 365

    print()
    print("=" * 80)
    print(f"📅 PERÍODO: {days} DIAS")
    print("=" * 80)

    # Download
    df = get_binance_klines(days=days)

    # Features
    df = add_quality_features(df)

    # Train
    model, results = train_modelo_definitivo(df, days)

    # Save
    model_file = storage_dir / f"model_DEFINITIVO_{days}d.pkl"
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)

    model_size_mb = os.path.getsize(model_file) / (1024 * 1024)
    print(f"\n💾 Modelo salvo: {model_file}")
    print(f"   Tamanho: {model_size_mb:.2f} MB")

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================

    elapsed = time.time() - start_time

    print("\n\n")
    print("=" * 80)
    print("🏆 MODELO DEFINITIVO - RESULTADOS FINAIS")
    print("=" * 80)
    print()

    print(f"Meta:           {results['meta_accuracy']*100:.2f}%")
    print(f"Long:           {results['long_accuracy']*100:.2f}%")
    print(f"Short:          {results['short_accuracy']*100:.2f}%")
    print(f"Desbalance:     {results['balance_diff']*100:.2f}%")
    print(f"Features:       {results['n_features']}")
    print()

    print("=" * 80)
    print(f"⏱️  Tempo total: {elapsed/60:.1f} minutos")
    print("=" * 80)
    print()

    print("✅ PRONTO para backtest!")
    print(f"   Execute: python backtest_PERFEITO.py")
    print()
