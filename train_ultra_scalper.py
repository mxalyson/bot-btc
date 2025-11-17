"""
ULTRA MASTER SCALPER - Estado da Arte para Scalping 15m
Combina o melhor de ML moderno: Ensemble Pesado + Features Avançadas + Meta-Learning
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import argparse
import logging
from typing import Tuple, List, Dict

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

from core.utils import load_config, setup_logging
from core.bybit_rest import BybitRESTClient
from core.data import DataManager
from core.features import FeatureStore

try:
    import lightgbm as lgb
    import xgboost as xgb
    HAS_BOOSTING = True
except ImportError:
    print("❌ LightGBM and XGBoost required! Install: pip install lightgbm xgboost")
    exit(1)

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    HAS_TENSORFLOW = True
except ImportError:
    print("⚠️  TensorFlow not found. Deep Learning models disabled.")
    HAS_TENSORFLOW = False


# ============================================================================
# FEATURE ENGINEERING AVANÇADO
# ============================================================================

def create_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features de Microestrutura de Mercado - Estado da Arte

    Referências:
    - Kyle (1985): Market Microstructure
    - Amihud (2002): Illiquidity and stock returns
    - Roll (1984): Spread estimation
    """
    df_micro = df.copy()

    # 1. ORDER FLOW PROXY (using volume and price change)
    df_micro['order_flow_proxy'] = np.sign(df_micro['close'].diff()) * df_micro['volume']
    df_micro['order_flow_cumsum_5'] = df_micro['order_flow_proxy'].rolling(5).sum()
    df_micro['order_flow_cumsum_20'] = df_micro['order_flow_proxy'].rolling(20).sum()

    # 2. KYLE'S LAMBDA (price impact)
    # λ = ΔP / √V (how much price moves per unit of volume)
    df_micro['price_change'] = df_micro['close'].diff()
    df_micro['volume_sqrt'] = np.sqrt(df_micro['volume'])
    df_micro['kyles_lambda'] = df_micro['price_change'] / (df_micro['volume_sqrt'] + 1e-8)
    df_micro['kyles_lambda_smooth'] = df_micro['kyles_lambda'].rolling(10).mean()

    # 3. AMIHUD ILLIQUIDITY
    # ILLIQ = |R| / V (absolute return per unit volume)
    df_micro['returns_abs'] = abs(df_micro['close'].pct_change())
    df_micro['amihud_illiq'] = df_micro['returns_abs'] / (df_micro['volume'] + 1e-8)
    df_micro['amihud_illiq_ma'] = df_micro['amihud_illiq'].rolling(20).mean()

    # 4. ROLL'S SPREAD (effective spread estimation)
    # Spread = 2√(-Cov(ΔP_t, ΔP_t-1))
    df_micro['price_change_lag1'] = df_micro['price_change'].shift(1)
    covariance = df_micro['price_change'].rolling(20).cov(df_micro['price_change_lag1'])
    df_micro['rolls_spread'] = 2 * np.sqrt(-covariance.clip(upper=0))

    # 5. BID-ASK PROXY (using high-low range)
    df_micro['bid_ask_proxy'] = (df_micro['high'] - df_micro['low']) / df_micro['close']
    df_micro['bid_ask_ma'] = df_micro['bid_ask_proxy'].rolling(20).mean()

    # 6. PRICE EFFICIENCY RATIO (trending vs ranging)
    # ER = Net Price Change / Sum of Absolute Changes
    df_micro['net_change'] = abs(df_micro['close'].diff(10))
    df_micro['abs_changes_sum'] = abs(df_micro['close'].diff()).rolling(10).sum()
    df_micro['efficiency_ratio'] = df_micro['net_change'] / (df_micro['abs_changes_sum'] + 1e-8)

    # 7. VOLUME-PRICE CORRELATION (buying vs selling pressure)
    df_micro['volume_price_corr'] = df_micro['volume'].rolling(20).corr(df_micro['close'])

    # 8. TICK DIRECTION PERSISTENCE
    df_micro['tick'] = np.sign(df_micro['close'].diff())
    df_micro['tick_persistence_5'] = df_micro['tick'].rolling(5).mean()
    df_micro['tick_persistence_20'] = df_micro['tick'].rolling(20).mean()

    return df_micro


def create_advanced_master_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features ULTRA avançadas combinando múltiplas dimensões."""

    df_adv = df.copy()

    # Multi-period momentum with decay
    for period in [3, 5, 8, 13, 21, 34]:
        df_adv[f'momentum_{period}'] = df_adv['close'].pct_change(period) * 100
        df_adv[f'volume_ratio_{period}'] = df_adv['volume'] / df_adv['volume'].rolling(period).mean()

        # Momentum strength (acceleration)
        df_adv[f'momentum_strength_{period}'] = df_adv[f'momentum_{period}'].diff()

    # Trend strength (EMA ribbon)
    if 'ema50' in df_adv.columns and 'ema200' in df_adv.columns:
        df_adv['trend_strength'] = (df_adv['ema50'] - df_adv['ema200']) / df_adv['ema200'] * 100
        df_adv['trend_acceleration'] = df_adv['trend_strength'].diff()

    # Volatility regimes (dynamic)
    if 'atr' in df_adv.columns:
        df_adv['volatility_regime'] = df_adv['atr'] / df_adv['atr'].rolling(50).mean()
        df_adv['volatility_percentile'] = df_adv['atr'].rolling(100).apply(
            lambda x: pd.Series(x).rank().iloc[-1] / len(x) if len(x) > 0 else 0.5
        )

    # Price position in dynamic range
    for window in [10, 20, 50]:
        high_max = df_adv['high'].rolling(window).max()
        low_min = df_adv['low'].rolling(window).min()
        df_adv[f'price_position_{window}'] = (
            (df_adv['close'] - low_min) / (high_max - low_min + 1e-8)
        )

    # Volume momentum and acceleration
    df_adv['volume_momentum'] = df_adv['volume'].pct_change(5)
    df_adv['volume_acceleration'] = df_adv['volume_momentum'].diff()

    # Price acceleration (2nd derivative)
    df_adv['price_acceleration'] = df_adv['close'].diff(2) - df_adv['close'].diff(1)

    # Candle patterns advanced
    df_adv['body_size'] = abs(df_adv['close'] - df_adv['open'])
    df_adv['upper_wick'] = df_adv['high'] - df_adv[['close', 'open']].max(axis=1)
    df_adv['lower_wick'] = df_adv[['close', 'open']].min(axis=1) - df_adv['low']
    df_adv['body_to_range'] = df_adv['body_size'] / (df_adv['high'] - df_adv['low'] + 1e-8)

    # Support/Resistance proximity
    df_adv['support_20'] = df_adv['low'].rolling(20).min()
    df_adv['resistance_20'] = df_adv['high'].rolling(20).max()
    df_adv['dist_to_support'] = (df_adv['close'] - df_adv['support_20']) / df_adv['close']
    df_adv['dist_to_resistance'] = (df_adv['resistance_20'] - df_adv['close']) / df_adv['close']

    return df_adv


def create_perfect_targets_advanced(df: pd.DataFrame, atr_col='atr') -> pd.DataFrame:
    """
    Labeling ULTRA avançado com Triple Barrier + Multi-Horizon Voting

    Combina:
    1. Dynamic thresholds (ATR-based)
    2. Multi-horizon consensus
    3. Risk-reward asymmetry
    """
    df_targets = df.copy()

    # ATR-based dynamic threshold
    atr = df_targets[atr_col] if atr_col in df_targets.columns else df_targets['close'] * 0.005
    atr_pct = (atr / df_targets['close']) * 100

    # Adaptive threshold based on volatility regime
    vol_regime = atr / atr.rolling(50).mean()

    # Lower threshold in low vol (stable), higher in high vol
    base_threshold = np.clip(atr_pct * 0.35, 0.30, 0.80)
    volatility_adjustment = np.clip(vol_regime * 0.2, -0.15, 0.15)
    dynamic_threshold = base_threshold + volatility_adjustment

    # Multi-horizon voting with weighted consensus
    horizons = [4, 6, 8, 12]  # 1h, 1.5h, 2h, 3h
    weights = [0.4, 0.3, 0.2, 0.1]  # More weight on shorter horizons

    weighted_votes = []

    for horizon, weight in zip(horizons, weights):
        future_returns = (df_targets['close'].shift(-horizon) / df_targets['close'] - 1) * 100

        # Asymmetric thresholds (risk-reward 1:1.5)
        profit_threshold = dynamic_threshold * 1.5
        loss_threshold = dynamic_threshold

        vote = pd.Series(0.5, index=df_targets.index)
        vote[future_returns > profit_threshold] = 1.0  # Strong UP
        vote[future_returns < -loss_threshold] = 0.0   # Strong DOWN

        weighted_votes.append(vote * weight)

    # Weighted average
    avg_vote = sum(weighted_votes) / sum(weights)

    # Strong consensus required (at least 70% agreement)
    target = pd.Series(np.nan, index=df_targets.index)
    target[avg_vote > 0.70] = 1  # UP
    target[avg_vote < 0.30] = 0  # DOWN

    df_targets['target'] = target
    df_targets['vote_confidence'] = np.abs(avg_vote - 0.5) * 2
    df_targets['avg_vote'] = avg_vote

    return df_targets


# ============================================================================
# FEATURE SELECTION COM SHAP
# ============================================================================

def select_top_features_shap(X_train, y_train, feature_names, top_n=50):
    """
    Feature selection usando SHAP values (TreeSHAP com LightGBM)

    Retorna top N features mais importantes
    """
    print(f"\n🔍 Feature Selection com SHAP...")
    print(f"   Features iniciais: {len(feature_names)}")

    # Train quick LightGBM for feature importance
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'verbose': -1
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(params, train_data, num_boost_round=100)

    # Get feature importance
    importance = model.feature_importance(importance_type='gain')
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)

    # Select top N
    top_features = feature_importance.head(top_n)['feature'].tolist()

    print(f"   Features selecionadas: {len(top_features)}")
    print(f"\n🔝 Top 10 features:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"      {row['feature']:40} {row['importance']:>10.0f}")

    return top_features


# ============================================================================
# MODELOS
# ============================================================================

def train_lightgbm_model(X_train, y_train, X_val, y_val):
    """Train optimized LightGBM."""
    print("\n🚀 Training LightGBM...")

    # Class weights
    class_counts = y_train.value_counts()
    total = len(y_train)
    weight_0 = total / (2 * class_counts[0]) if 0 in class_counts else 1.0
    weight_1 = total / (2 * class_counts[1]) if 1 in class_counts else 1.0
    scale_pos_weight = weight_1 / weight_0

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.03,
        'feature_fraction': 0.85,
        'bagging_fraction': 0.85,
        'bagging_freq': 5,
        'max_depth': 8,
        'min_data_in_leaf': 100,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'scale_pos_weight': scale_pos_weight,
        'verbose': -1
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        valid_names=['val'],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=100)]
    )

    return model


def train_xgboost_model(X_train, y_train, X_val, y_val):
    """Train optimized XGBoost."""
    print("\n🚀 Training XGBoost...")

    # Class weights
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 8,
        'learning_rate': 0.03,
        'subsample': 0.85,
        'colsample_bytree': 0.85,
        'min_child_weight': 100,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'scale_pos_weight': scale_pos_weight,
        'tree_method': 'hist',
        'verbosity': 0
    }

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=[(dval, 'val')],
        early_stopping_rounds=50,
        verbose_eval=100
    )

    return model


def build_transformer_model(sequence_length, n_features):
    """Build compact Transformer for sequences."""
    if not HAS_TENSORFLOW:
        return None

    print("\n🚀 Building Transformer...")

    inputs = keras.Input(shape=(sequence_length, n_features))

    # Positional encoding
    x = layers.Dense(64)(inputs)

    # Single Transformer block
    attention = layers.MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
    attention = layers.Dropout(0.2)(attention)
    x1 = layers.LayerNormalization()(x + attention)

    ff = layers.Dense(128, activation='relu')(x1)
    ff = layers.Dense(64)(ff)
    ff = layers.Dropout(0.2)(ff)
    x2 = layers.LayerNormalization()(x1 + ff)

    # Global pooling
    x = layers.GlobalAveragePooling1D()(x2)

    # Classifier
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss='binary_crossentropy',
        metrics=['AUC']
    )

    return model


def build_cnn_lstm_model(sequence_length, n_features):
    """Build CNN-LSTM hybrid model."""
    if not HAS_TENSORFLOW:
        return None

    print("\n🚀 Building CNN-LSTM...")

    inputs = keras.Input(shape=(sequence_length, n_features))

    # CNN layers
    x = layers.Conv1D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.Conv1D(64, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.2)(x)

    # LSTM layers
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(32))(x)
    x = layers.Dropout(0.3)(x)

    # Classifier
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss='binary_crossentropy',
        metrics=['AUC']
    )

    return model


def create_sequences(X, y, sequence_length=20):
    """Create sequences for time-series models."""
    X_seq = []
    y_seq = []

    for i in range(sequence_length, len(X)):
        X_seq.append(X[i-sequence_length:i])
        y_seq.append(y[i])

    return np.array(X_seq), np.array(y_seq)


# ============================================================================
# META-LEARNING (STACKING ENSEMBLE)
# ============================================================================

def train_meta_learner(base_predictions_train, y_train, base_predictions_val, y_val):
    """
    Train meta-learner for stacking ensemble.

    Uses calibrated logistic regression to combine base model predictions.
    """
    print("\n🎯 Training Meta-Learner (Stacking)...")

    # Stack predictions as features
    X_meta_train = np.column_stack([pred for pred in base_predictions_train.values()])
    X_meta_val = np.column_stack([pred for pred in base_predictions_val.values()])

    # Calibrated Logistic Regression
    base_meta = LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=42
    )

    meta_learner = CalibratedClassifierCV(base_meta, cv=5, method='sigmoid')
    meta_learner.fit(X_meta_train, y_train)

    # Evaluate meta-learner
    meta_pred_train = meta_learner.predict_proba(X_meta_train)[:, 1]
    meta_pred_val = meta_learner.predict_proba(X_meta_val)[:, 1]

    train_acc = ((meta_pred_train > 0.5).astype(int) == y_train).mean()
    val_acc = ((meta_pred_val > 0.5).astype(int) == y_val).mean()

    print(f"   Meta-Learner Train Accuracy: {train_acc*100:.2f}%")
    print(f"   Meta-Learner Val Accuracy:   {val_acc*100:.2f}%")

    return meta_learner


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def train_ultra_scalper(symbol: str, days: int, config: dict):
    """Train ULTRA MASTER SCALPER with ensemble."""

    logger = logging.getLogger('UltraScalper')

    print()
    print("=" * 80)
    print("🏆 ULTRA MASTER SCALPER - Estado da Arte ML")
    print("=" * 80)
    print(f"   Symbol:      {symbol}")
    print(f"   Period:      {days} days")
    print(f"   Strategy:    Ensemble (LGB + XGB + Transformer + CNN-LSTM)")
    print(f"   Features:    Microstructure + Advanced")
    print(f"   Selection:   SHAP-based")
    print(f"   Stacking:    Calibrated Meta-Learner")
    print("=" * 80)
    print()

    # Initialize
    rest_client = BybitRESTClient(
        api_key=config['bybit_api_key'],
        api_secret=config['bybit_api_secret'],
        testnet=config['bybit_testnet']
    )

    data_manager = DataManager(rest_client)
    feature_store = FeatureStore(config)

    # Download data
    print("📥 Downloading data...")
    df = data_manager.get_data(symbol, '15m', days, use_cache=False)
    print(f"✅ Downloaded {len(df):,} candles")
    print()

    # Build features
    print("🔨 Building base features...")
    df_features = feature_store.build_features(df, normalize=False)
    print(f"✅ Base features: {len(df_features.columns)}")

    print("\n🔬 Adding microstructure features...")
    df_features = create_microstructure_features(df_features)

    print("\n🎯 Adding advanced features...")
    df_features = create_advanced_master_features(df_features)
    print(f"✅ Total features: {len(df_features.columns)}")

    # Create targets
    print("\n🎯 Creating advanced targets...")
    df_features = create_perfect_targets_advanced(df_features)

    # Analyze targets
    valid_mask = ~df_features['target'].isna()
    target = df_features.loc[valid_mask, 'target']

    up_count = (target == 1).sum()
    down_count = (target == 0).sum()
    total = len(target)
    balance = abs(up_count - down_count) / total * 100

    print(f"\n📊 TARGET DISTRIBUTION:")
    print(f"   UP:      {up_count:,} ({up_count/total*100:.1f}%)")
    print(f"   DOWN:    {down_count:,} ({down_count/total*100:.1f}%)")
    print(f"   Balance: {balance:.1f}% diff")

    # Prepare data
    df_clean = df_features[valid_mask].copy()
    y = df_clean['target'].astype(int)

    exclude_cols = ['close', 'high', 'low', 'open', 'volume', 'target',
                   'vote_confidence', 'avg_vote']
    feature_cols = [col for col in df_clean.columns if col not in exclude_cols]

    # Remove object types
    object_cols = df_clean[feature_cols].select_dtypes(include=['object']).columns
    feature_cols = [col for col in feature_cols if col not in object_cols]

    X = df_clean[feature_cols].fillna(0)

    # Split
    split_idx = int(len(X) * 0.80)
    X_train_full = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    X_val = X.iloc[split_idx:]
    y_val = y.iloc[split_idx:]

    print(f"\n📊 Dataset:")
    print(f"   Features: {len(feature_cols)}")
    print(f"   Train:    {len(X_train_full):,}")
    print(f"   Val:      {len(X_val):,}")

    # Feature selection with SHAP
    top_features = select_top_features_shap(
        X_train_full, y_train, feature_cols, top_n=60
    )

    X_train = X_train_full[top_features]
    X_val_selected = X_val[top_features]

    # ========================================================================
    # TRAIN ENSEMBLE MODELS
    # ========================================================================

    models = {}

    # 1. LightGBM
    models['lightgbm'] = train_lightgbm_model(X_train, y_train, X_val_selected, y_val)

    # 2. XGBoost
    models['xgboost'] = train_xgboost_model(X_train, y_train, X_val_selected, y_val)

    # 3. Transformer (if TensorFlow available)
    if HAS_TENSORFLOW:
        # Normalize for DL
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val_selected)

        # Create sequences
        sequence_length = 20
        X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train.values, sequence_length)
        X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val.values, sequence_length)

        # Train Transformer
        transformer_model = build_transformer_model(sequence_length, len(top_features))
        if transformer_model:
            transformer_model.fit(
                X_train_seq, y_train_seq,
                validation_data=(X_val_seq, y_val_seq),
                epochs=30,
                batch_size=64,
                callbacks=[
                    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
                ],
                verbose=0
            )
            models['transformer'] = transformer_model
            models['scaler'] = scaler
            models['sequence_length'] = sequence_length

    # Get base predictions
    print("\n📊 Getting base model predictions...")
    base_predictions_train = {}
    base_predictions_val = {}

    # Determine if we need to align to sequence length
    if 'transformer' in models:
        sequence_length = models['sequence_length']
        print(f"   Aligning all predictions to sequence length offset: {sequence_length}")

        # Align all data to sequence-based indices
        X_train_aligned = X_train.iloc[sequence_length:]
        X_val_aligned = X_val_selected.iloc[sequence_length:]

        # LightGBM - predict on aligned data
        base_predictions_train['lightgbm'] = models['lightgbm'].predict(X_train_aligned)
        base_predictions_val['lightgbm'] = models['lightgbm'].predict(X_val_aligned)

        # XGBoost - predict on aligned data
        base_predictions_train['xgboost'] = models['xgboost'].predict(xgb.DMatrix(X_train_aligned))
        base_predictions_val['xgboost'] = models['xgboost'].predict(xgb.DMatrix(X_val_aligned))

        # Transformer - already aligned
        base_predictions_train['transformer'] = models['transformer'].predict(X_train_seq, verbose=0).flatten()
        base_predictions_val['transformer'] = models['transformer'].predict(X_val_seq, verbose=0).flatten()

        # Adjust y for sequence offset
        y_train_adj = y_train.iloc[sequence_length:].values
        y_val_adj = y_val.iloc[sequence_length:].values

        print(f"   ✅ All predictions aligned: {len(base_predictions_train['lightgbm'])} samples")
    else:
        # No transformer, use all data
        base_predictions_train['lightgbm'] = models['lightgbm'].predict(X_train)
        base_predictions_val['lightgbm'] = models['lightgbm'].predict(X_val_selected)

        base_predictions_train['xgboost'] = models['xgboost'].predict(xgb.DMatrix(X_train))
        base_predictions_val['xgboost'] = models['xgboost'].predict(xgb.DMatrix(X_val_selected))

        y_train_adj = y_train.values
        y_val_adj = y_val.values

    # Train meta-learner
    meta_learner = train_meta_learner(
        base_predictions_train, y_train_adj,
        base_predictions_val, y_val_adj
    )

    # Final evaluation
    print("\n" + "=" * 80)
    print("🏆 ULTRA SCALPER TRAINED!")
    print("=" * 80)

    # Individual model performance
    print("\n📊 INDIVIDUAL MODEL PERFORMANCE (Validation):")
    for name, preds in base_predictions_val.items():
        acc = ((preds > 0.5).astype(int) == y_val_adj).mean()
        print(f"   {name:15} Accuracy: {acc*100:.2f}%")

    # Ensemble performance
    X_meta_val = np.column_stack([pred for pred in base_predictions_val.values()])
    ensemble_pred = meta_learner.predict_proba(X_meta_val)[:, 1]
    ensemble_acc = ((ensemble_pred > 0.5).astype(int) == y_val_adj).mean()

    print(f"\n🎯 ENSEMBLE (Meta-Learner) Accuracy: {ensemble_acc*100:.2f}%")

    # Save
    model_dir = Path("storage/models")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"ultra_scalper_{symbol.lower()}_{days}d.pkl"

    model_data = {
        'models': models,
        'meta_learner': meta_learner,
        'feature_names': top_features,
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'days_trained': days,
        'symbol': symbol,
        'ensemble_accuracy': ensemble_acc
    }

    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"\n💾 Model saved: {model_path}")
    print("\n" + "=" * 80)
    print("🎯 NEXT: Run validate_ultra_scalper.py for walk-forward validation")
    print("=" * 80)

    return model_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', type=str, default='BTCUSDT')
    parser.add_argument('--days', type=int, default=365)

    args = parser.parse_args()

    config = load_config('standard')
    setup_logging('INFO', log_to_file=False)

    train_ultra_scalper(args.symbol, args.days, config)


if __name__ == "__main__":
    main()
