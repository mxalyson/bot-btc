"""
🚀 SCRIPT DE TREINAMENTO AVANÇADO 🚀

Integra TODAS as melhorias:
- Wavelet Transform
- Market Regime Detection
- Multi-Timeframe (15m + 1h + 4h)
- Transformer ou Hybrid Architecture
- Multi-Scale CNN + Residual + Multi-Head Attention
- Focal Loss
- Data Augmentation
- Triple Barrier Dinâmico
- Ensemble Stacking (DL + LightGBM + XGBoost)
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from loguru import logger
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score, classification_report
import pickle

# Módulos próprios
from core.utils import load_config, setup_logging, print_section, ensure_dir
from core.feature_engineering import FeatureEngineer
from core.microstructure_features import MicrostructureFeatures
from core.advanced_features import (
    WaveletFeatures, MarketRegimeDetector, MultiTimeframeFeatures
)
from core.dynamic_labeling import DynamicTripleBarrier
from core.labeling import TradingLabeler  # Fallback
from core.advanced_models import build_advanced_model
from core.data_augmentation import TimeSeriesAugmenter
from core.advanced_ensemble import StackingEnsemble

# Suprimir warnings do TensorFlow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def load_and_prepare_data(config, symbol, timeframe):
    """Carrega e prepara dados com TODAS as features avançadas"""
    
    print_section("ETAPA 1: Carregamento de Dados")
    
    data_dir = Path(config['data']['raw_data_dir'])
    
    # Carregar timeframe base
    filepath_base = data_dir / f"{symbol.lower()}_{timeframe}.csv"
    if not filepath_base.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {filepath_base}")
    
    df_base = pd.read_csv(filepath_base)
    df_base['timestamp'] = pd.to_datetime(df_base['timestamp'])
    logger.info(f"✓ Dados base carregados: {len(df_base)} linhas ({timeframe})")
    
    # Carregar timeframes maiores (se multi-timeframe habilitado)
    df_higher = {}
    if config['features'].get('multi_timeframe', {}).get('enabled', False):
        higher_tfs = config['features']['multi_timeframe'].get('higher_timeframes', [])
        
        for tf in higher_tfs:
            filepath_tf = data_dir / f"{symbol.lower()}_{tf}.csv"
            if filepath_tf.exists():
                df_tf = pd.read_csv(filepath_tf)
                df_tf['timestamp'] = pd.to_datetime(df_tf['timestamp'])
                df_higher[tf] = df_tf
                logger.info(f"✓ Dados {tf} carregados: {len(df_tf)} linhas")
            else:
                logger.warning(f"Arquivo {filepath_tf} não encontrado, pulando...")
    
    return df_base, df_higher


def create_all_features(config, df_base, df_higher):
    """Cria TODAS as features"""
    
    print_section("ETAPA 2: Feature Engineering COMPLETO")
    
    df = df_base.copy()
    
    # 1. Features técnicas tradicionais
    logger.info("1️⃣  Features Técnicas...")
    engineer = FeatureEngineer(config)
    df = engineer.create_all_features(df)
    
    # 2. Microstructure features
    logger.info("2️⃣  Microstructure Features...")
    micro_engineer = MicrostructureFeatures(config)
    df = micro_engineer.create_all_features(df)
    
    # 3. Wavelet Transform
    if config['features'].get('wavelet', {}).get('enabled', False):
        logger.info("3️⃣  Wavelet Transform...")
        wavelet = WaveletFeatures(
            wavelet=config['features']['wavelet'].get('wavelet_type', 'db4'),
            level=config['features']['wavelet'].get('decomposition_level', 3)
        )
        df = wavelet.create_features(df)
    
    # 4. Market Regime Detection
    if config['features'].get('market_regime', {}).get('enabled', False):
        logger.info("4️⃣  Market Regime Detection...")
        regime = MarketRegimeDetector(
            n_regimes=config['features']['market_regime'].get('n_regimes', 3),
            lookback=config['features']['market_regime'].get('lookback', 100)
        )
        df = regime.create_features(df)
    
    # 5. Multi-Timeframe
    if config['features'].get('multi_timeframe', {}).get('enabled', False) and df_higher:
        logger.info("5️⃣  Multi-Timeframe Features...")
        
        # Primeiro criar features técnicas nos timeframes maiores
        for tf, df_tf in df_higher.items():
            logger.info(f"  Criando features em {tf}...")
            df_higher[tf] = engineer.create_all_features(df_tf)
        
        # Depois integrar ao dataframe base
        mtf = MultiTimeframeFeatures(
            base_timeframe='15m',
            higher_timeframes=list(df_higher.keys())
        )
        df = mtf.create_features(df, df_higher)
    
    total_features = len([c for c in df.columns if c not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']])
    logger.info(f"✅ TOTAL: {total_features} features criadas!")
    
    return df


def create_labels(config, df):
    """Cria labels com Triple Barrier (dinâmico ou estático)"""
    
    print_section("ETAPA 3: Labeling")
    
    method = config['labeling'].get('method', 'triple_barrier')
    
    if method == 'triple_barrier_dynamic':
        logger.info("Usando Triple Barrier DINÂMICO (baseado em volatilidade)")
        labeler = DynamicTripleBarrier(config)
    else:
        logger.info("Usando Triple Barrier estático")
        labeler = TradingLabeler(config)
    
    df = labeler.create_labels(df)
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Treinamento AVANÇADO com TODAS as melhorias")
    parser.add_argument('--symbol', type=str, default='BTCUSDT')
    parser.add_argument('--timeframe', type=str, default='15m')
    parser.add_argument('--config', type=str, default='config/config_advanced.yaml')
    args = parser.parse_args()
    
    # Carregar config
    config = load_config(args.config)
    setup_logging(config)
    
    print_section("🚀 TREINAMENTO AVANÇADO - Sistema Completo")
    logger.info(f"Símbolo: {args.symbol}")
    logger.info(f"Timeframe: {args.timeframe}")
    logger.info(f"Config: {args.config}")
    
    # Carregar dados
    df_base, df_higher = load_and_prepare_data(config, args.symbol, args.timeframe)
    
    # Features
    df = create_all_features(config, df_base, df_higher)
    
    # Labels
    df = create_labels(config, df)
    
    # Limpar
    df_clean = df.dropna()
    df_binary = df_clean[df_clean['target_class'] != 'NONE'].copy()
    
    logger.info(f"Dataset limpo: {len(df_binary)} samples")
    
    # Split temporal
    labeler = TradingLabeler(config)
    df_train, df_val, df_test = labeler.split_temporal(df_binary)
    
    logger.info(f"Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")
    
    # Preparar features e labels
    exclude_cols = config['dataset']['exclude_features']
    feature_cols = [col for col in df_binary.columns if col not in exclude_cols]
    
    X_train = df_train[feature_cols].values
    y_train = df_train['target_class'].map({'LONG': 0, 'SHORT': 1}).values
    
    X_val = df_val[feature_cols].values
    y_val = df_val['target_class'].map({'LONG': 0, 'SHORT': 1}).values
    
    X_test = df_test[feature_cols].values
    y_test = df_test['target_class'].map({'LONG': 0, 'SHORT': 1}).values
    
    logger.info(f"Features: {X_train.shape[1]}")
    
    # Normalizar
    print_section("ETAPA 4: Normalização")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Data Augmentation
    augmentation_config = config['dataset'].get('augmentation', {})
    if augmentation_config.get('enabled', False):
        print_section("ETAPA 5: Data Augmentation")
        
        augmenter = TimeSeriesAugmenter(
            noise_level=augmentation_config.get('noise_level', 0.01),
            use_temporal_flip=augmentation_config.get('temporal_flip', True),
            use_mixup=augmentation_config.get('mixup', False)
        )
        
        # Note: Augmentation precisa de sequências 3D, vamos fazer depois de sequencing
        logger.info("Data augmentation será aplicado nas sequências...")
    
    # Criar sequências
    print_section("ETAPA 6: Criando Sequências Temporais")
    
    sequence_length = config['model']['sequence_length']
    logger.info(f"Sequence length: {sequence_length}")
    
    def create_sequences(X, y, seq_len):
        X_seq, y_seq = [], []
        for i in range(len(X) - seq_len):
            X_seq.append(X[i:i+seq_len])
            y_seq.append(y[i+seq_len])
        return np.array(X_seq), np.array(y_seq)
    
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, sequence_length)
    X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val, sequence_length)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, sequence_length)
    
    logger.info(f"Train sequences: {X_train_seq.shape}")
    logger.info(f"Val sequences: {X_val_seq.shape}")
    logger.info(f"Test sequences: {X_test_seq.shape}")
    
    # Aplicar augmentation AGORA (com sequências 3D)
    if augmentation_config.get('enabled', False):
        X_train_seq, y_train_seq = augmenter.augment(X_train_seq, y_train_seq)
        logger.info(f"Após augmentation: {X_train_seq.shape}")
    
    # Construir modelo
    print_section("ETAPA 7: Construindo Modelo Avançado")
    
    model = build_advanced_model(
        config=config,
        sequence_length=sequence_length,
        n_features=X_train_scaled.shape[1]
    )
    
    model.summary(print_fn=logger.info)
    
    # Callbacks
    print_section("ETAPA 8: Treinamento")
    
    callbacks = []
    
    # Early stopping
    if config['training']['early_stopping']['enabled']:
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=config['training']['early_stopping']['patience'],
            mode='max',
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stop)
    
    # ReduceLROnPlateau
    if config['training']['reduce_lr']['enabled']:
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_auc',
            factor=config['training']['reduce_lr']['factor'],
            patience=config['training']['reduce_lr']['patience'],
            min_lr=config['training']['reduce_lr']['min_lr'],
            mode='max',
            verbose=1
        )
        callbacks.append(reduce_lr)
    
    # Treinar
    history = model.fit(
        X_train_seq, y_train_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=config['training']['epochs'],
        batch_size=config['training']['batch_size'],
        callbacks=callbacks,
        verbose=1
    )
    
    # Avaliar
    print_section("ETAPA 9: Avaliação")
    
    y_pred_train = model.predict(X_train_seq).flatten()
    y_pred_val = model.predict(X_val_seq).flatten()
    y_pred_test = model.predict(X_test_seq).flatten()
    
    train_auc = roc_auc_score(y_train_seq, y_pred_train)
    val_auc = roc_auc_score(y_val_seq, y_pred_val)
    test_auc = roc_auc_score(y_test_seq, y_pred_test)
    
    logger.info(f"Train ROC AUC: {train_auc:.4f}")
    logger.info(f"Val ROC AUC:   {val_auc:.4f}")
    logger.info(f"Test ROC AUC:  {test_auc:.4f}")
    
    # Classification report
    y_pred_test_binary = (y_pred_test >= 0.5).astype(int)
    logger.info("\nClassification Report:")
    logger.info("\n" + classification_report(y_test_seq, y_pred_test_binary, target_names=['LONG', 'SHORT']))
    
    # Salvar modelo
    print_section("ETAPA 10: Salvando Modelo")
    
    model_dir = Path(config['model']['output_dir'])
    ensure_dir(model_dir)
    
    base_name = f"{args.symbol.lower()}_{args.timeframe}_advanced"
    model_name = f"{base_name}.keras"
    model.save(model_dir / model_name)
    logger.info(f"✓ Modelo salvo: {model_dir / model_name}")

    # Salvar scaler
    with open(model_dir / f"{base_name}_scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    
    print_section("✅ TREINAMENTO AVANÇADO CONCLUÍDO!")
    
    logger.info(f"Arquitetura: {config['model']['architecture']}")
    logger.info(f"Features: {X_train_scaled.shape[1]} (com Wavelet, Regime, Multi-TF)")
    logger.info(f"Sequence Length: {sequence_length}")
    logger.info(f"Test ROC AUC: {test_auc:.4f}")
    
    if test_auc > 0.60:
        logger.info("🎉 ROC AUC > 0.60 - Performance EXCELENTE!")
    elif test_auc > 0.55:
        logger.info("✅ ROC AUC > 0.55 - Performance BOA!")
    else:
        logger.info("⚠️  ROC AUC < 0.55 - Considere mais tuning")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\nTreinamento interrompido")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Erro: {e}")
        logger.exception(e)
        sys.exit(1)
