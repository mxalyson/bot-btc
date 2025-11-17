"""
Script de Treinamento Deep Learning para Scalping 15m
Modelo Híbrido: CNN + LSTM + Attention + Microstructure Features

Este script implementa as melhores práticas de research:
- Features de microestrutura (73% da performance)
- Arquitetura híbrida CNN+LSTM+Attention
- Multi-timeframe features
- Regularização adequada
"""

import sys
import argparse
from pathlib import Path

# Adicionar diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from loguru import logger
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import pickle

from core.utils import (
    load_config, setup_logging, print_section,
    ensure_dir
)
from core.data_loader import BybitDataLoader
from core.feature_engineering import FeatureEngineer
from core.microstructure_features import MicrostructureFeatures
from core.labeling import TradingLabeler
from core.deep_learning_model import DeepScalpingModel, TENSORFLOW_AVAILABLE


def prepare_data_for_dl(
    df: pd.DataFrame,
    config: dict,
    label_mapping: dict = {'LONG': 0, 'SHORT': 1}
) -> tuple:
    """
    Prepara dados para Deep Learning.
    """
    logger.info("Preparando dados para Deep Learning...")

    # Separar features e labels
    exclude_cols = config['dataset']['exclude_features']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    X = df[feature_cols].values
    y = df['target_class'].map(label_mapping).values

    logger.info(f"Features: {len(feature_cols)}")
    logger.info(f"Samples: {len(X)}")
    logger.info(f"Feature columns: {feature_cols[:10]}...")

    return X, y, feature_cols


def main():
    parser = argparse.ArgumentParser(
        description="Treinamento Deep Learning para Scalping 15m"
    )
    parser.add_argument(
        '--symbol', type=str, default='BTCUSDT',
        help='Símbolo para treinar'
    )
    parser.add_argument(
        '--timeframe', type=str, default='15m',
        help='Timeframe (5m, 15m, 1h, etc)'
    )
    parser.add_argument(
        '--config', type=str, default='config/config_deep_learning.yaml',
        help='Arquivo de configuração'
    )
    parser.add_argument(
        '--sequence-length', type=int, default=20,
        help='Comprimento da sequência para LSTM (número de candles)'
    )
    parser.add_argument(
        '--epochs', type=int, default=100,
        help='Número máximo de epochs'
    )
    parser.add_argument(
        '--batch-size', type=int, default=64,
        help='Batch size'
    )

    args = parser.parse_args()

    # Verificar TensorFlow
    if not TENSORFLOW_AVAILABLE:
        logger.error("TensorFlow não está instalado!")
        logger.error("Instale com: pip install tensorflow")
        sys.exit(1)

    # Carregar configuração
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        logger.warning(f"Config {args.config} não encontrado, usando config_advanced.yaml")
        config = load_config('config/config_advanced.yaml')

    setup_logging(config)

    print_section("DEEP LEARNING para Scalping 15m")
    print_section("Arquitetura: CNN + Bidirectional LSTM + Attention")

    logger.info(f"Símbolo: {args.symbol}")
    logger.info(f"Timeframe: {args.timeframe}")
    logger.info(f"Sequence length: {args.sequence_length}")
    logger.info(f"Max epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")

    # ========== ETAPA 1: DADOS ==========
    print_section("ETAPA 1: Carregamento de Dados")

    # Tentar carregar dados existentes (incluindo sintéticos)
    data_dir = Path(config['data']['raw_data_dir'])
    symbol_lower = args.symbol.lower()
    symbol_upper = args.symbol.upper()

    # Tentar diferentes variações de nome de arquivo
    possible_files = [
        data_dir / f"{symbol_lower}_{args.timeframe}.csv",
        data_dir / f"{symbol_upper}_{args.timeframe}.csv",
    ]

    df = None
    for filepath in possible_files:
        if filepath.exists():
            logger.info(f"Carregando dados de: {filepath}")
            df = pd.read_csv(filepath)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            logger.info(f"✓ Dados carregados: {len(df)} linhas")
            break

    # Se não encontrou, sair com erro
    if df is None or len(df) == 0:
        logger.error("❌ Dados não encontrados!")
        logger.error(f"Procurados em: {[str(f) for f in possible_files]}")
        logger.error("Gere dados sintéticos com: python scripts/generate_synthetic_data.py")
        sys.exit(1)

    logger.info(f"Dataset inicial: {df.shape}")
    logger.info(f"Período: {df['timestamp'].min()} a {df['timestamp'].max()}")

    # ========== ETAPA 2: FEATURES TRADICIONAIS ==========
    print_section("ETAPA 2: Feature Engineering Tradicional")

    engineer = FeatureEngineer(config)
    df = engineer.create_all_features(df)

    logger.info(f"Dataset com features técnicas: {df.shape}")

    # ========== ETAPA 3: FEATURES DE MICROESTRUTURA ==========
    print_section("ETAPA 3: Features de Microestrutura (CRÍTICO!)")

    micro_engineer = MicrostructureFeatures(config)
    df = micro_engineer.create_all_features(df)

    logger.info(f"Dataset com microestrutura: {df.shape}")
    logger.info(f"Total de features: {len([c for c in df.columns if c not in ['timestamp','open','high','low','close','volume']])}")

    # ========== ETAPA 4: LABELS ==========
    print_section("ETAPA 4: Criação de Labels (Triple Barrier)")

    labeler = TradingLabeler(config)
    df = labeler.create_labels(df)

    logger.info(f"Dataset com labels: {df.shape}")
    logger.info("Distribuição inicial:")
    for label in ['LONG', 'SHORT', 'NONE']:
        count = (df['target_class'] == label).sum()
        pct = 100 * count / len(df)
        logger.info(f"  {label}: {count} ({pct:.2f}%)")

    # Remover NaNs
    df_clean = df.dropna()
    removed = len(df) - len(df_clean)
    logger.info(f"NaNs removidos: {removed} linhas")

    # Remover NONE (classificação binária)
    df_binary = df_clean[df_clean['target_class'] != 'NONE'].copy()
    logger.info(f"Classe NONE removida: {len(df_clean) - len(df_binary)} linhas")

    logger.info("Distribuição final (LONG vs SHORT):")
    for label in ['LONG', 'SHORT']:
        count = (df_binary['target_class'] == label).sum()
        pct = 100 * count / len(df_binary)
        logger.info(f"  {label}: {count} ({pct:.2f}%)")

    # ========== ETAPA 5: SPLIT TEMPORAL ==========
    print_section("ETAPA 5: Divisão Temporal do Dataset")

    df_train, df_val, df_test = labeler.split_temporal(df_binary)

    logger.info(f"Train: {len(df_train)} samples")
    logger.info(f"Val:   {len(df_val)} samples")
    logger.info(f"Test:  {len(df_test)} samples")

    # ========== ETAPA 6: PREPARAÇÃO PARA DL ==========
    print_section("ETAPA 6: Preparação para Deep Learning")

    label_mapping = {'LONG': 0, 'SHORT': 1}

    X_train, y_train, feature_cols = prepare_data_for_dl(df_train, config, label_mapping)
    X_val, y_val, _ = prepare_data_for_dl(df_val, config, label_mapping)
    X_test, y_test, _ = prepare_data_for_dl(df_test, config, label_mapping)

    logger.info(f"Train: X={X_train.shape}, y={y_train.shape}")
    logger.info(f"Val:   X={X_val.shape}, y={y_val.shape}")
    logger.info(f"Test:  X={X_test.shape}, y={y_test.shape}")

    # Normalização (RobustScaler para outliers)
    logger.info("Normalizando features com RobustScaler...")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # ========== ETAPA 7: CONSTRUÇÃO DO MODELO ==========
    print_section("ETAPA 7: Construção do Modelo Deep Learning")

    model = DeepScalpingModel(config)
    model.scaler = scaler

    # Mostrar arquitetura
    model.build_model(
        sequence_length=args.sequence_length,
        n_features=X_train_scaled.shape[1]
    )
    model.summary()

    # ========== ETAPA 8: TREINAMENTO ==========
    print_section("ETAPA 8: Treinamento do Modelo")

    results = model.train(
        X_train_scaled, y_train,
        X_val_scaled, y_val,
        sequence_length=args.sequence_length,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    # ========== ETAPA 9: AVALIAÇÃO NO TEST SET ==========
    print_section("ETAPA 9: Avaliação no Conjunto de Teste")

    test_results = model.evaluate(
        X_test_scaled, y_test,
        sequence_length=args.sequence_length
    )

    logger.info("Métricas de Teste:")
    logger.info(f"  Loss:      {test_results['loss']:.4f}")
    logger.info(f"  Accuracy:  {test_results['accuracy']:.4f}")
    logger.info(f"  ROC AUC:   {test_results['auc']:.4f}")
    logger.info(f"  Precision: {test_results['precision']:.4f}")
    logger.info(f"  Recall:    {test_results['recall']:.4f}")

    # Predições detalhadas
    y_pred_proba = model.predict(X_test_scaled, sequence_length=args.sequence_length)
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Ajustar y_test para match com predições (sequences são menores)
    y_test_seq = y_test[args.sequence_length:]

    # Confusion matrix
    logger.info("\nMatriz de Confusão:")
    cm = confusion_matrix(y_test_seq, y_pred)
    logger.info(f"\n{cm}")
    logger.info(f"         Pred LONG  Pred SHORT")
    logger.info(f"LONG     {cm[0,0]:6d}     {cm[0,1]:6d}   (Recall: {cm[0,0]/(cm[0,0]+cm[0,1]):.2%})")
    logger.info(f"SHORT    {cm[1,0]:6d}     {cm[1,1]:6d}   (Recall: {cm[1,1]/(cm[1,0]+cm[1,1]):.2%})")

    # Classification report
    logger.info("\nClassification Report:")
    logger.info("\n" + classification_report(
        y_test_seq, y_pred,
        target_names=['LONG', 'SHORT'],
        digits=4
    ))

    # ========== ETAPA 10: SALVAR MODELO ==========
    print_section("ETAPA 10: Salvando Modelo")

    model_dir = Path(config.get('model', {}).get('output_dir', 'models'))
    ensure_dir(model_dir)

    model_name = f"{args.symbol.lower()}_{args.timeframe}_deep_learning"
    model_path = model_dir / model_name

    model.save(str(model_path))

    # Salvar metadata
    metadata = {
        'symbol': args.symbol,
        'timeframe': args.timeframe,
        'sequence_length': args.sequence_length,
        'n_features': X_train_scaled.shape[1],
        'feature_cols': feature_cols,
        'label_mapping': label_mapping,
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'train_metrics': {
            'accuracy': results['train_accuracy'],
            'auc': results['train_auc']
        },
        'val_metrics': {
            'accuracy': results['val_accuracy'],
            'auc': results['val_auc']
        },
        'test_metrics': {
            'accuracy': test_results['accuracy'],
            'auc': test_results['auc'],
            'precision': test_results['precision'],
            'recall': test_results['recall']
        },
        'epochs_trained': results['epochs_trained']
    }

    metadata_path = model_dir / f"{model_name}_metadata.pkl"
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)

    logger.info(f"✓ Modelo salvo: {model_path}")
    logger.info(f"✓ Metadata salvo: {metadata_path}")

    # ========== RESUMO FINAL ==========
    print_section("RESUMO DO TREINAMENTO DEEP LEARNING")

    logger.info(f"Símbolo: {args.symbol}")
    logger.info(f"Timeframe: {args.timeframe}")
    logger.info(f"Arquitetura: CNN + BiLSTM + Attention")
    logger.info(f"Sequence length: {args.sequence_length}")
    logger.info(f"Features: {X_train_scaled.shape[1]} (incluindo microestrutura)")
    logger.info("")
    logger.info("Dados:")
    logger.info(f"  Train: {len(X_train)} samples")
    logger.info(f"  Val:   {len(X_val)} samples")
    logger.info(f"  Test:  {len(X_test)} samples")
    logger.info("")
    logger.info("Resultados Finais:")
    logger.info(f"  Train ROC AUC: {results['train_auc']:.4f}")
    logger.info(f"  Val ROC AUC:   {results['val_auc']:.4f}")
    logger.info(f"  Test ROC AUC:  {test_results['auc']:.4f} {'✅' if test_results['auc'] > 0.55 else '❌'}")
    logger.info("")

    if test_results['auc'] > 0.55:
        logger.info("🎉 ROC AUC > 0.55 - Modelo tem edge competitivo!")
        logger.info("Próximo passo: Backtesting")
    elif test_results['auc'] > 0.52:
        logger.info("🟡 ROC AUC > 0.52 - Modelo marginal, considere mais tuning")
    else:
        logger.info("❌ ROC AUC < 0.52 - Modelo sem edge, não use em produção")

    print_section("TREINAMENTO DEEP LEARNING CONCLUÍDO!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\nTreinamento interrompido pelo usuário")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Erro durante o treinamento: {e}")
        logger.exception(e)
        sys.exit(1)
