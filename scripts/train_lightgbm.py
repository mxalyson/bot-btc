"""
Script de Treinamento LightGBM para Scalping 15m
Modelo tree-based para complementar Deep Learning

LightGBM é escolhido por:
- Velocidade 3-15x maior que XGBoost
- Melhor com datasets grandes
- Leaf-wise growth (menos overfitting)
- Excelente com features numéricas
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from loguru import logger
import lightgbm as lgb
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_fscore_support
import pickle
import json

from core.utils import (
    load_config, setup_logging, print_section,
    ensure_dir
)
from core.feature_engineering import FeatureEngineer
from core.microstructure_features import MicrostructureFeatures
from core.labeling import TradingLabeler


def main():
    parser = argparse.ArgumentParser(
        description="Treinamento LightGBM para Scalping 15m"
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
        '--n-estimators', type=int, default=500,
        help='Número de árvores'
    )
    parser.add_argument(
        '--max-depth', type=int, default=8,
        help='Profundidade máxima das árvores'
    )
    parser.add_argument(
        '--learning-rate', type=float, default=0.05,
        help='Learning rate'
    )

    args = parser.parse_args()

    # Carregar configuração
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        logger.warning(f"Config {args.config} não encontrado, usando config_advanced.yaml")
        config = load_config('config/config_advanced.yaml')

    setup_logging(config)

    print_section("LIGHTGBM para Scalping 15m")
    print_section("Tree-based complementar ao Deep Learning")

    logger.info(f"Símbolo: {args.symbol}")
    logger.info(f"Timeframe: {args.timeframe}")
    logger.info(f"N estimators: {args.n_estimators}")
    logger.info(f"Max depth: {args.max_depth}")
    logger.info(f"Learning rate: {args.learning_rate}")

    # ========== ETAPA 1: DADOS ==========
    print_section("ETAPA 1: Carregamento de Dados")

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

    if df is None or len(df) == 0:
        logger.error("❌ Dados não encontrados!")
        logger.error(f"Procurados em: {[str(f) for f in possible_files]}")
        sys.exit(1)

    logger.info(f"Dataset inicial: {df.shape}")
    logger.info(f"Período: {df['timestamp'].min()} a {df['timestamp'].max()}")

    # ========== ETAPA 2: FEATURES TRADICIONAIS ==========
    print_section("ETAPA 2: Feature Engineering Tradicional")

    engineer = FeatureEngineer(config)
    df = engineer.create_all_features(df)

    logger.info(f"Dataset com features técnicas: {df.shape}")

    # ========== ETAPA 3: FEATURES DE MICROESTRUTURA ==========
    print_section("ETAPA 3: Features de Microestrutura")

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

    # ========== ETAPA 6: PREPARAÇÃO ==========
    print_section("ETAPA 6: Preparação de Features")

    exclude_cols = config['dataset']['exclude_features']
    feature_cols = [col for col in df_binary.columns if col not in exclude_cols]

    X_train = df_train[feature_cols].values
    y_train = df_train['target_class'].map({'LONG': 0, 'SHORT': 1}).values

    X_val = df_val[feature_cols].values
    y_val = df_val['target_class'].map({'LONG': 0, 'SHORT': 1}).values

    X_test = df_test[feature_cols].values
    y_test = df_test['target_class'].map({'LONG': 0, 'SHORT': 1}).values

    logger.info(f"Train: X={X_train.shape}, y={y_train.shape}")
    logger.info(f"Val:   X={X_val.shape}, y={y_val.shape}")
    logger.info(f"Test:  X={X_test.shape}, y={y_test.shape}")

    # Normalização
    logger.info("Normalizando features com RobustScaler...")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # ========== ETAPA 7: TREINAMENTO LIGHTGBM ==========
    print_section("ETAPA 7: Treinamento LightGBM")

    # Parâmetros LightGBM
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 2 ** args.max_depth,
        'max_depth': args.max_depth,
        'learning_rate': args.learning_rate,
        'n_estimators': args.n_estimators,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': 42,
        'verbose': -1,
        'n_jobs': -1
    }

    logger.info("Parâmetros LightGBM:")
    for key, value in params.items():
        logger.info(f"  {key}: {value}")

    # Criar datasets LightGBM
    train_data = lgb.Dataset(X_train_scaled, label=y_train)
    val_data = lgb.Dataset(X_val_scaled, label=y_val, reference=train_data)

    # Treinar
    logger.info("Iniciando treinamento...")

    callbacks = [
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=50)
    ]

    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=callbacks
    )

    logger.info(f"✓ Treinamento concluído! Best iteration: {model.best_iteration}")

    # ========== ETAPA 8: AVALIAÇÃO ==========
    print_section("ETAPA 8: Avaliação no Conjunto de Teste")

    # Predições
    y_pred_proba_train = model.predict(X_train_scaled, num_iteration=model.best_iteration)
    y_pred_proba_val = model.predict(X_val_scaled, num_iteration=model.best_iteration)
    y_pred_proba_test = model.predict(X_test_scaled, num_iteration=model.best_iteration)

    # ROC AUC
    train_auc = roc_auc_score(y_train, y_pred_proba_train)
    val_auc = roc_auc_score(y_val, y_pred_proba_val)
    test_auc = roc_auc_score(y_test, y_pred_proba_test)

    logger.info(f"Train ROC AUC: {train_auc:.4f}")
    logger.info(f"Val ROC AUC:   {val_auc:.4f}")
    logger.info(f"Test ROC AUC:  {test_auc:.4f}")

    # Predições binárias
    y_pred_test = (y_pred_proba_test >= 0.5).astype(int)

    # Métricas detalhadas
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred_test, average='binary'
    )

    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall:    {recall:.4f}")
    logger.info(f"F1 Score:  {f1:.4f}")

    # Confusion matrix
    logger.info("\nMatriz de Confusão:")
    cm = confusion_matrix(y_test, y_pred_test)
    logger.info(f"\n{cm}")
    logger.info(f"         Pred LONG  Pred SHORT")
    logger.info(f"LONG     {cm[0,0]:6d}     {cm[0,1]:6d}   (Recall: {cm[0,0]/(cm[0,0]+cm[0,1]):.2%})")
    logger.info(f"SHORT    {cm[1,0]:6d}     {cm[1,1]:6d}   (Recall: {cm[1,1]/(cm[1,0]+cm[1,1]):.2%})")

    # Classification report
    logger.info("\nClassification Report:")
    logger.info("\n" + classification_report(
        y_test, y_pred_test,
        target_names=['LONG', 'SHORT'],
        digits=4
    ))

    # Feature importance
    logger.info("\nTop 20 Features Mais Importantes:")
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)

    for i, row in feature_importance.head(20).iterrows():
        logger.info(f"  {row['feature']:<30} {row['importance']:>10.0f}")

    # ========== ETAPA 9: SALVAR MODELO ==========
    print_section("ETAPA 9: Salvando Modelo")

    model_dir = Path(config.get('model', {}).get('output_dir', 'models'))
    ensure_dir(model_dir)

    model_name = f"{args.symbol.lower()}_{args.timeframe}_lightgbm"
    model_path = model_dir / f"{model_name}.txt"

    # Salvar modelo LightGBM
    model.save_model(str(model_path))

    # Salvar scaler
    scaler_path = model_dir / f"{model_name}_scaler.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    # Salvar metadata
    metadata = {
        'symbol': args.symbol,
        'timeframe': args.timeframe,
        'n_features': X_train_scaled.shape[1],
        'feature_cols': feature_cols,
        'label_mapping': {'LONG': 0, 'SHORT': 1},
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'params': params,
        'best_iteration': int(model.best_iteration),
        'train_metrics': {
            'auc': float(train_auc)
        },
        'val_metrics': {
            'auc': float(val_auc)
        },
        'test_metrics': {
            'auc': float(test_auc),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        },
        'feature_importance': feature_importance.head(50).to_dict('records')
    }

    metadata_path = model_dir / f"{model_name}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=float)

    logger.info(f"✓ Modelo LightGBM salvo: {model_path}")
    logger.info(f"✓ Scaler salvo: {scaler_path}")
    logger.info(f"✓ Metadata salvo: {metadata_path}")

    # ========== RESUMO FINAL ==========
    print_section("RESUMO DO TREINAMENTO LIGHTGBM")

    logger.info(f"Símbolo: {args.symbol}")
    logger.info(f"Timeframe: {args.timeframe}")
    logger.info(f"Algoritmo: LightGBM (Leaf-wise GBDT)")
    logger.info(f"Features: {X_train_scaled.shape[1]} (incluindo microestrutura)")
    logger.info(f"Árvores: {model.best_iteration}")
    logger.info("")
    logger.info("Dados:")
    logger.info(f"  Train: {len(X_train)} samples")
    logger.info(f"  Val:   {len(X_val)} samples")
    logger.info(f"  Test:  {len(X_test)} samples")
    logger.info("")
    logger.info("Resultados Finais:")
    logger.info(f"  Train ROC AUC: {train_auc:.4f}")
    logger.info(f"  Val ROC AUC:   {val_auc:.4f}")
    logger.info(f"  Test ROC AUC:  {test_auc:.4f} {'✅' if test_auc > 0.55 else '❌'}")
    logger.info("")

    if test_auc > 0.55:
        logger.info("🎉 ROC AUC > 0.55 - Modelo tem edge competitivo!")
        if test_auc > val_auc + 0.02:
            logger.info("⚠️  Possível overfitting (test > val)")
        else:
            logger.info("Próximo passo: Criar Ensemble com Deep Learning")
    elif test_auc > 0.52:
        logger.info("🟡 ROC AUC > 0.52 - Modelo marginal, considere mais tuning")
    else:
        logger.info("❌ ROC AUC < 0.52 - Modelo sem edge, não use em produção")

    print_section("TREINAMENTO LIGHTGBM CONCLUÍDO!")


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
