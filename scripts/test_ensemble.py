"""
Script de Teste do Ensemble DL + LightGBM
Combina predições dos dois modelos para melhorar performance
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from loguru import logger
import pickle
import lightgbm as lgb
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_fscore_support

from core.utils import load_config, setup_logging, print_section
from core.feature_engineering import FeatureEngineer
from core.microstructure_features import MicrostructureFeatures
from core.labeling import TradingLabeler
from core.deep_learning_model import DeepScalpingModel


def main():
    config = load_config('config/config_deep_learning.yaml')
    setup_logging(config)

    print_section("ENSEMBLE: Deep Learning + LightGBM")

    # ========== CARREGAR DADOS ==========
    print_section("ETAPA 1: Carregando Dados")

    data_dir = Path(config['data']['raw_data_dir'])
    filepath = data_dir / "btcusdt_15m.csv"

    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    logger.info(f"✓ Dados carregados: {len(df)} linhas")

    # Features
    engineer = FeatureEngineer(config)
    df = engineer.create_all_features(df)

    micro_engineer = MicrostructureFeatures(config)
    df = micro_engineer.create_all_features(df)

    # Labels
    labeler = TradingLabeler(config)
    df = labeler.create_labels(df)

    df_clean = df.dropna()
    df_binary = df_clean[df_clean['target_class'] != 'NONE'].copy()

    # Split
    df_train, df_val, df_test = labeler.split_temporal(df_binary)
    logger.info(f"Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

    # Preparar dados
    exclude_cols = config['dataset']['exclude_features']
    feature_cols = [col for col in df_binary.columns if col not in exclude_cols]

    X_test = df_test[feature_cols].values
    y_test = df_test['target_class'].map({'LONG': 0, 'SHORT': 1}).values

    logger.info(f"Test set: X={X_test.shape}, y={y_test.shape}")

    # ========== CARREGAR MODELOS ==========
    print_section("ETAPA 2: Carregando Modelos")

    # Deep Learning
    logger.info("Carregando Deep Learning...")
    dl_model = DeepScalpingModel(config)
    dl_model.load('models/btcusdt_15m_deep_learning')

    # Normalizar para DL (usa scaler próprio)
    scaler_dl = RobustScaler()
    # Re-criar o scaler (idealmente seria carregado do treino)
    X_train = df_train[feature_cols].values
    scaler_dl.fit(X_train)
    X_test_dl = scaler_dl.transform(X_test)

    logger.info("✓ Deep Learning carregado")

    # LightGBM
    logger.info("Carregando LightGBM...")
    lgb_model = lgb.Booster(model_file='models/btcusdt_15m_lightgbm.txt')

    # Carregar scaler do LightGBM
    with open('models/btcusdt_15m_lightgbm_scaler.pkl', 'rb') as f:
        scaler_lgb = pickle.load(f)

    X_test_lgb = scaler_lgb.transform(X_test)
    logger.info("✓ LightGBM carregado")

    # ========== PREDIÇÕES INDIVIDUAIS ==========
    print_section("ETAPA 3: Predições Individuais")

    # DL (precisa de sequences)
    logger.info("Gerando predições DL...")
    sequence_length = 20
    dl_predictions_full = dl_model.predict(X_test_dl, sequence_length=sequence_length)
    logger.info(f"DL: {len(dl_predictions_full)} predições (após sequencing)")

    # Ajustar y_test para match
    y_test_seq = y_test[sequence_length:]

    # LightGBM (usa todos os dados)
    logger.info("Gerando predições LightGBM...")
    lgb_predictions_full = lgb_model.predict(X_test_lgb, num_iteration=lgb_model.best_iteration)
    # Ajustar para match com DL
    lgb_predictions = lgb_predictions_full[sequence_length:]
    logger.info(f"LGB: {len(lgb_predictions)} predições (após align)")

    dl_predictions = dl_predictions_full

    # Avaliar individuais
    dl_auc = roc_auc_score(y_test_seq, dl_predictions)
    lgb_auc = roc_auc_score(y_test_seq, lgb_predictions)

    logger.info(f"DL ROC AUC:  {dl_auc:.4f}")
    logger.info(f"LGB ROC AUC: {lgb_auc:.4f}")

    # ========== ENSEMBLE ==========
    print_section("ETAPA 4: Ensemble (Weighted Average)")

    # Testar diferentes pesos
    weight_combinations = [
        (0.5, 0.5),  # Igual
        (0.6, 0.4),  # Mais DL
        (0.7, 0.3),  # Ainda mais DL
        (0.4, 0.6),  # Mais LGB
        (0.3, 0.7),  # Ainda mais LGB
    ]

    logger.info("Testando diferentes pesos:")
    best_auc = 0
    best_weights = None
    best_ensemble_preds = None

    for dl_weight, lgb_weight in weight_combinations:
        ensemble_preds = dl_weight * dl_predictions + lgb_weight * lgb_predictions
        ensemble_auc = roc_auc_score(y_test_seq, ensemble_preds)

        logger.info(f"  DL={dl_weight:.1f}, LGB={lgb_weight:.1f}: ROC AUC = {ensemble_auc:.4f}")

        if ensemble_auc > best_auc:
            best_auc = ensemble_auc
            best_weights = (dl_weight, lgb_weight)
            best_ensemble_preds = ensemble_preds

    # ========== RESULTADOS FINAIS ==========
    print_section("RESULTADOS FINAIS")

    logger.info("Comparação de Performance:")
    logger.info(f"  Deep Learning:    {dl_auc:.4f}")
    logger.info(f"  LightGBM:         {lgb_auc:.4f}")
    logger.info(f"  Ensemble (best):  {best_auc:.4f}")
    logger.info(f"  Melhor peso: DL={best_weights[0]:.1f}, LGB={best_weights[1]:.1f}")
    logger.info("")

    improvement_vs_dl = ((best_auc - dl_auc) / dl_auc) * 100
    improvement_vs_lgb = ((best_auc - lgb_auc) / lgb_auc) * 100

    logger.info(f"Melhoria sobre DL:  {improvement_vs_dl:+.2f}%")
    logger.info(f"Melhoria sobre LGB: {improvement_vs_lgb:+.2f}%")

    # Predições binárias
    y_pred_dl = (dl_predictions >= 0.5).astype(int)
    y_pred_lgb = (lgb_predictions >= 0.5).astype(int)
    y_pred_ensemble = (best_ensemble_preds >= 0.5).astype(int)

    # Métricas detalhadas
    logger.info("\nMétricas Detalhadas (threshold=0.5):")

    for name, y_pred in [("DL", y_pred_dl), ("LGB", y_pred_lgb), ("Ensemble", y_pred_ensemble)]:
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test_seq, y_pred, average='binary'
        )
        accuracy = (y_pred == y_test_seq).mean()

        logger.info(f"\n{name}:")
        logger.info(f"  Accuracy:  {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall:    {recall:.4f}")
        logger.info(f"  F1 Score:  {f1:.4f}")

    # ========== CONCLUSÃO ==========
    print_section("CONCLUSÃO")

    if best_auc > max(dl_auc, lgb_auc):
        logger.info("✅ Ensemble MELHOROU a performance!")
        logger.info(f"   Ganho absoluto: {best_auc - max(dl_auc, lgb_auc):.4f}")
        logger.info(f"   Ganho relativo: {improvement_vs_dl:.2f}%")
        logger.info("")
        logger.info("Próximo passo: Backtesting com ensemble")
    else:
        logger.info("⚠️  Ensemble não melhorou significativamente")
        logger.info("   Considere:")
        logger.info("   - Stacking com meta-learner")
        logger.info("   - Max confidence selection")
        logger.info("   - Calibração de probabilidades")

    print_section("TESTE DE ENSEMBLE CONCLUÍDO!")


if __name__ == "__main__":
    main()
