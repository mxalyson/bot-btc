"""
Script de Treinamento Avançado para Scalping
Otimizado para máxima performance em trading de alta frequência
"""

import sys
import argparse
from pathlib import Path

# Adicionar diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from loguru import logger
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, f1_score

from core.utils import (
    load_config, setup_logging, print_section,
    ensure_dir, save_model_metadata
)
from core.data_loader import BybitDataLoader
from core.feature_engineering import FeatureEngineer
from core.labeling import TradingLabeler
from core.model_trainer import ModelTrainer


class AdvancedTrainer:
    """Treinamento avançado com otimizações para scalping"""

    def __init__(self, config):
        self.config = config
        self.trainer = ModelTrainer(config)

    def convert_to_binary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Converte problema de 3 classes para binário (LONG vs SHORT).
        Remove classe NONE que tem pouquíssimas amostras.
        """
        logger.info("Convertendo para problema binário (LONG vs SHORT)...")

        # Contar classes
        none_count = (df['target_class'] == 'NONE').sum()
        total = len(df)

        # Remover NONE
        df_binary = df[df['target_class'] != 'NONE'].copy()

        logger.info(f"✓ Removidas {none_count} amostras NONE ({100*none_count/total:.2f}%)")
        logger.info(f"✓ Dataset binário: {len(df_binary)} amostras")

        # Atualizar label mapping para binário
        self.trainer.label_mapping = {'LONG': 0, 'SHORT': 1}
        self.trainer.inverse_label_mapping = {0: 'LONG', 1: 'SHORT'}

        # Atualizar config do modelo para 2 classes
        self.config['model']['xgboost']['num_class'] = 2
        self.config['model']['xgboost']['objective'] = 'binary:logistic'

        return df_binary

    def select_features(self, df_train: pd.DataFrame, df_val: pd.DataFrame,
                       df_test: pd.DataFrame, top_n: int = 40) -> tuple:
        """
        Seleciona as top N features mais importantes.
        """
        logger.info(f"Selecionando top {top_n} features...")

        # Treinar modelo rápido para obter importâncias
        temp_config = self.config.copy()
        temp_config['model']['xgboost']['n_estimators'] = 50
        temp_trainer = ModelTrainer(temp_config)

        # Treinar
        temp_trainer.train(df_train, df_val)

        # Obter importâncias
        importance_df = temp_trainer.get_feature_importance(top_n=100)

        # Selecionar top N
        top_features = importance_df.head(top_n)['feature'].tolist()

        logger.info(f"✓ Top {top_n} features selecionadas")

        # Atualizar exclude_features no config
        all_features = [col for col in df_train.columns
                       if col not in self.config['dataset']['exclude_features']]

        features_to_exclude = [f for f in all_features if f not in top_features]

        # Adicionar às exclusões (mantendo as originais)
        original_exclude = self.config['dataset']['exclude_features']
        self.config['dataset']['exclude_features'] = original_exclude + features_to_exclude

        logger.info(f"Features mantidas: {len(top_features)}")
        logger.info(f"Top 10: {', '.join(top_features[:10])}")

        return df_train, df_val, df_test

    def optimize_threshold(self, trainer, X_val, y_val):
        """
        Otimiza threshold de decisão para maximizar F1 score.
        """
        logger.info("Otimizando threshold de decisão...")

        # Obter probabilidades
        y_proba = trainer.model.predict_proba(X_val)[:, 1]

        # Testar diferentes thresholds
        best_threshold = 0.5
        best_f1 = 0

        for threshold in np.arange(0.3, 0.7, 0.05):
            y_pred = (y_proba >= threshold).astype(int)
            f1 = f1_score(y_val, y_pred)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        logger.info(f"✓ Melhor threshold: {best_threshold:.2f} (F1: {best_f1:.4f})")

        return best_threshold

    def train_ensemble(self, df_train, df_val, df_test):
        """
        Treina ensemble de XGBoost + LightGBM.
        """
        logger.info("Treinando ensemble de modelos...")

        # XGBoost
        logger.info("=" * 80)
        logger.info("Treinando XGBoost...")
        logger.info("=" * 80)

        config_xgb = self.config.copy()
        config_xgb['model']['type'] = 'xgboost'
        trainer_xgb = ModelTrainer(config_xgb)

        results_xgb = trainer_xgb.train(df_train, df_val)

        # LightGBM (se disponível)
        trainer_lgbm = None
        try:
            import lightgbm as lgb

            logger.info("=" * 80)
            logger.info("Treinando LightGBM...")
            logger.info("=" * 80)

            config_lgbm = self.config.copy()
            config_lgbm['model']['type'] = 'lightgbm'
            config_lgbm['model']['lightgbm']['num_class'] = 2
            config_lgbm['model']['lightgbm']['objective'] = 'binary'

            trainer_lgbm = ModelTrainer(config_lgbm)
            results_lgbm = trainer_lgbm.train(df_train, df_val)

        except ImportError:
            logger.warning("LightGBM não disponível - usando apenas XGBoost")

        return trainer_xgb, trainer_lgbm


def main():
    parser = argparse.ArgumentParser(description="Treinamento Avançado para Scalping")
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Símbolo para treinar')
    parser.add_argument('--timeframe', type=str, default='15m', help='Timeframe (5m, 15m, 1h, etc)')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Arquivo de configuração')
    parser.add_argument('--top-features', type=int, default=30, help='Número de features a usar')

    args = parser.parse_args()

    # Carregar configuração
    config = load_config(args.config)
    setup_logging(config)

    print_section("Pipeline de Treinamento AVANÇADO para Scalping")

    logger.info(f"Símbolo: {args.symbol}")
    logger.info(f"Timeframe: {args.timeframe}")
    logger.info(f"Top features: {args.top_features}")
    logger.info(f"Configuração: {args.config}")

    # ========== ETAPA 1: DADOS ==========
    print_section("ETAPA 1: Carregamento de Dados")

    loader = BybitDataLoader(config)

    # Tentar carregar dados existentes
    try:
        df = loader.load_data(args.symbol, args.timeframe)
        logger.info(f"✓ Dados carregados: {len(df)} linhas")
    except FileNotFoundError:
        logger.info("Dados não encontrados. Baixando...")
        df = loader.download_ohlcv(
            args.symbol,
            args.timeframe,
            lookback_days=config['data']['lookback_days']
        )
        loader.save_data(df, args.symbol, args.timeframe)

    logger.info(f"Dataset inicial: {df.shape}")
    logger.info(f"Período: {df['timestamp'].min()} a {df['timestamp'].max()}")

    # ========== ETAPA 2: FEATURES ==========
    print_section("ETAPA 2: Feature Engineering")

    engineer = FeatureEngineer(config)
    df = engineer.create_all_features(df)

    logger.info(f"Dataset com features: {df.shape}")

    # ========== ETAPA 3: LABELS ==========
    print_section("ETAPA 3: Criação de Labels")

    labeler = TradingLabeler(config)
    df = labeler.create_labels(df)

    logger.info(f"Dataset com labels: {df.shape}")
    logger.info("Distribuição de labels:")
    for label in ['LONG', 'SHORT', 'NONE']:
        count = (df['target_class'] == label).sum()
        pct = 100 * count / len(df)
        logger.info(f"  {label}: {count} ({pct:.2f}%)")

    # Remover NaNs
    df_clean = df.dropna()
    removed = len(df) - len(df_clean)
    logger.info(f"NaNs removidos: {removed} linhas")
    logger.info(f"Dataset final: {len(df_clean)} linhas")

    # ========== ETAPA 4: CONVERSÃO PARA BINÁRIO ==========
    print_section("ETAPA 4: Conversão para Problema Binário")

    trainer_adv = AdvancedTrainer(config)
    df_binary = trainer_adv.convert_to_binary(df_clean)

    logger.info("Distribuição final:")
    for label in ['LONG', 'SHORT']:
        count = (df_binary['target_class'] == label).sum()
        pct = 100 * count / len(df_binary)
        logger.info(f"  {label}: {count} ({pct:.2f}%)")

    # ========== ETAPA 5: SPLIT ==========
    print_section("ETAPA 5: Divisão do Dataset")

    df_train, df_val, df_test = labeler.split_temporal(df_binary)

    # ========== ETAPA 6: FEATURE SELECTION ==========
    print_section("ETAPA 6: Seleção de Features")

    df_train, df_val, df_test = trainer_adv.select_features(
        df_train, df_val, df_test,
        top_n=args.top_features
    )

    # ========== ETAPA 7: TREINAMENTO ==========
    print_section("ETAPA 7: Treinamento do Modelo")

    # Adicionar scale_pos_weight para balancear classes
    long_count = (df_train['target_class'] == 'LONG').sum()
    short_count = (df_train['target_class'] == 'SHORT').sum()
    scale_pos_weight = short_count / long_count

    config['model']['xgboost']['scale_pos_weight'] = scale_pos_weight
    logger.info(f"Class balancing: scale_pos_weight = {scale_pos_weight:.2f}")

    # Treinar modelo final
    trainer_final = ModelTrainer(config)
    results = trainer_final.train(df_train, df_val)

    # ========== ETAPA 8: OTIMIZAÇÃO DE THRESHOLD ==========
    print_section("ETAPA 8: Otimização de Threshold")

    X_val, _ = trainer_final.prepare_features(df_val, fit_scaler=False)
    y_val = trainer_final.prepare_labels(df_val, X_val.index)

    best_threshold = trainer_adv.optimize_threshold(trainer_final, X_val, y_val)

    # ========== ETAPA 9: TESTE ==========
    print_section("ETAPA 9: Avaliação no Conjunto de Teste")

    X_test, _ = trainer_final.prepare_features(df_test, fit_scaler=False)
    y_test = trainer_final.prepare_labels(df_test, X_test.index)

    test_metrics = trainer_final.evaluate(X_test, y_test)

    # ========== ETAPA 10: SALVAR ==========
    print_section("ETAPA 10: Salvando Modelo")

    model_name = f"{args.symbol.lower()}_{args.timeframe}_advanced"
    model_path = trainer_final.save_model(model_name)

    # Salvar threshold otimizado
    metadata_extra = {
        'optimal_threshold': float(best_threshold),
        'top_features': args.top_features,
        'binary_classification': True,
        'scale_pos_weight': float(scale_pos_weight)
    }

    # ========== RESUMO ==========
    print_section("RESUMO DO TREINAMENTO")

    logger.info(f"Símbolo: {args.symbol}")
    logger.info(f"Timeframe: {args.timeframe}")
    logger.info(f"Modelo: XGBoost Binário Otimizado")
    logger.info(f"Features selecionadas: {args.top_features}")
    logger.info(f"Threshold ótimo: {best_threshold:.2f}")
    logger.info("")
    logger.info("Dados:")
    logger.info(f"  Train: {len(df_train)} linhas")
    logger.info(f"  Val:   {len(df_val)} linhas")
    logger.info(f"  Test:  {len(df_test)} linhas")
    logger.info("")
    logger.info("Métricas de Teste:")
    logger.info(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    logger.info(f"  F1 (macro): {test_metrics['f1_macro']:.4f}")
    logger.info(f"  Precision: {test_metrics['precision_macro']:.4f}")
    logger.info(f"  Recall:    {test_metrics['recall_macro']:.4f}")
    if test_metrics['roc_auc']:
        logger.info(f"  ROC AUC:   {test_metrics['roc_auc']:.4f}")
    logger.info("")
    logger.info(f"Modelo salvo em: {model_path}")

    print_section("TREINAMENTO AVANÇADO CONCLUÍDO COM SUCESSO!")


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
