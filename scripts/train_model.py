#!/usr/bin/env python3
"""
Script CLI para treinamento de modelos de ML.

Uso:
    python scripts/train_model.py --symbol BTCUSDT --timeframe 5m
    python scripts/train_model.py --symbol BTCUSDT --timeframe 5m --model-name my_model
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Adicionar diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.utils import load_config, setup_logging, print_section, validate_config
from core.data_loader import BybitDataLoader
from core.feature_engineering import FeatureEngineer
from core.labeling import TradingLabeler
from core.model_trainer import ModelTrainer
from loguru import logger


def parse_args():
    """Parse argumentos da linha de comando."""
    parser = argparse.ArgumentParser(
        description='Treinamento de modelo de ML para trading',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  # Treinar modelo para BTC 5m
  python scripts/train_model.py --symbol BTCUSDT --timeframe 5m

  # Treinar com nome customizado
  python scripts/train_model.py --symbol ETHUSDT --timeframe 1m --model-name eth_1m_model

  # Treinar sem validação
  python scripts/train_model.py --symbol BTCUSDT --timeframe 5m --no-validation

  # Usar dados já processados (pular feature engineering)
  python scripts/train_model.py --symbol BTCUSDT --timeframe 5m --use-processed
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Caminho para arquivo de configuração (default: config/config.yaml)'
    )

    parser.add_argument(
        '--symbol',
        type=str,
        required=True,
        help='Símbolo do ativo (ex: BTCUSDT, ETHUSDT)'
    )

    parser.add_argument(
        '--timeframe',
        type=str,
        required=True,
        help='Timeframe (ex: 1m, 5m, 15m)'
    )

    parser.add_argument(
        '--model-name',
        type=str,
        help='Nome do modelo a ser salvo (default: auto-gerado com timestamp)'
    )

    parser.add_argument(
        '--use-processed',
        action='store_true',
        help='Usar dados já processados (pular download e feature engineering)'
    )

    parser.add_argument(
        '--no-validation',
        action='store_true',
        help='Não usar conjunto de validação'
    )

    parser.add_argument(
        '--download',
        action='store_true',
        help='Forçar download de novos dados (mesmo se já existirem)'
    )

    return parser.parse_args()


def load_or_download_data(
    loader: BybitDataLoader,
    symbol: str,
    timeframe: str,
    config: dict,
    force_download: bool = False
):
    """
    Carrega dados existentes ou faz download.
    """
    # Limpar símbolo
    clean_symbol = symbol.replace(':USDT', '').replace('/', '')

    # Tentar carregar dados existentes
    if not force_download:
        try:
            logger.info(f"Tentando carregar dados existentes...")
            df = loader.load_data(clean_symbol, timeframe, data_type="raw")
            logger.info(f"✓ Dados carregados: {len(df)} linhas")
            return df
        except FileNotFoundError:
            logger.info("Dados não encontrados localmente. Fazendo download...")

    # Download
    lookback_days = config['data']['lookback_days']
    logger.info(f"Baixando {symbol} {timeframe} - últimos {lookback_days} dias...")

    df = loader.download_ohlcv(symbol, timeframe, lookback_days=lookback_days)

    if df.empty:
        raise ValueError(f"Nenhum dado baixado para {symbol} {timeframe}")

    # Salvar
    loader.save_data(df, symbol, timeframe, data_type="raw")

    return df


def main():
    """Função principal."""
    args = parse_args()

    # Carregar configuração
    try:
        config = load_config(args.config)
        validate_config(config)
    except FileNotFoundError:
        print(f"Erro: Arquivo de configuração não encontrado: {args.config}")
        sys.exit(1)
    except ValueError as e:
        print(f"Erro na configuração: {e}")
        sys.exit(1)

    # Setup logging
    setup_logging(config)

    print_section("Pipeline de Treinamento de ML")
    logger.info(f"Símbolo: {args.symbol}")
    logger.info(f"Timeframe: {args.timeframe}")
    logger.info(f"Configuração: {args.config}")

    try:
        # === ETAPA 1: CARREGAR/BAIXAR DADOS ===
        print_section("ETAPA 1: Carregamento de Dados")

        loader = BybitDataLoader(config)
        df = load_or_download_data(
            loader,
            args.symbol,
            args.timeframe,
            config,
            force_download=args.download
        )

        logger.info(f"Dataset inicial: {df.shape}")
        logger.info(f"Período: {df['timestamp'].min()} a {df['timestamp'].max()}")

        # === ETAPA 2: FEATURE ENGINEERING ===
        print_section("ETAPA 2: Feature Engineering")

        engineer = FeatureEngineer(config)
        df = engineer.create_all_features(df)

        logger.info(f"Dataset com features: {df.shape}")
        logger.info(f"Total de features: {len(engineer.get_feature_names(df))}")

        # === ETAPA 3: LABELING ===
        print_section("ETAPA 3: Criação de Labels")

        labeler = TradingLabeler(config)
        df = labeler.create_labels(df)

        logger.info(f"Dataset com labels: {df.shape}")

        # Distribuição de labels
        label_dist = df['target_class'].value_counts()
        logger.info("Distribuição de labels:")
        for label, count in label_dist.items():
            pct = 100 * count / len(df)
            logger.info(f"  {label}: {count} ({pct:.2f}%)")

        # Remover NaNs
        if config['dataset'].get('drop_na', True):
            before = len(df)
            df = df.dropna()
            after = len(df)
            logger.info(f"NaNs removidos: {before - after} linhas")
            logger.info(f"Dataset final: {after} linhas")

        # === ETAPA 4: DIVISÃO TEMPORAL ===
        print_section("ETAPA 4: Divisão do Dataset")

        df_train, df_val, df_test = labeler.split_temporal(
            df,
            train_split=config['dataset']['train_split'],
            val_split=config['dataset']['val_split'],
            test_split=config['dataset']['test_split']
        )

        # === ETAPA 5: TREINAMENTO ===
        print_section("ETAPA 5: Treinamento do Modelo")

        trainer = ModelTrainer(config)

        # Treinar com ou sem validação
        if args.no_validation:
            logger.info("Treinando sem conjunto de validação")
            results = trainer.train(df_train, df_val=None)
        else:
            logger.info("Treinando com conjunto de validação")
            results = trainer.train(df_train, df_val=df_val)

        # === ETAPA 6: AVALIAÇÃO NO TESTE ===
        print_section("ETAPA 6: Avaliação no Conjunto de Teste")

        # Preparar dados de teste
        X_test, _ = trainer.prepare_features(df_test, fit_scaler=False)
        y_test = trainer.prepare_labels(df_test, X_test.index)

        # Avaliar
        test_metrics = trainer.evaluate(X_test, y_test)
        results['test_metrics'] = test_metrics

        # === ETAPA 7: SALVAR MODELO ===
        print_section("ETAPA 7: Salvando Modelo")

        # Gerar nome do modelo
        if args.model_name:
            model_name = args.model_name
        else:
            # Auto-gerar: symbol_timeframe_timestamp
            clean_symbol = args.symbol.replace(':USDT', '').replace('/', '').lower()
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_name = f"{clean_symbol}_{args.timeframe}_{timestamp}"

        model_path = trainer.save_model(model_name)

        # === RESUMO FINAL ===
        print_section("RESUMO DO TREINAMENTO")

        logger.info(f"Símbolo: {args.symbol}")
        logger.info(f"Timeframe: {args.timeframe}")
        logger.info(f"Modelo: {config['model']['type']}")
        logger.info(f"\nDados:")
        logger.info(f"  Train: {len(df_train)} linhas")
        logger.info(f"  Val:   {len(df_val)} linhas")
        logger.info(f"  Test:  {len(df_test)} linhas")
        logger.info(f"\nMétricas de Teste:")
        logger.info(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
        logger.info(f"  F1 (macro): {test_metrics['f1_macro']:.4f}")
        logger.info(f"  Precision: {test_metrics['precision_macro']:.4f}")
        logger.info(f"  Recall:    {test_metrics['recall_macro']:.4f}")

        if test_metrics['roc_auc'] is not None:
            logger.info(f"  ROC AUC:   {test_metrics['roc_auc']:.4f}")

        logger.info(f"\nModelo salvo em: {model_path}")

        print_section("TREINAMENTO CONCLUÍDO COM SUCESSO!")

    except Exception as e:
        logger.error(f"Erro durante o treinamento: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
