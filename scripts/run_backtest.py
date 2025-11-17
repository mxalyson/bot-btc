"""
Script Completo de Backtesting
Testa modelo Deep Learning com backtesting real

Etapas:
1. Carrega dados
2. Carrega modelo treinado
3. Gera predições
4. Executa backtesting
5. Analisa resultados
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import argparse
from loguru import logger

from core.utils import load_config, setup_logging, print_section
from core.backtesting import Backtester
from core.deep_learning_model import DeepScalpingModel
from core.feature_engineering import FeatureEngineer
from core.microstructure_features import MicrostructureFeatures
from core.labeling import TradingLabeler
from sklearn.preprocessing import RobustScaler


def main():
    parser = argparse.ArgumentParser(description="Backtesting de modelos")
    parser.add_argument('--symbol', type=str, default='BTCUSDT')
    parser.add_argument('--timeframe', type=str, default='15m')
    parser.add_argument('--model-path', type=str, default='models/btcusdt_15m_deep_learning')
    parser.add_argument('--initial-capital', type=float, default=10000.0)
    parser.add_argument('--fee-rate', type=float, default=0.0006)  # 0.06%
    parser.add_argument('--threshold', type=float, default=0.5)

    args = parser.parse_args()

    # Config
    config = load_config('config/config_deep_learning.yaml')
    setup_logging(config)

    print_section("BACKTESTING - Deep Learning Model")

    logger.info(f"Modelo: {args.model_path}")
    logger.info(f"Capital inicial: ${args.initial_capital:,.2f}")
    logger.info(f"Fee rate: {args.fee_rate*100:.2f}%")

    # ========== ETAPA 1: DADOS ==========
    print_section("ETAPA 1: Carregamento de Dados")

    data_dir = Path(config['data']['raw_data_dir'])
    filepath = data_dir / f"{args.symbol.lower()}_{args.timeframe}.csv"

    if not filepath.exists():
        logger.error(f"Dados não encontrados: {filepath}")
        sys.exit(1)

    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    logger.info(f"✓ Dados carregados: {len(df)} linhas")
    logger.info(f"  Período: {df['timestamp'].min()} a {df['timestamp'].max()}")

    # ========== ETAPA 2: FEATURES ==========
    print_section("ETAPA 2: Feature Engineering")

    # Features técnicas
    engineer = FeatureEngineer(config)
    df = engineer.create_all_features(df)

    # Features de microestrutura
    micro_engineer = MicrostructureFeatures(config)
    df = micro_engineer.create_all_features(df)

    logger.info(f"✓ Features criadas: {df.shape[1]} colunas")

    # ========== ETAPA 3: LABELS ==========
    print_section("ETAPA 3: Labels")

    labeler = TradingLabeler(config)
    df = labeler.create_labels(df)

    # Remover NaNs e NONE
    df_clean = df.dropna()
    df_binary = df_clean[df_clean['target_class'] != 'NONE'].copy()

    logger.info(f"✓ Dataset limpo: {len(df_binary)} linhas")

    # ========== ETAPA 4: PREPARAÇÃO ==========
    print_section("ETAPA 4: Preparação para Predição")

    exclude_cols = config['dataset']['exclude_features']
    feature_cols = [col for col in df_binary.columns if col not in exclude_cols]

    X = df_binary[feature_cols].values
    y = df_binary['target_class'].map({'LONG': 0, 'SHORT': 1}).values

    # Normalizar
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    logger.info(f"✓ Features normalizadas: {X_scaled.shape}")

    # ========== ETAPA 5: CARREGAR MODELO ==========
    print_section("ETAPA 5: Carregando Modelo")

    model = DeepScalpingModel(config)
    model.load(args.model_path)

    logger.info(f"✓ Modelo carregado de: {args.model_path}")

    # ========== ETAPA 6: PREDIÇÕES ==========
    print_section("ETAPA 6: Gerando Predições")

    sequence_length = 20
    predictions = model.predict(X_scaled, sequence_length=sequence_length)

    logger.info(f"✓ Predições geradas: {len(predictions)}")

    # ========== ETAPA 7: BACKTESTING ==========
    print_section("ETAPA 7: Backtesting")

    # Ajustar df para match com predições
    df_backtest = df_binary.iloc[sequence_length:].reset_index(drop=True)
    y_backtest = y[sequence_length:]

    backtester = Backtester(
        initial_capital=args.initial_capital,
        fee_rate=args.fee_rate,
        slippage=0.0001,
        position_size=1.0
    )

    metrics = backtester.run(
        df=df_backtest,
        predictions=predictions,
        actual_labels=y_backtest,
        threshold=args.threshold
    )

    # ========== ETAPA 8: RESULTADOS ==========
    print_section("ETAPA 8: Análise de Resultados")

    backtester.print_report(metrics)

    # Salvar métricas
    import json
    output_file = Path('backtest_results.json')
    with open(output_file, 'w') as f:
        # Converter numpy types para Python types
        metrics_json = {k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                        for k, v in metrics.items()}
        json.dump(metrics_json, f, indent=2)

    logger.info(f"✓ Resultados salvos em: {output_file}")

    print_section("BACKTESTING CONCLUÍDO!")


if __name__ == "__main__":
    main()
