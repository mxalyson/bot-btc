"""
Script de Otimização de Parâmetros para Backtesting
Testa múltiplas combinações de threshold e position size
para encontrar configuração ótima
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from loguru import logger
import json

from core.utils import load_config, setup_logging, print_section
from core.backtesting import Backtester
from core.deep_learning_model import DeepScalpingModel
from core.feature_engineering import FeatureEngineer
from core.microstructure_features import MicrostructureFeatures
from core.labeling import TradingLabeler
from sklearn.preprocessing import RobustScaler


def run_single_backtest(
    df_backtest,
    predictions,
    y_backtest,
    threshold,
    position_size,
    initial_capital=10000,
    fee_rate=0.0006
):
    """
    Executa um backtest com parâmetros específicos.
    """
    backtester = Backtester(
        initial_capital=initial_capital,
        fee_rate=fee_rate,
        slippage=0.0001,
        position_size=position_size
    )

    metrics = backtester.run(
        df=df_backtest,
        predictions=predictions,
        actual_labels=y_backtest,
        threshold=threshold
    )

    return metrics


def main():
    logger.info("="*80)
    logger.info("OTIMIZAÇÃO DE PARÂMETROS - BACKTESTING")
    logger.info("="*80)

    # Config
    config = load_config('config/config_deep_learning.yaml')
    setup_logging(config)

    # ========== CARREGAR DADOS E MODELO (igual ao run_backtest.py) ==========
    print_section("Carregando Dados e Modelo")

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

    # Preparar dados
    exclude_cols = config['dataset']['exclude_features']
    feature_cols = [col for col in df_binary.columns if col not in exclude_cols]

    X = df_binary[feature_cols].values
    y = df_binary['target_class'].map({'LONG': 0, 'SHORT': 1}).values

    # Normalizar
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # Carregar modelo
    model = DeepScalpingModel(config)
    model.load('models/btcusdt_15m_deep_learning')
    logger.info("✓ Modelo carregado")

    # Predições
    sequence_length = 20
    predictions = model.predict(X_scaled, sequence_length=sequence_length)

    # Ajustar df para match com predições
    df_backtest = df_binary.iloc[sequence_length:].reset_index(drop=True)
    y_backtest = y[sequence_length:]

    logger.info(f"✓ Setup completo: {len(predictions)} predições")

    # ========== GRID SEARCH ==========
    print_section("GRID SEARCH - Testando Combinações")

    # Parâmetros para testar
    thresholds = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    position_sizes = [0.3, 0.5, 0.7, 1.0]

    results = []
    total_tests = len(thresholds) * len(position_sizes)
    current_test = 0

    for threshold in thresholds:
        for position_size in position_sizes:
            current_test += 1
            logger.info(f"[{current_test}/{total_tests}] Testing threshold={threshold}, position_size={position_size}")

            try:
                metrics = run_single_backtest(
                    df_backtest=df_backtest,
                    predictions=predictions,
                    y_backtest=y_backtest,
                    threshold=threshold,
                    position_size=position_size
                )

                # Adicionar parâmetros ao resultado
                metrics['threshold'] = threshold
                metrics['position_size'] = position_size

                # Calcular score customizado
                # Queremos: ROI positivo, Sharpe alto, drawdown baixo, trades razoáveis
                score = 0

                if metrics.get('total_return', -1) > 0:
                    score += 2
                if metrics.get('sharpe_ratio', 0) > 1.0:
                    score += 2
                if metrics.get('sharpe_ratio', 0) > 2.0:
                    score += 1
                if metrics.get('max_drawdown', -1) > -0.2:
                    score += 2
                if metrics.get('profit_factor', 0) > 1.5:
                    score += 2
                if 100 < metrics.get('total_trades', 0) < 5000:  # Trades razoáveis
                    score += 1

                metrics['optimization_score'] = score

                results.append(metrics)

                logger.info(f"  → Return: {metrics.get('total_return', 0)*100:.2f}%, "
                          f"Sharpe: {metrics.get('sharpe_ratio', 0):.2f}, "
                          f"Trades: {metrics.get('total_trades', 0)}, "
                          f"Score: {score}/10")

            except Exception as e:
                logger.error(f"  → Erro: {e}")
                continue

    # ========== ANÁLISE DE RESULTADOS ==========
    print_section("ANÁLISE DE RESULTADOS")

    if not results:
        logger.error("Nenhum resultado válido!")
        return

    # Ordenar por score
    results_sorted = sorted(results, key=lambda x: x.get('optimization_score', 0), reverse=True)

    # Top 10
    logger.info("🏆 TOP 10 CONFIGURAÇÕES:")
    logger.info("")
    logger.info(f"{'Rank':<5} {'Threshold':<10} {'PosSize':<10} {'Return%':<10} {'Sharpe':<10} {'Trades':<10} {'Score':<10}")
    logger.info("-" * 80)

    for i, res in enumerate(results_sorted[:10], 1):
        logger.info(
            f"{i:<5} "
            f"{res['threshold']:<10.2f} "
            f"{res['position_size']:<10.2f} "
            f"{res.get('total_return', 0)*100:<10.2f} "
            f"{res.get('sharpe_ratio', 0):<10.2f} "
            f"{res.get('total_trades', 0):<10} "
            f"{res.get('optimization_score', 0):<10}"
        )

    # Melhor configuração
    best = results_sorted[0]

    print_section("MELHOR CONFIGURAÇÃO ENCONTRADA")

    logger.info(f"Threshold: {best['threshold']}")
    logger.info(f"Position Size: {best['position_size']}")
    logger.info("")
    logger.info(f"Retorno Total: {best.get('total_return', 0)*100:.2f}%")
    logger.info(f"Retorno Anualizado: {best.get('annualized_return', 0)*100:.2f}%")
    logger.info(f"Sharpe Ratio: {best.get('sharpe_ratio', 0):.2f}")
    logger.info(f"Max Drawdown: {best.get('max_drawdown', 0)*100:.2f}%")
    logger.info(f"Win Rate: {best.get('win_rate', 0)*100:.2f}%")
    logger.info(f"Profit Factor: {best.get('profit_factor', 0):.2f}")
    logger.info(f"Total Trades: {best.get('total_trades', 0)}")
    logger.info(f"Total Fees: ${best.get('total_fees', 0):,.2f}")
    logger.info("")
    logger.info(f"Score de Otimização: {best.get('optimization_score', 0)}/10")

    # Salvar resultados
    output_file = Path('optimization_results.json')
    with open(output_file, 'w') as f:
        json.dump(results_sorted, f, indent=2, default=float)

    logger.info(f"✓ Resultados salvos em: {output_file}")

    # Salvar melhor configuração separadamente
    best_config_file = Path('best_backtest_config.json')
    with open(best_config_file, 'w') as f:
        json.dump({
            'threshold': best['threshold'],
            'position_size': best['position_size'],
            'metrics': {k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                       for k, v in best.items()}
        }, f, indent=2)

    logger.info(f"✓ Melhor config salva em: {best_config_file}")

    print_section("OTIMIZAÇÃO CONCLUÍDA!")

    if best.get('total_return', -1) > 0:
        logger.info("🎉 Configuração lucrativa encontrada!")
        logger.info("   Próximo passo: Validar com dados reais e preparar para paper trading")
    else:
        logger.info("⚠️  Nenhuma configuração lucrativa encontrada")
        logger.info("   Próximos passos:")
        logger.info("   1. Treinar ensemble DL + LightGBM")
        logger.info("   2. Otimizar features e modelo")
        logger.info("   3. Considerar timeframes maiores (30m, 1h)")


if __name__ == "__main__":
    main()
