"""
📊 COMPARE BEFORE vs AFTER
Script para comparar performance com e sem filtro de confiança

Gera relatório completo com:
- Métricas lado-a-lado
- Gráficos comparativos
- Análise de melhoria

USO:
    python examples/compare_before_after.py --demo
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import argparse
from typing import Dict

from validation.confidence_filter import ConfidenceFilter
from validation.optimize_confidence_threshold import ThresholdOptimizer


def generate_demo_backtest_data(n_samples: int = 1000):
    """Gera dados simulados de backtest"""

    # Simula probabilidades do modelo
    probas = np.random.beta(2, 2, n_samples)

    # Simula retornos correlacionados com probabilidade
    # Quanto maior a confiança, melhor o retorno esperado
    base_returns = (probas - 0.5) * 0.08  # -4% a +4%
    noise = np.random.normal(0, 0.02, n_samples)
    returns = base_returns + noise

    # Simula regimes
    regimes = np.random.choice(
        ['medium_bear', 'high_vol_bear', 'low_vol_bear', 'high_vol_bull', 'medium_bull', 'low_vol_bull'],
        n_samples
    )

    return probas, returns, regimes


def calculate_metrics(returns: np.ndarray, label: str = "") -> Dict:
    """Calcula métricas de trading"""

    if len(returns) == 0:
        return {
            'label': label,
            'n_trades': 0,
            'win_rate': 0.0,
            'total_return': 0.0,
            'avg_return': 0.0,
            'sharpe': 0.0,
            'max_dd': 0.0,
            'profit_factor': 0.0
        }

    wins = returns > 0
    losses = returns < 0

    # Equity curve
    cumulative = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative - running_max

    # Profit factor
    gross_profit = np.sum(returns[wins]) if np.any(wins) else 0
    gross_loss = abs(np.sum(returns[losses])) if np.any(losses) else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    return {
        'label': label,
        'n_trades': len(returns),
        'win_rate': np.mean(wins),
        'total_return': np.sum(returns),
        'avg_return': np.mean(returns),
        'sharpe': np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252),
        'max_dd': np.min(drawdown),
        'profit_factor': profit_factor
    }


def print_comparison_table(before: Dict, after: Dict):
    """Imprime tabela comparativa"""

    print("\n" + "="*100)
    print("📊 PERFORMANCE COMPARISON")
    print("="*100)

    print(f"\n{'Metric':<20s} {'Before':<20s} {'After':<20s} {'Change':<20s} {'Status':<10s}")
    print("-"*100)

    # Trades
    trades_change = after['n_trades'] - before['n_trades']
    trades_pct = (trades_change / before['n_trades'] * 100) if before['n_trades'] > 0 else 0
    print(f"{'Trades':<20s} {before['n_trades']:<20d} {after['n_trades']:<20d} "
          f"{trades_change:>6d} ({trades_pct:>6.1f}%){'':<5s} {'📉' if trades_change < 0 else '📈':<10s}")

    # Win Rate
    wr_change = (after['win_rate'] - before['win_rate']) * 100
    wr_status = '✅ Better' if wr_change > 0 else '⚠️ Worse' if wr_change < -2 else '➖ Same'
    print(f"{'Win Rate':<20s} {before['win_rate']:<19.1%} {after['win_rate']:<19.1%} "
          f"{wr_change:>+6.1f}pp{'':<8s} {wr_status:<10s}")

    # Total Return
    ret_change = (after['total_return'] - before['total_return']) * 100
    ret_pct = (after['total_return'] / before['total_return'] - 1) * 100 if before['total_return'] != 0 else 0
    ret_status = '✅ Better' if ret_change > 0 else '⚠️ Worse'
    print(f"{'Total Return':<20s} {before['total_return']:<19.1%} {after['total_return']:<19.1%} "
          f"{ret_change:>+6.1f}pp{'':<8s} {ret_status:<10s}")

    # Sharpe
    sharpe_change = after['sharpe'] - before['sharpe']
    sharpe_pct = (sharpe_change / before['sharpe'] * 100) if before['sharpe'] != 0 else 0
    sharpe_status = '✅ Better' if sharpe_change > 0.2 else '⚠️ Worse' if sharpe_change < -0.2 else '➖ Same'
    print(f"{'Sharpe Ratio':<20s} {before['sharpe']:<20.2f} {after['sharpe']:<20.2f} "
          f"{sharpe_change:>+6.2f} ({sharpe_pct:>+6.1f}%){'':<2s} {sharpe_status:<10s}")

    # Max DD
    dd_change = (after['max_dd'] - before['max_dd']) * 100
    dd_status = '✅ Better' if dd_change > 0 else '⚠️ Worse'  # Menor DD negativo é melhor
    print(f"{'Max Drawdown':<20s} {before['max_dd']:<19.1%} {after['max_dd']:<19.1%} "
          f"{dd_change:>+6.1f}pp{'':<8s} {dd_status:<10s}")

    # Profit Factor
    pf_change = after['profit_factor'] - before['profit_factor']
    pf_status = '✅ Better' if pf_change > 0.1 else '⚠️ Worse' if pf_change < -0.1 else '➖ Same'
    print(f"{'Profit Factor':<20s} {before['profit_factor']:<20.2f} {after['profit_factor']:<20.2f} "
          f"{pf_change:>+6.2f}{'':<12s} {pf_status:<10s}")

    print("-"*100)


def print_summary(before: Dict, after: Dict):
    """Imprime resumo da análise"""

    print("\n" + "="*100)
    print("💡 SUMMARY & RECOMMENDATIONS")
    print("="*100)

    # Calcula score de melhoria
    improvements = 0
    total_metrics = 0

    # Win Rate
    if after['win_rate'] > before['win_rate']:
        improvements += 1
    total_metrics += 1

    # Sharpe
    if after['sharpe'] > before['sharpe']:
        improvements += 1
    total_metrics += 1

    # Max DD (menos negativo é melhor)
    if after['max_dd'] > before['max_dd']:
        improvements += 1
    total_metrics += 1

    # Profit Factor
    if after['profit_factor'] > before['profit_factor']:
        improvements += 1
    total_metrics += 1

    improvement_score = improvements / total_metrics

    print(f"\n📊 Improvement Score: {improvement_score:.0%} ({improvements}/{total_metrics} metrics improved)")

    if improvement_score >= 0.75:
        print("\n✅ RECOMMENDATION: DEPLOY CONFIDENCE FILTER")
        print("   The filter significantly improves performance across most metrics.")
        print("   Expected benefits:")
        print(f"   • Win Rate: +{(after['win_rate'] - before['win_rate'])*100:.1f}pp")
        print(f"   • Sharpe: +{(after['sharpe'] - before['sharpe']):.2f}")
        print(f"   • Trade Quality: Higher (fewer low-quality signals)")

    elif improvement_score >= 0.50:
        print("\n⚠️  RECOMMENDATION: TEST IN PAPER TRADING")
        print("   The filter shows mixed results. Test in paper trading first.")
        print("   Consider:")
        print("   • Adjusting threshold (try 0.55, 0.60, 0.65, 0.70)")
        print("   • Testing with different time periods")
        print("   • Analyzing which regimes benefit most")

    else:
        print("\n❌ RECOMMENDATION: DO NOT DEPLOY YET")
        print("   The filter does not improve performance significantly.")
        print("   Possible reasons:")
        print("   • Model probabilities not well-calibrated")
        print("   • Threshold too high/low")
        print("   • Need more data for optimization")
        print("\n   Try:")
        print("   • Re-train model with probability calibration")
        print("   • Use threshold optimizer to find better threshold")
        print("   • Test on longer time period")

    # Trade reduction analysis
    trade_reduction = 1 - (after['n_trades'] / before['n_trades']) if before['n_trades'] > 0 else 0
    print(f"\n📉 Trade Reduction: {trade_reduction:.1%}")

    if trade_reduction > 0.7:
        print("   ⚠️  WARNING: Filtering out >70% of trades")
        print("   This might be too aggressive. Consider:")
        print("   • Lowering threshold (try 0.55 or 0.58)")
        print("   • Checking if model is too conservative")

    elif trade_reduction > 0.5:
        print("   ✅ Good: Filtering ~50-70% of trades")
        print("   This is typical for confidence filtering")
        print("   Focus on quality over quantity")

    else:
        print("   ⚠️  Low filtering (<50% of trades)")
        print("   Filter may not be selective enough. Consider:")
        print("   • Increasing threshold (try 0.65 or 0.68)")
        print("   • Checking if model probabilities are well-distributed")

    print("="*100)


def main():
    """Main function"""

    parser = argparse.ArgumentParser(description='Compare before/after confidence filter')
    parser.add_argument('--demo', action='store_true', help='Use simulated data')
    parser.add_argument('--samples', type=int, default=1000, help='Number of samples (demo mode)')
    parser.add_argument('--threshold', type=float, default=None, help='Confidence threshold (None = auto-optimize)')

    args = parser.parse_args()

    print("\n" + "="*100)
    print("📊 BEFORE vs AFTER COMPARISON - CONFIDENCE FILTER")
    print("="*100)

    # Generate or load data
    if args.demo:
        print(f"\n📊 Generating {args.samples} simulated trades...")
        probas, returns, regimes = generate_demo_backtest_data(args.samples)
        print(f"   ✅ Data generated")
    else:
        print("❌ Real data mode not implemented yet. Use --demo")
        return

    # ========================================================================
    # BEFORE: Without filter
    # ========================================================================

    print("\n" + "="*80)
    print("1️⃣  BEFORE: Without Confidence Filter")
    print("="*80)

    # All trades with proba >= 0.5
    mask_before = probas >= 0.5
    returns_before = returns[mask_before]

    metrics_before = calculate_metrics(returns_before, label="Before")

    print(f"   Trades: {metrics_before['n_trades']}")
    print(f"   Win Rate: {metrics_before['win_rate']:.1%}")
    print(f"   Total Return: {metrics_before['total_return']:.2%}")
    print(f"   Sharpe: {metrics_before['sharpe']:.2f}")
    print(f"   Max DD: {metrics_before['max_dd']:.2%}")
    print(f"   Profit Factor: {metrics_before['profit_factor']:.2f}")

    # ========================================================================
    # OPTIMIZE THRESHOLD (if not provided)
    # ========================================================================

    if args.threshold is None:
        print("\n" + "="*80)
        print("🔍 OPTIMIZING THRESHOLD")
        print("="*80)

        optimizer = ThresholdOptimizer(
            min_threshold=0.50,
            max_threshold=0.80,
            step=0.02,
            min_trades=50
        )

        best_threshold, best_result = optimizer.optimize(
            probas,
            returns,
            objective='sharpe'
        )

        print(f"\n   ✅ Optimal Threshold: {best_threshold:.1%}")
        print(f"      Expected Sharpe: {best_result.sharpe:.2f}")

        threshold = best_threshold
    else:
        threshold = args.threshold
        print(f"\n💡 Using provided threshold: {threshold:.1%}")

    # ========================================================================
    # AFTER: With filter
    # ========================================================================

    print("\n" + "="*80)
    print(f"2️⃣  AFTER: With Confidence Filter (threshold={threshold:.1%})")
    print("="*80)

    # Filter by optimized threshold
    mask_after = probas >= threshold
    returns_after = returns[mask_after]

    metrics_after = calculate_metrics(returns_after, label="After")

    print(f"   Trades: {metrics_after['n_trades']}")
    print(f"   Win Rate: {metrics_after['win_rate']:.1%}")
    print(f"   Total Return: {metrics_after['total_return']:.2%}")
    print(f"   Sharpe: {metrics_after['sharpe']:.2f}")
    print(f"   Max DD: {metrics_after['max_dd']:.2%}")
    print(f"   Profit Factor: {metrics_after['profit_factor']:.2f}")

    # ========================================================================
    # COMPARISON
    # ========================================================================

    print_comparison_table(metrics_before, metrics_after)
    print_summary(metrics_before, metrics_after)

    print("\n✅ COMPARISON COMPLETE\n")


if __name__ == "__main__":
    main()
