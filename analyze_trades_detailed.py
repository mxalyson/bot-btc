"""
ANÁLISE DETALHADA DE TRADES - Multi-Período
Mostra cada trade individual do sistema V1 em diferentes períodos
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
from typing import Dict, List
import argparse
import pickle
import yaml
from datetime import datetime, timedelta

from core.utils import load_config
from core.bybit_rest import BybitRESTClient
from core.data import DataManager
from core.features import FeatureStore

from train_ultra_scalper import (
    create_microstructure_features,
    create_advanced_master_features,
    create_sequences
)

from validate_optimized_ultra_scalper import OptimizedUltraValidator


def analyze_period(validator, df, period_name, days):
    """Analisa um período específico e retorna trades detalhados."""

    print("\n" + "=" * 80)
    print(f"📊 ANÁLISE: {period_name} ({days} dias)")
    print("=" * 80)

    # Reset counters
    validator.blocked_trades_count = 0
    validator.blocked_by_regime = {}

    # Run backtest
    stats = validator.backtest_optimized(df)

    if stats['total_trades'] == 0:
        print("⚠️  Nenhum trade neste período")
        return None

    # Get trades DataFrame
    trades_df = stats['trades']

    # Summary
    print(f"\n📈 RESUMO:")
    print(f"   Trades: {stats['total_trades']}")
    print(f"   Win Rate: {stats['win_rate']*100:.1f}%")
    print(f"   ROI: {stats['roi']:+.2f}%")
    print(f"   Sharpe: {stats['sharpe_ratio']:.2f}")
    print(f"   Max DD: {stats['max_drawdown']:.1f}%")
    print(f"   Profit Factor: {stats['profit_factor']:.2f}")
    print(f"   Bloqueados: {stats['blocked_trades']}")

    # Top 10 best trades
    print(f"\n🏆 TOP 10 MELHORES TRADES:")
    top_trades = trades_df.nlargest(10, 'pnl_pct')
    for idx, trade in top_trades.iterrows():
        print(f"   {trade['entry_time']} | {trade['direction'].upper():5} | "
              f"Entry: ${trade['entry_price']:,.2f} | Exit: ${trade['exit_price']:,.2f} | "
              f"P&L: {trade['pnl_pct']:+.2f}% | {trade['exit_reason']}")

    # Bottom 10 worst trades
    print(f"\n❌ TOP 10 PIORES TRADES:")
    worst_trades = trades_df.nsmallest(10, 'pnl_pct')
    for idx, trade in worst_trades.iterrows():
        print(f"   {trade['entry_time']} | {trade['direction'].upper():5} | "
              f"Entry: ${trade['entry_price']:,.2f} | Exit: ${trade['exit_price']:,.2f} | "
              f"P&L: {trade['pnl_pct']:+.2f}% | {trade['exit_reason']}")

    # Performance by regime
    print(f"\n🎯 PERFORMANCE POR REGIME:")
    regime_stats = trades_df.groupby('regime').agg({
        'pnl_pct': ['count', 'mean', lambda x: (x > 0).mean()],
        'pnl_amount': 'sum'
    }).round(2)
    regime_stats.columns = ['Trades', 'Avg P&L%', 'Win Rate', 'Total P&L']
    print(regime_stats.to_string())

    # Performance by exit reason
    print(f"\n🚪 PERFORMANCE POR SAÍDA:")
    exit_stats = trades_df.groupby('exit_reason').agg({
        'pnl_pct': ['count', 'mean', lambda x: (x > 0).mean()]
    }).round(2)
    exit_stats.columns = ['Trades', 'Avg P&L%', 'Win Rate']
    print(exit_stats.to_string())

    # Confidence analysis
    print(f"\n🎲 ANÁLISE POR CONFIANÇA:")
    trades_df['confidence_bucket'] = pd.cut(
        trades_df['ml_confidence'],
        bins=[0, 0.3, 0.5, 0.7, 1.0],
        labels=['Low (0-30%)', 'Med (30-50%)', 'High (50-70%)', 'Ultra (70%+)']
    )
    conf_stats = trades_df.groupby('confidence_bucket').agg({
        'pnl_pct': ['count', 'mean', lambda x: (x > 0).mean()]
    }).round(2)
    conf_stats.columns = ['Trades', 'Avg P&L%', 'Win Rate']
    print(conf_stats.to_string())

    # Daily performance
    print(f"\n📅 PERFORMANCE DIÁRIA:")
    trades_df['date'] = pd.to_datetime(trades_df['entry_time']).dt.date
    daily_stats = trades_df.groupby('date').agg({
        'pnl_amount': ['count', 'sum'],
        'pnl_pct': lambda x: (x > 0).mean()
    }).round(2)
    daily_stats.columns = ['Trades', 'P&L $', 'Win Rate']
    daily_stats['Cumulative $'] = daily_stats['P&L $'].cumsum()
    print(daily_stats.to_string())

    return {
        'period': period_name,
        'days': days,
        'stats': stats,
        'trades_df': trades_df
    }


def main():
    parser = argparse.ArgumentParser(description='Análise Detalhada de Trades - Multi-Período')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Trading symbol')
    parser.add_argument('--periods', type=str, default='7,30,90,180,365',
                        help='Períodos para análise (dias, separados por vírgula)')
    parser.add_argument('--initial-capital', type=float, default=10000, help='Initial capital')
    parser.add_argument('--export-csv', action='store_true', help='Exportar trades para CSV')

    args = parser.parse_args()

    # Parse periods
    periods = [int(p) for p in args.periods.split(',')]

    # Load configs
    config = load_config('standard')
    config['initial_capital'] = args.initial_capital

    opt_config_path = Path(__file__).parent / 'config_optimized.yaml'
    with open(opt_config_path, 'r', encoding='utf-8') as f:
        opt_config = yaml.safe_load(f)

    print("=" * 80)
    print("🔍 ANÁLISE DETALHADA DE TRADES - SISTEMA V1 OTIMIZADO")
    print("=" * 80)
    print(f"Symbol: {args.symbol}")
    print(f"Períodos: {periods} dias")
    print(f"Initial Capital: ${args.initial_capital:,.2f}")
    print()

    # Download data (maximum period)
    max_days = max(periods)
    print(f"📥 Downloading {max_days} days of data...")

    rest_client = BybitRESTClient(
        api_key=config['bybit_api_key'],
        api_secret=config['bybit_api_secret'],
        testnet=config['bybit_testnet']
    )

    dm = DataManager(rest_client)
    df_full = dm.get_data(args.symbol, '15m', max_days, use_cache=False)

    if df_full.empty:
        print("❌ No data")
        return

    print(f"✅ Downloaded {len(df_full):,} candles\n")

    # Build features
    print("🔨 Building features...")
    fs = FeatureStore(config)
    df_full = fs.build_features(df_full, normalize=False)
    df_full = create_microstructure_features(df_full)
    df_full = create_advanced_master_features(df_full)
    print(f"✅ Features ready\n")

    # Load model
    model_path = "storage/models/ultra_scalper_btcusdt_365d.pkl"
    validator = OptimizedUltraValidator(config, opt_config, model_path)

    # Analyze each period
    all_results = []

    for days in periods:
        # Get last N days
        df_period = df_full.tail(days * 96).copy()  # 96 candles per day (15m)

        result = analyze_period(validator, df_period, f"Últimos {days} dias", days)

        if result:
            all_results.append(result)

            # Export to CSV if requested
            if args.export_csv:
                csv_filename = f"trades_{args.symbol}_{days}d.csv"
                result['trades_df'].to_csv(csv_filename, index=False)
                print(f"   💾 Exported to {csv_filename}")

    # Comparison summary
    print("\n" + "=" * 80)
    print("📊 COMPARAÇÃO ENTRE PERÍODOS")
    print("=" * 80)

    comparison = []
    for result in all_results:
        stats = result['stats']
        comparison.append({
            'Período': result['period'],
            'Trades': stats['total_trades'],
            'Win Rate %': f"{stats['win_rate']*100:.1f}",
            'ROI %': f"{stats['roi']:+.2f}",
            'Sharpe': f"{stats['sharpe_ratio']:.2f}",
            'Max DD %': f"{stats['max_drawdown']:.1f}",
            'Profit Factor': f"{stats['profit_factor']:.2f}"
        })

    df_comparison = pd.DataFrame(comparison)
    print(df_comparison.to_string(index=False))

    # Overall insights
    print("\n" + "=" * 80)
    print("💡 INSIGHTS")
    print("=" * 80)

    # Check consistency
    win_rates = [r['stats']['win_rate'] for r in all_results]
    rois = [r['stats']['roi'] for r in all_results]

    avg_wr = np.mean(win_rates) * 100
    std_wr = np.std(win_rates) * 100
    avg_roi = np.mean(rois)

    print(f"✅ Win Rate Médio: {avg_wr:.1f}% (±{std_wr:.1f}%)")
    print(f"✅ ROI Médio: {avg_roi:+.2f}%")

    if std_wr < 5:
        print("✅ ALTA CONSISTÊNCIA: Win rate estável entre períodos!")
    elif std_wr < 10:
        print("⚠️  MÉDIA CONSISTÊNCIA: Alguma variação entre períodos")
    else:
        print("❌ BAIXA CONSISTÊNCIA: Grande variação entre períodos")

    # Check if profitable in all periods
    all_profitable = all(r['stats']['roi'] > 0 for r in all_results)
    if all_profitable:
        print("✅ LUCRATIVO EM TODOS OS PERÍODOS!")
    else:
        losing_periods = [r['period'] for r in all_results if r['stats']['roi'] <= 0]
        print(f"⚠️  Períodos com prejuízo: {', '.join(losing_periods)}")

    print("\n" + "=" * 80)
    print("✅ ANÁLISE COMPLETA")
    print("=" * 80)


if __name__ == "__main__":
    main()
