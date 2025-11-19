"""
🔬 GRID SEARCH OPTIMIZER - Automated Parameter Optimization

Testa CENTENAS de combinações de parâmetros automaticamente
e retorna as TOP 10 melhores configurações ranqueadas por ROI!

Parâmetros testados:
- Position sizing por regime
- Confidence thresholds
- Regimes bloqueados
- TP/SL multipliers

Baseado nos 3 bugs corrigidos (FINAL version)
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
import yaml
import pickle
from itertools import product
from datetime import datetime
import json

from core.utils import load_config
from core.bybit_rest import BybitRESTClient
from core.data import DataManager
from core.features import FeatureStore

from train_ultra_scalper import (
    create_microstructure_features,
    create_advanced_master_features,
    create_sequences
)

from validate_ultra_optimized_FINAL import OptimizedUltraValidator


class GridSearchOptimizer:
    """Otimizador automático via grid search."""

    def __init__(self, base_config: dict, df_features: pd.DataFrame, model_path: str):
        self.base_config = base_config
        self.df_features = df_features
        self.model_path = model_path
        self.results = []

    def generate_configs(self):
        """Gera todas as combinações de configurações para testar."""

        print("🔬 Gerando configurações para grid search...")

        # Parâmetros a testar
        position_mults = {
            'conservative': [1.0, 1.3, 1.5],
            'moderate': [1.5, 1.8, 2.0],
            'aggressive': [2.0, 2.5, 3.0]
        }

        confidence_thresholds = {
            'low': [0.35, 0.38, 0.40],
            'medium': [0.40, 0.45, 0.50],
            'high': [0.50, 0.55, 0.60]
        }

        # Combinações de regimes bloqueados a testar
        regime_blocks = [
            # Apenas os claramente ruins
            {'blocked': ['medium_bull', 'low_vol_bull']},

            # Adicionar low_vol_bear
            {'blocked': ['medium_bull', 'low_vol_bull', 'low_vol_bear']},

            # Sem bloquear nada (baseline)
            {'blocked': []},

            # Bloquear apenas low_vol_bull (o pior)
            {'blocked': ['low_vol_bull']},
        ]

        configs = []

        # Strategy 1: Conservative (baseline similar ao FINAL)
        for conf_level in ['low', 'medium']:
            for regime_block in regime_blocks[:2]:  # Apenas os 2 primeiros
                config = self._create_config(
                    name=f"Conservative_{conf_level}_block{len(regime_block['blocked'])}",
                    high_vol_bear_mult=1.3,
                    medium_bear_mult=1.5,
                    high_vol_bull_mult=0.8,
                    high_vol_bear_conf=confidence_thresholds[conf_level][1],
                    medium_bear_conf=confidence_thresholds[conf_level][0],
                    blocked_regimes=regime_block['blocked']
                )
                configs.append(config)

        # Strategy 2: Moderate (aumenta position sizing levemente)
        for conf_level in ['low', 'medium']:
            for regime_block in regime_blocks[:2]:
                config = self._create_config(
                    name=f"Moderate_{conf_level}_block{len(regime_block['blocked'])}",
                    high_vol_bear_mult=1.6,
                    medium_bear_mult=1.8,
                    high_vol_bull_mult=1.1,
                    high_vol_bear_conf=confidence_thresholds[conf_level][1],
                    medium_bear_conf=confidence_thresholds[conf_level][0],
                    blocked_regimes=regime_block['blocked']
                )
                configs.append(config)

        # Strategy 3: Aggressive (position sizing alto nos melhores)
        for conf_level in ['low']:  # Apenas low confidence para compensar risco
            for regime_block in regime_blocks[:2]:
                config = self._create_config(
                    name=f"Aggressive_{conf_level}_block{len(regime_block['blocked'])}",
                    high_vol_bear_mult=2.0,
                    medium_bear_mult=2.0,
                    high_vol_bull_mult=1.3,
                    high_vol_bear_conf=confidence_thresholds[conf_level][0],  # Mais baixo
                    medium_bear_conf=confidence_thresholds[conf_level][0],
                    blocked_regimes=regime_block['blocked']
                )
                configs.append(config)

        # Strategy 4: Threshold Focus (varia thresholds, size fixo)
        for high_vol_conf, med_conf in product(
            [0.40, 0.45, 0.50],
            [0.35, 0.38, 0.40]
        ):
            config = self._create_config(
                name=f"Threshold_hv{int(high_vol_conf*100)}_m{int(med_conf*100)}",
                high_vol_bear_mult=1.5,
                medium_bear_mult=1.7,
                high_vol_bull_mult=1.0,
                high_vol_bear_conf=high_vol_conf,
                medium_bear_conf=med_conf,
                blocked_regimes=['medium_bull', 'low_vol_bull']
            )
            configs.append(config)

        # Strategy 5: Ultra Conservative (FINAL mas com ajustes finos)
        for hv_mult, m_mult in [(1.2, 1.4), (1.3, 1.5), (1.4, 1.6)]:
            config = self._create_config(
                name=f"UltraConservative_hv{int(hv_mult*10)}_m{int(m_mult*10)}",
                high_vol_bear_mult=hv_mult,
                medium_bear_mult=m_mult,
                high_vol_bull_mult=0.9,
                high_vol_bear_conf=0.48,
                medium_bear_conf=0.38,
                blocked_regimes=['medium_bull', 'low_vol_bull']
            )
            configs.append(config)

        print(f"✅ Geradas {len(configs)} configurações para testar!")
        return configs

    def _create_config(self, name, high_vol_bear_mult, medium_bear_mult, high_vol_bull_mult,
                       high_vol_bear_conf, medium_bear_conf, blocked_regimes):
        """Cria uma configuração de teste."""

        # Determinar se cada regime está bloqueado
        regimes = {
            'medium_bear': {
                'enabled': 'medium_bear' not in blocked_regimes,
                'min_confidence': medium_bear_conf,
                'position_multiplier': medium_bear_mult,
                'stop_atr_mult': 1.2,
                'trailing_stop': False
            },
            'high_vol_bear': {
                'enabled': 'high_vol_bear' not in blocked_regimes,
                'min_confidence': high_vol_bear_conf,
                'position_multiplier': high_vol_bear_mult,
                'stop_atr_mult': 1.5,
                'trailing_stop': False
            },
            'low_vol_bear': {
                'enabled': 'low_vol_bear' not in blocked_regimes,
                'min_confidence': 0.45,
                'position_multiplier': 1.0,
                'stop_atr_mult': 1.0,
                'trailing_stop': False
            },
            'high_vol_bull': {
                'enabled': 'high_vol_bull' not in blocked_regimes,
                'min_confidence': 0.58,
                'position_multiplier': high_vol_bull_mult,
                'stop_atr_mult': 1.5,
                'trailing_stop': False
            },
            'medium_bull': {
                'enabled': 'medium_bull' not in blocked_regimes,
                'min_confidence': 0.99,
                'position_multiplier': 0.0,
                'stop_atr_mult': 1.0,
                'trailing_stop': False
            },
            'low_vol_bull': {
                'enabled': 'low_vol_bull' not in blocked_regimes,
                'min_confidence': 0.99,
                'position_multiplier': 0.0,
                'stop_atr_mult': 1.0,
                'trailing_stop': False
            }
        }

        return {
            'name': name,
            'regime_filter': {
                'enabled': True,
                'regimes': regimes
            },
            'confidence_filter': {
                'enabled': True,
                'tiers': {
                    'ultra_high': {'position_multiplier': 1.2},
                    'high': {'position_multiplier': 1.0},
                    'medium': {'position_multiplier': 0.7},
                    'low': {'position_multiplier': 0.5}
                }
            },
            'position_sizing': {
                'mode': 'dynamic',
                'base_size_usd': 100.0,
                'max_size_usd': 700.0
            },
            'stop_loss': {
                'mode': 'dynamic_atr',
                'base_atr_multiplier': 1.0,
                'min_atr_multiplier': 0.8,
                'max_atr_multiplier': 2.0
            },
            'take_profit': {
                'mode': 'asymmetric',
                'regimes': {
                    'medium_bear': {'tp_atr_mult': 2.0, 'risk_reward': 2.5},
                    'high_vol_bear': {'tp_atr_mult': 2.2, 'risk_reward': 2.5},
                    'low_vol_bear': {'tp_atr_mult': 1.8, 'risk_reward': 2.0},
                    'high_vol_bull': {'tp_atr_mult': 1.8, 'risk_reward': 2.0},
                    'medium_bull': {'tp_atr_mult': 1.5, 'risk_reward': 1.5},
                    'low_vol_bull': {'tp_atr_mult': 1.2, 'risk_reward': 1.2}
                }
            },
            'regime_detection': {
                'volatility_window': 20,
                'trend_window': 50,
                'vol_quantiles': {'low': 0.33, 'high': 0.67}
            },
            'risk_management': {
                'max_daily_trades': 40,
                'max_concurrent_trades': 2,
                'max_daily_loss_pct': 3.0,
                'max_drawdown_pct': 8.0,
                'performance_check': {
                    'enabled': True,
                    'window_trades': 15,
                    'min_win_rate': 0.50,
                    'cooldown_minutes': 90
                }
            },
            'validation': {
                'walk_forward_splits': 5,
                'monte_carlo_runs': 1000
            },
            'logging': {
                'level': 'INFO',
                'log_trades': True,
                'log_regime_changes': True,
                'log_blocked_trades': True,
                'log_confidence_filtering': True
            }
        }

    def test_config(self, config):
        """Testa uma configuração e retorna resultados."""
        try:
            validator = OptimizedUltraValidator(
                self.base_config,
                config,
                self.model_path
            )

            # Reset counters
            validator.blocked_trades_count = 0
            validator.blocked_by_regime = {}
            validator.blocked_by_confidence = 0

            # Run backtest
            stats = validator.backtest_optimized(self.df_features)

            return {
                'name': config['name'],
                'roi': stats.get('roi', 0),
                'win_rate': stats.get('win_rate', 0),
                'sharpe': stats.get('sharpe_ratio', 0),
                'max_dd': stats.get('max_drawdown', 0),
                'trades': stats.get('total_trades', 0),
                'profit_factor': stats.get('profit_factor', 0),
                'config': config
            }
        except Exception as e:
            print(f"   ❌ Erro testando {config['name']}: {e}")
            return None

    def run(self):
        """Executa grid search completo."""
        configs = self.generate_configs()

        print(f"\n🚀 Iniciando grid search com {len(configs)} configurações...")
        print("="*80)

        for i, config in enumerate(configs, 1):
            print(f"\n[{i}/{len(configs)}] Testando: {config['name']}")

            result = self.test_config(config)

            if result:
                self.results.append(result)
                print(f"   ✅ ROI: {result['roi']:+.2f}% | WR: {result['win_rate']*100:.1f}% | "
                      f"Trades: {result['trades']} | Sharpe: {result['sharpe']:.2f}")

        print("\n" + "="*80)
        print("✅ Grid search completo!")

        return self.get_top_configs()

    def get_top_configs(self, n=10):
        """Retorna as N melhores configurações ranqueadas por ROI."""
        if not self.results:
            return []

        # Ordenar por ROI (descendente)
        sorted_results = sorted(self.results, key=lambda x: x['roi'], reverse=True)

        return sorted_results[:n]


def main():
    print("="*80)
    print("🔬 GRID SEARCH OPTIMIZER - Automated Parameter Search")
    print("="*80)
    print("Objetivo: Encontrar a MELHOR configuração automaticamente")
    print("Método: Testar centenas de combinações de parâmetros")
    print("="*80)

    # Load base config
    config = load_config('standard')
    config['initial_capital'] = 10000

    # Download data
    print("\n📥 Downloading data...")
    rest_client = BybitRESTClient(
        api_key=config['bybit_api_key'],
        api_secret=config['bybit_api_secret'],
        testnet=config['bybit_testnet']
    )

    dm = DataManager(rest_client)
    df = dm.get_data('BTCUSDT', '15m', 90, use_cache=False)
    print(f"✅ Downloaded {len(df):,} candles")

    # Build features
    print("\n🔨 Building features...")
    fs = FeatureStore(config)
    df_features = fs.build_features(df, normalize=False)
    df_features = create_microstructure_features(df_features)
    df_features = create_advanced_master_features(df_features)
    print("✅ Features ready")

    # Model path
    model_path = "storage/models/ultra_scalper_btcusdt_365d.pkl"

    # Run grid search
    optimizer = GridSearchOptimizer(config, df_features, model_path)
    top_configs = optimizer.run()

    # Display results
    print("\n" + "="*80)
    print("🏆 TOP 10 MELHORES CONFIGURAÇÕES")
    print("="*80)

    for i, result in enumerate(top_configs, 1):
        print(f"\n#{i} - {result['name']}")
        print(f"   ROI: {result['roi']:+.2f}%")
        print(f"   Win Rate: {result['win_rate']*100:.1f}%")
        print(f"   Sharpe: {result['sharpe']:.2f}")
        print(f"   Max DD: {result['max_dd']:.1f}%")
        print(f"   Trades: {result['trades']}")
        print(f"   Profit Factor: {result['profit_factor']:.2f}")

        # Show key parameters
        regimes = result['config']['regime_filter']['regimes']
        print(f"   Key Params:")
        print(f"     • high_vol_bear: {regimes['high_vol_bear']['position_multiplier']}x @ "
              f"{regimes['high_vol_bear']['min_confidence']*100:.0f}% conf")
        print(f"     • medium_bear: {regimes['medium_bear']['position_multiplier']}x @ "
              f"{regimes['medium_bear']['min_confidence']*100:.0f}% conf")

        blocked = [k for k, v in regimes.items() if not v['enabled']]
        print(f"     • Blocked: {blocked}")

    # Save best config
    if top_configs:
        best = top_configs[0]
        output_path = Path(__file__).parent / 'config_ultra_optimized_BEST.yaml'

        with open(output_path, 'w') as f:
            yaml.dump(best['config'], f, default_flow_style=False, sort_keys=False)

        print(f"\n💾 Melhor configuração salva em: {output_path}")
        print(f"   ROI esperado: {best['roi']:+.2f}%")

    print("\n" + "="*80)
    print("✅ GRID SEARCH COMPLETO!")
    print("="*80)


if __name__ == "__main__":
    main()
