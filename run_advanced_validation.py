"""
🎯 ADVANCED VALIDATION PIPELINE
Script completo que integra todos os métodos de validação avançados

INTEGRA:
1. Confidence Filtering
2. Threshold Optimization
3. Purged K-Fold CV
4. Ensemble Scoring
5. Monte Carlo Simulation

USO:
    python run_advanced_validation.py --help

EXEMPLO:
    # Com dados simulados (demo)
    python run_advanced_validation.py --demo

    # Com seu modelo real
    python run_advanced_validation.py --model path/to/model.pkl --data path/to/data.csv
"""

import argparse
import numpy as np
import pandas as pd
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

from confidence_filter import ConfidenceFilter
from optimize_confidence_threshold import ThresholdOptimizer
from purged_kfold import PurgedKFold, CombinatorialPurgedCV, PurgedWalkForward
from ensemble_scoring import EnsembleScorer, TradeQuality


class AdvancedValidationPipeline:
    """
    Pipeline completo de validação avançada para estratégias de trading
    """

    def __init__(
        self,
        model=None,
        initial_capital: float = 10000,
        risk_free_rate: float = 0.02
    ):
        """
        Args:
            model: Modelo de ML treinado (deve ter predict_proba)
            initial_capital: Capital inicial
            risk_free_rate: Taxa livre de risco (para Sharpe)
        """
        self.model = model
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate

        # Componentes
        self.confidence_filter = None
        self.threshold_optimizer = None
        self.ensemble_scorer = None

        # Resultados
        self.results = {}

    def run_full_pipeline(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        actual_returns: np.ndarray,
        regimes: pd.Series,
        run_cpcv: bool = False
    ) -> Dict:
        """
        Executa pipeline completo de validação

        Args:
            X: Features
            y: Targets (0/1)
            actual_returns: Retornos reais dos trades
            regimes: Regime de mercado para cada sample
            run_cpcv: Se True, roda CPCV (pode ser lento)

        Returns:
            Dict com todos os resultados
        """
        print("\n" + "="*80)
        print("🚀 ADVANCED VALIDATION PIPELINE")
        print("="*80)

        results = {}

        # ========================================================================
        # STEP 1: BASELINE (sem filtro)
        # ========================================================================
        print("\n📊 STEP 1: BASELINE VALIDATION (sem filtro de confiança)")
        print("-"*80)

        baseline = self._validate_baseline(X, y, actual_returns)
        results['baseline'] = baseline

        print(f"  Trades: {baseline['n_trades']}")
        print(f"  Win Rate: {baseline['win_rate']:.1%}")
        print(f"  ROI: {baseline['roi']:.2%}")
        print(f"  Sharpe: {baseline['sharpe']:.2f}")
        print(f"  Max DD: {baseline['max_dd']:.2%}")

        # ========================================================================
        # STEP 2: OPTIMIZE CONFIDENCE THRESHOLD
        # ========================================================================
        print("\n📊 STEP 2: OPTIMIZING CONFIDENCE THRESHOLD")
        print("-"*80)

        if self.model and hasattr(self.model, 'predict_proba'):
            probas = self.model.predict_proba(X)[:, 1]
        else:
            # Simula probabilidades
            probas = np.random.uniform(0.5, 0.9, len(X))

        optimizer = ThresholdOptimizer(
            min_threshold=0.50,
            max_threshold=0.80,
            step=0.02,
            min_trades=30
        )

        best_threshold, best_result = optimizer.optimize(
            probas,
            actual_returns,
            objective='sharpe'
        )

        print(f"\n  ✅ Best Threshold: {best_threshold:.1%}")
        print(f"     Trades: {best_result.total_trades}")
        print(f"     Win Rate: {best_result.win_rate:.1%}")
        print(f"     ROI: {best_result.roi:.2%}")
        print(f"     Sharpe: {best_result.sharpe:.2f}")
        print(f"     Improvement: {best_result.sharpe - baseline['sharpe']:+.2f} Sharpe")

        results['threshold_optimization'] = {
            'best_threshold': best_threshold,
            'best_result': best_result,
            'all_results': optimizer.results
        }

        # ========================================================================
        # STEP 3: PURGED K-FOLD VALIDATION
        # ========================================================================
        print("\n📊 STEP 3: PURGED K-FOLD CROSS-VALIDATION")
        print("-"*80)

        pkf_results = self._run_purged_kfold(
            X, y, actual_returns, probas, best_threshold
        )

        results['purged_kfold'] = pkf_results

        print(f"\n  Folds: {len(pkf_results['folds'])}")
        print(f"  Avg Win Rate: {pkf_results['avg_win_rate']:.1%}")
        print(f"  Avg ROI: {pkf_results['avg_roi']:.2%}")
        print(f"  Avg Sharpe: {pkf_results['avg_sharpe']:.2f}")
        print(f"  Consistency: {pkf_results['consistency']:.1%} (% positive folds)")

        # ========================================================================
        # STEP 4: REGIME-BASED VALIDATION
        # ========================================================================
        print("\n📊 STEP 4: REGIME-BASED VALIDATION")
        print("-"*80)

        regime_results = self._validate_by_regime(
            X, y, actual_returns, probas, regimes, best_threshold
        )

        results['regime_validation'] = regime_results

        print(f"\n  Tested {len(regime_results)} regimes:")
        for regime, metrics in regime_results.items():
            print(f"    {regime:20s}: WR={metrics['win_rate']:.1%}, ROI={metrics['roi']:+6.1%}, Sharpe={metrics['sharpe']:5.2f}")

        # ========================================================================
        # STEP 5: ENSEMBLE SCORING
        # ========================================================================
        print("\n📊 STEP 5: ENSEMBLE SCORING SYSTEM")
        print("-"*80)

        ensemble_results = self._test_ensemble_scoring(
            probas, regimes, actual_returns
        )

        results['ensemble'] = ensemble_results

        print(f"\n  Signals above threshold: {ensemble_results['signals_passed']}/{ensemble_results['total_signals']}")
        print(f"  Pass rate: {ensemble_results['pass_rate']:.1%}")
        print(f"  Avg score (passed): {ensemble_results['avg_score_passed']:.1f}/100")
        print(f"  Quality distribution:")
        for quality, count in ensemble_results['quality_dist'].items():
            print(f"    {quality:15s}: {count}")

        # ========================================================================
        # STEP 6: COMBINATORIAL PURGED CV (opcional - lento)
        # ========================================================================
        if run_cpcv:
            print("\n📊 STEP 6: COMBINATORIAL PURGED CV (pode levar alguns minutos...)")
            print("-"*80)

            cpcv_results = self._run_cpcv(
                X, y, actual_returns, probas, best_threshold
            )

            results['cpcv'] = cpcv_results

            print(f"\n  Combinations tested: {cpcv_results['n_combinations']}")
            print(f"  Mean Sharpe: {cpcv_results['mean_sharpe']:.2f}")
            print(f"  Std Sharpe: {cpcv_results['std_sharpe']:.2f}")
            print(f"  Sharpe range: {cpcv_results['min_sharpe']:.2f} to {cpcv_results['max_sharpe']:.2f}")

        # ========================================================================
        # SUMMARY
        # ========================================================================
        print("\n" + "="*80)
        print("📊 VALIDATION SUMMARY")
        print("="*80)

        print(f"\n{'Method':<30s} {'WR':>8s} {'ROI':>10s} {'Sharpe':>10s} {'Status':>12s}")
        print("-"*80)

        print(f"{'Baseline (no filter)':<30s} "
              f"{baseline['win_rate']:>7.1%} "
              f"{baseline['roi']:>9.1%} "
              f"{baseline['sharpe']:>10.2f} "
              f"{'📍 Ref':>12s}")

        status = '✅ Better' if best_result.sharpe > baseline['sharpe'] else '⚠️ Worse'
        print(f"{'Optimized Threshold':<30s} "
              f"{best_result.win_rate:>7.1%} "
              f"{best_result.roi:>9.1%} "
              f"{best_result.sharpe:>10.2f} "
              f"{status:>12s}")

        consistency_pct = f"{pkf_results['consistency']:.0%}"
        print(f"{'Purged K-Fold (avg)':<30s} "
              f"{pkf_results['avg_win_rate']:>7.1%} "
              f"{pkf_results['avg_roi']:>9.1%} "
              f"{pkf_results['avg_sharpe']:>10.2f} "
              f"{consistency_pct + ' stable':>12s}")

        print("="*80)

        # Save results
        self.results = results

        return results

    def _validate_baseline(self, X, y, returns) -> Dict:
        """Validação baseline sem filtros"""
        wins = returns > 0
        return {
            'n_trades': len(returns),
            'win_rate': np.mean(wins),
            'roi': np.sum(returns),
            'sharpe': np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252),
            'max_dd': self._calculate_max_dd(returns)
        }

    def _run_purged_kfold(self, X, y, returns, probas, threshold) -> Dict:
        """Executa Purged K-Fold"""
        pkf = PurgedKFold(n_splits=5, embargo_td=pd.Timedelta(hours=1))

        fold_results = []

        for fold_idx, (train_idx, test_idx) in enumerate(pkf.split(X)):
            # Filtra por threshold no test set
            test_probas = probas[test_idx]
            test_returns = returns[test_idx]

            mask = test_probas >= threshold
            filtered_returns = test_returns[mask]

            if len(filtered_returns) > 0:
                wins = filtered_returns > 0
                fold_result = {
                    'fold': fold_idx + 1,
                    'n_trades': len(filtered_returns),
                    'win_rate': np.mean(wins),
                    'roi': np.sum(filtered_returns),
                    'sharpe': np.mean(filtered_returns) / (np.std(filtered_returns) + 1e-10) * np.sqrt(252)
                }
                fold_results.append(fold_result)

        # Calcula médias
        avg_wr = np.mean([f['win_rate'] for f in fold_results])
        avg_roi = np.mean([f['roi'] for f in fold_results])
        avg_sharpe = np.mean([f['sharpe'] for f in fold_results])

        # Consistency = % de folds com ROI positivo
        positive_folds = sum(1 for f in fold_results if f['roi'] > 0)
        consistency = positive_folds / len(fold_results) if fold_results else 0

        return {
            'folds': fold_results,
            'avg_win_rate': avg_wr,
            'avg_roi': avg_roi,
            'avg_sharpe': avg_sharpe,
            'consistency': consistency
        }

    def _validate_by_regime(self, X, y, returns, probas, regimes, threshold) -> Dict:
        """Valida performance por regime"""
        unique_regimes = regimes.unique()
        regime_results = {}

        for regime in unique_regimes:
            mask_regime = (regimes == regime).values
            regime_probas = probas[mask_regime]
            regime_returns = returns[mask_regime]

            # Filtra por threshold
            mask_conf = regime_probas >= threshold
            filtered_returns = regime_returns[mask_conf]

            if len(filtered_returns) > 0:
                wins = filtered_returns > 0
                regime_results[regime] = {
                    'n_trades': len(filtered_returns),
                    'win_rate': np.mean(wins),
                    'roi': np.sum(filtered_returns),
                    'sharpe': np.mean(filtered_returns) / (np.std(filtered_returns) + 1e-10) * np.sqrt(252)
                }

        return regime_results

    def _test_ensemble_scoring(self, probas, regimes, returns) -> Dict:
        """Testa sistema de ensemble scoring"""
        scorer = EnsembleScorer(min_score=55.0)

        signals = []
        for i in range(len(probas)):
            # Simula volatilidade e momentum
            vol = np.random.uniform(0.015, 0.04)
            momentum = np.random.uniform(-0.02, 0.02)

            signal = scorer.score_trade(
                model_proba=probas[i],
                regime=regimes.iloc[i],
                volatility=vol,
                current_dd=0.0,
                price_momentum=momentum
            )
            signals.append(signal)

        # Análise
        passed = [s for s in signals if s.should_trade]
        quality_dist = {}
        for q in TradeQuality:
            quality_dist[q.value] = sum(1 for s in signals if s.quality == q)

        return {
            'total_signals': len(signals),
            'signals_passed': len(passed),
            'pass_rate': len(passed) / len(signals),
            'avg_score_passed': np.mean([s.score for s in passed]) if passed else 0,
            'quality_dist': quality_dist
        }

    def _run_cpcv(self, X, y, returns, probas, threshold) -> Dict:
        """Executa Combinatorial Purged CV"""
        cpcv = CombinatorialPurgedCV(
            n_splits=5,
            n_test_splits=1,
            embargo_td=pd.Timedelta(hours=1)
        )

        sharpes = []

        for train_idx, test_idx in cpcv.split(X):
            test_probas = probas[test_idx]
            test_returns = returns[test_idx]

            mask = test_probas >= threshold
            filtered_returns = test_returns[mask]

            if len(filtered_returns) > 5:
                sharpe = np.mean(filtered_returns) / (np.std(filtered_returns) + 1e-10) * np.sqrt(252)
                sharpes.append(sharpe)

        return {
            'n_combinations': len(sharpes),
            'mean_sharpe': np.mean(sharpes),
            'std_sharpe': np.std(sharpes),
            'min_sharpe': np.min(sharpes),
            'max_sharpe': np.max(sharpes)
        }

    def _calculate_max_dd(self, returns) -> float:
        """Calcula max drawdown"""
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        return np.min(drawdown)


# ============================================================================
# MAIN
# ============================================================================

def generate_demo_data(n_samples: int = 2000) -> Tuple:
    """Gera dados simulados para demonstração"""
    print("📊 Gerando dados simulados...")

    # Timestamps
    dates = pd.date_range(start='2024-01-01', periods=n_samples, freq='5min')

    # Features
    X = pd.DataFrame({
        'rsi': np.random.uniform(20, 80, n_samples),
        'macd': np.random.randn(n_samples),
        'volume': np.random.uniform(0.5, 2.0, n_samples),
        'volatility': np.random.uniform(0.01, 0.05, n_samples),
    }, index=dates)

    # Targets
    y = pd.Series(np.random.randint(0, 2, n_samples), index=dates)

    # Returns (correlacionados com features)
    base_returns = (X['rsi'] - 50) / 1000 + X['macd'] * 0.01
    noise = np.random.normal(0, 0.02, n_samples)
    returns = base_returns + noise

    # Regimes
    regimes = pd.Series(
        np.random.choice(
            ['medium_bear', 'high_vol_bear', 'low_vol_bear', 'high_vol_bull', 'medium_bull', 'low_vol_bull'],
            n_samples
        ),
        index=dates
    )

    print(f"  ✅ {n_samples} samples gerados")
    return X, y, returns.values, regimes


def main():
    """Função principal"""
    parser = argparse.ArgumentParser(description='Advanced Validation Pipeline')
    parser.add_argument('--demo', action='store_true', help='Run with simulated data')
    parser.add_argument('--model', type=str, help='Path to trained model (.pkl)')
    parser.add_argument('--data', type=str, help='Path to data (.csv)')
    parser.add_argument('--cpcv', action='store_true', help='Run CPCV (slow)')
    parser.add_argument('--samples', type=int, default=2000, help='Number of samples (demo mode)')

    args = parser.parse_args()

    if args.demo:
        print("\n🎯 RUNNING IN DEMO MODE WITH SIMULATED DATA")
        X, y, returns, regimes = generate_demo_data(n_samples=args.samples)
        model = None
    elif args.model and args.data:
        print(f"\n📂 Loading model from: {args.model}")
        print(f"📂 Loading data from: {args.data}")
        # TODO: Implementar loading de modelo e dados reais
        raise NotImplementedError("Real model/data loading not implemented yet. Use --demo mode.")
    else:
        print("❌ Error: Use --demo for demo mode, or provide --model and --data")
        parser.print_help()
        return

    # Cria e roda pipeline
    pipeline = AdvancedValidationPipeline(model=model)

    results = pipeline.run_full_pipeline(
        X=X,
        y=y,
        actual_returns=returns,
        regimes=regimes,
        run_cpcv=args.cpcv
    )

    print("\n✅ PIPELINE COMPLETE!")
    print("\n💡 RECOMMENDATIONS:")
    print("  1. Use otimized threshold in production")
    print("  2. Monitor regime performance continuously")
    print("  3. Implement ensemble scoring for better filtering")
    print("  4. Re-run validation monthly with new data")
    print("="*80)


if __name__ == "__main__":
    main()
