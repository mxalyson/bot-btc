"""
🎯 CONFIDENCE THRESHOLD OPTIMIZER
Encontra o threshold ótimo de probabilidade para maximizar métricas

Testa múltiplos thresholds e compara:
- Win Rate
- ROI
- Sharpe Ratio
- Max Drawdown
- Número de trades

Uso:
    python optimize_confidence_threshold.py --model <path> --data <path>
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, asdict

try:
    from .confidence_filter import ConfidenceFilter
except ImportError:
    from confidence_filter import ConfidenceFilter


@dataclass
class ThresholdResult:
    """Resultado de teste de um threshold"""
    threshold: float
    total_trades: int
    win_rate: float
    roi: float
    sharpe: float
    max_drawdown: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    avg_confidence: float
    expectancy: float  # Expectativa matemática por trade


class ThresholdOptimizer:
    """
    Otimizador de threshold de confiança

    Testa range de thresholds e encontra o ótimo baseado em objetivo
    """

    def __init__(
        self,
        min_threshold: float = 0.50,
        max_threshold: float = 0.85,
        step: float = 0.01,
        min_trades: int = 50  # Mínimo de trades para considerar válido
    ):
        """
        Args:
            min_threshold: Threshold mínimo a testar
            max_threshold: Threshold máximo a testar
            step: Incremento entre testes
            min_trades: Número mínimo de trades para considerar resultado válido
        """
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.step = step
        self.min_trades = min_trades
        self.results: List[ThresholdResult] = []

    def backtest_with_threshold(
        self,
        predictions_proba: np.ndarray,
        actual_returns: np.ndarray,
        threshold: float,
        initial_capital: float = 10000
    ) -> ThresholdResult:
        """
        Backtesta estratégia com threshold específico

        Args:
            predictions_proba: Probabilidades do modelo (0-1)
            actual_returns: Retornos reais dos trades (ex: 0.02 = 2%)
            threshold: Threshold a testar
            initial_capital: Capital inicial

        Returns:
            ThresholdResult com métricas
        """
        # Filtra trades baseado no threshold
        mask = predictions_proba >= threshold
        filtered_returns = actual_returns[mask]
        filtered_probas = predictions_proba[mask]

        n_trades = len(filtered_returns)

        if n_trades == 0:
            return ThresholdResult(
                threshold=threshold,
                total_trades=0,
                win_rate=0.0,
                roi=0.0,
                sharpe=0.0,
                max_drawdown=0.0,
                profit_factor=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                avg_confidence=0.0,
                expectancy=0.0
            )

        # Calcula métricas
        wins = filtered_returns > 0
        losses = filtered_returns < 0

        win_rate = np.mean(wins) if n_trades > 0 else 0
        total_return = np.sum(filtered_returns)
        roi = total_return

        # Sharpe Ratio (anualizado assumindo trades diários)
        if n_trades > 1:
            sharpe = np.mean(filtered_returns) / (np.std(filtered_returns) + 1e-10) * np.sqrt(252)
        else:
            sharpe = 0.0

        # Max Drawdown
        cumulative = np.cumsum(filtered_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        max_dd = np.min(drawdown) if len(drawdown) > 0 else 0

        # Profit Factor
        gross_profit = np.sum(filtered_returns[wins]) if np.any(wins) else 0
        gross_loss = abs(np.sum(filtered_returns[losses])) if np.any(losses) else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Médias
        avg_win = np.mean(filtered_returns[wins]) if np.any(wins) else 0
        avg_loss = np.mean(filtered_returns[losses]) if np.any(losses) else 0
        avg_confidence = np.mean(filtered_probas)

        # Expectância (Kelly Criterion)
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

        return ThresholdResult(
            threshold=threshold,
            total_trades=n_trades,
            win_rate=win_rate,
            roi=roi,
            sharpe=sharpe,
            max_drawdown=max_dd,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_confidence=avg_confidence,
            expectancy=expectancy
        )

    def optimize(
        self,
        predictions_proba: np.ndarray,
        actual_returns: np.ndarray,
        objective: str = 'sharpe',
        initial_capital: float = 10000
    ) -> Tuple[float, ThresholdResult]:
        """
        Otimiza threshold baseado em objetivo

        Args:
            predictions_proba: Probabilidades do modelo
            actual_returns: Retornos reais
            objective: Métrica a otimizar ('sharpe', 'roi', 'win_rate', 'expectancy')
            initial_capital: Capital inicial

        Returns:
            (best_threshold, best_result)
        """
        self.results = []

        thresholds = np.arange(self.min_threshold, self.max_threshold + self.step, self.step)

        print(f"\n🔍 Testando {len(thresholds)} thresholds de {self.min_threshold:.0%} a {self.max_threshold:.0%}...")
        print(f"Objetivo: Maximizar {objective.upper()}")
        print("="*80)

        for threshold in thresholds:
            result = self.backtest_with_threshold(
                predictions_proba,
                actual_returns,
                threshold,
                initial_capital
            )

            # Só considera se tem trades suficientes
            if result.total_trades >= self.min_trades:
                self.results.append(result)

        if not self.results:
            print("❌ Nenhum threshold gerou trades suficientes!")
            return 0.50, None

        # Encontra melhor resultado baseado no objetivo
        objective_map = {
            'sharpe': lambda r: r.sharpe,
            'roi': lambda r: r.roi,
            'win_rate': lambda r: r.win_rate,
            'expectancy': lambda r: r.expectancy,
            'profit_factor': lambda r: r.profit_factor
        }

        if objective not in objective_map:
            raise ValueError(f"Objetivo inválido. Use: {list(objective_map.keys())}")

        best_result = max(self.results, key=objective_map[objective])

        return best_result.threshold, best_result

    def print_results(self, top_n: int = 10):
        """Imprime top N resultados"""
        if not self.results:
            print("❌ Nenhum resultado para exibir")
            return

        print("\n" + "="*100)
        print(f"📊 TOP {top_n} THRESHOLDS POR SHARPE RATIO")
        print("="*100)

        # Ordena por Sharpe
        sorted_results = sorted(self.results, key=lambda r: r.sharpe, reverse=True)

        print(f"{'Threshold':>10} {'Trades':>8} {'WR':>7} {'ROI':>9} {'Sharpe':>8} {'MaxDD':>9} {'PF':>6} {'Expect':>8}")
        print("-"*100)

        for result in sorted_results[:top_n]:
            print(
                f"{result.threshold:>9.1%} "
                f"{result.total_trades:>8d} "
                f"{result.win_rate:>6.1%} "
                f"{result.roi:>8.1%} "
                f"{result.sharpe:>8.2f} "
                f"{result.max_drawdown:>8.1%} "
                f"{result.profit_factor:>6.2f} "
                f"{result.expectancy:>7.2%}"
            )

        print("="*100)

    def plot_optimization(self, save_path: str = None):
        """Cria visualização dos resultados"""
        if not self.results:
            print("❌ Nenhum resultado para plotar")
            return

        df = pd.DataFrame([asdict(r) for r in self.results])

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('🎯 Confidence Threshold Optimization', fontsize=16, fontweight='bold')

        # 1. Sharpe vs Threshold
        ax = axes[0, 0]
        ax.plot(df['threshold'], df['sharpe'], 'o-', color='blue', markersize=4)
        best_sharpe_idx = df['sharpe'].idxmax()
        ax.plot(df.loc[best_sharpe_idx, 'threshold'], df.loc[best_sharpe_idx, 'sharpe'],
                'r*', markersize=20, label=f"Best: {df.loc[best_sharpe_idx, 'threshold']:.1%}")
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Sharpe Ratio')
        ax.set_title('Sharpe Ratio vs Threshold')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # 2. Win Rate vs Threshold
        ax = axes[0, 1]
        ax.plot(df['threshold'], df['win_rate'] * 100, 'o-', color='green', markersize=4)
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Win Rate (%)')
        ax.set_title('Win Rate vs Threshold')
        ax.grid(True, alpha=0.3)

        # 3. ROI vs Threshold
        ax = axes[0, 2]
        ax.plot(df['threshold'], df['roi'] * 100, 'o-', color='purple', markersize=4)
        ax.set_xlabel('Threshold')
        ax.set_ylabel('ROI (%)')
        ax.set_title('ROI vs Threshold')
        ax.grid(True, alpha=0.3)

        # 4. Trades vs Threshold
        ax = axes[1, 0]
        ax.plot(df['threshold'], df['total_trades'], 'o-', color='orange', markersize=4)
        ax.axhline(y=self.min_trades, color='r', linestyle='--', label=f'Min trades: {self.min_trades}')
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Number of Trades')
        ax.set_title('Number of Trades vs Threshold')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # 5. Max Drawdown vs Threshold
        ax = axes[1, 1]
        ax.plot(df['threshold'], df['max_drawdown'] * 100, 'o-', color='red', markersize=4)
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Max Drawdown (%)')
        ax.set_title('Max Drawdown vs Threshold')
        ax.grid(True, alpha=0.3)

        # 6. Expectancy vs Threshold
        ax = axes[1, 2]
        ax.plot(df['threshold'], df['expectancy'] * 100, 'o-', color='brown', markersize=4)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Expectancy (%)')
        ax.set_title('Expectancy per Trade vs Threshold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Gráfico salvo em: {save_path}")

        plt.show()

    def get_recommendations(self) -> Dict[str, float]:
        """
        Retorna recomendações de threshold para diferentes perfis

        Returns:
            Dict com thresholds recomendados
        """
        if not self.results:
            return {}

        df = pd.DataFrame([asdict(r) for r in self.results])

        recommendations = {}

        # 1. Melhor Sharpe (balanceado)
        best_sharpe_idx = df['sharpe'].idxmax()
        recommendations['balanced_sharpe'] = df.loc[best_sharpe_idx, 'threshold']

        # 2. Melhor ROI (agressivo)
        best_roi_idx = df['roi'].idxmax()
        recommendations['aggressive_roi'] = df.loc[best_roi_idx, 'threshold']

        # 3. Melhor Win Rate (conservador)
        best_wr_idx = df['win_rate'].idxmax()
        recommendations['conservative_wr'] = df.loc[best_wr_idx, 'threshold']

        # 4. Melhor Expectancy (matemático)
        best_exp_idx = df['expectancy'].idxmax()
        recommendations['mathematical_expectancy'] = df.loc[best_exp_idx, 'threshold']

        # 5. Melhor DD (proteção)
        # Menor drawdown negativo
        best_dd_idx = df['max_drawdown'].idxmax()  # Menos negativo
        recommendations['protective_dd'] = df.loc[best_dd_idx, 'threshold']

        return recommendations


# ============================================================================
# EXEMPLO DE USO
# ============================================================================

def simulate_trading_data(n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simula dados de trading para teste

    Returns:
        (probabilities, returns)
    """
    # Simula probabilidades do modelo
    probas = np.random.beta(2, 2, n_samples)  # Distribuição beta

    # Simula retornos (correlacionados com probabilidade)
    # Quanto maior a probabilidade, melhor o retorno esperado
    noise = np.random.normal(0, 0.02, n_samples)
    returns = (probas - 0.5) * 0.08 + noise  # Expectativa positiva para proba > 0.5

    return probas, returns


if __name__ == "__main__":
    """Exemplo de uso do otimizador"""

    print("🎯 CONFIDENCE THRESHOLD OPTIMIZER - DEMO")
    print("="*80)

    # Gera dados simulados
    print("\n📊 Gerando dados simulados...")
    probas, returns = simulate_trading_data(n_samples=2000)
    print(f"   Gerados {len(probas)} trades simulados")
    print(f"   Retorno médio: {np.mean(returns):.2%}")
    print(f"   Win rate geral: {np.mean(returns > 0):.1%}")

    # Cria otimizador
    optimizer = ThresholdOptimizer(
        min_threshold=0.50,
        max_threshold=0.80,
        step=0.01,
        min_trades=50
    )

    # Otimiza para Sharpe
    print("\n" + "="*80)
    print("🎯 OTIMIZANDO PARA SHARPE RATIO")
    best_threshold, best_result = optimizer.optimize(
        probas,
        returns,
        objective='sharpe'
    )

    print(f"\n✅ MELHOR THRESHOLD: {best_threshold:.1%}")
    print(f"   Trades: {best_result.total_trades}")
    print(f"   Win Rate: {best_result.win_rate:.1%}")
    print(f"   ROI: {best_result.roi:.2%}")
    print(f"   Sharpe: {best_result.sharpe:.2f}")
    print(f"   Max DD: {best_result.max_drawdown:.2%}")
    print(f"   Profit Factor: {best_result.profit_factor:.2f}")
    print(f"   Expectancy: {best_result.expectancy:.2%}")

    # Mostra top 10
    optimizer.print_results(top_n=10)

    # Recomendações
    print("\n" + "="*80)
    print("💡 RECOMENDAÇÕES POR PERFIL")
    print("="*80)
    recs = optimizer.get_recommendations()
    for profile, threshold in recs.items():
        print(f"  {profile:30s}: {threshold:.1%}")

    # Plot (comentado para não requerer display)
    # optimizer.plot_optimization(save_path='threshold_optimization.png')

    print("\n✅ Otimização completa!")
