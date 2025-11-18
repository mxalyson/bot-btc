"""
🎯 CONFIDENCE FILTER MODULE
Adiciona filtro de probabilidade ao modelo para melhorar Win Rate e reduzir trades ruins

Uso:
    from confidence_filter import ConfidenceFilter

    cf = ConfidenceFilter(model, threshold=0.62)
    should_trade, confidence = cf.predict(X)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class ConfidenceMetrics:
    """Métricas de performance com filtro de confiança"""
    total_signals: int
    filtered_signals: int
    filter_rate: float
    avg_confidence: float
    confidence_distribution: Dict[str, int]


class ConfidenceFilter:
    """
    Filtro de confiança para sinais de trading

    Features:
    - Threshold fixo ou adaptativo
    - Ajuste baseado em regime de mercado
    - Ajuste baseado em drawdown
    - Estatísticas de filtro
    """

    def __init__(
        self,
        threshold: float = 0.62,
        adaptive: bool = True,
        regime_multipliers: Optional[Dict[str, float]] = None,
        dd_adjustment: bool = True
    ):
        """
        Args:
            threshold: Probabilidade mínima para tradear (0-1)
            adaptive: Se True, ajusta threshold baseado em condições
            regime_multipliers: Multiplicadores por regime {'medium_bear': 0.95, 'low_vol_bull': 1.15}
            dd_adjustment: Se True, aumenta threshold durante drawdown
        """
        self.base_threshold = threshold
        self.threshold = threshold
        self.adaptive = adaptive
        self.dd_adjustment = dd_adjustment

        # Multiplicadores por regime (menor = mais permissivo)
        self.regime_multipliers = regime_multipliers or {
            'medium_bear': 0.92,      # Melhor regime - reduz threshold
            'high_vol_bear': 0.95,    # Bom regime - reduz um pouco
            'low_vol_bear': 0.98,     # OK
            'high_vol_bull': 1.00,    # Neutro
            'medium_bull': 1.03,      # Ruim - aumenta threshold
            'low_vol_bull': 1.15,     # Péssimo - muito mais seletivo
        }

        # Estatísticas
        self.stats = {
            'total_signals': 0,
            'filtered_out': 0,
            'accepted': 0,
            'confidences': []
        }

    def set_threshold(self, threshold: float):
        """Atualiza threshold base"""
        self.base_threshold = threshold
        self.threshold = threshold

    def adjust_for_regime(self, regime: str) -> float:
        """Ajusta threshold baseado no regime de mercado"""
        multiplier = self.regime_multipliers.get(regime, 1.0)
        return self.base_threshold * multiplier

    def adjust_for_drawdown(self, current_dd: float, max_acceptable_dd: float = 0.10) -> float:
        """
        Aumenta threshold durante drawdown para reduzir risco

        Args:
            current_dd: Drawdown atual (0.05 = 5%)
            max_acceptable_dd: DD máximo aceitável antes de parar

        Returns:
            Threshold ajustado
        """
        if current_dd <= 0:
            return self.threshold

        # Se DD > 50% do máximo aceitável, aumenta threshold progressivamente
        dd_ratio = abs(current_dd) / max_acceptable_dd

        if dd_ratio > 0.5:
            # Aumenta threshold de 0% a 20% baseado no DD
            adjustment = 1.0 + (dd_ratio - 0.5) * 0.4
            return min(self.threshold * adjustment, 0.95)  # Max 95%

        return self.threshold

    def predict(
        self,
        model,
        X,
        regime: Optional[str] = None,
        current_dd: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Faz predição com filtro de confiança

        Args:
            model: Modelo treinado com método predict_proba
            X: Features
            regime: Regime atual do mercado
            current_dd: Drawdown atual

        Returns:
            predictions: Array de sinais filtrados (0 ou 1)
            confidences: Array de probabilidades
        """
        # Get probabilidades
        if hasattr(model, 'predict_proba'):
            probas = model.predict_proba(X)
            # Pega probabilidade da classe 1 (buy)
            confidences = probas[:, 1] if probas.shape[1] > 1 else probas.ravel()
        else:
            # Fallback se modelo não tem predict_proba
            predictions = model.predict(X)
            confidences = np.where(predictions == 1, 0.7, 0.3)  # Assume confiança média

        # Ajusta threshold se adaptativo
        active_threshold = self.base_threshold

        if self.adaptive:
            if regime:
                active_threshold = self.adjust_for_regime(regime)

            if self.dd_adjustment and current_dd > 0:
                active_threshold = self.adjust_for_drawdown(current_dd)

        self.threshold = active_threshold

        # Aplica filtro
        predictions = np.where(confidences >= active_threshold, 1, 0)

        # Atualiza estatísticas
        self.stats['total_signals'] += len(predictions)
        original_signals = np.sum(confidences >= 0.5)  # Sinais sem filtro
        filtered_signals = np.sum(predictions == 1)
        self.stats['filtered_out'] += (original_signals - filtered_signals)
        self.stats['accepted'] += filtered_signals
        self.stats['confidences'].extend(confidences.tolist())

        return predictions, confidences

    def get_metrics(self) -> ConfidenceMetrics:
        """Retorna métricas de filtro"""
        total = self.stats['total_signals']
        filtered = self.stats['filtered_out']

        if total == 0:
            return ConfidenceMetrics(0, 0, 0.0, 0.0, {})

        # Distribuição de confiança
        confidences = np.array(self.stats['confidences'])
        distribution = {
            'very_low_0.0-0.4': np.sum(confidences < 0.4),
            'low_0.4-0.5': np.sum((confidences >= 0.4) & (confidences < 0.5)),
            'medium_0.5-0.6': np.sum((confidences >= 0.5) & (confidences < 0.6)),
            'high_0.6-0.7': np.sum((confidences >= 0.6) & (confidences < 0.7)),
            'very_high_0.7+': np.sum(confidences >= 0.7),
        }

        return ConfidenceMetrics(
            total_signals=total,
            filtered_signals=filtered,
            filter_rate=filtered / total if total > 0 else 0,
            avg_confidence=float(np.mean(confidences)),
            confidence_distribution=distribution
        )

    def reset_stats(self):
        """Reseta estatísticas"""
        self.stats = {
            'total_signals': 0,
            'filtered_out': 0,
            'accepted': 0,
            'confidences': []
        }

    def print_report(self):
        """Imprime relatório de performance do filtro"""
        metrics = self.get_metrics()

        print("\n" + "="*80)
        print("📊 CONFIDENCE FILTER REPORT")
        print("="*80)
        print(f"Threshold: {self.threshold:.1%} (Base: {self.base_threshold:.1%})")
        print(f"Adaptive: {self.adaptive}")
        print(f"\nTotal Signals: {metrics.total_signals:,}")
        print(f"Filtered Out: {metrics.filtered_signals:,} ({metrics.filter_rate:.1%})")
        print(f"Accepted: {self.stats['accepted']:,}")
        print(f"Average Confidence: {metrics.avg_confidence:.1%}")

        print(f"\n📈 Confidence Distribution:")
        for range_name, count in metrics.confidence_distribution.items():
            pct = count / metrics.total_signals * 100 if metrics.total_signals > 0 else 0
            bar = "█" * int(pct / 2)
            print(f"  {range_name:20s}: {count:5d} ({pct:5.1f}%) {bar}")

        print("="*80)


class DynamicConfidenceFilter(ConfidenceFilter):
    """
    Versão avançada que aprende threshold ótimo baseado em performance
    """

    def __init__(self, initial_threshold: float = 0.60):
        super().__init__(threshold=initial_threshold, adaptive=True)
        self.performance_history = []
        self.threshold_history = []

    def update_from_performance(self, win_rate: float, sharpe: float, target_wr: float = 0.55):
        """
        Ajusta threshold baseado em performance recente

        Args:
            win_rate: Win rate recente
            sharpe: Sharpe ratio recente
            target_wr: Win rate alvo
        """
        self.performance_history.append({'wr': win_rate, 'sharpe': sharpe})

        # Se WR muito baixo, aumenta threshold
        if win_rate < target_wr - 0.05:  # 5% abaixo do alvo
            self.base_threshold = min(self.base_threshold * 1.02, 0.95)

        # Se WR muito alto mas poucos trades, reduz threshold
        elif win_rate > target_wr + 0.05 and len(self.performance_history) > 0:
            self.base_threshold = max(self.base_threshold * 0.98, 0.50)

        self.threshold_history.append(self.base_threshold)


# ============================================================================
# EXEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    """Exemplo de como usar o ConfidenceFilter"""

    # Simula um modelo
    class DummyModel:
        def predict_proba(self, X):
            # Retorna probabilidades aleatórias
            n = len(X)
            return np.column_stack([
                np.random.uniform(0.2, 0.8, n),  # Classe 0
                np.random.uniform(0.2, 0.8, n)   # Classe 1
            ])

    model = DummyModel()
    X = np.random.randn(1000, 10)

    # Teste 1: Filtro fixo
    print("\n🧪 TEST 1: Fixed Threshold")
    cf = ConfidenceFilter(threshold=0.65, adaptive=False)
    predictions, confidences = cf.predict(model, X)
    print(f"Sinais gerados: {np.sum(predictions)}/1000")
    cf.print_report()

    # Teste 2: Filtro adaptativo por regime
    print("\n🧪 TEST 2: Adaptive by Regime")
    cf_adaptive = ConfidenceFilter(threshold=0.62, adaptive=True)

    regimes = ['medium_bear', 'low_vol_bull', 'high_vol_bear']
    for regime in regimes:
        cf_adaptive.reset_stats()
        predictions, confidences = cf_adaptive.predict(model, X, regime=regime)
        print(f"\n{regime}: {np.sum(predictions)} sinais (threshold: {cf_adaptive.threshold:.1%})")

    # Teste 3: Ajuste por drawdown
    print("\n🧪 TEST 3: Drawdown Adjustment")
    cf_dd = ConfidenceFilter(threshold=0.60, dd_adjustment=True)

    drawdowns = [0.0, 0.03, 0.07, 0.12]
    for dd in drawdowns:
        adjusted = cf_dd.adjust_for_drawdown(dd)
        print(f"DD {dd:.1%}: Threshold ajustado para {adjusted:.1%}")

    print("\n✅ Testes completos!")
