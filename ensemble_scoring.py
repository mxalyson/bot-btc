"""
🎯 ENSEMBLE SCORING SYSTEM
Sistema de score composto que combina múltiplas validações

Combina:
- Confiança do modelo (probabilidade)
- Performance por regime
- Validação Walk-Forward
- Volatilidade
- Momentum/Tendência
- Drawdown atual

Usa pesos adaptativos para gerar score final de 0-100

Uso:
    from ensemble_scoring import EnsembleScorer, TradeSignal

    scorer = EnsembleScorer()
    signal = scorer.score_trade(model, X, regime, volatility, ...)

    if signal.should_trade:
        execute_trade()
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from enum import Enum


class TradeQuality(Enum):
    """Qualidade do sinal de trade"""
    EXCELLENT = "excellent"  # Score >= 80
    GOOD = "good"            # Score >= 65
    FAIR = "fair"            # Score >= 50
    POOR = "poor"            # Score >= 35
    REJECT = "reject"        # Score < 35


@dataclass
class TradeSignal:
    """Sinal de trade com score composto"""
    score: float  # 0-100
    should_trade: bool
    quality: TradeQuality

    # Components
    confidence_score: float
    regime_score: float
    volatility_score: float
    momentum_score: float
    drawdown_penalty: float

    # Metadata
    model_proba: float
    regime: str
    volatility: float
    current_dd: float

    # Breakdown
    component_weights: Dict[str, float] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)

    def __str__(self):
        return (
            f"TradeSignal(score={self.score:.1f}, quality={self.quality.value}, "
            f"should_trade={self.should_trade}, proba={self.model_proba:.2%})"
        )

    def get_detailed_report(self) -> str:
        """Retorna relatório detalhado do score"""
        report = []
        report.append("="*80)
        report.append(f"🎯 TRADE SIGNAL REPORT")
        report.append("="*80)
        report.append(f"Final Score: {self.score:.1f}/100")
        report.append(f"Quality: {self.quality.value.upper()}")
        report.append(f"Decision: {'✅ TRADE' if self.should_trade else '❌ SKIP'}")
        report.append("")
        report.append("📊 Score Components:")
        report.append(f"  Confidence:   {self.confidence_score:5.1f} (weight: {self.component_weights.get('confidence', 0):.0%})")
        report.append(f"  Regime:       {self.regime_score:5.1f} (weight: {self.component_weights.get('regime', 0):.0%})")
        report.append(f"  Volatility:   {self.volatility_score:5.1f} (weight: {self.component_weights.get('volatility', 0):.0%})")
        report.append(f"  Momentum:     {self.momentum_score:5.1f} (weight: {self.component_weights.get('momentum', 0):.0%})")
        report.append(f"  DD Penalty:   {self.drawdown_penalty:5.1f}")
        report.append("")
        report.append("📈 Market Conditions:")
        report.append(f"  Model Probability: {self.model_proba:.1%}")
        report.append(f"  Regime: {self.regime}")
        report.append(f"  Volatility: {self.volatility:.4f}")
        report.append(f"  Current DD: {self.current_dd:.2%}")
        report.append("")
        if self.reasons:
            report.append("💡 Reasons:")
            for reason in self.reasons:
                report.append(f"  • {reason}")
        report.append("="*80)
        return "\n".join(report)


class EnsembleScorer:
    """
    Sistema de scoring composto para sinais de trading

    Combina múltiplas fontes de informação em um score final
    """

    def __init__(
        self,
        min_score: float = 50.0,
        component_weights: Optional[Dict[str, float]] = None,
        regime_performance: Optional[Dict[str, Dict]] = None,
        adaptive_weights: bool = True
    ):
        """
        Args:
            min_score: Score mínimo para tradear (0-100)
            component_weights: Pesos de cada componente
            regime_performance: Performance histórica por regime
            adaptive_weights: Se True, ajusta pesos baseado em condições
        """
        self.min_score = min_score
        self.adaptive_weights = adaptive_weights

        # Pesos padrão (devem somar 1.0)
        self.base_weights = component_weights or {
            'confidence': 0.40,   # Confiança do modelo (mais importante)
            'regime': 0.25,       # Performance do regime
            'volatility': 0.15,   # Adequação da volatilidade
            'momentum': 0.10,     # Alinhamento com tendência
            'technical': 0.10     # Indicadores técnicos
        }

        # Performance histórica por regime
        # Formato: {'regime_name': {'win_rate': 0.55, 'avg_roi': 0.15, 'sharpe': 2.5}}
        self.regime_performance = regime_performance or self._get_default_regime_performance()

        # Histórico de scores (para adaptação)
        self.score_history = []

    def _get_default_regime_performance(self) -> Dict[str, Dict]:
        """Retorna performance padrão por regime baseada nos seus resultados"""
        return {
            'medium_bear': {
                'win_rate': 0.569,
                'avg_roi': 0.2863,
                'sharpe': 3.5,
                'quality': 1.0  # Melhor regime
            },
            'high_vol_bear': {
                'win_rate': 0.505,
                'avg_roi': 0.2483,
                'sharpe': 3.0,
                'quality': 0.9
            },
            'low_vol_bear': {
                'win_rate': 0.512,
                'avg_roi': 0.1309,
                'sharpe': 2.0,
                'quality': 0.7
            },
            'high_vol_bull': {
                'win_rate': 0.426,
                'avg_roi': 0.1696,
                'sharpe': 1.5,
                'quality': 0.6
            },
            'medium_bull': {
                'win_rate': 0.405,
                'avg_roi': 0.0589,
                'sharpe': 0.8,
                'quality': 0.4
            },
            'low_vol_bull': {
                'win_rate': 0.331,
                'avg_roi': 0.0072,
                'sharpe': 0.1,
                'quality': 0.1  # Pior regime - deve bloquear
            }
        }

    def _score_confidence(self, model_proba: float) -> float:
        """
        Score baseado na probabilidade do modelo

        Returns:
            Score 0-100
        """
        if model_proba < 0.5:
            return 0.0

        # Mapeia 0.5-1.0 para 0-100
        # Função não-linear: mais peso para probabilidades altas
        normalized = (model_proba - 0.5) / 0.5
        score = 100 * (normalized ** 0.8)  # Exponencial para dar mais peso ao topo

        return min(score, 100.0)

    def _score_regime(self, regime: str) -> float:
        """
        Score baseado na performance histórica do regime

        Returns:
            Score 0-100
        """
        if regime not in self.regime_performance:
            return 50.0  # Neutro se regime desconhecido

        perf = self.regime_performance[regime]

        # Combina múltiplas métricas
        # Win rate contribui 50%
        wr_score = perf['win_rate'] * 100

        # Sharpe contribui 30% (normalizado para 0-100)
        sharpe_norm = min(perf['sharpe'] / 4.0, 1.0)  # Sharpe 4+ = 100
        sharpe_score = sharpe_norm * 100

        # Quality score contribui 20%
        quality_score = perf['quality'] * 100

        final_score = (
            wr_score * 0.5 +
            sharpe_score * 0.3 +
            quality_score * 0.2
        )

        return min(final_score, 100.0)

    def _score_volatility(
        self,
        current_vol: float,
        optimal_vol_range: tuple = (0.015, 0.035)
    ) -> float:
        """
        Score baseado na adequação da volatilidade

        Args:
            current_vol: Volatilidade atual (ex: 0.02 = 2%)
            optimal_vol_range: Range ótimo de volatilidade

        Returns:
            Score 0-100
        """
        min_vol, max_vol = optimal_vol_range

        if current_vol < min_vol:
            # Muito baixa - scalping difícil
            score = (current_vol / min_vol) * 50
        elif current_vol > max_vol:
            # Muito alta - risco excessivo
            excess = current_vol - max_vol
            score = max(100 - (excess / max_vol) * 100, 0)
        else:
            # Dentro do range ótimo
            # Score máximo no meio do range
            mid = (min_vol + max_vol) / 2
            distance = abs(current_vol - mid) / (max_vol - min_vol)
            score = 100 - (distance * 20)  # Penalidade suave

        return min(max(score, 0), 100)

    def _score_momentum(
        self,
        price_momentum: Optional[float] = None,
        volume_momentum: Optional[float] = None
    ) -> float:
        """
        Score baseado em momentum/tendência

        Args:
            price_momentum: Retorno recente (ex: 0.02 = 2%)
            volume_momentum: Volume relativo (ex: 1.5 = 150% da média)

        Returns:
            Score 0-100
        """
        if price_momentum is None and volume_momentum is None:
            return 50.0  # Neutro se não tem dados

        scores = []

        if price_momentum is not None:
            # Momentum positivo forte = bom
            # Normaliza para 0-100
            price_score = 50 + (price_momentum * 1000)  # Assume momentum ~ -0.05 a +0.05
            scores.append(min(max(price_score, 0), 100))

        if volume_momentum is not None:
            # Volume acima da média = bom
            # 1.5x volume = score 75, 2x = score 100
            volume_score = min(volume_momentum * 50, 100)
            scores.append(volume_score)

        return np.mean(scores) if scores else 50.0

    def _calculate_drawdown_penalty(self, current_dd: float, max_acceptable_dd: float = 0.10) -> float:
        """
        Penalidade baseada em drawdown atual

        Args:
            current_dd: Drawdown atual (positivo, ex: 0.05 = 5%)
            max_acceptable_dd: DD máximo aceitável

        Returns:
            Penalidade 0-50 (subtraída do score final)
        """
        if current_dd <= 0:
            return 0.0

        # Penalidade cresce exponencialmente com DD
        dd_ratio = current_dd / max_acceptable_dd

        if dd_ratio < 0.3:
            # DD pequeno - penalidade mínima
            penalty = dd_ratio * 10
        elif dd_ratio < 0.7:
            # DD moderado - penalidade linear
            penalty = 3 + (dd_ratio - 0.3) * 25
        else:
            # DD alto - penalidade severa
            penalty = 13 + (dd_ratio - 0.7) * 50

        return min(penalty, 50.0)

    def _adjust_weights_adaptive(
        self,
        base_weights: Dict[str, float],
        regime: str,
        volatility: float,
        current_dd: float
    ) -> Dict[str, float]:
        """
        Ajusta pesos baseado em condições de mercado

        Returns:
            Pesos ajustados (somam 1.0)
        """
        weights = base_weights.copy()

        # Em regimes ruins, aumenta peso da confiança
        if regime in ['low_vol_bull', 'medium_bull']:
            weights['confidence'] += 0.10
            weights['regime'] -= 0.10

        # Em alta volatilidade, aumenta peso do regime
        if volatility > 0.04:
            weights['regime'] += 0.08
            weights['volatility'] += 0.02
            weights['confidence'] -= 0.10

        # Durante drawdown, aumenta conservadorismo
        if current_dd > 0.05:
            weights['confidence'] += 0.15
            weights['momentum'] -= 0.10
            weights['technical'] -= 0.05

        # Normaliza para somar 1.0
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}

        return weights

    def score_trade(
        self,
        model_proba: float,
        regime: str,
        volatility: float,
        current_dd: float = 0.0,
        price_momentum: Optional[float] = None,
        volume_momentum: Optional[float] = None,
        technical_score: Optional[float] = None
    ) -> TradeSignal:
        """
        Calcula score composto para um trade

        Args:
            model_proba: Probabilidade do modelo (0-1)
            regime: Regime de mercado
            volatility: Volatilidade atual
            current_dd: Drawdown atual
            price_momentum: Momentum de preço
            volume_momentum: Momentum de volume
            technical_score: Score de indicadores técnicos (0-100)

        Returns:
            TradeSignal com decisão e breakdown
        """
        # Calcula scores individuais
        confidence_score = self._score_confidence(model_proba)
        regime_score = self._score_regime(regime)
        volatility_score = self._score_volatility(volatility)
        momentum_score = self._score_momentum(price_momentum, volume_momentum)
        tech_score = technical_score if technical_score is not None else 50.0

        # Ajusta pesos se adaptativo
        if self.adaptive_weights:
            weights = self._adjust_weights_adaptive(
                self.base_weights,
                regime,
                volatility,
                current_dd
            )
        else:
            weights = self.base_weights

        # Calcula score composto
        composite_score = (
            confidence_score * weights['confidence'] +
            regime_score * weights['regime'] +
            volatility_score * weights['volatility'] +
            momentum_score * weights['momentum'] +
            tech_score * weights['technical']
        )

        # Aplica penalidade de drawdown
        dd_penalty = self._calculate_drawdown_penalty(current_dd)
        final_score = max(composite_score - dd_penalty, 0)

        # Determina qualidade
        if final_score >= 80:
            quality = TradeQuality.EXCELLENT
        elif final_score >= 65:
            quality = TradeQuality.GOOD
        elif final_score >= 50:
            quality = TradeQuality.FAIR
        elif final_score >= 35:
            quality = TradeQuality.POOR
        else:
            quality = TradeQuality.REJECT

        # Decisão de trade
        should_trade = final_score >= self.min_score

        # Gera razões
        reasons = []
        if model_proba >= 0.70:
            reasons.append(f"High model confidence ({model_proba:.1%})")
        if regime_score >= 70:
            reasons.append(f"Favorable regime ({regime})")
        if regime_score < 30:
            reasons.append(f"Unfavorable regime ({regime})")
        if current_dd > 0.07:
            reasons.append(f"High drawdown ({current_dd:.1%}) - reduced score")
        if volatility > 0.04:
            reasons.append(f"High volatility ({volatility:.2%})")
        if not should_trade:
            reasons.append(f"Score {final_score:.1f} below minimum {self.min_score:.1f}")

        # Cria sinal
        signal = TradeSignal(
            score=final_score,
            should_trade=should_trade,
            quality=quality,
            confidence_score=confidence_score,
            regime_score=regime_score,
            volatility_score=volatility_score,
            momentum_score=momentum_score,
            drawdown_penalty=dd_penalty,
            model_proba=model_proba,
            regime=regime,
            volatility=volatility,
            current_dd=current_dd,
            component_weights=weights,
            reasons=reasons
        )

        # Adiciona ao histórico
        self.score_history.append(final_score)

        return signal

    def get_statistics(self) -> Dict:
        """Retorna estatísticas dos scores"""
        if not self.score_history:
            return {}

        scores = np.array(self.score_history)
        return {
            'mean': np.mean(scores),
            'median': np.median(scores),
            'std': np.std(scores),
            'min': np.min(scores),
            'max': np.max(scores),
            'above_threshold': np.mean(scores >= self.min_score),
            'total_signals': len(scores)
        }


# ============================================================================
# EXEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    """Demonstração do sistema de scoring"""

    print("🎯 ENSEMBLE SCORING SYSTEM - DEMO")
    print("="*80)

    # Cria scorer
    scorer = EnsembleScorer(min_score=55.0, adaptive_weights=True)

    # Teste 1: Trade excelente
    print("\n📊 TEST 1: EXCELLENT TRADE")
    signal = scorer.score_trade(
        model_proba=0.78,
        regime='medium_bear',
        volatility=0.025,
        current_dd=0.02,
        price_momentum=0.015,
        volume_momentum=1.8
    )
    print(signal)
    print(signal.get_detailed_report())

    # Teste 2: Trade ruim
    print("\n📊 TEST 2: POOR TRADE")
    signal = scorer.score_trade(
        model_proba=0.55,
        regime='low_vol_bull',
        volatility=0.008,
        current_dd=0.08,
        price_momentum=-0.01,
        volume_momentum=0.7
    )
    print(signal)
    print(signal.get_detailed_report())

    # Teste 3: Trade médio
    print("\n📊 TEST 3: FAIR TRADE")
    signal = scorer.score_trade(
        model_proba=0.65,
        regime='high_vol_bear',
        volatility=0.042,
        current_dd=0.04,
        price_momentum=0.005,
        volume_momentum=1.2
    )
    print(signal)

    # Simula múltiplos sinais
    print("\n📊 SIMULATING 100 RANDOM SIGNALS...")
    print("-"*80)

    for _ in range(100):
        scorer.score_trade(
            model_proba=np.random.uniform(0.5, 0.9),
            regime=np.random.choice(list(scorer.regime_performance.keys())),
            volatility=np.random.uniform(0.01, 0.05),
            current_dd=np.random.uniform(0, 0.12),
            price_momentum=np.random.uniform(-0.02, 0.02),
            volume_momentum=np.random.uniform(0.5, 2.0)
        )

    stats = scorer.get_statistics()
    print(f"\n📈 STATISTICS:")
    print(f"  Total signals: {stats['total_signals']}")
    print(f"  Mean score: {stats['mean']:.1f}")
    print(f"  Median score: {stats['median']:.1f}")
    print(f"  Std dev: {stats['std']:.1f}")
    print(f"  Range: {stats['min']:.1f} - {stats['max']:.1f}")
    print(f"  Above threshold (55): {stats['above_threshold']:.1%}")

    print("\n✅ Demo completo!")
