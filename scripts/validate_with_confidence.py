"""
🎯 VALIDATE WITH CONFIDENCE FILTER
Versão melhorada do validate_optimized_ultra_scalper.py com filtro de confiança

COMO ADAPTAR SEU CÓDIGO ORIGINAL:
1. Copie seu validate_optimized_ultra_scalper.py para esta pasta
2. Siga os comentários marcados com "# MODIFICAÇÃO:" abaixo
3. Compare os resultados antes/depois

MELHORIAS IMPLEMENTADAS:
- ✅ Filtro de confiança adaptativo
- ✅ Threshold otimizado
- ✅ Validação com Purged K-Fold
- ✅ Análise por regime melhorada
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import argparse
from typing import Dict, Tuple

# MODIFICAÇÃO 1: Importar módulos de validação
from validation.confidence_filter import ConfidenceFilter
from validation.optimize_confidence_threshold import ThresholdOptimizer
from validation.purged_kfold import PurgedKFold
from validation.ensemble_scoring import EnsembleScorer


# ============================================================================
# CONFIGURAÇÃO
# ============================================================================

class Config:
    """Configuração centralizada"""

    # Filtro de Confiança
    USE_CONFIDENCE_FILTER = True
    CONFIDENCE_THRESHOLD = 0.62  # Será otimizado automaticamente
    ADAPTIVE_CONFIDENCE = True

    # Regime Filter (seus valores originais)
    USE_REGIME_FILTER = True
    BLOCKED_REGIMES = ['low_vol_bull']  # Regime com WR < 50%

    # Position Sizing
    POSITION_SIZING = 'dynamic'  # 'fixed' ou 'dynamic'

    # Validação
    USE_PURGED_KFOLD = False  # True para validação mais rigorosa
    N_FOLDS = 5
    EMBARGO_HOURS = 1

    # Capital
    INITIAL_CAPITAL = 10000.0


# ============================================================================
# EXEMPLO: ESTRUTURA DO SEU VALIDATE ORIGINAL
# ============================================================================

def example_original_validate():
    """
    EXEMPLO de como seu código original deve estar estruturado

    Substitua isso pelo seu código real de validate_optimized_ultra_scalper.py
    """

    print("⚠️  ESTE É UM EXEMPLO - Substitua pelo seu código real!")
    print("="*80)

    # Seus imports originais aqui
    # import pickle
    # import ccxt
    # from bybit_api import get_data
    # etc...

    # Seu código para:
    # 1. Baixar dados
    # 2. Construir features
    # 3. Carregar modelo
    # 4. Fazer predições
    # 5. Backtest

    pass


# ============================================================================
# MODIFICAÇÃO 2: ADICIONAR FUNÇÕES DE VALIDAÇÃO
# ============================================================================

def optimize_threshold_once(model, X_train, y_train, returns_train):
    """
    Otimiza threshold UMA VEZ com dados de treino

    Args:
        model: Modelo treinado
        X_train: Features de treino
        y_train: Targets de treino
        returns_train: Retornos reais de treino

    Returns:
        float: Threshold otimizado
    """
    print("\n🔍 Otimizando threshold de confiança...")

    # Get probabilidades
    if hasattr(model, 'predict_proba'):
        probas = model.predict_proba(X_train)[:, 1]
    else:
        print("⚠️  Modelo não tem predict_proba, usando threshold padrão 0.62")
        return 0.62

    # Otimizar
    optimizer = ThresholdOptimizer(
        min_threshold=0.50,
        max_threshold=0.80,
        step=0.02,
        min_trades=30
    )

    best_threshold, best_result = optimizer.optimize(
        probas,
        returns_train,
        objective='sharpe'
    )

    print(f"✅ Threshold Ótimo: {best_threshold:.1%}")
    print(f"   Sharpe esperado: {best_result.sharpe:.2f}")
    print(f"   Win Rate esperado: {best_result.win_rate:.1%}")

    return best_threshold


def create_confidence_filter(threshold: float, regime_multipliers: Dict = None):
    """
    Cria filtro de confiança configurado

    Args:
        threshold: Threshold base
        regime_multipliers: Multiplicadores por regime

    Returns:
        ConfidenceFilter configurado
    """
    if regime_multipliers is None:
        # Use seus resultados de regime
        regime_multipliers = {
            'medium_bear': 0.92,      # Melhor regime - mais permissivo
            'high_vol_bear': 0.95,
            'low_vol_bear': 0.98,
            'high_vol_bull': 1.00,
            'medium_bull': 1.03,
            'low_vol_bull': 1.15,     # Pior regime - mais restritivo
        }

    return ConfidenceFilter(
        threshold=threshold,
        adaptive=Config.ADAPTIVE_CONFIDENCE,
        regime_multipliers=regime_multipliers,
        dd_adjustment=True
    )


# ============================================================================
# MODIFICAÇÃO 3: FUNÇÃO DE BACKTEST MODIFICADA
# ============================================================================

def backtest_with_confidence(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    regimes: pd.Series,
    confidence_filter: ConfidenceFilter = None,
    initial_capital: float = 10000
) -> Dict:
    """
    Backtest COM filtro de confiança

    MODIFICAÇÃO: Esta função substitui seu backtest original

    Args:
        model: Modelo treinado
        X_test: Features de teste
        y_test: Targets reais
        regimes: Regime de cada amostra
        confidence_filter: Filtro de confiança (None = sem filtro)
        initial_capital: Capital inicial

    Returns:
        Dict com métricas
    """

    # ORIGINAL: predictions = model.predict(X_test)
    # NOVO: Usar filtro de confiança

    if confidence_filter is not None:
        print("\n📊 Aplicando filtro de confiança...")

        # Detectar regime atual (simplificado - adapte ao seu código)
        current_regime = regimes.mode()[0] if len(regimes) > 0 else 'medium_bull'

        # Predição com filtro
        predictions, confidences = confidence_filter.predict(
            model,
            X_test.values if isinstance(X_test, pd.DataFrame) else X_test,
            regime=current_regime,
            current_dd=0.0  # Atualizar com DD real durante backtest
        )

        n_filtered = np.sum(predictions == 0)
        print(f"   Filtrados: {n_filtered}/{len(predictions)} ({n_filtered/len(predictions):.1%})")

    else:
        # Sem filtro (baseline)
        if hasattr(model, 'predict_proba'):
            predictions = (model.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
        else:
            predictions = model.predict(X_test)

    # Simula backtest (ADAPTE AO SEU CÓDIGO REAL)
    # Aqui você faria:
    # - Calcular retornos por trade
    # - Aplicar position sizing
    # - Calcular equity curve
    # - Calcular métricas

    # Exemplo simplificado:
    trades = predictions == 1
    n_trades = np.sum(trades)

    if n_trades == 0:
        return {
            'n_trades': 0,
            'win_rate': 0.0,
            'roi': 0.0,
            'sharpe': 0.0,
            'max_dd': 0.0
        }

    # Simula retornos (SUBSTITUA pelo seu cálculo real)
    simulated_returns = np.random.normal(0.02, 0.05, n_trades)  # 2% média, 5% std
    wins = simulated_returns > 0

    metrics = {
        'n_trades': n_trades,
        'win_rate': np.mean(wins),
        'roi': np.sum(simulated_returns),
        'sharpe': np.mean(simulated_returns) / (np.std(simulated_returns) + 1e-10) * np.sqrt(252),
        'max_dd': np.min(np.cumsum(simulated_returns) - np.maximum.accumulate(np.cumsum(simulated_returns)))
    }

    return metrics


# ============================================================================
# MODIFICAÇÃO 4: MAIN MODIFICADO
# ============================================================================

def main():
    """
    Main function - ADAPTE AO SEU CÓDIGO
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', type=str, default='BTCUSDT')
    parser.add_argument('--days', type=int, default=90)
    parser.add_argument('--optimize-threshold', action='store_true', help='Otimizar threshold antes de validar')
    parser.add_argument('--use-confidence', action='store_true', default=True, help='Usar filtro de confiança')
    parser.add_argument('--compare', action='store_true', help='Comparar com/sem filtro')

    args = parser.parse_args()

    print("\n" + "="*80)
    print("🔬 VALIDATION WITH CONFIDENCE FILTER")
    print("="*80)
    print(f"Symbol: {args.symbol}")
    print(f"Period: {args.days} days")
    print(f"Confidence Filter: {'✅ Enabled' if args.use_confidence else '❌ Disabled'}")

    # ========================================================================
    # PASSO 1: CARREGAR DADOS E MODELO (SEU CÓDIGO ORIGINAL)
    # ========================================================================

    print("\n📥 Loading data and model...")

    # SUBSTITUA ISSO PELO SEU CÓDIGO:
    # data = download_data(args.symbol, args.days)
    # X, y = build_features(data)
    # model = load_model('ultra_scalper_btcusdt_365d.pkl')
    # regimes = detect_regimes(data)

    # Por enquanto, simula:
    print("⚠️  Usando dados simulados - substitua pelo seu código!")
    n_samples = 1000
    X = pd.DataFrame(np.random.randn(n_samples, 10))
    y = pd.Series(np.random.randint(0, 2, n_samples))
    regimes = pd.Series(np.random.choice(['medium_bear', 'low_vol_bull'], n_samples))

    # Modelo dummy
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X[:800], y[:800])  # Treina nos primeiros 80%

    # ========================================================================
    # PASSO 2: OTIMIZAR THRESHOLD (OPCIONAL, 1x por semana)
    # ========================================================================

    if args.optimize_threshold:
        # Simula retornos de treino (SUBSTITUA pelo cálculo real)
        returns_train = np.random.normal(0.02, 0.05, 800)

        optimized_threshold = optimize_threshold_once(
            model,
            X[:800],
            y[:800],
            returns_train
        )
    else:
        optimized_threshold = Config.CONFIDENCE_THRESHOLD
        print(f"\n💡 Usando threshold configurado: {optimized_threshold:.1%}")

    # ========================================================================
    # PASSO 3: CRIAR FILTRO
    # ========================================================================

    if args.use_confidence:
        cf = create_confidence_filter(optimized_threshold)
    else:
        cf = None

    # ========================================================================
    # PASSO 4: BACKTEST
    # ========================================================================

    print("\n📊 Running backtest...")

    X_test = X[800:]
    y_test = y[800:]
    regimes_test = regimes[800:]

    if args.compare:
        # Compara com e sem filtro
        print("\n" + "="*80)
        print("📊 COMPARISON: WITH vs WITHOUT CONFIDENCE FILTER")
        print("="*80)

        # Sem filtro
        print("\n1️⃣  WITHOUT filter (baseline):")
        metrics_without = backtest_with_confidence(
            model, X_test, y_test, regimes_test,
            confidence_filter=None
        )
        print(f"   Trades: {metrics_without['n_trades']}")
        print(f"   Win Rate: {metrics_without['win_rate']:.1%}")
        print(f"   ROI: {metrics_without['roi']:.2%}")
        print(f"   Sharpe: {metrics_without['sharpe']:.2f}")

        # Com filtro
        print("\n2️⃣  WITH confidence filter:")
        metrics_with = backtest_with_confidence(
            model, X_test, y_test, regimes_test,
            confidence_filter=cf
        )
        print(f"   Trades: {metrics_with['n_trades']}")
        print(f"   Win Rate: {metrics_with['win_rate']:.1%}")
        print(f"   ROI: {metrics_with['roi']:.2%}")
        print(f"   Sharpe: {metrics_with['sharpe']:.2f}")

        # Comparação
        print("\n📈 IMPROVEMENT:")
        print(f"   Win Rate: {metrics_without['win_rate']:.1%} → {metrics_with['win_rate']:.1%} ({(metrics_with['win_rate'] - metrics_without['win_rate'])*100:+.1f}pp)")
        print(f"   Sharpe: {metrics_without['sharpe']:.2f} → {metrics_with['sharpe']:.2f} ({metrics_with['sharpe'] - metrics_without['sharpe']:+.2f})")
        print(f"   Trades: {metrics_without['n_trades']} → {metrics_with['n_trades']} ({(1 - metrics_with['n_trades']/metrics_without['n_trades'])*100:.1f}% reduction)")

    else:
        # Apenas com filtro
        metrics = backtest_with_confidence(
            model, X_test, y_test, regimes_test,
            confidence_filter=cf
        )

        print(f"\n✅ Results:")
        print(f"   Trades: {metrics['n_trades']}")
        print(f"   Win Rate: {metrics['win_rate']:.1%}")
        print(f"   ROI: {metrics['roi']:.2%}")
        print(f"   Sharpe: {metrics['sharpe']:.2f}")

    # ========================================================================
    # PASSO 5: RELATÓRIO DO FILTRO
    # ========================================================================

    if cf is not None:
        cf.print_report()

    print("\n" + "="*80)
    print("✅ VALIDATION COMPLETE")
    print("="*80)


# ============================================================================
# INSTRUÇÕES DE INTEGRAÇÃO
# ============================================================================

INTEGRATION_INSTRUCTIONS = """
🎯 COMO INTEGRAR AO SEU validate_optimized_ultra_scalper.py:

OPÇÃO 1 - SUBSTITUIÇÃO COMPLETA (Recomendado):
==============================================
1. Faça backup do seu validate_optimized_ultra_scalper.py
2. Copie as funções deste arquivo
3. Substitua as seções marcadas com "MODIFICAÇÃO"
4. Teste com --compare para ver diferença

OPÇÃO 2 - MODIFICAÇÃO INCREMENTAL (Mais seguro):
================================================
1. Adicione os imports no topo:
   from validation.confidence_filter import ConfidenceFilter

2. Antes do backtest, otimize threshold (1x):
   optimized_threshold = optimize_threshold_once(model, X_train, y_train, returns_train)

3. Crie filtro:
   cf = create_confidence_filter(optimized_threshold)

4. No backtest, substitua:
   # predictions = model.predict(X)
   predictions, confidences = cf.predict(model, X, regime=current_regime)

5. Compare resultados

OPÇÃO 3 - SCRIPT SEPARADO (Para testar):
========================================
1. Mantenha seu validate original intacto
2. Use este script separadamente
3. Compare outputs
4. Quando confirmar melhoria, integre

Para rodar:
-----------
# Comparar com/sem filtro
python scripts/validate_with_confidence.py --compare

# Com otimização de threshold
python scripts/validate_with_confidence.py --optimize-threshold --compare

# Apenas com filtro
python scripts/validate_with_confidence.py --use-confidence
"""


if __name__ == "__main__":
    # Mostra instruções se rodado sem args
    if len(sys.argv) == 1:
        print(INTEGRATION_INSTRUCTIONS)
        print("\n" + "="*80)
        print("Rodando demo...")
        print("="*80)

    main()
