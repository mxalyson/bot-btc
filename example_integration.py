"""
🎯 EXEMPLO DE INTEGRAÇÃO COM SEU BOT EXISTENTE

Este arquivo mostra como integrar os módulos de validação avançada
ao seu validate_optimized_ultra_scalper.py

ANTES DE COMEÇAR:
1. Leia VALIDATION_GUIDE.md
2. Teste os módulos individualmente
3. Adapte este exemplo ao seu código
"""

import numpy as np
import pandas as pd
from confidence_filter import ConfidenceFilter
from optimize_confidence_threshold import ThresholdOptimizer
from purged_kfold import PurgedKFold
from ensemble_scoring import EnsembleScorer


# ============================================================================
# EXEMPLO 1: ADICIONAR FILTRO DE CONFIANÇA AO SEU VALIDATE
# ============================================================================

def example_1_add_confidence_filter():
    """
    Como adicionar filtro de confiança ao seu backtest existente
    """
    print("\n" + "="*80)
    print("EXEMPLO 1: ADICIONAR CONFIDENCE FILTER")
    print("="*80)

    # Simula seu modelo e dados
    class DummyModel:
        def predict_proba(self, X):
            n = len(X)
            return np.column_stack([
                np.random.uniform(0.3, 0.7, n),
                np.random.uniform(0.3, 0.7, n)
            ])

    model = DummyModel()
    X = pd.DataFrame(np.random.randn(1000, 10))

    # CÓDIGO ORIGINAL (sem filtro):
    print("\n📍 ANTES (sem filtro):")
    predictions_original = model.predict_proba(X)[:, 1] >= 0.5  # Threshold fixo 50%
    n_trades_original = np.sum(predictions_original)
    print(f"   Trades gerados: {n_trades_original}")

    # CÓDIGO NOVO (com filtro):
    print("\n✅ DEPOIS (com filtro otimizado):")

    # 1. Criar filtro
    cf = ConfidenceFilter(
        threshold=0.62,  # Usar threshold otimizado
        adaptive=True,   # Adaptar por regime
        regime_multipliers={
            'medium_bear': 0.92,     # Reduz threshold (melhor regime)
            'low_vol_bull': 1.15,    # Aumenta threshold (pior regime)
        }
    )

    # 2. Predição com filtro
    current_regime = 'medium_bear'
    current_dd = 0.03

    predictions_filtered, confidences = cf.predict(
        model,
        X,
        regime=current_regime,
        current_dd=current_dd
    )

    n_trades_filtered = np.sum(predictions_filtered)
    print(f"   Trades gerados: {n_trades_filtered}")
    print(f"   Redução: {(1 - n_trades_filtered/n_trades_original)*100:.1f}%")

    # 3. Ver estatísticas
    cf.print_report()

    # COMO INTEGRAR NO SEU CÓDIGO:
    print("\n💡 CÓDIGO PARA ADICIONAR AO SEU validate_optimized_ultra_scalper.py:")
    print("""
    # No início do arquivo:
    from confidence_filter import ConfidenceFilter

    # Dentro da função validate():
    cf = ConfidenceFilter(threshold=OPTIMIZED_THRESHOLD, adaptive=True)

    # Substituir:
    # predictions = model.predict(X_test)

    # Por:
    predictions, confidences = cf.predict(
        model,
        X_test,
        regime=current_regime,
        current_dd=current_drawdown
    )
    """)


# ============================================================================
# EXEMPLO 2: OTIMIZAR THRESHOLD ANTES DO BACKTEST
# ============================================================================

def example_2_optimize_threshold():
    """
    Como otimizar threshold antes de rodar backtest completo
    """
    print("\n" + "="*80)
    print("EXEMPLO 2: OTIMIZAR THRESHOLD")
    print("="*80)

    # Simula dados históricos
    n_samples = 2000
    probas = np.random.beta(2, 2, n_samples)  # Probabilidades do modelo
    returns = (probas - 0.5) * 0.08 + np.random.normal(0, 0.02, n_samples)  # Retornos

    print(f"\n📊 Dados: {n_samples} trades históricos")

    # Criar otimizador
    optimizer = ThresholdOptimizer(
        min_threshold=0.50,
        max_threshold=0.80,
        step=0.02,
        min_trades=50
    )

    # Otimizar para Sharpe
    best_threshold, best_result = optimizer.optimize(
        probas,
        returns,
        objective='sharpe'
    )

    print(f"\n✅ Threshold Ótimo: {best_threshold:.1%}")
    print(f"   Win Rate: {best_result.win_rate:.1%}")
    print(f"   Sharpe: {best_result.sharpe:.2f}")
    print(f"   Trades: {best_result.total_trades}")

    # Recomendações por perfil
    print("\n💡 Recomendações por perfil:")
    recs = optimizer.get_recommendations()
    for profile, threshold in recs.items():
        print(f"   {profile:30s}: {threshold:.1%}")

    # COMO USAR:
    print("\n💡 WORKFLOW RECOMENDADO:")
    print("""
    1. Rodar otimizador com dados dos últimos 3-6 meses
    2. Escolher threshold baseado no perfil (conservador/balanceado/agressivo)
    3. Usar threshold no backtest e produção
    4. Re-otimizar mensalmente ou após retreino
    """)


# ============================================================================
# EXEMPLO 3: VALIDAR COM PURGED K-FOLD
# ============================================================================

def example_3_purged_validation():
    """
    Como substituir walk-forward por Purged K-Fold
    """
    print("\n" + "="*80)
    print("EXEMPLO 3: PURGED K-FOLD VALIDATION")
    print("="*80)

    # Simula dados com timestamps
    dates = pd.date_range('2024-01-01', periods=1000, freq='5min')
    X = pd.DataFrame(np.random.randn(1000, 10), index=dates)
    y = pd.Series(np.random.randint(0, 2, 1000), index=dates)

    # Criar validador
    pkf = PurgedKFold(
        n_splits=5,
        embargo_td=pd.Timedelta(hours=1)  # 1h de embargo após cada test set
    )

    print(f"\n📊 Validando com {pkf.get_n_splits()} folds...")

    # Simula modelo
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=10, random_state=42)

    # Validação
    fold_results = []
    for fold, (train_idx, test_idx) in enumerate(pkf.split(X, y)):
        # Treinar
        model.fit(X.iloc[train_idx], y.iloc[train_idx])

        # Testar
        y_pred = model.predict(X.iloc[test_idx])
        accuracy = np.mean(y_pred == y.iloc[test_idx])

        fold_results.append({
            'fold': fold + 1,
            'train_size': len(train_idx),
            'test_size': len(test_idx),
            'accuracy': accuracy
        })

        print(f"   Fold {fold+1}: Train={len(train_idx):4d}, Test={len(test_idx):4d}, Acc={accuracy:.1%}")

    avg_acc = np.mean([r['accuracy'] for r in fold_results])
    print(f"\n✅ Accuracy Média: {avg_acc:.1%}")

    # DIFERENÇA DO WALK-FORWARD NORMAL:
    print("\n💡 DIFERENÇA DO SEU WALK-FORWARD ATUAL:")
    print("""
    Walk-Forward Normal:
    - Pode vazar informação entre folds
    - Amostras adjacentes são similares
    - Sharpe pode estar inflado

    Purged K-Fold:
    - Remove samples entre train/test (purge + embargo)
    - Previne leakage temporal
    - Sharpe mais realista (pode cair 10-20%)

    ⚠️  Se Sharpe cair muito (>30%), modelo está overfitted!
    """)


# ============================================================================
# EXEMPLO 4: ENSEMBLE SCORING EM PRODUÇÃO
# ============================================================================

def example_4_ensemble_scoring():
    """
    Como usar ensemble scoring para decisões de trade
    """
    print("\n" + "="*80)
    print("EXEMPLO 4: ENSEMBLE SCORING EM PRODUÇÃO")
    print("="*80)

    # Performance histórica por regime (do seu backtest)
    regime_performance = {
        'medium_bear': {'win_rate': 0.569, 'avg_roi': 0.2863, 'sharpe': 3.5, 'quality': 1.0},
        'high_vol_bear': {'win_rate': 0.505, 'avg_roi': 0.2483, 'sharpe': 3.0, 'quality': 0.9},
        'low_vol_bear': {'win_rate': 0.512, 'avg_roi': 0.1309, 'sharpe': 2.0, 'quality': 0.7},
        'high_vol_bull': {'win_rate': 0.426, 'avg_roi': 0.1696, 'sharpe': 1.5, 'quality': 0.6},
        'medium_bull': {'win_rate': 0.405, 'avg_roi': 0.0589, 'sharpe': 0.8, 'quality': 0.4},
        'low_vol_bull': {'win_rate': 0.331, 'avg_roi': 0.0072, 'sharpe': 0.1, 'quality': 0.1},
    }

    # Criar scorer
    scorer = EnsembleScorer(
        min_score=55.0,  # Score mínimo para tradear
        regime_performance=regime_performance,
        adaptive_weights=True
    )

    # Simula situações de mercado
    scenarios = [
        {
            'name': 'EXCELENTE',
            'model_proba': 0.75,
            'regime': 'medium_bear',
            'volatility': 0.025,
            'current_dd': 0.01,
        },
        {
            'name': 'RUIM',
            'model_proba': 0.58,
            'regime': 'low_vol_bull',
            'volatility': 0.008,
            'current_dd': 0.08,
        },
        {
            'name': 'MÉDIO',
            'model_proba': 0.65,
            'regime': 'high_vol_bear',
            'volatility': 0.042,
            'current_dd': 0.03,
        }
    ]

    for scenario in scenarios:
        print(f"\n📊 Cenário: {scenario['name']}")
        print(f"   Probabilidade: {scenario['model_proba']:.1%}")
        print(f"   Regime: {scenario['regime']}")

        signal = scorer.score_trade(
            model_proba=scenario['model_proba'],
            regime=scenario['regime'],
            volatility=scenario['volatility'],
            current_dd=scenario['current_dd']
        )

        print(f"\n   Score: {signal.score:.1f}/100")
        print(f"   Quality: {signal.quality.value.upper()}")
        print(f"   Decisão: {'✅ TRADEAR' if signal.should_trade else '❌ PULAR'}")

    # CÓDIGO DE INTEGRAÇÃO:
    print("\n💡 CÓDIGO PARA INTEGRAR NO BOT:")
    print("""
    # Setup (uma vez)
    scorer = EnsembleScorer(min_score=55.0, regime_performance=REGIME_STATS)

    # Em cada iteração do bot:
    signal = scorer.score_trade(
        model_proba=model.predict_proba(X_current)[0, 1],
        regime=detect_current_regime(),
        volatility=calculate_current_volatility(),
        current_dd=calculate_current_drawdown()
    )

    # Decisão multi-nível:
    if signal.quality.value == 'excellent':
        position_size = BASE_SIZE * 1.5  # Aumenta posição
        execute_trade(position_size)
    elif signal.quality.value == 'good':
        position_size = BASE_SIZE
        execute_trade(position_size)
    elif signal.quality.value == 'fair':
        position_size = BASE_SIZE * 0.5  # Reduz posição
        execute_trade(position_size)
    else:
        # Skip poor/reject
        pass
    """)


# ============================================================================
# EXEMPLO 5: PIPELINE COMPLETO
# ============================================================================

def example_5_complete_pipeline():
    """
    Pipeline completo de otimização e validação
    """
    print("\n" + "="*80)
    print("EXEMPLO 5: PIPELINE COMPLETO")
    print("="*80)

    print("""
    WORKFLOW RECOMENDADO PARA SEU BOT:

    1️⃣  DESENVOLVIMENTO (quando criar/atualizar modelo):

        a) Coletar dados históricos (6+ meses)

        b) Treinar modelo base

        c) Otimizar threshold:
           optimizer = ThresholdOptimizer()
           best_threshold, _ = optimizer.optimize(probas, returns)

        d) Validar com Purged K-Fold:
           pkf = PurgedKFold(n_splits=5)
           for train, test in pkf.split(X):
               validate_fold(train, test)

        e) Analisar por regime:
           for regime in regimes:
               validate_regime(regime)

        f) Se passar validação → Deploy


    2️⃣  PRODUÇÃO (trading ao vivo):

        a) Usar ConfidenceFilter OU EnsembleScorer

        b) Filtrar trades:
           if proba >= best_threshold:
               trade()

           # OU (mais avançado):
           signal = scorer.score_trade(...)
           if signal.should_trade:
               trade()

        c) Logar todas decisões


    3️⃣  MONITORAMENTO (diário):

        a) Comparar live vs backtest metrics

        b) Detectar regime changes

        c) Alertar se performance diverge


    4️⃣  MANUTENÇÃO (mensal):

        a) Re-otimizar threshold com dados recentes

        b) Re-validar com Purged K-Fold

        c) Ajustar filtros se necessário

        d) Retreinar modelo se drift detectado
    """)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("🎯 EXEMPLOS DE INTEGRAÇÃO - ADVANCED VALIDATION")
    print("="*80)

    # Rodar exemplos
    example_1_add_confidence_filter()
    example_2_optimize_threshold()
    example_3_purged_validation()
    example_4_ensemble_scoring()
    example_5_complete_pipeline()

    print("\n" + "="*80)
    print("✅ EXEMPLOS COMPLETOS!")
    print("="*80)
    print("\n💡 PRÓXIMOS PASSOS:")
    print("   1. Leia VALIDATION_GUIDE.md")
    print("   2. Teste run_advanced_validation.py --demo")
    print("   3. Adapte os exemplos acima ao seu código")
    print("   4. Valide com seus dados reais")
    print("   5. Deploy gradual (paper → small → full)")
    print("="*80 + "\n")
