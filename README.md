# 🤖 Bot BTC - Advanced Trading Validation Framework

Sistema avançado de validação para estratégias de trading quantitativas com proteção contra overfitting.

## 🎯 O Que Este Framework Faz

Este framework implementa os métodos mais avançados de validação usados por fundos quantitativos profissionais:

1. **Confidence Filtering** - Filtra trades por probabilidade do modelo
2. **Threshold Optimization** - Encontra threshold ótimo automaticamente
3. **Purged K-Fold CV** - Validação temporal sem leakage de informação
4. **Ensemble Scoring** - Sistema de score composto multi-fator
5. **Regime-Based Validation** - Performance por condição de mercado

## 🚀 Quick Start

### Instalação

```bash
pip install -r requirements.txt
```

### Demo Rápida

```bash
# Testar pipeline completo com dados simulados
python run_advanced_validation.py --demo --samples 1000

# Testar módulos individuais
python confidence_filter.py
python optimize_confidence_threshold.py
python purged_kfold.py
python ensemble_scoring.py
```

## 📚 Documentação

**Leia o guia completo:** [VALIDATION_GUIDE.md](VALIDATION_GUIDE.md)

## 📦 Módulos

| Módulo | Propósito | Status |
|--------|-----------|--------|
| `confidence_filter.py` | Filtro adaptativo por probabilidade | ✅ Testado |
| `optimize_confidence_threshold.py` | Otimizador de threshold | ✅ Testado |
| `purged_kfold.py` | Cross-validation sem leakage | ✅ Testado |
| `ensemble_scoring.py` | Score composto multi-fator | ✅ Testado |
| `run_advanced_validation.py` | Pipeline integrado | ✅ Testado |

## 🎓 Para Quem É Este Framework

- ✅ Traders quantitativos que querem validação robusta
- ✅ Desenvolvedores de bots de trading
- ✅ Pesquisadores em ML para finanças
- ✅ Quem quer evitar overfitting em backtests

## 📊 Resultados Esperados

**Antes (sem filtros):**
- Win Rate: ~50%
- Trades: Muitos (ruído)
- Sharpe: Baixo
- Overfitting: Alto risco

**Depois (com filtros otimizados):**
- Win Rate: 55-60% ⬆️
- Trades: Menos, mais qualidade
- Sharpe: +30-50% ⬆️
- Overfitting: Protegido

## 📖 Referências

Baseado em métodos de:
- Marcos López de Prado - "Advances in Financial Machine Learning"
- Ernest Chan - "Algorithmic Trading"
- David Aronson - "Evidence-Based Technical Analysis"

## 📄 Licença

MIT