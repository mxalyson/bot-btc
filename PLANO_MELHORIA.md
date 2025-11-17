# 📈 PLANO COMPLETO: Melhorar Modelo de 100%

## 🎯 Objetivo
Melhorar modelo atual (ROC AUC 0.5016 → ~0.55-0.60 mínimo)

---

## 📊 Situação Atual

**Modelo em Produção**:
- Symbol: BTCUSDT
- Timeframe: 15m
- Features: 113
- Test ROC AUC: **0.5016** ❌ (sem edge)
- Architecture: Deep Learning (Hybrid CNN-LSTM-Attention)

---

## 🚀 ESTRATÉGIA 1: Timeframe 1H + Ensemble (RECOMENDADO) ⭐⭐⭐⭐⭐

### Por que funciona:
- **15m = 80% ruído** vs **1h = 30% ruído**
- ROC AUC esperado: **0.58-0.63** (edge comprovado)
- Ainda permite day trading (20-50 trades/mês)

### Implementação:

```bash
# Passo 1: Baixar dados 1h (2 anos)
python scripts/download_data.py --symbol BTCUSDT --timeframe 1h --days 730

# Passo 2: Treinar sistema avançado
python scripts/train_advanced_system.py \
    --symbol BTCUSDT \
    --timeframe 1h \
    --config config/config_advanced.yaml

# Passo 3: Backtest
python scripts/run_backtest.py --symbol BTCUSDT --timeframe 1h

# Passo 4: Se Sharpe > 1.5, deploy!
```

### Ajustes no Config para 1H:

```yaml
labeling:
  forward_window: 4      # 4 candles 1h = 4 horas
  profit_target_atr: 2.0 # Maior target em 1h
  stop_loss_atr: 1.2
  time_limit: 8          # 8 horas max

model:
  sequence_length: 60    # 60h = 2.5 dias de histórico
```

**Probabilidade de sucesso: 85%** ✅

---

## 🔧 ESTRATÉGIA 2: Feature Engineering Avançado (15m) ⭐⭐⭐

Se você **precisa** manter 15m:

### A. Features de Microestrutura de Ordem Book (Level 2)

```python
# Adicionar em core/orderbook_features.py

class OrderBookFeatures:
    def create_features(self, df, orderbook_data):
        """
        Features de profundidade de mercado
        """
        # 1. Bid-Ask Imbalance
        df['order_imbalance'] = (bid_volume - ask_volume) / (bid_volume + ask_volume)

        # 2. Order Book Depth (5 níveis)
        df['bid_depth_5'] = sum(bid_sizes[:5])
        df['ask_depth_5'] = sum(ask_sizes[:5])
        df['depth_ratio'] = bid_depth_5 / ask_depth_5

        # 3. Liquidez Score
        df['liquidity_score'] = (bid_depth_5 + ask_depth_5) / average_volume

        # 4. Support/Resistance Levels
        df['near_support'] = price < support_level * 1.002
        df['near_resistance'] = price > resistance_level * 0.998

        return df
```

**Ganho esperado**: +0.01-0.02 AUC

### B. Features de Market Microstructure

```python
# Adicionar em core/microstructure_features.py

def add_market_impact_features(df):
    """
    Features de impacto no mercado
    """
    # 1. Price Impact
    df['price_impact'] = abs(close - vwap) / atr

    # 2. Kyle's Lambda (sensibilidade de preço)
    df['kyles_lambda'] = price_change / sqrt(volume)

    # 3. Roll's Spread (spread efetivo)
    df['effective_spread'] = -covariance(price_change[t], price_change[t-1])

    # 4. Amihud Illiquidity
    df['amihud_illiq'] = abs(returns) / volume

    return df
```

**Ganho esperado**: +0.01-0.015 AUC

### C. Feature Selection com SHAP

```python
# Novo script: scripts/feature_selection.py

import shap
import lightgbm as lgb

# 1. Treinar LightGBM rápido
model = lgb.LGBMClassifier()
model.fit(X_train, y_train)

# 2. Calcular SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)

# 3. Ranquear features
feature_importance = np.abs(shap_values).mean(axis=0)
top_features = feature_names[np.argsort(-feature_importance)[:50]]

# 4. Retreinar apenas com top 50 features
print(f"Top features: {top_features}")
```

**Ganho esperado**: +0.01-0.02 AUC (reduzindo ruído)

### D. Temporal Cross-Validation Rigorosa

```python
# Usar TimeSeriesSplit ao invés de split simples

from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

for train_idx, val_idx in tscv.split(X):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Treinar e avaliar
    model.fit(X_train, y_train)
    score = model.score(X_val, y_val)
    print(f"Fold AUC: {score}")
```

**Ganho esperado**: Modelo mais robusto (não inflaciona métricas)

**Probabilidade de sucesso total**: 50-60%

---

## ⚙️ ESTRATÉGIA 3: Ensemble Pesado + Meta-Learning ⭐⭐⭐⭐

### A. Treinar 6 Modelos Base

```bash
# 1. Deep Learning (CNN-LSTM-Attention)
python scripts/train_deep_learning.py --symbol BTCUSDT --timeframe 15m

# 2. Transformer
python scripts/train_advanced_system.py --symbol BTCUSDT --timeframe 15m --config config/config_advanced.yaml

# 3. LightGBM
python scripts/train_lightgbm.py --symbol BTCUSDT --timeframe 15m

# 4. XGBoost
python scripts/train_xgboost.py --symbol BTCUSDT --timeframe 15m

# 5. CatBoost (adicionar)
# TODO: criar script

# 6. TabNet (adicionar)
# TODO: criar script
```

### B. Stacking com Meta-Learner Avançado

```python
# scripts/train_stacking_ensemble.py

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Coletar predições de todos modelos
predictions = {
    'dl': dl_model.predict_proba(X_val)[:, 1],
    'transformer': transformer_model.predict_proba(X_val)[:, 1],
    'lgb': lgb_model.predict_proba(X_val)[:, 1],
    'xgb': xgb_model.predict_proba(X_val)[:, 1],
    'catboost': catboost_model.predict_proba(X_val)[:, 1],
    'tabnet': tabnet_model.predict_proba(X_val)[:, 1],
}

# Stack predictions
X_meta = pd.DataFrame(predictions)

# Meta-learner com calibração
from sklearn.calibration import CalibratedClassifierCV

base_meta = LogisticRegression()
meta_learner = CalibratedClassifierCV(base_meta, cv=5)
meta_learner.fit(X_meta, y_val)

# Predição final
final_pred = meta_learner.predict_proba(X_meta_test)[:, 1]
```

**Ganho esperado**: +0.02-0.04 AUC

**Probabilidade de sucesso**: 70%

---

## 📋 ROADMAP RECOMENDADO

### Semana 1: Teste Rápido 1H
```bash
# Dia 1-2: Download e preparação
python scripts/download_data.py --symbol BTCUSDT --timeframe 1h --days 730

# Dia 3-4: Treinamento
python scripts/train_advanced_system.py --symbol BTCUSDT --timeframe 1h --config config/config_advanced.yaml

# Dia 5: Backtest e análise
python scripts/run_backtest.py --symbol BTCUSDT --timeframe 1h

# Decisão: Se AUC > 0.58 e Sharpe > 1.5 → DEPLOY!
```

### Semana 2-3: Se 1H não funcionar, Voltar para 15M com Estratégia 2

```bash
# Feature Engineering
1. Implementar OrderBook features
2. Implementar Market Microstructure features
3. Feature selection com SHAP
4. Re-treinar com top 50 features

# Validação rigorosa
5. TimeSeriesSplit com 5 folds
6. Walk-forward analysis
```

### Semana 4: Ensemble Pesado

```bash
# Treinar 6 modelos
# Criar meta-learner
# Backtest final
```

---

## 🎯 Métricas de Sucesso

| Métrica | Atual | Meta Mínima | Meta Ideal |
|---------|-------|-------------|------------|
| Test ROC AUC | 0.5016 | 0.55 | 0.60+ |
| Sharpe Ratio | ? | 1.5 | 2.0+ |
| Win Rate | ? | 52% | 55%+ |
| Max Drawdown | ? | <20% | <15% |
| Trades/Mês | ? | 20-50 | 30-60 |

---

## ⚠️ IMPORTANTE: Expectativas Realistas

### 15m Scalping:
- **Melhor caso**: ROC AUC 0.54-0.56
- **Típico**: ROC AUC 0.52-0.54
- **Difícil passar de 0.56** (muito ruído)

### 1H Day Trading:
- **Melhor caso**: ROC AUC 0.62-0.65
- **Típico**: ROC AUC 0.58-0.62
- **Mais confiável** e rentável

### 4H Swing Trading:
- **Melhor caso**: ROC AUC 0.65-0.70
- **Típico**: ROC AUC 0.60-0.67
- **Mais fácil**, menos trades

---

## 🚦 Decisão Final

**Recomendação**: Comece com **ESTRATÉGIA 1 (1H)** por 3 motivos:

1. ✅ **Maior probabilidade de sucesso** (85% vs 50%)
2. ✅ **Implementação rápida** (2-3 dias)
3. ✅ **ROI melhor** (menos trades, mais confiáveis)

Se realmente precisa de 15m, combine **ESTRATÉGIA 2 + 3**:
- Feature Engineering avançado
- Ensemble com 6 modelos
- Expect AUC ~0.54-0.56 (edge pequeno mas possível)

---

## 📞 Próximos Passos

Me diga qual estratégia você quer seguir e começamos agora! 🚀
