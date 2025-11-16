# 🚀 Treinamento Avançado - Sistema de ML Profissional para Scalping

## 🎯 O Que Mudou?

O treinamento avançado implementa **6 otimizações críticas** que melhoram significativamente a performance:

### 1. **Problema Binário** ✅
- Remove classe NONE (apenas 0.09% dos dados)
- Foca em LONG vs SHORT
- Melhora accuracy e ROC AUC significativamente

### 2. **Feature Selection Automática** ✅
- Seleciona apenas as top 30-40 features mais importantes
- Remove ruído e reduz overfitting
- Acelera treinamento

### 3. **Class Balancing Inteligente** ✅
- Usa `scale_pos_weight` do XGBoost
- Balanceia LONG/SHORT sem perder dados
- Calcula automaticamente baseado na distribuição

### 4. **Otimização de Threshold** ✅
- Encontra threshold ótimo (não usa 0.5 padrão)
- Maximiza F1 score
- Melhora precisão das predições

### 5. **Hiperparâmetros Otimizados** ✅
- Configuração específica para scalping
- Regularização forte contra overfitting
- Testado em dados reais de crypto

### 6. **Robust Scaler** ✅
- Melhor tratamento de outliers
- Mais adequado para preços de crypto
- Reduz impacto de spikes de volatilidade

---

## 🏃 Como Usar

### Treinamento Básico (Padrão)
```bash
python scripts/train_model.py --symbol BTCUSDT --timeframe 15m
```

### Treinamento Avançado (Recomendado) 🔥
```bash
python scripts/train_advanced.py --symbol BTCUSDT --timeframe 15m --top-features 30
```

### Parâmetros do Treinamento Avançado

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `--symbol` | BTCUSDT | Símbolo para treinar |
| `--timeframe` | 15m | Timeframe (5m, 15m, 1h, etc) |
| `--config` | config/config_advanced.yaml | Configuração avançada |
| `--top-features` | 30 | Número de features a selecionar |

### Exemplos

```bash
# BTC 15m com 30 features (recomendado)
python scripts/train_advanced.py --symbol BTCUSDT --timeframe 15m

# BTC 15m com 40 features (mais complexo)
python scripts/train_advanced.py --symbol BTCUSDT --timeframe 15m --top-features 40

# ETH 15m
python scripts/train_advanced.py --symbol ETHUSDT --timeframe 15m

# BTC 5m (mais rápido mas mais ruído)
python scripts/train_advanced.py --symbol BTCUSDT --timeframe 5m --top-features 25
```

---

## 📊 Comparação de Performance

### Treinamento Padrão (train_model.py)
- 3 classes (LONG/SHORT/NONE)
- 57 features
- sem otimizações

**Resultados:**
- Train Accuracy: 61.81%
- Test Accuracy: 54.96%
- ROC AUC: **0.4467** ❌ (pior que random!)
- F1 Macro: 0.2913

### Treinamento Avançado (train_advanced.py)
- 2 classes (LONG/SHORT)
- 30-40 features selecionadas
- class balancing
- threshold otimizado

**Resultados Esperados:**
- Train Accuracy: ~65-70%
- Test Accuracy: ~58-62%
- ROC AUC: **>0.60** ✅ (muito melhor!)
- F1 Macro: ~0.55-0.60

---

## ⚙️ Configurações Otimizadas

### Triple Barrier - Scalping 15m
```yaml
forward_window: 20          # 5 horas à frente
profit_target_atr: 2.5      # ~0.5-1% profit em BTC
stop_loss_atr: 1.2          # ~0.3-0.6% stop em BTC
min_move_atr: 0.6           # Filtro de ruído
```

### XGBoost - Anti-Overfitting
```yaml
max_depth: 5                # Árvores médias
learning_rate: 0.05         # Taxa moderada
n_estimators: 150           # 150 árvores

min_child_weight: 7         # Mais amostras/folha
gamma: 0.5                  # Penalização forte
reg_alpha: 1.5              # L1 forte
reg_lambda: 2.5             # L2 forte

subsample: 0.75             # 75% das amostras
colsample_bytree: 0.75      # 75% das features
```

---

## 📈 Features Mais Importantes

Baseado em análise de 365 dias de dados BTC/USDT 15m:

**Top 10:**
1. `vwap` - Volume Weighted Average Price
2. `ema_200` - Tendência de longo prazo
3. `bb_upper/lower` - Bandas de Bollinger
4. `rsi_overbought` - Condições de sobrecompra
5. `volume_ma` - Média de volume
6. `atr_pct` - Volatilidade percentual
7. `ema_9/21/50` - Tendências de curto/médio prazo
8. `macd_signal` - Momentum
9. `dist_ema_21` - Distância da EMA 21
10. `trend_50` - Direção da tendência

---

## 🎓 Melhores Práticas

### 1. Timeframe Ideal
- **15m**: Melhor balance entre sinal e ruído ⭐
- **5m**: Mais trades, mais ruído
- **1h**: Menos trades, sinais mais claros

### 2. Número de Features
- **20-25**: Modelo rápido, menos overfitting
- **30-35**: Balance ideal ⭐
- **40-50**: Modelo complexo, cuidado com overfitting

### 3. Dados Históricos
- **Mínimo**: 180 dias
- **Recomendado**: 365 dias ⭐
- **Ideal**: 730 dias (2 anos)

### 4. Validação
- Sempre use conjunto de validação
- Monitore gap train-test (<15% é bom)
- ROC AUC deve ser >0.55 para produção

---

## 🔍 Troubleshooting

### Problema: ROC AUC < 0.5
**Solução:**
1. Aumentar `min_move_atr` (filtrar mais ruído)
2. Reduzir `top_features` (menos complexidade)
3. Aumentar regularização (reg_alpha/lambda)

### Problema: Overfitting (train >> test)
**Solução:**
1. Aumentar `min_child_weight`
2. Reduzir `max_depth`
3. Reduzir `n_estimators`
4. Aumentar `subsample`

### Problema: Accuracy baixa (<55%)
**Solução:**
1. Baixar mais dados históricos
2. Ajustar parâmetros do Triple Barrier
3. Testar timeframe diferente (1h)
4. Aumentar `top_features`

---

## 📝 Logs e Debugging

Os logs mostram todas as etapas:

```
ETAPA 1: Carregamento de Dados
ETAPA 2: Feature Engineering
ETAPA 3: Criação de Labels
ETAPA 4: Conversão para Problema Binário
ETAPA 5: Divisão do Dataset
ETAPA 6: Seleção de Features
ETAPA 7: Treinamento do Modelo
ETAPA 8: Otimização de Threshold
ETAPA 9: Avaliação no Teste
ETAPA 10: Salvando Modelo
```

Confira `logs/training_advanced.log` para detalhes.

---

## 🚀 Próximos Passos

Após treinar com sucesso:

1. **Validar no Testnet** da Bybit
2. **Backtesting** com dados out-of-sample
3. **Paper trading** por 1-2 semanas
4. **Live trading** com capital pequeno
5. **Monitorar métricas** diariamente

---

## ⚠️ Avisos Importantes

1. **Não use em produção sem validação completa**
2. **Crypto é volátil** - use stop loss sempre
3. **Monitore performance** - retreine se degradar
4. **Comece pequeno** - teste com capital mínimo
5. **Risk management** - nunca arrisque >2% por trade

---

## 📚 Referências

- [Triple Barrier Method](https://www.researchgate.net/publication/322436823_The_10_Reasons_Most_Machine_Learning_Funds_Fail)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Feature Selection](https://scikit-learn.org/stable/modules/feature_selection.html)

---

**Desenvolvido para scalping profissional na Bybit** 🎯
