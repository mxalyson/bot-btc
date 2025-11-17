# 🚀 Otimização Específica para Scalping 15m

## 🎯 Objetivo

Maximizar performance em **trades rápidos** no timeframe 15m mantendo:
- Múltiplos trades por dia
- Risco controlado
- ROC AUC > 0.55 (mínimo aceitável para scalping)

---

## ⚙️ Mudanças Implementadas

### 1. Triple Barrier Calibrado para 15m

#### ❌ Antes (Ruim para Scalping):
```yaml
forward_window: 20        # 5 horas - MUITO LONGO
profit_target_atr: 2.5    # ~0.7-1.0% - MUITO AGRESSIVO
stop_loss_atr: 1.2        # ~0.35-0.5%
min_move_atr: 0.6         # Filtra demais
```

**Problema**:
- Janela de 5h não é scalping, é swing trading!
- Alvos de 2.5x ATR raramente atingidos em 15m
- Resultado: Labels ruins, modelo não aprende

#### ✅ Agora (Otimizado para Scalping):
```yaml
forward_window: 8         # 2 horas - SCALPING REAL
profit_target_atr: 1.5    # ~0.3-0.5% - REALISTA
stop_loss_atr: 1.0        # ~0.2-0.35% - APERTADO
min_move_atr: 0.3         # Captura mais trades
```

**Benefícios**:
- 2 horas = tempo real de scalping
- Alvos atingíveis em movimentos normais de 15m
- Mais trades = mais dados para treinar
- R:R ainda favorável (1.5:1.0)

---

### 2. XGBoost Ultra-Regularizado

#### ❌ Antes (Overfitting):
```yaml
max_depth: 5              # Árvores médias
learning_rate: 0.05       # Taxa moderada
n_estimators: 150         # Muitas árvores
min_child_weight: 7       # Moderado
gamma: 0.5                # Moderado
reg_alpha: 1.5            # L1 moderado
reg_lambda: 2.5           # L2 moderado
subsample: 0.75           # 75%
```

**Problema**:
- Train ROC AUC 0.72 vs Test 0.51 = 21% overfitting!
- Modelo decora ruído do 15m

#### ✅ Agora (Anti-Overfitting Extremo):
```yaml
max_depth: 3              # Árvores RASAS (menos complexidade)
learning_rate: 0.03       # Taxa LENTA (aprende devagar)
n_estimators: 100         # Menos árvores
min_child_weight: 10      # Exige MUITAS amostras
gamma: 0.8                # Penalização FORTE
reg_alpha: 2.0            # L1 FORTE
reg_lambda: 3.0           # L2 FORTE
subsample: 0.65           # Apenas 65% (muito agressivo)
```

**Benefícios**:
- Árvores rasas = menos memorização
- Regularização forte = força generalização
- Subsample baixo = mais diversidade
- **Expectativa**: Gap train-test < 10%

---

## 📊 Resultados Esperados

### Antes (Baseline):
```
Train ROC AUC: 0.7182
Val ROC AUC:   0.5084
Test ROC AUC:  0.5076  ❌ (random)

Gap: 21% (overfitting severo)
```

### Depois (Meta):
```
Train ROC AUC: 0.58-0.62
Val ROC AUC:   0.55-0.58
Test ROC AUC:  0.55-0.58  ✅ (útil)

Gap: < 8% (overfitting controlado)
```

**Por quê ROC AUC menor no train?**
- Regularização forte IMPEDE overfitting
- Modelo mais simples, generaliza melhor
- **Trade-off correto**: Preferimos 58% real a 72% falso!

---

## 🔄 Retreinar Agora

```bash
# Executar treinamento otimizado
python scripts/train_advanced.py --symbol BTCUSDT --timeframe 15m --top-features 25
```

**Nota**: Reduzi de 30 para 25 features (menos complexidade)

---

## 📈 Análise dos Novos Labels

### Distribuição Esperada (Com Novos Parâmetros):

**Antes** (forward_window=20, profit=2.5):
- SHORT: 62% (muito desbalanceado!)
- LONG: 38%

**Agora** (forward_window=8, profit=1.5):
- SHORT: 50-55% (mais balanceado)
- LONG: 45-50%
- Mais trades atingem alvos

---

## 🎯 Estratégia de Scalping Realista

### Timeframe: 15m
- **Holding time**: 2-4 horas (8-16 candles)
- **Profit target**: 0.3-0.5% (~1.5x ATR)
- **Stop loss**: 0.2-0.35% (~1.0x ATR)
- **Trades/dia**: 3-8 trades

### Risk/Reward: 1.5:1
```
Win: +0.4% (média)
Loss: -0.25% (média)
Breakeven: 38.5% win rate (modelo deve ter >50%)
```

### Performance Mínima Necessária:

| Métrica | Mínimo | Bom | Excelente |
|---------|--------|-----|-----------|
| Win Rate | 52% | 55% | 60% |
| ROC AUC | 0.55 | 0.60 | 0.65 |
| Profit Factor | 1.2 | 1.5 | 2.0 |

---

## 🧪 Testes Adicionais Recomendados

### 1. Diferentes Combinações de Alvos

Testar variações do Triple Barrier:

```yaml
# Conservador (mais win rate, menos lucro)
profit_target_atr: 1.2
stop_loss_atr: 1.0

# Balanceado (atual)
profit_target_atr: 1.5
stop_loss_atr: 1.0

# Agressivo (menos win rate, mais lucro)
profit_target_atr: 2.0
stop_loss_atr: 0.8
```

### 2. Forward Windows Diferentes

```yaml
# Super rápido (30min-1h)
forward_window: 4

# Rápido (2h) - ATUAL
forward_window: 8

# Moderado (3h)
forward_window: 12
```

### 3. Mais Features Específicas para 15m

Adicionar:
- **Microestrutura**: Spread, depth, imbalance
- **Order flow**: Buy/sell pressure
- **Intraday patterns**: Hour of day, session
- **Recent price action**: Last 3-5 candles patterns

---

## 📝 Checklist de Otimização

- [x] Triple Barrier ajustado para 2h (forward_window=8)
- [x] Profit target realista (1.5x ATR)
- [x] Stop loss apertado (1.0x ATR)
- [x] Min move reduzido (0.3 ATR)
- [x] XGBoost ultra-regularizado (max_depth=3)
- [x] Subsample agressivo (65%)
- [x] Features reduzidas (25 ao invés de 30)
- [ ] Retreinar modelo
- [ ] Analisar nova distribuição LONG/SHORT
- [ ] Verificar gap train-test < 10%
- [ ] ROC AUC test > 0.55
- [ ] Backtesting se ROC AUC > 0.55

---

## ⚠️ Expectativas Realistas

### Scalping 15m em Crypto é DIFÍCIL:

1. **Mercado extremamente ruidoso**
   - Movimentos aleatórios dominam
   - Muitos false breakouts
   - Volatilidade imprevisível

2. **Custos de transação importam MUITO**
   - Bybit futures: ~0.02-0.06% por trade
   - Slippage: 0.01-0.03%
   - Total: 0.03-0.09% por round trip
   - Com 0.4% target, fees = 7-22% do lucro!

3. **Win rate de 55-60% seria EXCELENTE**
   - Não espere 70-80%
   - Mercado profissional tem ~52-55%
   - Você está competindo com HFT bots

### Meta Realista:

```
Se conseguir:
- ROC AUC > 0.55
- Win rate > 52%
- Profit factor > 1.3

→ Você TEM um edge competitivo! ✅
```

---

## 🚀 Próximo Passo IMEDIATO

Execute agora:

```bash
python scripts/train_advanced.py --symbol BTCUSDT --timeframe 15m --top-features 25
```

Aguarde e analise:
1. **Distribuição LONG/SHORT** (deve estar ~50/50)
2. **ROC AUC test** (meta > 0.55)
3. **Gap train-test** (meta < 10%)
4. **Confusion matrix** (balanceada?)

Se ROC AUC > 0.55 → **BACKTESTING**
Se ROC AUC < 0.55 → Ajustar mais ou considerar 1h

---

**Última atualização**: 2025-11-16 21:10
**Status**: Configuração otimizada para scalping 15m real
