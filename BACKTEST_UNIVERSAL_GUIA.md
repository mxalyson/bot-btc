# 🚀 BACKTEST UNIVERSAL - GUIA COMPLETO

## ✅ PROBLEMA RESOLVIDO!

Criei um **backtest UNIVERSAL** que resolve TODOS os problemas:

1. ✅ Não quebra com KeyError (initial_capital tem default)
2. ✅ Aceita qualquer modelo .pkl
3. ✅ Permite ajustar thresholds dinamicamente
4. ✅ Configurável via linha de comando
5. ✅ Flexível para qualquer período/capital/risco

---

## 📦 ATUALIZAR REPOSITÓRIO

**IMPORTANTE:** Se o backtest_V6.py ainda está com erro, faça:

```bash
git pull
```

Isso vai atualizar para a versão corrigida.

---

## 🎯 USO DO BACKTEST UNIVERSAL

### Uso Básico (padrão)

```bash
python backtest_UNIVERSAL.py
```

**O que faz:**
- Carrega modelo: `storage/models/model_DEFINITIVO_4ML_365d.pkl`
- Usa thresholds do modelo (salvos no .pkl)
- Testa últimos 52 dias (~2 meses)
- Capital: $10,000
- Risco: 2% por trade

---

### Ajustar Thresholds (SEM RETREINAR!)

```bash
python backtest_UNIVERSAL.py --long-threshold 0.50 --short-threshold 0.50
```

**Útil quando:**
- Modelo tem threshold 0.35/0.65 mas você quer testar 0.50/0.50
- Quer balancear predições sem retreinar
- Testar diferentes níveis de confiança

**Exemplos:**

```bash
# Threshold neutro (balanceado)
python backtest_UNIVERSAL.py --long-threshold 0.50 --short-threshold 0.50

# Leve viés long
python backtest_UNIVERSAL.py --long-threshold 0.45 --short-threshold 0.55

# Forte viés long
python backtest_UNIVERSAL.py --long-threshold 0.35 --short-threshold 0.65

# Conservador (só trades com alta confiança)
python backtest_UNIVERSAL.py --long-threshold 0.60 --short-threshold 0.40
```

---

### Testar Diferentes Modelos

```bash
# Modelo 4ML
python backtest_UNIVERSAL.py --model storage/models/model_DEFINITIVO_4ML_365d.pkl

# Modelo 4ML BALANCED
python backtest_UNIVERSAL.py --model storage/models/model_DEFINITIVO_4ML_BALANCED_365d.pkl

# Modelo V6 (6 modelos)
python backtest_UNIVERSAL.py --model storage/models/model_DEFINITIVO_V6_365d.pkl

# Modelo de 540 dias
python backtest_UNIVERSAL.py --model storage/models/model_DEFINITIVO_4ML_540d.pkl

# Qualquer outro modelo
python backtest_UNIVERSAL.py --model meu_modelo_personalizado.pkl
```

---

### Ajustar Período de Teste

```bash
# Testar 30 dias (1 mês)
python backtest_UNIVERSAL.py --days 30

# Testar 90 dias (3 meses)
python backtest_UNIVERSAL.py --days 90

# Testar 180 dias (6 meses)
python backtest_UNIVERSAL.py --days 180

# Testar 365 dias (1 ano)
python backtest_UNIVERSAL.py --days 365
```

**Nota:** Backtests longos (>180 dias) podem demorar mais.

---

### Ajustar Capital e Risco

```bash
# Capital de $5,000 com risco de 1%
python backtest_UNIVERSAL.py --capital 5000 --risk-per-trade 1.0

# Capital de $50,000 com risco de 3%
python backtest_UNIVERSAL.py --capital 50000 --risk-per-trade 3.0

# Capital de $100,000 com risco de 0.5%
python backtest_UNIVERSAL.py --capital 100000 --risk-per-trade 0.5
```

---

### Ajustar Stop Loss / Take Profit

```bash
# SL mais apertado (1.0 ATR), TP normal (2.0 ATR)
python backtest_UNIVERSAL.py --sl-atr-mult 1.0 --tp-atr-mult 2.0

# SL normal (1.5 ATR), TP mais distante (3.0 ATR)
python backtest_UNIVERSAL.py --sl-atr-mult 1.5 --tp-atr-mult 3.0

# SL largo (2.0 ATR), TP curto (1.5 ATR) - scalping agressivo
python backtest_UNIVERSAL.py --sl-atr-mult 2.0 --tp-atr-mult 1.5
```

---

### Combinações Avançadas

**Exemplo 1: Teste completo com threshold balanceado**
```bash
python backtest_UNIVERSAL.py \
  --model storage/models/model_DEFINITIVO_4ML_365d.pkl \
  --long-threshold 0.50 \
  --short-threshold 0.50 \
  --days 90 \
  --capital 10000
```

**Exemplo 2: Backtest conservador**
```bash
python backtest_UNIVERSAL.py \
  --long-threshold 0.60 \
  --short-threshold 0.40 \
  --risk-per-trade 1.0 \
  --sl-atr-mult 1.0 \
  --tp-atr-mult 3.0
```

**Exemplo 3: Scalping agressivo**
```bash
python backtest_UNIVERSAL.py \
  --long-threshold 0.40 \
  --short-threshold 0.60 \
  --risk-per-trade 3.0 \
  --sl-atr-mult 1.5 \
  --tp-atr-mult 1.5 \
  --days 30
```

**Exemplo 4: Teste de longo prazo**
```bash
python backtest_UNIVERSAL.py \
  --model storage/models/model_DEFINITIVO_4ML_540d.pkl \
  --days 180 \
  --capital 50000 \
  --risk-per-trade 1.5
```

---

## 📊 INTERPRETANDO RESULTADOS

### Métricas Principais

**1. Win Rate**
- ✅ BOM: ≥ 48%
- ⚠️ MÉDIO: 45-48%
- ❌ RUIM: < 45%

**2. ROI**
- ✅ BOM: > 5%
- ⚠️ MÉDIO: 0-5%
- ❌ RUIM: < 0%

**3. Distribuição Longs/Shorts**
- ✅ BALANCEADO: 40-60% cada
- ⚠️ LEVE VIÉS: 30-40% ou 60-70%
- ❌ DESBALANCEADO: < 30% ou > 70%

**4. Risk/Reward**
- ✅ BOM: > 1.5
- ⚠️ MÉDIO: 1.0-1.5
- ❌ RUIM: < 1.0

### Critérios de Aprovação

O backtest mostra 7 critérios:

```
✅ CRITÉRIOS DE APROVAÇÃO
   ✅ Win Rate ≥ 48%
   ✅ ROI > 0%
   ✅ Longs 40-60%
   ✅ Shorts 40-60%
   ✅ Long WR 45-55%
   ✅ Short WR 45-55%
   ✅ Risk/Reward > 1.0
```

**Aprovação:**
- ≥ 5 critérios OK → ✅ APROVADO
- < 5 critérios OK → ❌ REPROVADO

---

## 🔧 TROUBLESHOOTING

### Problema: Predições muito desbalanceadas (>80% longs ou shorts)

**Solução:** Ajustar thresholds

```bash
# Se >80% longs, aumentar long threshold
python backtest_UNIVERSAL.py --long-threshold 0.55 --short-threshold 0.45

# Se >80% shorts, diminuir long threshold
python backtest_UNIVERSAL.py --long-threshold 0.45 --short-threshold 0.55

# Ou usar threshold neutro
python backtest_UNIVERSAL.py --long-threshold 0.50 --short-threshold 0.50
```

### Problema: Win Rate muito baixo (<45%)

**Possíveis causas:**
1. Modelo não generalizou bem
2. SL muito apertado
3. TP muito distante
4. Threshold inadequado

**Soluções:**
```bash
# Tentar SL mais largo
python backtest_UNIVERSAL.py --sl-atr-mult 2.0

# Tentar TP mais próximo
python backtest_UNIVERSAL.py --tp-atr-mult 1.5

# Tentar threshold mais conservador (menos trades, maior confiança)
python backtest_UNIVERSAL.py --long-threshold 0.55 --short-threshold 0.45
```

### Problema: ROI negativo mesmo com WR > 50%

**Causa:** Risk/Reward ruim (perdas maiores que ganhos)

**Solução:** Aumentar TP ou diminuir SL
```bash
python backtest_UNIVERSAL.py --sl-atr-mult 1.0 --tp-atr-mult 3.0
```

### Problema: KeyError 'initial_capital' no backtest_V6.py

**Solução:**
```bash
git pull  # Atualizar para versão corrigida
```

Ou use o backtest_UNIVERSAL.py que não tem esse problema:
```bash
python backtest_UNIVERSAL.py
```

---

## 📋 ARGUMENTOS DISPONÍVEIS

| Argumento | Tipo | Default | Descrição |
|-----------|------|---------|-----------|
| `--model` | str | model_DEFINITIVO_4ML_365d.pkl | Caminho do modelo .pkl |
| `--long-threshold` | float | (do modelo) | Threshold para long (0.0-1.0) |
| `--short-threshold` | float | (do modelo) | Threshold para short (0.0-1.0) |
| `--days` | int | 52 | Dias de dados (~2 meses) |
| `--capital` | float | 10000 | Capital inicial ($) |
| `--risk-per-trade` | float | 2.0 | Risco por trade (%) |
| `--sl-atr-mult` | float | 1.5 | Multiplicador ATR para SL |
| `--tp-atr-mult` | float | 2.0 | Multiplicador ATR para TP |
| `--symbol` | str | BTCUSDT | Par de trading |
| `--interval` | str | 15m | Timeframe |

---

## 🎯 WORKFLOW RECOMENDADO

### 1. Teste Rápido (Padrão)

```bash
python backtest_UNIVERSAL.py
```

Veja se modelo tem boa performance básica.

### 2. Ajuste de Threshold (se necessário)

Se predições desbalanceadas:
```bash
python backtest_UNIVERSAL.py --long-threshold 0.50 --short-threshold 0.50
```

### 3. Teste em Diferentes Períodos

```bash
# Curto prazo (30 dias)
python backtest_UNIVERSAL.py --days 30

# Médio prazo (90 dias)
python backtest_UNIVERSAL.py --days 90

# Longo prazo (180 dias)
python backtest_UNIVERSAL.py --days 180
```

### 4. Otimização de Parâmetros

Teste diferentes combinações de SL/TP:
```bash
python backtest_UNIVERSAL.py --sl-atr-mult 1.0 --tp-atr-mult 2.0
python backtest_UNIVERSAL.py --sl-atr-mult 1.5 --tp-atr-mult 2.5
python backtest_UNIVERSAL.py --sl-atr-mult 2.0 --tp-atr-mult 3.0
```

### 5. Aprovação Final

Se modelo passou em ≥ 5 critérios → ✅ **APROVADO para paper trading**

```bash
python bot.py --paper-trading
```

---

## 🚀 QUANDO USAR QUAL BACKTEST?

### backtest_UNIVERSAL.py ⭐ RECOMENDADO

**Use quando:**
- ✅ Quer testar rapidamente diferentes thresholds
- ✅ Precisa testar múltiplos modelos
- ✅ Quer configurar tudo via CLI
- ✅ Precisa de flexibilidade máxima
- ✅ Quer evitar KeyError e outros bugs

**Comando:**
```bash
python backtest_UNIVERSAL.py --long-threshold 0.50 --short-threshold 0.50
```

### backtest_V6.py

**Use quando:**
- ⚠️ Já tem config.json configurado perfeitamente
- ⚠️ Modelo específico V6 com 6 modelos
- ⚠️ Prefere editar código a usar CLI

**Requisito:** Atualizar repositório primeiro
```bash
git pull
python backtest_V6.py
```

---

## 💡 DICAS AVANÇADAS

### 1. Encontrar Threshold Ideal

Teste vários thresholds:
```bash
python backtest_UNIVERSAL.py --long-threshold 0.35 --short-threshold 0.65
python backtest_UNIVERSAL.py --long-threshold 0.40 --short-threshold 0.60
python backtest_UNIVERSAL.py --long-threshold 0.45 --short-threshold 0.55
python backtest_UNIVERSAL.py --long-threshold 0.50 --short-threshold 0.50
python backtest_UNIVERSAL.py --long-threshold 0.55 --short-threshold 0.45
```

Compare ROI e Win Rate, escolha o melhor.

### 2. Otimização de SL/TP

```bash
# Grid search manual
for sl in 1.0 1.5 2.0; do
  for tp in 1.5 2.0 2.5 3.0; do
    echo "Testing SL=$sl TP=$tp"
    python backtest_UNIVERSAL.py --sl-atr-mult $sl --tp-atr-mult $tp
  done
done
```

### 3. Validação Cruzada Temporal

Teste em períodos diferentes:
```bash
python backtest_UNIVERSAL.py --days 30   # Último mês
python backtest_UNIVERSAL.py --days 60   # Últimos 2 meses
python backtest_UNIVERSAL.py --days 90   # Últimos 3 meses
```

Se WR consistente em todos → modelo robusto!

---

## 🎉 PRONTO PARA USAR!

Execute agora:

```bash
python backtest_UNIVERSAL.py --long-threshold 0.50 --short-threshold 0.50
```

Aguarde ~30 segundos e veja os resultados! 🚀
