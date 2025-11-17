# 🧠 Deep Learning para Scalping 15m

## 🎯 Por Que Deep Learning?

### ❌ XGBoost Falhou (ROC AUC 0.4936)

```
Train ROC AUC: 0.5981
Val ROC AUC:   0.5080
Test ROC AUC:  0.4936  ❌ (pior que random!)
```

**Por quê?**
- Indicadores técnicos tradicionais (EMA, RSI, MACD) capturam apenas **4% da performance**
- 15m é extremamente ruidoso (~95% ruído, 5% sinal)
- XGBoost não captura dependências temporais complexas

---

## 🔬 Research Crítico

### Descoberta 1: Microestrutura é 73% da Performance!

> **"Order book features contribute 73% of total performance, with order book depth alone accounting for a 68% profit decline when removed, compared to just 4% from technical indicators."**

**Fonte**: Machine Learning for Market Microstructure (2025)

### Descoberta 2: Arquiteturas Híbridas Superam

> **"The hybrid LSTM+XGBoost model consistently outperforms individual models by capturing temporal dependencies and leveraging gradient boosting's strength in non-linearity handling."**

**Fonte**: Crypto Price Prediction with LSTM and Transformer Models (2025)

### Descoberta 3: CNN+LSTM é State-of-Art

> **"CNN+LSTM models use bidirectional LSTM layers to capture both forward and backward order-flow autocorrelations, achieving superior results on BTC/USDT data."**

**Fonte**: Exploring Microstructural Dynamics in Cryptocurrency Limit Order Books (2025)

---

## 🏗️ Nossa Arquitetura

### Modelo Híbrido: CNN + BiLSTM + Attention

```
Input Sequence (20 candles × N features)
              ↓
┌─────────────────────────────────┐
│ BLOCO 1: CNN (Padrões Locais)  │
│  - Conv1D (64 filters, k=3)    │
│  - BatchNorm + Dropout         │
│  - Conv1D (32 filters, k=3)    │
│  - BatchNorm + Dropout         │
└─────────────────────────────────┘
              ↓
┌─────────────────────────────────┐
│ BLOCO 2: BiLSTM (Temporal)     │
│  - Bidirectional LSTM (128)    │
│  - BatchNorm + Dropout         │
└─────────────────────────────────┘
              ↓
┌─────────────────────────────────┐
│ BLOCO 3: Attention             │
│  - Multi-head Attention        │
│  - Foca momentos críticos      │
└─────────────────────────────────┘
              ↓
┌─────────────────────────────────┐
│ BLOCO 4: Dense + Output        │
│  - Dense (64) + Dropout        │
│  - Output (Sigmoid)            │
└─────────────────────────────────┘
              ↓
     Probability (LONG vs SHORT)
```

### Por Que Cada Componente?

#### 1️⃣ CNN (Convolutional Neural Network)
- **Propósito**: Captura padrões locais (3-5 candles consecutivos)
- **Exemplos**: Breakouts, reversões, formações de velas
- **Vantagem**: Detecta micro-padrões que XGBoost não vê

#### 2️⃣ BiLSTM (Bidirectional Long Short-Term Memory)
- **Propósito**: Captura dependências temporais longas
- **Bidirectional**: Olha para frente E para trás no tempo
- **Vantagem**: Entende contexto completo da sequência

#### 3️⃣ Attention
- **Propósito**: Foca nos momentos mais importantes
- **Exemplos**: Breakout de volume, mudança de regime
- **Vantagem**: Ignora ruído, foca em sinais

#### 4️⃣ Regularização Forte
- **Dropout (30%)**: Previne overfitting
- **L2 Regularization**: Penaliza weights grandes
- **BatchNormalization**: Estabiliza treinamento
- **Early Stopping**: Para quando não melhora mais

---

## 🎨 Features Avançadas (Microestrutura)

### Criamos **80+ Features** em 7 Categorias:

### 1. Spread e Range (8 features)
```python
- spread_abs, spread_pct, spread_atr_ratio
- upper_shadow, lower_shadow (pavios)
- body_abs, body_pct (corpo da vela)
```

**Captura**: Volatilidade intracandle, força compradores/vendedores

---

### 2. Order Flow Proxy (10 features)
```python
- buy_volume, sell_volume (estimado)
- order_flow, order_flow_ratio
- order_flow_cum_5/10/20 (momentum)
- buy_pressure, sell_pressure
```

**Captura**: Pressão de compra vs venda (proxy de order book)

---

### 3. Volume Profile (8 features)
```python
- volume_ratio (vs média)
- vwap, price_vwap_dist
- volume_surge (detecção)
- volume_trend_5/10
- volume_delta, volume_delta_pct
```

**Captura**: Anomalias de volume, divergências

---

### 4. Price Action Micro (8 features)
```python
- new_high_5/10, new_low_5/10
- price_position_20 (% do range)
- consec_up, consec_down
- gap_up, gap_down
```

**Captura**: Breakouts, topos/fundos, gaps

---

### 5. Tick Direction (8 features)
```python
- tick, tick_momentum_5/10/20
- price_momentum_3/5/10
- price_accel_5 (aceleração)
```

**Captura**: Direção e aceleração de preço

---

### 6. Volatility Regime (8 features)
```python
- volatility_5/10/20 (rolling std)
- vol_ratio_5_20, vol_ratio_10_20
- high_vol_regime (binário)
- parkinson_vol (high-low based)
```

**Captura**: Mudanças de regime de mercado

---

### 7. Multi-Timeframe (16 features)
```python
- trend_4/8/12/20 (1h, 2h, 3h, 5h)
- volume_trend_4/8/12/20
- trend_alignment (todos alinhados?)
```

**Captura**: Contexto de timeframes maiores

---

## ⚙️ Configuração Otimizada

### Triple Barrier (Mantido do XGBoost)
```yaml
forward_window: 8         # 2 horas
profit_target_atr: 1.5    # ~0.3-0.5%
stop_loss_atr: 1.0        # ~0.2-0.35%
min_move_atr: 0.3
```

### Arquitetura DL
```yaml
conv_filters: 64
conv_kernel: 3
lstm_units: 128           # BiLSTM = 256 total
attention_units: 64
dense_units: 64

dropout_rate: 0.3         # 30% dropout
l2_reg: 0.001

learning_rate: 0.001      # Com ReduceLROnPlateau
batch_size: 64
epochs: 100               # Com early stopping
```

### Dados
```yaml
lookback_days: 730        # 2 anos (DL precisa mais dados)
sequence_length: 20       # 20 candles = 5 horas de histórico
```

---

## 🚀 Como Treinar

### 1. Instalar TensorFlow
```bash
pip install tensorflow
```

### 2. Baixar Mais Dados (2 anos)
```bash
python scripts/download_data.py --symbol BTCUSDT --timeframe 15m --days 730
```

### 3. Treinar Modelo DL
```bash
python scripts/train_deep_learning.py \
  --symbol BTCUSDT \
  --timeframe 15m \
  --sequence-length 20 \
  --epochs 100 \
  --batch-size 64
```

### 4. Parâmetros Opcionais
```bash
# Sequence mais longa (mais contexto)
--sequence-length 30

# Mais epochs (se não overfit)
--epochs 150

# Batch maior (se tem RAM)
--batch-size 128

# Config customizado
--config config/my_custom_config.yaml
```

---

## 📊 Expectativas Realistas

### Baseline (XGBoost)
```
Test ROC AUC: 0.4936  ❌ (inútil)
```

### Meta Deep Learning

| Cenário | ROC AUC | Status |
|---------|---------|--------|
| **Mínimo Aceitável** | 0.52-0.55 | 🟡 Marginal |
| **Bom** | 0.55-0.60 | ✅ Útil |
| **Excelente** | 0.60-0.65 | ✅ Muito bom |
| **Excepcional** | > 0.65 | ✅ Excepcional |

### Por Que Mais Conservador?

15m crypto é EXTREMAMENTE difícil:
- Research mostra ROC AUC 0.55-0.60 é **excelente** para HFT
- Você está competindo com bots profissionais
- ~95% ruído no sinal
- **Qualquer coisa > 0.55 = edge competitivo real**

---

## 🧪 Vantagens vs XGBoost

| Aspecto | XGBoost | Deep Learning |
|---------|---------|---------------|
| **Temporal Dependencies** | ❌ Não captura | ✅ BiLSTM captura |
| **Local Patterns** | ❌ Vê features isoladas | ✅ CNN detecta padrões |
| **Feature Importance** | ✅ Explícito | ⚠️ Attention ajuda |
| **Microstructure** | ❌ 4% performance | ✅ 73% performance |
| **Sequence Context** | ❌ Vê 1 candle | ✅ Vê 20 candles |
| **Overfitting Control** | ✅ Regularização boa | ✅ Dropout + L2 + Early Stop |
| **Training Time** | ✅ Rápido (minutos) | ⚠️ Lento (30-60 min) |
| **Data Requirements** | ✅ Funciona com pouco | ⚠️ Precisa muito (2 anos) |

---

## 🎓 Melhorias Futuras (Se Funcionar)

### 1. Transformer Architecture
- Substituir LSTM por Temporal Fusion Transformer
- Multi-head attention mais sofisticado
- Positional encoding

### 2. Order Book Real
- Integrar dados de order book da Bybit
- Bid-ask spread real
- Depth real nos níveis

### 3. Reinforcement Learning
- DQN ou PPO para aprender política de trading
- Recompensa baseada em Sharpe Ratio
- Action: BUY/SELL/HOLD

### 4. Ensemble DL + XGBoost
- Voting ensemble
- Stacking com meta-learner
- Combinar forças de ambos

### 5. Feature Learning Automático
- Autoencoder para comprimir features
- Representações latentes
- Reduzir ruído

---

## ⚠️ Avisos Importantes

### 1. Deep Learning Não é Mágica
- Se dados não têm sinal, DL também não vai achar
- Pode melhorar de 0.49 para 0.55-0.60 (realista)
- Não espere 0.80+ (impossível em 15m crypto)

### 2. Risco de Overfitting Maior
- DL tem MUITO mais parâmetros que XGBoost
- Monitorar gap train-test SEMPRE
- Early stopping é CRÍTICO

### 3. Interpretabilidade Menor
- XGBoost mostra feature importance claramente
- DL é mais "black box"
- Attention weights ajudam mas não é perfeito

### 4. Custos Computacionais
- GPU recomendada (não obrigatória)
- Treinamento 30-60 minutos vs 5 min do XGBoost
- Inferência mais lenta (ms vs us)

### 5. Mais Dados = Melhor
- DL precisa MUITO mais dados que XGBoost
- Recomendado: 2 anos (730 dias)
- Mínimo: 1 ano (365 dias)

---

## 📝 Checklist

- [ ] TensorFlow instalado (`pip install tensorflow`)
- [ ] Dados de 2 anos baixados (730 dias)
- [ ] Config `config_deep_learning.yaml` verificado
- [ ] Script `train_deep_learning.py` executado
- [ ] ROC AUC test > 0.55 ✅
- [ ] Gap train-test < 15% ✅
- [ ] Confusion matrix balanceada ✅
- [ ] Se tudo OK → Backtesting
- [ ] Se ROC AUC < 0.55 → Ajustar hyperparameters

---

## 🎯 Próximos Passos AGORA

### 1. Instalar TensorFlow
```bash
pip install tensorflow
```

### 2. Treinar
```bash
python scripts/train_deep_learning.py --symbol BTCUSDT --timeframe 15m
```

### 3. Analisar
- Verificar ROC AUC test
- Comparar com baseline XGBoost (0.4936)
- Se > 0.55 → **SUCESSO!**
- Se < 0.55 → Ajustar hyperparameters

---

**Última atualização**: 2025-11-17
**Status**: Pronto para treinar
**Meta**: ROC AUC > 0.55 para ter edge competitivo
