# 🏆 ULTRA MASTER SCALPER - Guia Completo

Sistema de ML estado da arte para scalping 15m com ensemble pesado e validação rigorosa.

---

## 🎯 O que foi implementado?

### **1. Ensemble Pesado (4 Modelos)**
- ✅ **LightGBM** - Gradient Boosting otimizado
- ✅ **XGBoost** - Gradient Boosting alternativo
- ✅ **Transformer** - Self-attention para padrões temporais
- ✅ **CNN-LSTM** - Hybrid deep learning
- ✅ **Meta-Learner** - Stacking com calibração

### **2. Features Avançadas (80+ features)**

#### Microestrutura de Mercado:
- **Kyle's Lambda** - Price impact (quanto o preço move por volume)
- **Amihud Illiquidity** - Liquidez do ativo
- **Roll's Spread** - Spread efetivo estimado
- **Order Flow Proxy** - Fluxo de ordens (compra vs venda)
- **Price Efficiency Ratio** - Trending vs ranging
- **Tick Persistence** - Persistência de direção

#### Features Avançadas:
- Multi-period momentum (3, 5, 8, 13, 21, 34 períodos)
- Volatility regimes (dinâmico)
- Price position em múltiplas janelas (10, 20, 50)
- Support/Resistance proximity
- Candle patterns avançados
- Volume acceleration

### **3. Labeling Inteligente**
- **Multi-horizon voting** (1h, 1.5h, 2h, 3h)
- **Dynamic thresholds** baseados em ATR e volatilidade
- **Asymmetric risk-reward** (1:1.5)
- **Consensus requirement** (70% agreement)

### **4. Feature Selection SHAP**
- Automático: seleciona top 60 features mais importantes
- Remove ruído e melhora generalização

### **5. Validação Ultra-Rigorosa**
- **Walk-Forward Validation** (5 folds)
- **Regime-Based Analysis** (bull/bear × low/high vol)
- **Monte Carlo Simulation** (1000 runs)

---

## 🚀 Como Usar

### **Passo 1: Treinar o Ultra Scalper**

```bash
# Treinar com 1 ano de dados (recomendado)
python train_ultra_scalper.py --symbol BTCUSDT --days 365

# Treinar com 2 anos (mais dados, melhor generalização)
python train_ultra_scalper.py --symbol BTCUSDT --days 730
```

**O que acontece:**
1. Baixa dados históricos da Bybit
2. Cria ~150 features avançadas
3. Seleciona top 60 com SHAP
4. Treina 4 modelos em paralelo:
   - LightGBM
   - XGBoost
   - Transformer
   - CNN-LSTM (opcional, requer TensorFlow)
5. Treina meta-learner de stacking
6. Salva modelo em `storage/models/ultra_scalper_btcusdt_365d.pkl`

**Tempo estimado:** 10-30 minutos (depende de CPU/GPU)

---

### **Passo 2: Validação Rigorosa**

```bash
# Validar com 6 meses de dados out-of-sample
python validate_ultra_scalper.py --symbol BTCUSDT --days 180 --model ultra_scalper_btcusdt_365d.pkl
```

**O que acontece:**
1. **Walk-Forward Validation** (5 folds)
   - Divide dados em 5 períodos sequenciais
   - Testa como se estivesse em produção

2. **Regime-Based Analysis**
   - Testa performance em:
     - Bull + Low Vol
     - Bull + High Vol
     - Bear + Low Vol
     - Bear + High Vol

3. **Monte Carlo Simulation**
   - 1000 simulações randomizando ordem dos trades
   - Estima distribuição de resultados
   - Probabilidade de lucro

**Output:**
```
📊 WALK-FORWARD RESULTS
Fold 1: 2024-01-01 to 2024-02-15
   Trades: 45  |  WR: 57.8%  |  ROI: +8.5%  |  Sharpe: 1.45  |  DD: -5.2%

Fold 2: 2024-02-15 to 2024-04-01
   Trades: 52  |  WR: 54.2%  |  ROI: +6.1%  |  Sharpe: 1.18  |  DD: -7.3%

📊 REGIME-BASED PERFORMANCE
medium_bull          (4521 samples)
   Trades: 85  |  WR: 62.4%  |  ROI: +15.2%

high_vol_bear        (2143 samples)
   Trades: 38  |  WR: 47.4%  |  ROI: -3.8%

📊 MONTE CARLO SIMULATION (1000 runs)
   Mean ROI:   +12.3%
   Median ROI: +11.8%
   5th %ile:   +3.2%
   95th %ile:  +22.1%
   Prob Profit: 78.5%
   Worst DD:   -12.4%
```

---

## 📊 Métricas de Sucesso

### **Mínimo Aceitável (15m scalping)**
- ✅ Win Rate: **52-55%**
- ✅ ROI anual: **15-30%**
- ✅ Sharpe Ratio: **> 1.0**
- ✅ Max Drawdown: **< 20%**
- ✅ Profit Factor: **> 1.3**

### **Bom (competitivo)**
- ✅ Win Rate: **55-58%**
- ✅ ROI anual: **30-60%**
- ✅ Sharpe Ratio: **> 1.5**
- ✅ Max Drawdown: **< 15%**
- ✅ Profit Factor: **> 1.5**

### **Excelente (top tier)**
- ✅ Win Rate: **58-62%**
- ✅ ROI anual: **60-120%**
- ✅ Sharpe Ratio: **> 2.0**
- ✅ Max Drawdown: **< 12%**
- ✅ Profit Factor: **> 2.0**

---

## ⚙️ Configurações Importantes

### **Risk Management (ajustar em config)**
```yaml
initial_capital: 10000        # Capital inicial
risk_per_trade_pct: 0.75      # 0.75% risco por trade
```

### **Stop Loss / Take Profit**
- **SL:** 1.5 × ATR
- **TP1:** 1.5 × ATR (1:1)
- **TP2:** 3.0 × ATR (1:2)
- **TP3:** 4.5 × ATR (1:3)

### **Confidence Threshold**
O validador testa níveis de 0% a 40%. Recomendações:

- **Mais trades, menos qualidade:** min_confidence = 0.05-0.10
- **Balanceado:** min_confidence = 0.10-0.15
- **Menos trades, mais qualidade:** min_confidence = 0.20-0.30

---

## 🔧 Troubleshooting

### **Erro: TensorFlow não encontrado**
```bash
pip install tensorflow
```

Se não quiser usar TensorFlow, o sistema funciona apenas com LightGBM + XGBoost (ainda muito bom!)

### **Modelo treinando muito devagar**
- Reduza `--days` de 730 para 365
- Desabilite Transformer (comente código TensorFlow)
- Use CPU com mais cores

### **Features dando erro**
Certifique-se de que `core/features.py` tem:
- ema50, ema200
- atr
- volume

### **Resultados ruins (AUC < 0.52)**
15m é muito difícil! Tente:
1. Aumentar dados de treino (730 dias)
2. Ajustar thresholds no labeling
3. Testar 1H timeframe (mais fácil)

---

## 📈 Comparação com Sistema Original

| Feature | Original | Ultra Scalper |
|---------|----------|---------------|
| **Modelos** | LightGBM only | Ensemble (4 modelos) |
| **Features** | ~113 | ~150 (+ microstructure) |
| **Selection** | Manual | SHAP automático |
| **Labeling** | Multi-horizon voting | Multi-horizon + Dynamic + Asymmetric |
| **Validation** | Simple split | Walk-Forward + Regime + Monte Carlo |
| **Meta-Learning** | Não | Stacking com calibração |
| **ROC AUC Esperado** | 0.50-0.52 | 0.52-0.56 |
| **Sharpe Esperado** | 0.8-1.2 | 1.2-1.8 |

---

## 💡 Próximos Passos

### **Se Resultados Bons (WR > 54%, Sharpe > 1.2)**
1. Fazer backtesting mais longo (1 ano out-of-sample)
2. Paper trading por 1 mês
3. Live trading com capital pequeno

### **Se Resultados Medianos (WR 52-54%)**
1. Ajustar confidence threshold
2. Testar apenas regimes específicos (ex: apenas bull markets)
3. Combinar com análise de ordem book em tempo real

### **Se Resultados Ruins (WR < 52%)**
1. **Testar 1H timeframe** (muito mais fácil)
2. Aumentar dados de treino para 2 anos
3. Revisar features (podem ter leak)

---

## 🎯 Features Futuras (Roadmap)

### **Curto Prazo**
- [ ] Order Book Level 2 features (bid-ask depth real)
- [ ] Funding rate features (perps)
- [ ] Liquidation heatmap proximity

### **Médio Prazo**
- [ ] AutoML (otimização automática de hiperparâmetros)
- [ ] Online learning (retreino automático semanal)
- [ ] Multi-timeframe prediction (15m + 1h combined)

### **Longo Prazo**
- [ ] Reinforcement Learning (PPO/A3C)
- [ ] Attention visualization (explicabilidade)
- [ ] Portfolio optimization (BTC + ETH + alts)

---

## 📞 Suporte

Dúvidas ou problemas?

1. Verifique logs em `storage/logs/`
2. Confira features com `df_features.info()`
3. Teste com dados sintéticos primeiro
4. Ajuste hiperparâmetros no código

---

## ⚠️ Disclaimers

1. **15m scalping é MUITO difícil** - Até profissionais lutam para ter edge consistente
2. **Past performance ≠ Future results** - Backtest bom não garante lucro real
3. **Sempre teste em paper trading primeiro** - Nunca vá direto pra live
4. **Gerencie risco** - Nunca arrisque mais de 1% do capital por trade
5. **Market regime changes** - Modelo pode degradar se mercado mudar

---

## 🏆 Resumo

O **Ultra Master Scalper** combina o melhor de ML moderno:

✅ Ensemble com 4 modelos (LGB + XGB + Transformer + CNN)
✅ Features de microestrutura de mercado
✅ Feature selection automática (SHAP)
✅ Labeling inteligente multi-horizon
✅ Validação rigorosa (Walk-Forward + Regime + Monte Carlo)
✅ Meta-learning com calibração

**Objetivo:** Aumentar ROC AUC de ~0.50 (random) para **0.52-0.56** (edge real) no 15m

**Expectativa realista:** Win Rate 54-58%, Sharpe 1.2-1.8, ROI 30-80%/ano

Boa sorte! 🚀
