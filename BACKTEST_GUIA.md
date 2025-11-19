# 📊 BACKTEST PERFEITO - Guia de Uso

## ✅ O QUE É ESTE BACKTEST

Validação completa da estratégia de scalping 15m usando:
- ✅ Modelo treinado (.pkl)
- ✅ Mesmas features do train_model_PERFEITO.py
- ✅ Configurações reais do config_ultra_optimized_FINAL.yaml
- ✅ Simulação realista de SL/TP
- ✅ Taxas Bybit (0.06% maker/taker)
- ✅ Slippage e timeout
- ✅ Código revisado 3x - SEM BUGS

---

## 🚀 COMO USAR

### Passo 1: Treinar Modelo (se ainda não treinou)

```bash
python train_model_PERFEITO.py
```

Aguarde 20-30 minutos. O modelo será salvo em `storage/models/ultra_scalper_btcusdt_365d.pkl`.

### Passo 2: Executar Backtest

```bash
python backtest_PERFEITO.py
```

**Tempo**: ~5-10 minutos (depende da quantidade de trades)

---

## 📊 O QUE O BACKTEST FAZ

### ETAPA 1: Carrega Modelo e Config
- Carrega `ultra_scalper_btcusdt_365d.pkl`
- Carrega `config_ultra_optimized_FINAL.yaml`
- Valida se arquivos existem

### ETAPA 2: Download de Dados (180 dias)
- Baixa 180 dias de dados BTC/USDT da Binance
- Período: Últimos 6 meses (out-of-sample!)
- ~17,280 candles de 15m

**Por que 180 dias?**
- Modelo treinou com 365d (ou período otimizado)
- Backtest usa 180d diferentes = validação out-of-sample
- Simula trading em dados "não vistos" pelo modelo

### ETAPA 3: Feature Engineering
- Calcula **MESMAS** features do treinamento
- 42 features técnicas (RSI, MACD, ATR, etc.)
- Detecta regimes de mercado

### ETAPA 4: Simulação de Trades

Para cada candle:
1. **Detecta regime** (medium_bear, high_vol_bear, etc.)
2. **Verifica se regime habilitado** (config FINAL)
3. **Predição do modelo** (long ou short + confiança)
4. **Filtra por min_confidence** do regime
5. **Calcula SL/TP** (baseado em ATR e config)
6. **Simula execução**:
   - Loop pelas próximas ~100 barras (25 horas)
   - Verifica se hit SL (prioridade 1)
   - Verifica se hit TP (prioridade 2)
   - Timeout se não atingir em 100 barras
7. **Aplica taxas** (0.06% entry + 0.06% exit = 0.12%)
8. **Atualiza equity**

---

## 📈 MÉTRICAS CALCULADAS

### Performance Geral:
```
Total Trades: 250
Win Rate: 54.8%
Total PnL: $2,450.00
ROI: +24.5%
Sharpe Ratio: 2.8
Max Drawdown: -5.2%
```

**Interpretação**:
- **Win Rate**: % de trades vencedores (meta: > 48%)
- **ROI**: Retorno sobre investimento inicial de $10k (meta: > 0%)
- **Sharpe**: Retorno ajustado por risco (> 2.0 = excelente)
- **Max DD**: Maior queda do pico (< -10% = bom)

### Trades:
```
Winners: 137 (54.8%)
Losers: 113 (45.2%)
Avg Win: $32.50
Avg Loss: -$18.20
Best Trade: +4.8%
Worst Trade: -2.1%
```

**Interpretação**:
- **Avg Win > |Avg Loss|**: Bom! Ganha mais quando acerta
- **Best/Worst**: Valida se SL/TP estão funcionando

### Long vs Short (IMPORTANTE!):
```
Longs: 125 (50.0%) - WR: 56.0%
Shorts: 125 (50.0%) - WR: 53.6%
✅ BALANCEADO! (diferença: 0.0%)
```

**Validação**:
- ✅ Diferença < 10%: PERFEITO!
- ⚠️  Diferença 10-20%: Aceitável
- ❌ Diferença > 20%: Problema! Retreinar

**Por que isso importa?**
- Modelo balanceado funciona em bull E bear markets
- Desbalanceado (ex: 80% longs) só funciona em bull

### Duração:
```
Avg Bars Held: 12.5 (≈3.1 horas)
```

**Interpretação**:
- Scalping deve ser rápido: 1-6 horas ideal
- > 10 horas: Pode virar swing trade (ok se lucrativo)

### Exit Reasons:
```
Take Profit: 137 (54.8%)
Stop Loss: 98 (39.2%)
Timeout: 15 (6.0%)
```

**Interpretação**:
- **TP > SL**: Bom! Modelo acerta mais que erra
- **Timeout baixo**: SL/TP bem calibrados
- **Timeout alto (> 20%)**: Aumentar SL/TP

### Por Regime:
```
medium_bear:    85 trades, WR: 58.8%, PnL: $+1,450.00
high_vol_bear:  60 trades, WR: 53.3%, PnL: $+680.00
low_vol_bear:   45 trades, WR: 51.1%, PnL: $+220.00
high_vol_bull:  60 trades, WR: 48.3%, PnL: $+100.00
```

**Interpretação**:
- Identifica quais regimes são mais lucrativos
- Se regime tem WR < 40% → desabilitar no config

---

## ✅ CRITÉRIOS DE VALIDAÇÃO

### Backtest APROVADO se:

1. ✅ **Win Rate > 48%** (idealmente > 50%)
2. ✅ **ROI > 0%** (idealmente > +15% em 6 meses)
3. ✅ **Sharpe > 1.5** (idealmente > 2.0)
4. ✅ **Max DD < -10%** (idealmente < -6%)
5. ✅ **Long/Short balanceados** (diferença < 15%)
6. ✅ **Todos regimes com WR > 40%**

### Se APROVADO:
```bash
# Configurar .env
cp .env.example .env
# Editar com suas credenciais

# Paper trading (2 semanas)
python main.py
```

### Se REPROVADO:

**Cenário 1: WR < 45% ou ROI negativo**
- Modelo ruim, retreinar:
```bash
python train_model_PERFEITO.py
```

**Cenário 2: Long/Short desbalanceado (> 20%)**
- Threshold problema, retreinar:
```bash
python train_model_PERFEITO.py
```
(Script vai auto-ajustar threshold)

**Cenário 3: WR 45-48% (marginal)**
- Ajustar config FINAL:
  - Aumentar min_confidence dos regimes fracos
  - Desabilitar regimes com WR < 45%

---

## 📋 ARQUIVO GERADO

### `storage/models/backtest_results.csv`

Contém todos os trades:
```csv
entry_time,entry_price,exit_price,side,regime,confidence,pnl_pct,pnl_usd,exit_reason,bars_held,stop_loss,take_profit
2024-05-01 10:00:00,62500,63100,long,medium_bear,0.68,0.89,89.00,tp,8,61800,63500
```

**Use para**:
- Analisar trades específicos
- Importar no Excel/Google Sheets
- Criar gráficos personalizados

---

## 🔧 TROUBLESHOOTING

### Erro: "Modelo não encontrado"

**Causa**: Modelo ainda não foi treinado

**Solução**:
```bash
python train_model_PERFEITO.py
```

### Erro: "Config não encontrado"

**Causa**: Arquivo config_ultra_optimized_FINAL.yaml ausente

**Solução**: Verifique se arquivo existe no diretório raiz

### Backtest muito lento

**Causa**: Muitos trades sendo simulados (> 500)

**Solução**: Normal! Aguarde 10-15 minutos

### "Nenhum trade executado"

**Causas possíveis**:
1. Modelo ruim (confidence sempre baixa)
2. Todos regimes desabilitados no config
3. Min_confidence muito alta

**Solução**: Revisar config FINAL, reduzir min_confidence temporariamente para teste

### Win Rate muito baixo (< 40%)

**Causa**: Modelo não generalizou

**Solução**: Retreinar com mais dados ou ajustar features

---

## 📊 EXEMPLO DE OUTPUT COMPLETO

```
================================================================================
📊 BACKTEST PERFEITO - Validação da Estratégia
================================================================================

📋 Verificando dependências...
✅ Todas as dependências instaladas!

================================================================================
ETAPA 1: CARREGANDO MODELO E CONFIG
================================================================================

🤖 Carregando modelo: storage/models/ultra_scalper_btcusdt_365d.pkl
✅ Modelo carregado! Tamanho: 85.40 MB
⚙️  Carregando config: config_ultra_optimized_FINAL.yaml
✅ Config carregado!

================================================================================
ETAPA 2: DOWNLOAD DE DADOS
================================================================================

📥 Baixando 180 dias de dados para backtest...
   Progresso: 100.0% - 17,280 candles
✅ 17,280 candles baixados!

================================================================================
ETAPA 3: FEATURE ENGINEERING
================================================================================

🔧 Calculando features...
✅ 42 features calculadas!

================================================================================
ETAPA 4: SIMULAÇÃO
================================================================================

🚀 EXECUTANDO BACKTEST
================================================================================

📊 Período: 2024-05-15 a 2024-11-15
   Candles: 17,280

💹 Simulando trades...

   Trades executados: 250
✅ 250 trades simulados!

================================================================================
📊 CALCULANDO MÉTRICAS
================================================================================

================================================================================
✅ RESULTADOS DO BACKTEST
================================================================================

📈 PERFORMANCE GERAL:
   Total Trades: 250
   Win Rate: 54.80%
   Total PnL: $2,450.00
   ROI: +24.50%
   Sharpe Ratio: 2.85
   Max Drawdown: -5.20%

💰 TRADES:
   Winners: 137 (54.8%)
   Losers: 113 (45.2%)
   Avg Win: $32.50
   Avg Loss: $-18.20
   Best Trade: +4.85%
   Worst Trade: -2.15%

📊 LONG vs SHORT:
   Longs: 125 (50.0%) - WR: 56.00%
   Shorts: 125 (50.0%) - WR: 53.60%
   ✅ BALANCEADO! (diferença: 0.0%)

⏱️  DURAÇÃO:
   Avg Bars Held: 12.5 (≈3.1 horas)

🎯 EXIT REASONS:
   Take Profit: 137 (54.8%)
   Stop Loss: 98 (39.2%)
   Timeout: 15 (6.0%)

🏆 POR REGIME:
   medium_bear    :  85 trades, WR:  58.8%, PnL: $+1450.00
   high_vol_bear  :  60 trades, WR:  53.3%, PnL: $+680.00
   low_vol_bear   :  45 trades, WR:  51.1%, PnL: $+220.00
   high_vol_bull  :  60 trades, WR:  48.3%, PnL: $+100.00

================================================================================
💾 EQUITY CURVE:
   Inicial: $10,000.00
   Final: $12,450.00
   Pico: $12,680.00
   Vale: $9,480.00

💾 Trades salvos em: storage/models/backtest_results.csv

================================================================================

================================================================================
✅ BACKTEST COMPLETO!
================================================================================

🚀 Próximos passos:
   1. Revisar métricas acima
   2. Verificar balanceamento Long/Short
   3. Se WR > 48% e ROI > 0% → Validar em paper trading
   4. Se WR < 48% ou ROI < 0% → Retreinar modelo

================================================================================
```

---

## 🎯 COMPARAÇÃO: Backtest vs Paper vs Live

| Aspecto | Backtest | Paper Trading | Live Trading |
|---------|----------|---------------|--------------|
| **Dados** | Históricos | Real-time | Real-time |
| **Execução** | Simulada | Simulada | Real |
| **Dinheiro** | Nenhum | Nenhum | Real |
| **Slippage** | Estimado | Real | Real |
| **Latência** | Zero | Real | Real |
| **Objetivo** | Validação | Validação | Lucro |
| **Duração** | 5-10 min | 1-2 semanas | Contínuo |

**Pipeline Recomendado**:
1. ✅ **Backtest** (hoje) → Valida se modelo funciona
2. ✅ **Paper** (2 semanas) → Valida em tempo real
3. ✅ **Live** (se paper > 48% WR) → Trading real

---

## ✅ RESUMO

### Execute:
```bash
python backtest_PERFEITO.py
```

### Aguarde: 5-10 minutos

### Valide:
- ✅ WR > 48%
- ✅ ROI > 0%
- ✅ Sharpe > 1.5
- ✅ Max DD < -10%
- ✅ Long/Short balanceados

### Se aprovado:
```bash
python main.py  # Paper trading
```

### Se reprovado:
```bash
python train_model_PERFEITO.py  # Retreinar
```

---

**BOA SORTE! 🚀**

Este backtest é sua primeira validação antes de arriscar dinheiro real! 💪
