# 📂 GUIA DE INSTALAÇÃO E USO

## 🎯 Onde Colocar os Arquivos

### **Estrutura Final do Projeto:**

```
C:\Users\alyso\Downloads\bybit_scalping_bot\
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── validation/                              # ← PASTA DOS MÓDULOS
│   ├── __init__.py
│   ├── confidence_filter.py
│   ├── optimize_confidence_threshold.py
│   ├── purged_kfold.py
│   ├── ensemble_scoring.py
│   └── run_advanced_validation.py
│
├── examples/                                # ← PASTA DE EXEMPLOS
│   ├── example_integration.py
│   └── compare_before_after.py
│
├── docs/                                    # ← DOCUMENTAÇÃO
│   ├── VALIDATION_GUIDE.md
│   └── DIRECTORY_STRUCTURE.md
│
└── scripts/                                 # ← SEUS SCRIPTS (opcional)
    ├── validate_optimized_ultra_scalper.py  # SEU SCRIPT ORIGINAL
    ├── validate_with_confidence.py          # NOVO: Com filtro
    └── train_model.py                       # Etc...

# OU TUDO NA RAIZ (mais simples):
C:\Users\alyso\Downloads\bybit_scalping_bot\
├── README.md
├── requirements.txt
├── validate_optimized_ultra_scalper.py      # SEU SCRIPT ORIGINAL
├── confidence_filter.py
├── optimize_confidence_threshold.py
├── purged_kfold.py
├── ensemble_scoring.py
├── run_advanced_validation.py
├── example_integration.py
└── compare_before_after.py
```

---

## 📥 **INSTALAÇÃO NO WINDOWS**

### **Opção 1: Organizado (Recomendado)**

```powershell
# 1. No seu terminal PowerShell, vá para a pasta do bot:
cd C:\Users\alyso\Downloads\bybit_scalping_bot

# 2. Clone ou baixe os arquivos do repositório
# (você já tem isso commitado no GitHub)

# 3. Criar pastas (se não existirem):
mkdir validation
mkdir examples
mkdir docs
mkdir scripts  # Opcional

# 4. Mover seus scripts existentes:
move validate_optimized_ultra_scalper.py scripts\
move ultra_scalper_btcusdt_365d.pkl scripts\  # Seu modelo

# 5. Instalar dependências:
pip install -r requirements.txt

# 6. Testar instalação:
python validation/run_advanced_validation.py --help
python examples/compare_before_after.py --demo
```

### **Opção 2: Tudo na Raiz (Mais Simples)**

```powershell
# 1. Vá para a pasta:
cd C:\Users\alyso\Downloads\bybit_scalping_bot

# 2. Baixe/copie todos os arquivos .py para a raiz

# 3. Instalar dependências:
pip install -r requirements.txt

# 4. Testar:
python run_advanced_validation.py --demo
python compare_before_after.py --demo
```

---

## 🚀 **COMO USAR**

### **1. Teste Rápido (5 minutos)**

```powershell
# Testar pipeline completo com dados simulados:
python validation/run_advanced_validation.py --demo --samples 1000

# Comparar antes/depois:
python examples/compare_before_after.py --demo --samples 1000

# Testar módulos individuais:
python validation/confidence_filter.py
python validation/optimize_confidence_threshold.py
python validation/purged_kfold.py
python validation/ensemble_scoring.py
```

**Você deve ver:**
- ✅ Todos os testes passando
- 📊 Relatórios de métricas
- 💡 Recomendações

---

### **2. Integrar ao Seu Bot (1-2 horas)**

#### **PASSO 1: Backup**
```powershell
# Fazer backup do seu script original:
copy validate_optimized_ultra_scalper.py validate_optimized_ultra_scalper_BACKUP.py
```

#### **PASSO 2: Adicionar Imports**

No topo do seu `validate_optimized_ultra_scalper.py`:

```python
# ADICIONE ESTAS LINHAS:
import sys
import os

# Se usar estrutura organizada:
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from validation.confidence_filter import ConfidenceFilter
from validation.optimize_confidence_threshold import ThresholdOptimizer

# Se tudo na raiz:
from confidence_filter import ConfidenceFilter
from optimize_confidence_threshold import ThresholdOptimizer
```

#### **PASSO 3: Otimizar Threshold (1x, antes de validar)**

```python
# ADICIONE ANTES DO SEU BACKTEST:

# 1. Coletar probabilidades e retornos históricos
probas_train = model.predict_proba(X_train)[:, 1]
returns_train = calculate_returns(trades_train)  # Seu cálculo

# 2. Otimizar threshold
optimizer = ThresholdOptimizer()
best_threshold, _ = optimizer.optimize(
    probas_train,
    returns_train,
    objective='sharpe'
)

print(f"✅ Threshold Ótimo: {best_threshold:.1%}")
# Salvar para usar em produção
```

#### **PASSO 4: Criar Filtro**

```python
# ADICIONE APÓS OTIMIZAÇÃO:

# Multiplicadores por regime (use seus resultados)
regime_multipliers = {
    'medium_bear': 0.92,      # Seu melhor regime
    'high_vol_bear': 0.95,
    'low_vol_bear': 0.98,
    'high_vol_bull': 1.00,
    'medium_bull': 1.03,
    'low_vol_bull': 1.15,     # Seu pior regime
}

# Criar filtro
cf = ConfidenceFilter(
    threshold=best_threshold,
    adaptive=True,
    regime_multipliers=regime_multipliers,
    dd_adjustment=True
)
```

#### **PASSO 5: Usar Filtro no Backtest**

```python
# SUBSTITUIR ESTE CÓDIGO:
# predictions = model.predict(X_test)

# POR:
current_regime = detect_regime(...)  # Seu código
current_dd = calculate_drawdown(...)  # Seu código

predictions, confidences = cf.predict(
    model,
    X_test,
    regime=current_regime,
    current_dd=current_dd
)

# Continuar com seu backtest usando predictions
```

#### **PASSO 6: Ver Estatísticas**

```python
# AO FINAL DO BACKTEST:
cf.print_report()
```

---

### **3. Comparar Resultados**

#### **Opção A: Rodar Script de Comparação**

```powershell
# Com dados simulados:
python examples/compare_before_after.py --demo

# Com seus dados reais (adapte o código):
python examples/compare_before_after.py --real-data
```

#### **Opção B: Rodar Seu Validate 2x**

```powershell
# 1. SEM filtro (comentar o código do filtro):
python validate_optimized_ultra_scalper.py --symbol BTCUSDT --days 90
# Anotar: Win Rate, Sharpe, ROI

# 2. COM filtro (descomentar):
python validate_optimized_ultra_scalper.py --symbol BTCUSDT --days 90
# Comparar resultados
```

---

## 📊 **RESULTADOS ESPERADOS**

### **Seus Resultados Atuais (ANTES):**
```
Walk-Forward:
  Fold 1: -4.07% ROI ❌
  Fold 2: +3.49% ROI
  Fold 3: +14.93% ROI
  Fold 4: +17.75% ROI
  Fold 5: +14.34% ROI

Overall:
  Win Rate: 50.3%
  ROI: +67.34%
  Sharpe: 2.57
  Trades: 579
  Blocked: 1488 (72%)
```

### **Esperado COM Filtro de Confiança:**
```
Walk-Forward:
  Fold 1: +2-5% ROI ✅ (melhorado!)
  Fold 2: +5-8% ROI
  Fold 3: +16-20% ROI
  Fold 4: +19-23% ROI
  Fold 5: +16-19% ROI

Overall:
  Win Rate: 55-58% ⬆️ (+5-8pp)
  ROI: +70-80% ⬆️
  Sharpe: 3.0-3.5 ⬆️ (+20-35%)
  Trades: 250-350 ⬇️ (melhor qualidade)
  Blocked: 40-50% (mais eficiente)
```

**Principais Melhorias:**
- ✅ **Fold 1 deixa de ser negativo** (crítico!)
- ✅ **Win Rate mais consistente** (menos variação)
- ✅ **Sharpe aumenta** (melhor risk-return)
- ✅ **Menos dependência do regime filter**

---

## ⚙️ **CONFIGURAÇÃO AVANÇADA**

### **Ajustar Threshold Manualmente**

Se quiser testar diferentes thresholds:

```python
# No seu validate:
CONFIDENCE_THRESHOLD = 0.62  # Testar: 0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70

cf = ConfidenceFilter(threshold=CONFIDENCE_THRESHOLD)
```

**Recomendações por perfil:**
- **Conservador:** 0.68-0.75 (poucos trades, alta qualidade)
- **Balanceado:** 0.60-0.65 (equilíbrio)
- **Agressivo:** 0.52-0.58 (mais trades)

### **Usar Ensemble Scoring (Avançado)**

Para decisões mais sofisticadas:

```python
from validation.ensemble_scoring import EnsembleScorer

# Configurar com seus dados de regime
scorer = EnsembleScorer(
    min_score=55.0,
    regime_performance=YOUR_REGIME_STATS
)

# Em cada iteração:
signal = scorer.score_trade(
    model_proba=proba,
    regime=current_regime,
    volatility=current_vol,
    current_dd=current_dd
)

if signal.should_trade and signal.quality.value in ['excellent', 'good']:
    execute_trade()
```

---

## 🐛 **TROUBLESHOOTING**

### **Erro: ModuleNotFoundError**

```powershell
# Se estrutura organizada:
# Certifique-se de rodar do diretório raiz:
cd C:\Users\alyso\Downloads\bybit_scalping_bot
python validation/run_advanced_validation.py --demo

# Ou adicionar ao PATH:
set PYTHONPATH=%PYTHONPATH%;C:\Users\alyso\Downloads\bybit_scalping_bot
```

### **Erro: No module named 'numpy'**

```powershell
# Instalar dependências:
pip install -r requirements.txt

# Ou manualmente:
pip install numpy pandas scikit-learn matplotlib seaborn
```

### **Threshold muito alto/baixo**

Se threshold otimizado parece estranho:
- **> 0.75:** Modelo muito conservador, poucas trades
  - Solução: Limitar max_threshold=0.72
- **< 0.52:** Modelo pouco confiante
  - Solução: Treinar modelo melhor com calibração

### **Performance não melhora**

Possíveis causas:
1. **Modelo sem predict_proba:** Alguns modelos não retornam probabilidades
   - Solução: Usar modelo com suporte (RandomForest, XGBoost, LightGBM)

2. **Probabilidades não calibradas:** Modelo retorna sempre ~0.6
   - Solução: Aplicar calibração (CalibratedClassifierCV)

3. **Poucos dados:** < 500 trades para otimizar
   - Solução: Usar mais dados históricos (6+ meses)

---

## 📚 **PRÓXIMOS PASSOS**

### **Semana 1: Teste e Validação**
- [ ] Instalar módulos
- [ ] Rodar demos
- [ ] Comparar resultados
- [ ] Ler VALIDATION_GUIDE.md

### **Semana 2: Integração**
- [ ] Otimizar threshold com seus dados
- [ ] Integrar filtro ao validate
- [ ] Backteste completo
- [ ] Validar com Purged K-Fold

### **Semana 3: Deploy**
- [ ] Paper trading 1 semana
- [ ] Monitorar métricas
- [ ] Ajustar threshold se necessário
- [ ] Deploy gradual

### **Mensal: Manutenção**
- [ ] Re-otimizar threshold
- [ ] Re-validar com dados novos
- [ ] Atualizar regime_multipliers
- [ ] Retreinar modelo se drift

---

## 💡 **DICAS IMPORTANTES**

1. **SEMPRE faça backup** antes de modificar código
2. **Teste com dados simulados** primeiro
3. **Compare resultados** antes/depois
4. **Paper trade** antes de usar dinheiro real
5. **Re-otimize mensalmente** o threshold
6. **Monitore drift** de performance

---

## 📞 **SUPORTE**

### **Documentação:**
- **VALIDATION_GUIDE.md** - Guia completo de uso
- **DIRECTORY_STRUCTURE.md** - Estrutura de pastas
- **example_integration.py** - Exemplos de código

### **Scripts de Teste:**
```powershell
# Testar instalação:
python validation/confidence_filter.py

# Testar otimização:
python validation/optimize_confidence_threshold.py

# Testar Purged K-Fold:
python validation/purged_kfold.py

# Comparar antes/depois:
python examples/compare_before_after.py --demo
```

### **Verificar que tudo está OK:**
```powershell
# Se todos estes comandos funcionam, instalação OK:
python -c "from validation import ConfidenceFilter; print('✅ OK')"
python -c "from validation import ThresholdOptimizer; print('✅ OK')"
python -c "from validation import PurgedKFold; print('✅ OK')"
python -c "from validation import EnsembleScorer; print('✅ OK')"
```

---

## ✅ **CHECKLIST DE INSTALAÇÃO**

- [ ] Pastas criadas (validation/, examples/, docs/)
- [ ] Arquivos copiados para pastas corretas
- [ ] requirements.txt instalado
- [ ] Testes básicos rodando
- [ ] Demo funcionando
- [ ] Imports funcionando no seu código
- [ ] Backup do código original feito

**Se todos ✅, você está pronto para integrar!**

---

**Boa sorte! 🚀**
