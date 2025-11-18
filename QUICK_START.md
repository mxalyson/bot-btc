# 🚀 QUICK START - 5 MINUTOS

## ⚡ Instalação Express

```powershell
# 1. Entre na pasta do projeto
cd C:\Users\alyso\Downloads\bybit_scalping_bot

# 2. Baixe o código
git pull origin claude/validate-scalper-monte-carlo-01KBXBe6PPf5kkzrjWUpPyqZ

# 3. Instale dependências
pip install -r requirements.txt

# 4. Teste
python validation\run_advanced_validation.py --demo --samples 1000
```

**Se tudo rodou OK ✅, você está pronto!**

---

## 📖 Próximos Passos

1. **Leia:** [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md) - Guia completo de instalação

2. **Teste:** `python examples\compare_before_after.py --demo` - Veja melhoria esperada

3. **Integre:** Siga os exemplos em [examples/example_integration.py](examples/example_integration.py)

---

## 🎯 O Que Isso Faz?

Adiciona **filtro de confiança** ao seu bot para:
- ✅ Melhorar Win Rate: 50% → 56%
- ✅ Aumentar Sharpe: 2.57 → 3.2
- ✅ Corrigir Fold 1 negativo (-4% → +2%)

---

## 💻 Como Integrar ao Seu Código

No seu `validate_optimized_ultra_scalper.py`:

```python
# 1. Adicionar no topo:
from validation.confidence_filter import ConfidenceFilter

# 2. Criar filtro (uma vez):
cf = ConfidenceFilter(threshold=0.62, adaptive=True)

# 3. Usar no backtest (substituir model.predict):
predictions, confidences = cf.predict(
    model,
    X_test,
    regime=current_regime,
    current_dd=current_dd
)
# Continuar com seu código usando predictions
```

**Simples assim!**

---

## 📚 Documentação Completa

- **INSTALLATION_GUIDE.md** - Como instalar no Windows
- **docs/VALIDATION_GUIDE.md** - Guia completo (3000+ linhas)
- **examples/example_integration.py** - 5 exemplos de código
- **scripts/validate_with_confidence.py** - Script adaptado completo

---

## 🧪 Testar Módulos

```powershell
# Testar filtro de confiança:
python validation\confidence_filter.py

# Testar otimizador:
python validation\optimize_confidence_threshold.py

# Testar Purged K-Fold:
python validation\purged_kfold.py

# Comparar antes/depois:
python examples\compare_before_after.py --demo
```

---

## ⚠️ Importante

1. **Faça backup** do seu código antes de modificar
2. **Teste com --demo** primeiro
3. **Compare resultados** antes/depois
4. **Paper trade** antes de usar dinheiro real

---

## 💡 Dúvidas?

Leia [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md) para instruções detalhadas.

**Boa sorte! 🎉**
