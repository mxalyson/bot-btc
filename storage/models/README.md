# 📦 ML Model Storage

## ⚠️ IMPORTANTE: Modelo ML Necessário!

Para o bot funcionar, você precisa do modelo ML treinado.

### 📥 Como obter o modelo:

**Opção 1: Copiar do seu ambiente Windows**

Se você já tem o modelo treinado:

```bash
# No Windows, copie o arquivo .pkl para este diretório
# Caminho Windows: C:\Users\alyso\Downloads\bybit_scalping_bot\storage\models\

# Arquivo necessário:
ultra_scalper_btcusdt_365d.pkl
```

**Opção 2: Treinar um novo modelo**

Se você NÃO tem o modelo, precisa treinar usando os scripts de validação:

1. Use um dos scripts de validação existentes (eles têm o código de treinamento)
2. O modelo será salvo automaticamente em `storage/models/`

### ✅ Verificar se o modelo existe:

```python
import os

model_path = 'storage/models/ultra_scalper_btcusdt_365d.pkl'

if os.path.exists(model_path):
    print("✅ Modelo encontrado!")
else:
    print("❌ Modelo NÃO encontrado! Bot não vai funcionar.")
```

### 📊 Detalhes do modelo esperado:

- **Nome**: `ultra_scalper_btcusdt_365d.pkl`
- **Tipo**: Ensemble (LightGBM + XGBoost + Transformer)
- **Treinamento**: 365 dias de dados BTC/USDT 15min
- **Tamanho**: ~50-200 MB (varia conforme features)

### 🔍 Onde encontrar o modelo original:

Se você rodou algum dos scripts de validação anteriormente:
- `validate_ultra_optimized_FINAL.py`
- `validate_ultra_optimized_v3.py`
- `validate_ultra_optimized_V5.py`

O modelo deve ter sido salvo em `storage/models/ultra_scalper_btcusdt_365d.pkl`

### 🚨 Se o modelo não existir:

O bot dará este erro ao iniciar:

```
❌ Failed to load model: [Errno 2] No such file or directory: 'storage/models/ultra_scalper_btcusdt_365d.pkl'
```

**Solução**: Copie o modelo do seu ambiente de desenvolvimento ou treine um novo.

---

**NOTA**: Modelos ML são grandes (50-200MB) e NÃO devem ser commitados no git por isso este diretório está vazio no repositório.
