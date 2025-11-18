# 📂 Estrutura de Diretórios Recomendada

```
bot-btc/
├── README.md
├── requirements.txt
├── .gitignore
│
├── validation/                    # ← NOVA PASTA: Módulos de validação
│   ├── __init__.py
│   ├── confidence_filter.py
│   ├── optimize_confidence_threshold.py
│   ├── purged_kfold.py
│   ├── ensemble_scoring.py
│   └── run_advanced_validation.py
│
├── examples/                      # ← NOVA PASTA: Exemplos
│   ├── example_integration.py
│   └── compare_before_after.py
│
├── docs/                          # ← NOVA PASTA: Documentação
│   └── VALIDATION_GUIDE.md
│
└── scripts/                       # ← PASTA PRINCIPAL: Seus scripts
    ├── validate_optimized_ultra_scalper.py  # SEU SCRIPT ORIGINAL
    ├── validate_with_confidence.py          # NOVO: Com filtro integrado
    └── train_model.py                       # Seu script de treino (se tiver)
```

## 📍 Onde Colocar os Arquivos

### Opção 1: Estrutura Organizada (RECOMENDADO)
- Criar pasta `validation/` para os módulos
- Criar pasta `examples/` para exemplos
- Criar pasta `docs/` para documentação
- Manter seus scripts na raiz ou em `scripts/`

### Opção 2: Tudo na Raiz (Simples)
- Deixar tudo na raiz do projeto
- Funciona, mas fica desorganizado

**Vou criar a Opção 1 para você agora:**
