# 🚀 QUICK START - MAINNET MODE

Guia rápido para colocar o bot funcionando em mainnet (paper ou live trading).

---

## ⚡ Instalação Automática (RECOMENDADO)

### Windows:

1. **Clone o repositório** (se ainda não clonou)
2. **Execute o script de início**:
   ```cmd
   start.bat
   ```

Isso vai:
- ✅ Criar ambiente virtual
- ✅ Instalar dependências
- ✅ Validar configuração
- ✅ Iniciar o bot

### Linux/Mac:

```bash
bash start.sh
```

---

## 🔧 Instalação Manual

### 1. Instalar Dependências

```bash
# Criar ambiente virtual (recomendado)
python -m venv venv

# Ativar ambiente virtual
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Instalar dependências
pip install -r requirements.txt
```

### 2. Copiar Modelo ML

**CRÍTICO**: O bot precisa do modelo ML treinado!

**Opção A: Copiar do ambiente Windows**

Se você já treinou o modelo:

```bash
# Execute o script de cópia
python copy_model_from_windows.py
```

O script vai procurar o modelo em:
- `C:\Users\alyso\Downloads\bybit_scalping_bot\storage\models\ultra_scalper_btcusdt_365d.pkl`
- `storage\models\ultra_scalper_btcusdt_365d.pkl`
- E outras localizações comuns

**Opção B: Treinar novo modelo**

Se você NÃO tem o modelo:

```bash
# Execute um dos scripts de validação (isso treina o modelo)
python validate_ultra_optimized_FINAL.py
```

O modelo será salvo automaticamente em `storage/models/`

### 3. Configurar .env

```bash
# Copiar template
cp .env.example .env

# Editar com suas credenciais
# Windows: notepad .env
# Linux/Mac: nano .env
```

**Campos obrigatórios**:

```bash
# Modo de trading (SEMPRE comece com paper!)
TRADING_MODE=paper

# API Bybit (pegar em https://www.bybit.com/app/user/api-management)
BYBIT_API_KEY=sua_api_key_aqui
BYBIT_API_SECRET=seu_api_secret_aqui
BYBIT_API_URL=https://api.bybit.com

# Telegram (criar bot com @BotFather)
TELEGRAM_BOT_TOKEN=seu_bot_token_aqui
TELEGRAM_CHAT_ID=seu_user_id_aqui
TELEGRAM_ENABLED=true

# Risk Management
MAX_POSITION_SIZE_USD=1000.0
DAILY_LOSS_LIMIT_PCT=5.0

# Paper Trading
PAPER_INITIAL_BALANCE=10000.0
```

### 4. Validar Configuração

```bash
python setup.py
```

**Output esperado**:

```
================================================================================
🚀 BTC SCALPER BOT - SETUP MAINNET
================================================================================

📋 VALIDANDO AMBIENTE...

1️⃣  Python Version
✅ Python 3.10.0

2️⃣  Estrutura de Diretórios
✅ Diretório 'core' existe
✅ Diretório 'storage' existe
✅ Diretório 'storage/models' existe
✅ Diretório 'logs' existe

3️⃣  Arquivos Core
✅ Arquivo 'main.py'
✅ Arquivo 'core/bybit_api.py'
✅ Arquivo 'core/trading_bot.py'
✅ Arquivo 'core/telegram_bot.py'
✅ Arquivo 'config_ultra_optimized_FINAL.yaml'

4️⃣  Configuração (.env)
✅ Arquivo .env existe
✅ TRADING_MODE configurado
✅ BYBIT_API_KEY configurado
✅ BYBIT_API_SECRET configurado
✅ TELEGRAM_BOT_TOKEN configurado
✅ TELEGRAM_CHAT_ID configurado

5️⃣  Modelo ML
✅ Modelo ML 'storage/models/ultra_scalper_btcusdt_365d.pkl'
✅ Modelo tem 125.3 MB (tamanho OK)

6️⃣  Dependências Python
✅ Pacote 'pandas' instalado
✅ Pacote 'numpy' instalado
✅ Pacote 'yaml' instalado
✅ Pacote 'requests' instalado
✅ Pacote 'dotenv' instalado
✅ Pacote 'telegram' instalado
✅ Pacote 'sklearn' instalado
✅ Pacote 'lightgbm' instalado
✅ Pacote 'xgboost' instalado

================================================================================
📊 RESUMO
================================================================================

✅ TUDO OK! Ambiente pronto para mainnet!

🚀 Próximos passos:

   1. Revise o arquivo .env (principalmente TRADING_MODE)
   2. Execute: python main.py
   3. Use /start no Telegram para controlar o bot

⚠️  IMPORTANTE: Sempre comece em PAPER TRADING MODE!

================================================================================
```

### 5. Executar Bot

```bash
python main.py
```

**Output esperado**:

```
================================================================================
🚀 BTC SCALPER BOT STARTING
================================================================================
✅ Connected to Bybit - Server time: 2025-11-19 12:30:45
📄 PAPER TRADING MODE - Initial balance: $10,000.00
✅ Loaded config from: config_ultra_optimized_FINAL.yaml
✅ Loaded ML model from: storage/models/ultra_scalper_btcusdt_365d.pkl
🤖 Trading Bot initialized - BTCUSDT 15m - Mode: paper
✅ Telegram handlers registered
🚀 Starting trading loop...
```

### 6. Controlar via Telegram

1. Abra Telegram
2. Fale com seu bot (o que criou com @BotFather)
3. Envie `/start`
4. Clique em "▶️ Start Bot"
5. Receba notificações!

---

## 🔍 Troubleshooting

### ❌ Erro: "No module named 'pandas'"

```bash
pip install -r requirements.txt
```

### ❌ Erro: "No such file or directory: 'storage/models/ultra_scalper_btcusdt_365d.pkl'"

Modelo ML não encontrado!

**Solução**:
1. Execute: `python copy_model_from_windows.py`
2. Ou treine novo modelo: `python validate_ultra_optimized_FINAL.py`

### ❌ Erro: "BYBIT_API_KEY must be set"

Arquivo `.env` não foi criado ou configurado.

**Solução**:
```bash
cp .env.example .env
# Edite .env com suas credenciais
```

### ❌ Erro: "Invalid signature"

API Secret está incorreto.

**Solução**:
1. Verifique se copiou o secret corretamente (sem espaços)
2. Gere nova API key na Bybit se necessário

### ❌ Erro: "Unauthorized access denied" (Telegram)

`TELEGRAM_CHAT_ID` está incorreto.

**Solução**:
1. Fale com @userinfobot no Telegram
2. Copie seu ID para `.env`

### ⚠️ Bot não abre trades

**Possíveis causas**:
1. Nenhum sinal ML válido
2. Confidence abaixo do threshold
3. Regime bloqueado
4. Position já aberta

**Solução**: Verifique logs
```bash
# Windows
type logs\trading_bot.log | findstr "ML Signal"

# Linux/Mac
tail -f logs/trading_bot.log | grep "ML Signal"
```

---

## 📋 Checklist Final

Antes de executar:

- [ ] Python 3.8+ instalado
- [ ] Dependências instaladas (`pip install -r requirements.txt`)
- [ ] Modelo ML copiado para `storage/models/`
- [ ] Arquivo `.env` criado e configurado
- [ ] API keys da Bybit configuradas
- [ ] Telegram bot criado e configurado
- [ ] `python setup.py` executado com sucesso
- [ ] TRADING_MODE=paper (SEMPRE comece com paper!)

Se tudo OK → Execute: `python main.py`

---

## 🎯 Próximos Passos

### Semana 1-2: Paper Trading
- [ ] Executar bot em paper mode
- [ ] Monitorar trades diariamente
- [ ] Verificar win rate ≥ 48%
- [ ] Verificar ROI positivo

### Semana 3: Análise
- [ ] Analisar performance vs backtest
- [ ] Decidir: ir para live ou não?

### Semana 4+: Live Trading (SE aprovado)
- [ ] Mudar para `TRADING_MODE=live`
- [ ] Começar com capital pequeno ($500-1000)
- [ ] Monitorar DIARIAMENTE

---

## ⚠️ AVISOS IMPORTANTES

### 🚨 SEMPRE comece em PAPER MODE!

```bash
# .env
TRADING_MODE=paper  ← SEMPRE isso primeiro!
```

Rode por **1-2 semanas** para validar antes de arriscar dinheiro real.

### 🚨 API V2 vs V5

O código usa Bybit API V2. Se der erro "endpoint not found":
- Bybit pode ter desativado V2
- Precisa migrar para V5 API
- Me avise que crio versão V5

### 🚨 Modelo ML

Modelos ML degradam com o tempo. Re-treine a cada 3-6 meses.

---

## 📞 Suporte

**Logs**: `logs/trading_bot.log`
**Trades**: `logs/trades.csv`
**Telegram**: Use `/help` no bot

**Docs completos**: `BOT_SETUP_GUIDE.md`

---

✅ **PRONTO PARA MAINNET!** 🚀

Execute: `python main.py` ou `start.bat` (Windows) ou `bash start.sh` (Linux/Mac)
