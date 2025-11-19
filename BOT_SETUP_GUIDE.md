# 🤖 BTC SCALPER BOT - Setup & Usage Guide

## 📋 Índice

1. [Visão Geral](#visão-geral)
2. [Instalação](#instalação)
3. [Configuração](#configuração)
4. [Como Usar](#como-usar)
5. [Comandos Telegram](#comandos-telegram)
6. [Modos de Operação](#modos-de-operação)
7. [Monitoramento](#monitoramento)
8. [Troubleshooting](#troubleshooting)
9. [FAQ](#faq)

---

## 🎯 Visão Geral

Bot de trading automatizado para BTC/USDT usando:
- ✅ Machine Learning (modelo treinado)
- ✅ Regime detection (6 regimes de mercado)
- ✅ Risk management (SL/TP automáticos)
- ✅ Paper trading (simulação realista)
- ✅ Live trading (dinheiro real)
- ✅ Telegram integration (controle remoto)

### Performance Esperada (baseado em backtest):
- **ROI**: +89.55% em 90 dias
- **Win Rate**: 50.8%
- **Sharpe Ratio**: 3.56
- **Drawdown Máximo**: -4.2%
- **~500 trades** em 90 dias (~5-6 trades/dia)

---

## 🚀 Instalação

### 1. Pré-requisitos

- Python 3.8+
- Conta Bybit (www.bybit.com)
- Telegram (opcional mas recomendado)

### 2. Clone o repositório

```bash
cd C:\Users\alyso\Downloads\bybit_scalping_bot
```

### 3. Instale dependências

```bash
pip install -r requirements.txt
```

**Requirements.txt**:
```
pandas>=2.0.0
numpy>=1.24.0
pyyaml>=6.0
requests>=2.31.0
python-dotenv>=1.0.0
python-telegram-bot>=20.0
scikit-learn>=1.3.0
lightgbm>=4.0.0
xgboost>=2.0.0
```

### 4. Estrutura de diretórios

```
bot-btc/
├── .env                           # Suas credenciais (criar)
├── .env.example                   # Template
├── main.py                        # Entry point
├── core/
│   ├── bybit_api.py              # API wrapper
│   ├── trading_bot.py            # Trading engine
│   └── telegram_bot.py           # Telegram bot
├── config_ultra_optimized_FINAL.yaml  # Config de trading
├── storage/models/
│   └── ultra_scalper_btcusdt_365d.pkl  # ML model
└── logs/
    ├── trading_bot.log           # Logs
    └── trades.csv                # Histórico de trades
```

---

## ⚙️ Configuração

### 1. Criar arquivo .env

```bash
cp .env.example .env
```

Edite `.env` com suas credenciais:

```bash
# TRADING MODE
TRADING_MODE=paper   # "paper" ou "live"

# BYBIT API
BYBIT_API_KEY=your_api_key_here
BYBIT_API_SECRET=your_api_secret_here
BYBIT_API_URL=https://api.bybit.com

# TELEGRAM
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_user_id
TELEGRAM_ENABLED=true

# TRADING CONFIG
SYMBOL=BTCUSDT
TIMEFRAME=15
CONFIG_PATH=config_ultra_optimized_FINAL.yaml
MODEL_PATH=storage/models/ultra_scalper_btcusdt_365d.pkl

# RISK MANAGEMENT
MAX_POSITION_SIZE_USD=1000.0
DAILY_LOSS_LIMIT_PCT=5.0
```

### 2. Obter API Keys da Bybit

1. Acesse: https://www.bybit.com/app/user/api-management
2. Clique em "Create New Key"
3. **Permissões**:
   - ✅ Read Position
   - ✅ Trade
   - ❌ Withdraw (NUNCA HABILITE!)
4. **IP Whitelist** (recomendado para live):
   - Adicione seu IP fixo
5. Copie API Key e Secret para `.env`

### 3. Criar Bot do Telegram

1. Abra Telegram e fale com **@BotFather**
2. Envie `/newbot`
3. Escolha um nome: `BTC Scalper Bot`
4. Escolha um username: `btc_scalper_12345_bot`
5. Copie o **token** para `.env`

**Obter seu Chat ID**:
1. Fale com **@userinfobot** no Telegram
2. Copie seu **ID** para `.env` (campo `TELEGRAM_CHAT_ID`)

---

## 🎮 Como Usar

### Modo 1: Paper Trading (RECOMENDADO PRIMEIRO!)

Paper trading usa **preços reais do mainnet** mas simula os trades (sem dinheiro real).

```bash
# Configurar .env
TRADING_MODE=paper
PAPER_INITIAL_BALANCE=10000.0

# Executar bot
python main.py
```

**Output esperado**:
```
================================================================================
🚀 BTC SCALPER BOT STARTING
================================================================================
✅ Connected to Bybit - Server time: 2025-11-19 00:15:32
📄 PAPER TRADING MODE - Initial balance: $10,000.00
✅ Loaded config from: config_ultra_optimized_FINAL.yaml
✅ Loaded ML model from: storage/models/ultra_scalper_btcusdt_365d.pkl
🤖 Trading Bot initialized - BTCUSDT 15m - Mode: paper
✅ Telegram bot initialized
🚀 Starting trading loop...
🚀 Telegram bot starting (async)...
```

**Deixe rodar por 1-2 semanas** para validar performance antes de usar dinheiro real!

### Modo 2: Live Trading (CUIDADO! DINHEIRO REAL!)

⚠️ **APENAS após validar em paper trading!**

```bash
# Configurar .env
TRADING_MODE=live

# Executar bot
python main.py
```

**Checklist antes de live:**
- [ ] Paper trading rodou por 1-2 semanas
- [ ] ROI em paper ≥ +80%
- [ ] Win rate em paper ≥ 48%
- [ ] IP whitelist configurado na Bybit
- [ ] Testou comandos do Telegram
- [ ] Definiu `MAX_POSITION_SIZE_USD` conservador
- [ ] Configurou `DAILY_LOSS_LIMIT_PCT`

---

## 💬 Comandos Telegram

### Controle do Bot

| Comando | Descrição |
|---------|-----------|
| `/start` | Mostra painel de controle |
| `/startbot` | Inicia o trading bot |
| `/stopbot` | Para o bot (não fecha positions) |
| `/help` | Lista todos os comandos |

### Monitoramento

| Comando | Descrição |
|---------|-----------|
| `/status` | Status do bot e posições |
| `/balance` | Saldo da conta |
| `/position` | Posição aberta atual |
| `/stats` | Estatísticas de trading |

### Segurança

| Comando | Descrição |
|---------|-----------|
| `/closeposition` | Fecha posição aberta manualmente |
| `/emergency` | **STOP TUDO** (para bot + fecha positions) |

### Exemplo de uso:

1. Inicie o bot via Python: `python main.py`
2. Abra o Telegram e fale com seu bot
3. Envie `/start` - você verá botões de controle
4. Clique em "▶️ Start Bot" ou envie `/startbot`
5. Bot começará a monitorar o mercado
6. Receba notificações quando trades forem abertos/fechados

### Notificações Automáticas:

O bot envia notificações para:
- ✅ Trade aberto (com detalhes de entry, SL, TP)
- ✅ Trade fechado (com PnL e motivo)
- ✅ Erros críticos
- ✅ Startup/shutdown

---

## 🔧 Modos de Operação

### 1. Paper Trading

**Características**:
- Usa preços reais do mainnet Bybit
- Simula execução de trades
- Simula slippage (0.05%)
- Simula fees (0.055%)
- Não gasta dinheiro real

**Quando usar**:
- Validar estratégia
- Testar configurações
- Aprender a usar o bot
- Verificar se backtest se confirma em forward testing

**Duração recomendada**: 1-2 semanas (mínimo 100 trades)

### 2. Live Trading

**Características**:
- Trades reais na Bybit
- Dinheiro real em risco
- Fees reais (0.055% taker)
- Slippage real

**Quando usar**:
- Após validação em paper trading
- Com capital que pode perder
- Com stops e limits configurados

**Recomendação de capital inicial**: $500 - $2,000

---

## 📊 Monitoramento

### 1. Logs em Tempo Real

```bash
# Linux/Mac
tail -f logs/trading_bot.log

# Windows PowerShell
Get-Content logs/trading_bot.log -Wait -Tail 50
```

### 2. Histórico de Trades (CSV)

Arquivo: `logs/trades.csv`

Colunas:
- `symbol`: Par (BTCUSDT)
- `side`: long ou short
- `entry_price`: Preço de entrada
- `exit_price`: Preço de saída
- `size_usd`: Tamanho da posição em USD
- `pnl`: Lucro/prejuízo em USD
- `pnl_pct`: Lucro/prejuízo em %
- `reason`: Motivo do fechamento (stop_loss, take_profit_1/2/3, manual)
- `opened_at`: Timestamp de abertura
- `closed_at`: Timestamp de fechamento
- `duration`: Duração do trade

### 3. Telegram

Use `/stats` para ver estatísticas em tempo real:
- Total de trades
- Win rate
- ROI
- Profit factor
- Avg win/loss

### 4. Métricas Importantes

**Métricas boas**:
- ✅ Win Rate ≥ 48%
- ✅ ROI mensal ≥ +25%
- ✅ Profit Factor ≥ 1.5
- ✅ Max DD ≤ -8%

**Sinais de alerta**:
- ⚠️ Win Rate < 45%
- ⚠️ 5+ perdas consecutivas
- ⚠️ DD > -10%
- ⚠️ ROI negativo por 1 semana

**Quando parar**:
- 🛑 Win Rate < 40% por 1 semana
- 🛑 DD > -15%
- 🛑 ROI < -10%
- 🛑 Modelo claramente não funciona

---

## 🔍 Troubleshooting

### Erro: "No module named 'pandas'"

```bash
pip install -r requirements.txt
```

### Erro: "BYBIT_API_KEY must be set"

Você não criou o arquivo `.env`. Copie de `.env.example`:

```bash
cp .env.example .env
# Edite .env com suas credenciais
```

### Erro: "API error: Invalid signature"

API Secret está incorreto. Verifique:
1. Copiou o secret corretamente (sem espaços)
2. API key não foi deletada na Bybit
3. Timestamp do seu PC está correto

### Erro: "Unauthorized access denied" (Telegram)

Seu `TELEGRAM_CHAT_ID` está incorreto. Pegue de novo com @userinfobot.

### Bot não abre trades

Possíveis causas:
1. **Nenhum sinal válido**: ML model não vê oportunidades
2. **Confidence baixa**: Todos os sinais < threshold
3. **Regime bloqueado**: Mercado em regime não permitido
4. **Position já aberta**: Bot só abre 1 position por vez

Verifique logs:
```bash
tail -f logs/trading_bot.log | grep "ML Signal"
```

### Bot abre muitos trades ruins

1. Reduza `MAX_POSITION_SIZE_USD` (menor risco)
2. Aumente confidence thresholds no YAML
3. Bloqueie mais regimes (edite config YAML)
4. Reduza leverage (se aplicável)

### Drawdown muito alto

1. **PARE O BOT** imediatamente (`/emergency`)
2. Analise trades em `logs/trades.csv`
3. Reduza position sizing em 50%
4. Considere voltar para paper trading

---

## ❓ FAQ

### 1. O bot funciona 24/7?

Sim, o bot roda continuamente checando o mercado a cada 60 segundos (configurável em `CHECK_INTERVAL_SECONDS`).

### 2. Preciso deixar o computador ligado?

Sim. Ou use um VPS (servidor na nuvem) para rodar 24/7.

**Opções de VPS**:
- AWS EC2 (free tier por 1 ano)
- DigitalOcean ($5/mês)
- Vultr ($5/mês)
- Contabo (~$5/mês)

### 3. Quanto capital preciso?

**Paper trading**: $0 (virtual)

**Live trading**: Mínimo $500, recomendado $1,000-$2,000

Com $1,000:
- Position size: $100-$200 por trade
- Risk por trade: 1-2%
- ~5-10% do capital em risco por vez

### 4. Qual é o risco?

**Worst case scenario** (baseado em backtest):
- Max DD: -4.2% em 1 dia
- Potencial perda: ~5-10% em semana ruim

**Mitigação**:
- Use `DAILY_LOSS_LIMIT_PCT=5.0`
- Use `CIRCUIT_BREAKER_ENABLED=true`
- Monitore diariamente via Telegram
- Comece com capital pequeno

### 5. Posso usar em outras moedas além de BTC?

O modelo foi treinado especificamente para BTCUSDT 15min. Para outras moedas:
1. Precisaria treinar novo modelo
2. Ajustar config YAML
3. Validar em backtest
4. Validar em paper trading

### 6. Posso usar outros timeframes?

Modelo treinado para 15min. Outros timeframes (5min, 1h) precisariam:
1. Novo treinamento
2. Novos thresholds
3. Nova validação

### 7. O que fazer se Bybit cair?

O bot tentará reconectar automaticamente (3 tentativas com backoff exponencial).

Se Bybit ficar offline por muito tempo:
1. Posições abertas ficam na exchange
2. Bot não consegue monitorar
3. Use app mobile da Bybit para fechar manualmente se necessário

### 8. Posso rodar múltiplos bots?

Sim, mas cada um precisa:
- Conta Bybit separada (ou subaccount)
- API keys separadas
- Telegram bot separado
- Arquivo `.env` separado

### 9. Bot pode perder todo meu dinheiro?

**Teoricamente sim** (risco sempre existe), mas mitigações:
- Stop loss em TODOS os trades
- Daily loss limit
- Circuit breaker
- Max position size
- Sem leverage extremo

**Nunca invista dinheiro que não pode perder!**

### 10. Resultados de backtest garantem lucro futuro?

❌ **NÃO!** Backtest mostra performance passada.

Mercado pode mudar:
- Volatilidade diferente
- Tendências diferentes
- Eventos inesperados (news, dumps)

**Sempre valide em paper trading primeiro!**

---

## 🎯 Roadmap de Uso Recomendado

### Semana 1-2: Paper Trading + Aprendizado
- [ ] Instalar bot
- [ ] Configurar .env
- [ ] Rodar em paper trading
- [ ] Testar comandos Telegram
- [ ] Acompanhar trades diariamente
- [ ] Ajustar configs se necessário

### Semana 3: Análise de Resultados
- [ ] Verificar win rate (meta: ≥48%)
- [ ] Verificar ROI (meta: ≥+20% em 2 semanas)
- [ ] Analisar drawdowns
- [ ] Comparar com backtest
- [ ] Decidir: seguir para live ou não?

### Semana 4+: Live Trading (se aprovado)
- [ ] Começar com capital pequeno ($500-1000)
- [ ] Position size conservador ($100-200)
- [ ] Monitorar DIARIAMENTE
- [ ] Ajustar configs baseado em performance
- [ ] Aumentar capital gradualmente (se funcionar)

### Manutenção Contínua
- [ ] Revisar trades semanalmente
- [ ] Atualizar model se necessário (a cada 3-6 meses)
- [ ] Ajustar thresholds baseado em market conditions
- [ ] Backup de logs e trades mensalmente

---

## 🚨 Avisos Importantes

1. **Trading é arriscado**: Você pode perder dinheiro
2. **Backtest ≠ Futuro**: Performance passada não garante resultados futuros
3. **Sempre use paper trading primeiro**: Mínimo 1-2 semanas
4. **Monitore regularmente**: Não deixe bot sem supervisão por semanas
5. **Começe pequeno**: Aumente capital gradualmente conforme ganha confiança
6. **Stop loss existe por motivo**: Nunca desabilite
7. **ML models degradam**: Re-treine periodicamente (a cada 3-6 meses)
8. **News events**: Bot não sabe de news, pode perder em eventos extremos
9. **Bugs acontecem**: Sempre tenha plano B (acesso ao Bybit app)
10. **Nunca invista dinheiro que não pode perder!**

---

## 📞 Suporte

**Issues**: Use o Telegram para notificações e controle

**Logs**: Sempre check `logs/trading_bot.log` primeiro

**Trades history**: Analise `logs/trades.csv` para entender performance

---

## ✅ Checklist Final

Antes de colocar bot em produção:

- [ ] Instalou todas as dependências
- [ ] Criou e configurou `.env` corretamente
- [ ] Testou conexão com Bybit API
- [ ] Configurou Telegram bot
- [ ] Rodou em paper trading por 1-2 semanas
- [ ] Win rate ≥ 48% em paper trading
- [ ] ROI positivo em paper trading
- [ ] Entendeu todos os comandos do Telegram
- [ ] Configurou daily loss limits
- [ ] Testou comando `/emergency`
- [ ] Backupou arquivos de config
- [ ] Leu este guia completamente

**Se todos checkboxes OK → PRONTO PARA LIVE!** 🚀

---

**Boa sorte e bons trades!** 📈💰

*Criado por: Claude Code*
*Versão: 1.0*
*Data: 2025-11-19*
