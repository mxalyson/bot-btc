# Bybit Scalping ML Pipeline

Pipeline completo de Machine Learning para gerar sinais de scalping em contratos perpétuos de futuros na Bybit.

## Características

- Download automatizado de dados históricos da Bybit via CCXT
- Feature engineering com +50 indicadores técnicos
- Labeling inteligente usando Triple Barrier Method
- Modelos de ML (XGBoost/LightGBM) otimizados para scalping
- Pipeline completo: dados → features → labels → treinamento → modelo
- Configuração flexível via YAML
- Logging detalhado em todas as etapas

## Estrutura do Projeto

```
bot-btc/
├── config/
│   └── config.yaml              # Configurações do pipeline
├── core/
│   ├── __init__.py
│   ├── data_loader.py           # Download de dados da Bybit
│   ├── feature_engineering.py   # Criação de features
│   ├── labeling.py              # Geração de labels (Triple Barrier)
│   ├── model_trainer.py         # Treinamento de modelos
│   └── utils.py                 # Funções utilitárias
├── data/
│   ├── raw/                     # Dados brutos (OHLCV)
│   └── processed/               # Dados processados
├── models/                      # Modelos treinados (.pkl + .yaml)
├── notebooks/                   # Jupyter notebooks (análise)
├── scripts/
│   ├── download_data.py         # CLI para download
│   └── train_model.py           # CLI para treinamento
├── logs/                        # Arquivos de log
├── requirements.txt             # Dependências Python
└── README.md                    # Este arquivo
```

## Instalação

### 1. Clonar o repositório

```bash
git clone <seu-repositorio>
cd bot-btc
```

### 2. Criar ambiente virtual

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

### 3. Instalar dependências

```bash
pip install -r requirements.txt
```

## Configuração

### Arquivo de Configuração

Edite `config/config.yaml` para ajustar:

- **Símbolos**: Quais ativos treinar (ex: BTCUSDT, ETHUSDT)
- **Timeframes**: Intervalos de tempo (1m, 5m, 15m)
- **Período de dados**: Quantos dias de histórico buscar
- **Features**: Indicadores técnicos e parâmetros
- **Labeling**: Método e thresholds para criar labels
- **Modelo**: XGBoost ou LightGBM com hiperparâmetros
- **Treinamento**: Normalização, divisão train/val/test

### API da Bybit (Opcional)

Para dados públicos (OHLCV), não é necessário API key. Mas se quiser configurar:

```yaml
bybit:
  api_key: "sua_api_key"
  api_secret: "seu_secret"
  testnet: false
```

## Uso

### 1. Download de Dados

#### Baixar dados de um símbolo específico

```bash
python scripts/download_data.py --symbol BTCUSDT --timeframe 5m --days 180
```

#### Baixar múltiplos timeframes

```bash
python scripts/download_data.py --symbol ETHUSDT --timeframe 1m 5m 15m --days 90
```

#### Baixar todos os símbolos configurados

```bash
python scripts/download_data.py --all
```

### 2. Treinamento de Modelo

#### Treinar modelo para BTC 5m

```bash
python scripts/train_model.py --symbol BTCUSDT --timeframe 5m
```

#### Treinar com nome customizado

```bash
python scripts/train_model.py --symbol ETHUSDT --timeframe 1m --model-name eth_1m_model
```

#### Forçar novo download de dados

```bash
python scripts/train_model.py --symbol BTCUSDT --timeframe 5m --download
```

## Pipeline Detalhado

### Etapa 1: Download de Dados

O módulo `data_loader.py` usa CCXT para:

- Conectar na Bybit (futuros perpétuos)
- Baixar dados OHLCV (Open, High, Low, Close, Volume)
- Remover duplicatas
- Verificar continuidade (detectar gaps)
- Salvar em CSV em `data/raw/`

### Etapa 2: Feature Engineering

O módulo `feature_engineering.py` cria +50 features:

#### Preço
- Log returns
- Percent change
- Range, body, shadows dos candles

#### Médias Móveis
- EMAs (9, 21, 50, 200)
- Distância do preço às EMAs
- Cruzamentos (9x21, 21x50)

#### Indicadores Técnicos
- **RSI**: Relative Strength Index
- **Stochastic**: %K e %D
- **MACD**: MACD line, signal, histogram
- **Bollinger Bands**: Upper, middle, lower, position, width
- **ATR**: Average True Range (volatilidade)

#### Volume
- Volume médio
- Razão volume/média
- Spikes de volume
- VWAP (Volume Weighted Average Price)

#### Price Action
- Candles bullish/bearish
- Dojis
- Engulfing patterns
- Higher highs / Lower lows

#### Momentum
- Rate of Change (múltiplos períodos)
- Aceleração de preço
- Tendência (regressão linear)

**IMPORTANTE**: Todas as features são calculadas SEM look-ahead bias (apenas dados passados).

### Etapa 3: Labeling

O módulo `labeling.py` implementa o **Triple Barrier Method**:

Para cada candle:

1. Define barreiras baseadas em ATR:
   - **Upper barrier**: preço + (1.5 × ATR) → take profit
   - **Lower barrier**: preço - (1.0 × ATR) → stop loss

2. Observa os próximos N candles (ex: 10)

3. Determina label:
   - **LONG**: Se atingir upper barrier primeiro
   - **SHORT**: Se atingir lower barrier primeiro
   - **NONE**: Se não atingir barreiras ou movimento insuficiente

4. Calcula retorno esperado para cada trade

Este método é superior ao fixed horizon porque:
- Considera risco (stop loss)
- Adapta-se à volatilidade (ATR)
- Foca em oportunidades reais de scalping

### Etapa 4: Divisão Temporal

Divide o dataset SEM embaralhamento (séries temporais):

- **Train**: 70% (primeiros dados)
- **Validation**: 15% (meio)
- **Test**: 15% (dados mais recentes)

### Etapa 5: Treinamento

O módulo `model_trainer.py`:

1. **Preparação**:
   - Normalização de features (StandardScaler/MinMaxScaler)
   - Remoção de NaNs e infinitos
   - Codificação de labels (LONG=0, SHORT=1, NONE=2)

2. **Treinamento**:
   - XGBoost ou LightGBM (configurável)
   - Early stopping para evitar overfitting
   - Balanceamento de classes (opcional)

3. **Avaliação**:
   - Accuracy, Precision, Recall, F1-score
   - Confusion matrix
   - ROC AUC (multiclass)
   - Feature importance

4. **Salvamento**:
   - Modelo em `.pkl` (modelo + scaler + metadados)
   - Metadados em `.yaml` (config, features, data)

## Métricas de Avaliação

### Métricas Principais

- **Accuracy**: Acurácia geral do modelo
- **Precision**: Quando o modelo prevê LONG/SHORT, quão frequentemente está correto
- **Recall**: Quantas oportunidades reais de LONG/SHORT o modelo captura
- **F1-score**: Média harmônica de precision e recall

### Interpretação para Trading

- **Precision alta**: Poucos sinais falsos (menos perdas)
- **Recall alto**: Captura mais oportunidades (mais trades)
- **F1 balanceado**: Bom trade-off entre segurança e oportunidades

Para scalping, **precision** é geralmente mais importante que **recall** (melhor ter poucos sinais bons do que muitos sinais ruins).

## Exemplo Completo

### 1. Download de dados (BTC e ETH, últimos 180 dias)

```bash
# Editar config/config.yaml:
# symbols: [BTCUSDT, ETHUSDT]
# timeframes: [1m, 5m]
# lookback_days: 180

python scripts/download_data.py --all
```

### 2. Treinar modelo para BTC 5m

```bash
python scripts/train_model.py --symbol BTCUSDT --timeframe 5m
```

### 3. Verificar resultados

```bash
# Modelo salvo em models/
ls -lh models/

# Logs em logs/
tail -f logs/training.log
```

## Usando o Modelo Treinado

Exemplo de como carregar e usar o modelo (para integração em bot):

```python
import joblib
import pandas as pd
from core.feature_engineering import FeatureEngineer
from core.utils import load_config

# Carregar modelo
model_data = joblib.load('models/btcusdt_5m_20240101_120000.pkl')
model = model_data['model']
scaler = model_data['scaler']
feature_names = model_data['feature_names']
inverse_label_mapping = model_data['inverse_label_mapping']

# Carregar configuração
config = load_config()

# Criar features de novos dados (live)
engineer = FeatureEngineer(config)
df_live = engineer.create_all_features(df_live)  # df_live = seus dados OHLCV atuais

# Preparar features
X = df_live[feature_names].iloc[-1:].values  # Última linha
X_scaled = scaler.transform(X)

# Predição
y_pred = model.predict(X_scaled)[0]
y_proba = model.predict_proba(X_scaled)[0]

signal = inverse_label_mapping[y_pred]
confidence = y_proba[y_pred]

print(f"Sinal: {signal} (confiança: {confidence:.2%})")

# Lógica de trading:
# if signal == 'LONG' and confidence > 0.7:
#     # Abrir posição LONG
# elif signal == 'SHORT' and confidence > 0.7:
#     # Abrir posição SHORT
```

## Customização

### Adicionar Novos Indicadores

Edite `core/feature_engineering.py`:

```python
def _add_my_custom_indicator(self, df: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona seu indicador customizado.
    """
    # Sua lógica aqui
    df['my_indicator'] = ...
    return df
```

E adicione no método `create_all_features()`.

### Ajustar Labeling

Edite `config/config.yaml`:

```yaml
labeling:
  method: "triple_barrier"
  forward_window: 15        # Aumentar para scalping mais longo
  profit_target_atr: 2.0    # Aumentar take profit
  stop_loss_atr: 1.5        # Ajustar stop loss
  min_move_atr: 0.5         # Movimento mínimo
```

### Tuning de Hiperparâmetros

Edite `config/config.yaml`:

```yaml
model:
  xgboost:
    max_depth: 8              # Aumentar para modelo mais complexo
    learning_rate: 0.03       # Diminuir para treino mais lento/cuidadoso
    n_estimators: 500         # Mais árvores
```

## Boas Práticas

### 1. Evitar Overfitting

- Use validation set
- Early stopping
- Não tune excessivamente com dados de teste
- Regularização (L1/L2)

### 2. Walk-Forward Analysis

Para avaliação mais robusta, implemente walk-forward:

```python
# Treinar em janelas deslizantes
for i in range(0, len(df), window_size):
    train = df[i:i+train_window]
    test = df[i+train_window:i+train_window+test_window]
    # Treinar e avaliar
```

### 3. Backtesting

Sempre faça backtest completo:

- Custos de transação (fees)
- Slippage
- Latência
- Tamanho de posição
- Gestão de risco

### 4. Dados Limpos

- Verifique gaps de dados
- Remova outliers extremos (se aplicável)
- Sincronize timestamps entre diferentes símbolos

## Troubleshooting

### Erro: "Nenhum dado baixado"

- Verifique conexão com internet
- Verifique se símbolo está correto (ex: BTCUSDT, não BTC-USDT)
- Tente reduzir `lookback_days`

### Erro: "Feature X not found"

- Execute feature engineering antes de labeling
- Verifique se ATR foi calculado (necessário para labeling)

### Modelo com accuracy baixa

- Aumente `lookback_days` (mais dados)
- Ajuste parâmetros de labeling (targets muito agressivos?)
- Tente outros indicadores
- Verifique distribuição de labels (muito desbalanceado?)

### Muitos sinais NONE

- Reduza `profit_target_atr` (targets mais fáceis)
- Aumente `forward_window` (mais tempo para atingir target)
- Reduza `min_move_atr`

## Roadmap / Próximos Passos

- [ ] Implementar ensemble de modelos
- [ ] Adicionar features de order book (microestrutura)
- [ ] Implementar walk-forward optimization
- [ ] Backtesting engine integrado
- [ ] API para inferência real-time
- [ ] Dashboard de monitoramento (Streamlit/Dash)
- [ ] Suporte para mais exchanges (Binance, OKX)
- [ ] Auto-tuning de hiperparâmetros (Optuna)

## Contribuindo

Contribuições são bem-vindas! Por favor:

1. Faça fork do projeto
2. Crie uma branch (`git checkout -b feature/nova-feature`)
3. Commit suas mudanças (`git commit -am 'Adiciona nova feature'`)
4. Push para a branch (`git push origin feature/nova-feature`)
5. Abra um Pull Request

## Avisos Legais

**AVISO**: Este projeto é apenas para fins educacionais e de pesquisa.

- Trading de futuros é EXTREMAMENTE arriscado
- Você pode perder TODO o seu capital
- Não somos responsáveis por perdas financeiras
- Sempre teste em testnet primeiro
- Use gestão de risco adequada
- Consulte um profissional financeiro

**Nunca invista mais do que você pode perder.**

## Licença

MIT License - veja LICENSE para detalhes.

## Contato

Para dúvidas, sugestões ou problemas:

- Abra uma issue no GitHub
- Email: [seu-email]

---

**Happy Trading! 🚀📈**
