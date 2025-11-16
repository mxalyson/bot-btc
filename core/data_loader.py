"""
Módulo para download e carregamento de dados históricos da Bybit.
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
from loguru import logger
import time

from core.utils import ensure_dir, get_data_path


class BybitDataLoader:
    """
    Classe para download de dados históricos da Bybit usando CCXT.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa o data loader.

        Args:
            config: Dicionário de configuração
        """
        self.config = config
        self.data_config = config['data']
        self.bybit_config = config.get('bybit', {})

        # Inicializar exchange
        self.exchange = self._initialize_exchange()

        # Mapeamento de timeframes para milliseconds
        self.timeframe_to_ms = {
            '1m': 60 * 1000,
            '3m': 3 * 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '30m': 30 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000,
        }

    def _initialize_exchange(self) -> ccxt.Exchange:
        """
        Inicializa conexão com a exchange Bybit.

        Returns:
            Objeto exchange do CCXT
        """
        api_key = self.bybit_config.get('api_key', '')
        api_secret = self.bybit_config.get('api_secret', '')
        testnet = self.bybit_config.get('testnet', False)

        exchange_config = {
            'enableRateLimit': True,
            'options': {
                'defaultType': 'linear',  # futuros perpétuos
            }
        }

        if api_key and api_secret:
            exchange_config['apiKey'] = api_key
            exchange_config['secret'] = api_secret

        if testnet:
            exchange_config['urls'] = {
                'api': {
                    'public': 'https://api-testnet.bybit.com',
                    'private': 'https://api-testnet.bybit.com',
                }
            }

        exchange = ccxt.bybit(exchange_config)
        logger.info(f"Exchange Bybit inicializada (testnet: {testnet})")

        return exchange

    def download_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        lookback_days: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Baixa dados OHLCV da Bybit.

        Args:
            symbol: Símbolo do ativo (ex: BTC/USDT:USDT para perpétuos)
            timeframe: Timeframe (1m, 5m, 15m, etc)
            lookback_days: Número de dias para buscar (alternativa a start_date)
            start_date: Data inicial (formato: YYYY-MM-DD)
            end_date: Data final (formato: YYYY-MM-DD)

        Returns:
            DataFrame com colunas: timestamp, open, high, low, close, volume
        """
        logger.info(f"Baixando dados: {symbol} | {timeframe}")

        # Determinar período
        if lookback_days:
            end_dt = datetime.utcnow()
            start_dt = end_dt - timedelta(days=lookback_days)
        elif start_date and end_date:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        else:
            # Padrão: últimos 180 dias
            end_dt = datetime.utcnow()
            start_dt = end_dt - timedelta(days=180)

        since = int(start_dt.timestamp() * 1000)
        end_ts = int(end_dt.timestamp() * 1000)

        # Formatar símbolo para Bybit perpétuos
        # Bybit usa formato BTC/USDT:USDT para linear perpetuals
        if ':' not in symbol:
            # Converter BTCUSDT para BTC/USDT:USDT
            if 'USDT' in symbol:
                base = symbol.replace('USDT', '')
                formatted_symbol = f"{base}/USDT:USDT"
            elif 'USD' in symbol:
                base = symbol.replace('USD', '')
                formatted_symbol = f"{base}/USD:USD"
            else:
                # Fallback: assumir que já está no formato correto
                formatted_symbol = symbol
        else:
            formatted_symbol = symbol

        all_candles = []
        current_since = since

        # Loop para baixar todos os dados (CCXT tem limite por request)
        while current_since < end_ts:
            try:
                candles = self.exchange.fetch_ohlcv(
                    formatted_symbol,
                    timeframe=timeframe,
                    since=current_since,
                    limit=1000  # Máximo por request
                )

                if not candles:
                    break

                all_candles.extend(candles)

                # Atualizar timestamp para próxima iteração
                current_since = candles[-1][0] + self.timeframe_to_ms.get(timeframe, 60000)

                logger.debug(f"Baixados {len(candles)} candles até {datetime.fromtimestamp(current_since/1000)}")

                # Rate limiting
                time.sleep(self.exchange.rateLimit / 1000)

                # Break se ultrapassou end_ts
                if current_since >= end_ts:
                    break

            except Exception as e:
                logger.error(f"Erro ao baixar dados: {e}")
                break

        if not all_candles:
            logger.warning(f"Nenhum dado baixado para {symbol} {timeframe}")
            return pd.DataFrame()

        # Converter para DataFrame
        df = pd.DataFrame(
            all_candles,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )

        # Converter timestamp para datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        # Remover duplicatas (por timestamp)
        df = df.drop_duplicates(subset=['timestamp'], keep='last')

        # Ordenar por timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Verificar dados contínuos
        self._check_data_continuity(df, timeframe)

        logger.info(f"✓ {len(df)} candles baixados de {df['timestamp'].min()} até {df['timestamp'].max()}")

        return df

    def _check_data_continuity(self, df: pd.DataFrame, timeframe: str) -> None:
        """
        Verifica se há gaps nos dados.

        Args:
            df: DataFrame com dados OHLCV
            timeframe: Timeframe dos dados
        """
        if len(df) < 2:
            return

        expected_delta = pd.Timedelta(milliseconds=self.timeframe_to_ms.get(timeframe, 60000))

        # Calcular diferença entre timestamps consecutivos
        time_diffs = df['timestamp'].diff()

        # Encontrar gaps (onde diferença > esperado)
        gaps = time_diffs[time_diffs > expected_delta * 1.5]  # 1.5x tolerância

        if len(gaps) > 0:
            logger.warning(f"⚠ Encontrados {len(gaps)} gaps nos dados")
            for idx in gaps.index[:5]:  # Mostrar primeiros 5 gaps
                logger.warning(f"  Gap em {df.loc[idx, 'timestamp']}: {time_diffs.loc[idx]}")
        else:
            logger.info("✓ Dados contínuos (sem gaps)")

    def save_data(self, df: pd.DataFrame, symbol: str, timeframe: str, data_type: str = "raw") -> Path:
        """
        Salva dados em arquivo CSV.

        Args:
            df: DataFrame com dados
            symbol: Símbolo do ativo
            timeframe: Timeframe
            data_type: Tipo de dados ("raw" ou "processed")

        Returns:
            Path do arquivo salvo
        """
        # Limpar símbolo (remover :USDT se presente)
        clean_symbol = symbol.replace(':USDT', '').replace('/', '')

        file_path = get_data_path(clean_symbol, timeframe, data_type)

        df.to_csv(file_path, index=False)
        logger.info(f"✓ Dados salvos em: {file_path}")

        return file_path

    def load_data(self, symbol: str, timeframe: str, data_type: str = "raw") -> pd.DataFrame:
        """
        Carrega dados de arquivo CSV.

        Args:
            symbol: Símbolo do ativo
            timeframe: Timeframe
            data_type: Tipo de dados ("raw" ou "processed")

        Returns:
            DataFrame com dados
        """
        # Limpar símbolo
        clean_symbol = symbol.replace(':USDT', '').replace('/', '')

        file_path = get_data_path(clean_symbol, timeframe, data_type)

        if not file_path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

        df = pd.read_csv(file_path)

        # Converter timestamp para datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        logger.info(f"✓ Dados carregados de: {file_path} ({len(df)} linhas)")

        return df

    def download_multiple_symbols(
        self,
        symbols: List[str],
        timeframes: List[str],
        lookback_days: int
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Baixa dados para múltiplos símbolos e timeframes.

        Args:
            symbols: Lista de símbolos
            timeframes: Lista de timeframes
            lookback_days: Dias de histórico

        Returns:
            Dicionário aninhado: {symbol: {timeframe: DataFrame}}
        """
        results = {}

        total = len(symbols) * len(timeframes)
        current = 0

        for symbol in symbols:
            results[symbol] = {}

            for timeframe in timeframes:
                current += 1
                logger.info(f"[{current}/{total}] Processando {symbol} {timeframe}")

                try:
                    df = self.download_ohlcv(symbol, timeframe, lookback_days=lookback_days)

                    if not df.empty:
                        # Salvar automaticamente
                        self.save_data(df, symbol, timeframe, data_type="raw")
                        results[symbol][timeframe] = df
                    else:
                        logger.warning(f"Dados vazios para {symbol} {timeframe}")

                except Exception as e:
                    logger.error(f"Erro ao processar {symbol} {timeframe}: {e}")
                    continue

        logger.info(f"✓ Download concluído: {current}/{total} combinações processadas")

        return results


if __name__ == "__main__":
    # Teste do módulo
    from core.utils import load_config, setup_logging

    config = load_config()
    setup_logging(config)

    loader = BybitDataLoader(config)

    # Teste: baixar dados de BTC
    df = loader.download_ohlcv('BTCUSDT', '5m', lookback_days=7)
    print(df.head())
    print(df.tail())
