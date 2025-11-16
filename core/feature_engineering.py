"""
Módulo para criação de features (indicadores técnicos) a partir de dados OHLCV.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from loguru import logger


class FeatureEngineer:
    """
    Classe para criar features de trading a partir de dados OHLCV.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa o feature engineer.

        Args:
            config: Dicionário de configuração
        """
        self.config = config
        self.features_config = config['features']

    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria todas as features configuradas.

        Args:
            df: DataFrame com dados OHLCV (timestamp, open, high, low, close, volume)

        Returns:
            DataFrame com features adicionadas
        """
        logger.info("Criando features técnicas...")

        # Criar cópia para não modificar original
        df = df.copy()

        # 1. Features de preço normalizado
        df = self._add_price_features(df)

        # 2. Médias móveis
        df = self._add_moving_averages(df)

        # 3. RSI
        df = self._add_rsi(df)

        # 4. Stochastic
        df = self._add_stochastic(df)

        # 5. MACD
        df = self._add_macd(df)

        # 6. Bollinger Bands
        df = self._add_bollinger_bands(df)

        # 7. ATR
        df = self._add_atr(df)

        # 8. Volume features
        df = self._add_volume_features(df)

        # 9. Price action patterns
        df = self._add_price_patterns(df)

        # 10. Momentum features
        df = self._add_momentum_features(df)

        # Contar features criadas
        feature_cols = [col for col in df.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        logger.info(f"✓ {len(feature_cols)} features criadas")

        return df

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adiciona features básicas de preço.

        IMPORTANTE: Evita look-ahead bias usando apenas dados passados.
        """
        # Log returns (retorno logarítmico)
        df['returns'] = np.log(df['close'] / df['close'].shift(1))

        # Retorno percentual
        df['pct_change'] = df['close'].pct_change()

        # Range (high - low) normalizado pelo close
        df['range'] = (df['high'] - df['low']) / df['close']

        # Body (close - open) normalizado
        df['body'] = (df['close'] - df['open']) / df['close']

        # Upper shadow
        df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']

        # Lower shadow
        df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']

        return df

    def _add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adiciona médias móveis exponenciais (EMA).
        """
        periods = self.features_config.get('ema_periods', [9, 21, 50, 200])

        for period in periods:
            # EMA
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

            # Distância do preço à EMA (normalizada)
            df[f'dist_ema_{period}'] = (df['close'] - df[f'ema_{period}']) / df['close']

        # Cruzamentos de EMAs
        if 9 in periods and 21 in periods:
            df['ema_cross_9_21'] = np.where(df['ema_9'] > df['ema_21'], 1, -1)

        if 21 in periods and 50 in periods:
            df['ema_cross_21_50'] = np.where(df['ema_21'] > df['ema_50'], 1, -1)

        return df

    def _add_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adiciona Relative Strength Index (RSI).
        """
        period = self.features_config.get('rsi_period', 14)

        # Calcular mudanças de preço
        delta = df['close'].diff()

        # Separar ganhos e perdas
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        # Calcular RS e RSI
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # RSI normalizado (0 a 1)
        df['rsi_norm'] = df['rsi'] / 100

        # Zonas de sobrecompra/sobrevenda
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        df['rsi_oversold'] = (df['rsi'] < 30).astype(int)

        return df

    def _add_stochastic(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adiciona Stochastic Oscillator.
        """
        period = self.features_config.get('stoch_period', 14)
        smooth = self.features_config.get('stoch_smooth', 3)

        # Calcular %K
        low_min = df['low'].rolling(window=period).min()
        high_max = df['high'].rolling(window=period).max()

        df['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min)

        # Calcular %D (média móvel de %K)
        df['stoch_d'] = df['stoch_k'].rolling(window=smooth).mean()

        # Stochastic normalizado
        df['stoch_k_norm'] = df['stoch_k'] / 100
        df['stoch_d_norm'] = df['stoch_d'] / 100

        return df

    def _add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adiciona MACD (Moving Average Convergence Divergence).
        """
        fast = self.features_config.get('macd_fast', 12)
        slow = self.features_config.get('macd_slow', 26)
        signal = self.features_config.get('macd_signal', 9)

        # Calcular EMAs
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()

        # MACD line
        df['macd'] = ema_fast - ema_slow

        # Signal line
        df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()

        # Histogram
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # MACD normalizado pelo preço
        df['macd_norm'] = df['macd'] / df['close']
        df['macd_hist_norm'] = df['macd_hist'] / df['close']

        return df

    def _add_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adiciona Bollinger Bands.
        """
        period = self.features_config.get('bb_period', 20)
        std_dev = self.features_config.get('bb_std', 2)

        # Média móvel
        sma = df['close'].rolling(window=period).mean()

        # Desvio padrão
        std = df['close'].rolling(window=period).std()

        # Bandas
        df['bb_upper'] = sma + (std * std_dev)
        df['bb_middle'] = sma
        df['bb_lower'] = sma - (std * std_dev)

        # Posição do preço nas bandas (0 = lower, 1 = upper)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # Largura das bandas (volatilidade)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

        return df

    def _add_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adiciona Average True Range (ATR) - medida de volatilidade.
        """
        period = self.features_config.get('atr_period', 14)

        # True Range
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        # ATR (média móvel do True Range)
        df['atr'] = true_range.rolling(window=period).mean()

        # ATR normalizado pelo preço
        df['atr_pct'] = df['atr'] / df['close']

        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adiciona features de volume.
        """
        period = self.features_config.get('volume_ma_period', 20)

        # Volume médio
        df['volume_ma'] = df['volume'].rolling(window=period).mean()

        # Razão volume atual / volume médio
        df['volume_ratio'] = df['volume'] / df['volume_ma']

        # Volume spike (acima de 2x a média)
        df['volume_spike'] = (df['volume_ratio'] > 2).astype(int)

        # Volume weighted price
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).rolling(window=period).sum() / \
                     df['volume'].rolling(window=period).sum()

        # Distância do VWAP
        df['dist_vwap'] = (df['close'] - df['vwap']) / df['close']

        return df

    def _add_price_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adiciona padrões de price action.
        """
        # Candles verdes/vermelhos
        df['bullish_candle'] = (df['close'] > df['open']).astype(int)
        df['bearish_candle'] = (df['close'] < df['open']).astype(int)

        # Tamanho do corpo em relação ao range
        df['body_range_ratio'] = np.abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-10)

        # Doji (corpo pequeno)
        df['doji'] = (df['body_range_ratio'] < 0.1).astype(int)

        # Higher highs / Lower lows (sequência)
        df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
        df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)

        # Engulfing patterns (simplificado)
        df['bullish_engulfing'] = (
            (df['close'] > df['open']) &
            (df['close'].shift(1) < df['open'].shift(1)) &
            (df['close'] > df['open'].shift(1)) &
            (df['open'] < df['close'].shift(1))
        ).astype(int)

        df['bearish_engulfing'] = (
            (df['close'] < df['open']) &
            (df['close'].shift(1) > df['open'].shift(1)) &
            (df['close'] < df['open'].shift(1)) &
            (df['open'] > df['close'].shift(1))
        ).astype(int)

        return df

    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adiciona features de momentum.
        """
        # Rate of change (diferentes períodos)
        for period in [3, 5, 10, 20]:
            df[f'roc_{period}'] = df['close'].pct_change(period)

        # Aceleração do preço
        df['price_acceleration'] = df['returns'].diff()

        # Tendência (regressão linear simplificada)
        for window in [10, 20, 50]:
            df[f'trend_{window}'] = df['close'].rolling(window=window).apply(
                lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) == window else np.nan,
                raw=True
            )

        return df

    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """
        Retorna lista de nomes de features (excluindo colunas OHLCV originais).

        Args:
            df: DataFrame com features

        Returns:
            Lista de nomes de features
        """
        exclude = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        features = [col for col in df.columns if col not in exclude]

        return features


if __name__ == "__main__":
    # Teste do módulo
    from core.utils import load_config, setup_logging
    from core.data_loader import BybitDataLoader

    config = load_config()
    setup_logging(config)

    # Carregar dados de teste
    loader = BybitDataLoader(config)
    df = loader.download_ohlcv('BTCUSDT', '5m', lookback_days=7)

    # Criar features
    engineer = FeatureEngineer(config)
    df_features = engineer.create_all_features(df)

    print(f"Shape: {df_features.shape}")
    print(f"\nColunas: {df_features.columns.tolist()}")
    print(f"\nPrimeiras linhas:\n{df_features.head()}")

    # Verificar NaNs
    print(f"\nNaNs por coluna:\n{df_features.isna().sum()}")
