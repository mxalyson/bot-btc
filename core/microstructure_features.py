"""
Módulo de Features de Microestrutura
Aproxima características de order flow usando dados OHLCV
Baseado em research: order book features contribuem 73% da performance!
"""

import pandas as pd
import numpy as np
from loguru import logger


class MicrostructureFeatures:
    """
    Cria features avançadas de microestrutura do mercado.

    Embora não tenhamos order book histórico completo, podemos aproximar
    características importantes usando OHLCV e derivações inteligentes.
    """

    def __init__(self, config=None):
        self.config = config or {}

    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adiciona todas as features de microestrutura ao dataframe.
        """
        logger.info("Criando features de microestrutura...")

        df = df.copy()

        # 1. Spread e Range Features
        df = self._spread_features(df)

        # 2. Order Flow Proxy (Buy/Sell Pressure)
        df = self._order_flow_features(df)

        # 3. Volume Profile e Imbalance
        df = self._volume_profile_features(df)

        # 4. Price Action Microstructure
        df = self._price_action_features(df)

        # 5. Tick Direction e Momentum
        df = self._tick_features(df)

        # 6. Volatility Regime
        df = self._volatility_regime_features(df)

        # 7. Multi-timeframe Features
        df = self._multi_timeframe_features(df)

        logger.info(f"✓ {len([c for c in df.columns if c not in ['open','high','low','close','volume','timestamp']])} features de microestrutura criadas")

        return df

    def _spread_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Features de spread e range.
        Spread real = bid-ask spread (não temos)
        Proxy: high-low range como medida de volatilidade intracandle
        """
        # Spread absoluto (high - low)
        df['spread_abs'] = df['high'] - df['low']

        # Spread relativo (%)
        df['spread_pct'] = (df['spread_abs'] / df['close']) * 100

        # Spread normalizado por ATR
        atr = df['high'].rolling(14).max() - df['low'].rolling(14).min()
        df['spread_atr_ratio'] = df['spread_abs'] / (atr + 1e-8)

        # Upper shadow (resistência)
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['upper_shadow_pct'] = df['upper_shadow'] / df['spread_abs']

        # Lower shadow (suporte)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        df['lower_shadow_pct'] = df['lower_shadow'] / df['spread_abs']

        # Body (força da vela)
        df['body_abs'] = abs(df['close'] - df['open'])
        df['body_pct'] = df['body_abs'] / df['spread_abs']

        return df

    def _order_flow_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Proxy de order flow: buy pressure vs sell pressure.

        Lógica:
        - Se close > open: buy pressure
        - Se close < open: sell pressure
        - Volume ponderado pela direção
        """
        # Direção da vela
        df['candle_direction'] = np.where(df['close'] > df['open'], 1, -1)

        # Buy/Sell volume (aproximação)
        # Assumimos que volume é distribuído proporcionalmente ao movimento
        close_position = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
        df['buy_volume'] = df['volume'] * close_position
        df['sell_volume'] = df['volume'] * (1 - close_position)

        # Order flow balance
        df['order_flow'] = df['buy_volume'] - df['sell_volume']
        df['order_flow_ratio'] = df['buy_volume'] / (df['sell_volume'] + 1e-8)

        # Order flow momentum (cumulative)
        df['order_flow_cum_5'] = df['order_flow'].rolling(5).sum()
        df['order_flow_cum_10'] = df['order_flow'].rolling(10).sum()
        df['order_flow_cum_20'] = df['order_flow'].rolling(20).sum()

        # Buy/Sell pressure strength
        df['buy_pressure'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
        df['sell_pressure'] = (df['open'] - df['close']) / (df['high'] - df['low'] + 1e-8)

        return df

    def _volume_profile_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Features de perfil de volume.
        """
        # Volume relativo
        df['volume_ma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_ma_20'] + 1e-8)

        # Volume weighted price
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).rolling(20).sum() / \
                     (df['volume'].rolling(20).sum() + 1e-8)

        # Distance from VWAP
        df['price_vwap_dist'] = (df['close'] - df['vwap']) / df['vwap']

        # Volume surge detection
        df['volume_surge'] = (df['volume'] > df['volume_ma_20'] * 2).astype(int)

        # Volume trend
        df['volume_trend_5'] = df['volume'].rolling(5).mean() / df['volume_ma_20']
        df['volume_trend_10'] = df['volume'].rolling(10).mean() / df['volume_ma_20']

        # Delta volume (change)
        df['volume_delta'] = df['volume'].diff()
        df['volume_delta_pct'] = df['volume'].pct_change()

        return df

    def _price_action_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Features de price action micro.
        """
        # High/Low quebras
        df['new_high_5'] = (df['high'] == df['high'].rolling(5).max()).astype(int)
        df['new_low_5'] = (df['low'] == df['low'].rolling(5).min()).astype(int)
        df['new_high_10'] = (df['high'] == df['high'].rolling(10).max()).astype(int)
        df['new_low_10'] = (df['low'] == df['low'].rolling(10).min()).astype(int)

        # Price position in range
        range_20_high = df['high'].rolling(20).max()
        range_20_low = df['low'].rolling(20).min()
        df['price_position_20'] = (df['close'] - range_20_low) / (range_20_high - range_20_low + 1e-8)

        # Consecutive candles same direction
        df['consec_up'] = (df['close'] > df['open']).astype(int).rolling(5).sum()
        df['consec_down'] = (df['close'] < df['open']).astype(int).rolling(5).sum()

        # Gap detection
        df['gap_up'] = np.where(df['low'] > df['high'].shift(1), 1, 0)
        df['gap_down'] = np.where(df['high'] < df['low'].shift(1), 1, 0)

        return df

    def _tick_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tick direction e momentum features.
        """
        # Tick direction (simplified)
        df['tick'] = np.sign(df['close'].diff())

        # Tick momentum
        df['tick_momentum_5'] = df['tick'].rolling(5).sum()
        df['tick_momentum_10'] = df['tick'].rolling(10).sum()
        df['tick_momentum_20'] = df['tick'].rolling(20).sum()

        # Price momentum (%)
        df['price_momentum_3'] = df['close'].pct_change(3)
        df['price_momentum_5'] = df['close'].pct_change(5)
        df['price_momentum_10'] = df['close'].pct_change(10)

        # Acceleration
        df['price_accel_5'] = df['price_momentum_5'] - df['price_momentum_5'].shift(5)

        return df

    def _volatility_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detecção de regime de volatilidade.
        """
        # Rolling volatility (std)
        df['volatility_5'] = df['close'].pct_change().rolling(5).std()
        df['volatility_10'] = df['close'].pct_change().rolling(10).std()
        df['volatility_20'] = df['close'].pct_change().rolling(20).std()

        # Volatility ratio
        df['vol_ratio_5_20'] = df['volatility_5'] / (df['volatility_20'] + 1e-8)
        df['vol_ratio_10_20'] = df['volatility_10'] / (df['volatility_20'] + 1e-8)

        # High volatility regime
        vol_ma = df['volatility_20'].rolling(50).mean()
        df['high_vol_regime'] = (df['volatility_20'] > vol_ma * 1.5).astype(int)

        # Parkinson volatility (uses high-low)
        df['parkinson_vol'] = np.sqrt(
            (1 / (4 * np.log(2))) *
            np.log(df['high'] / df['low'])**2
        ).rolling(20).mean()

        return df

    def _multi_timeframe_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Features de múltiplos timeframes.
        Simula ter dados de 5m agregados em 15m.
        """
        # Trend em diferentes janelas (proxy para timeframes maiores)
        for window in [4, 8, 12, 20]:  # 1h, 2h, 3h, 5h em 15m
            # Price trend
            df[f'trend_{window}'] = (
                df['close'].rolling(window).mean() -
                df['close'].rolling(window).mean().shift(window)
            ) / df['close']

            # Volume trend
            df[f'volume_trend_{window}'] = (
                df['volume'].rolling(window).mean() /
                (df['volume'].rolling(window*2).mean() + 1e-8)
            )

        # Alignment (todos trends na mesma direção?)
        df['trend_alignment'] = (
            np.sign(df['trend_4']) +
            np.sign(df['trend_8']) +
            np.sign(df['trend_12'])
        ) / 3

        return df


def add_microstructure_features(df: pd.DataFrame, config=None) -> pd.DataFrame:
    """
    Função helper para adicionar features de microestrutura.
    """
    engineer = MicrostructureFeatures(config)
    return engineer.create_all_features(df)
