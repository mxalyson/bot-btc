"""
Advanced Features para Trading de Alta Frequência
Inclui: Wavelet Transform, Market Regime Detection, Multi-Timeframe
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from loguru import logger
import pywt  # PyWavelets
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


class WaveletFeatures:
    """
    Wavelet Transform para separar sinal de ruído

    Research mostra que wavelet decomposition melhora performance
    em séries temporais financeiras ao separar componentes de
    diferentes frequências (trend vs noise).
    """

    def __init__(self, wavelet='db4', level=3):
        """
        Args:
            wavelet: Tipo de wavelet ('db4', 'sym5', 'coif5')
            level: Níveis de decomposição
        """
        self.wavelet = wavelet
        self.level = level

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica Discrete Wavelet Transform no preço close.

        Decompõe sinal em:
        - Aproximação (cA): Componente de baixa frequência (trend)
        - Detalhes (cD): Componentes de alta frequência (ruído/oscilações)
        """
        logger.info(f"Criando Wavelet features (wavelet={self.wavelet}, level={self.level})...")

        close = df['close'].values

        try:
            # Discrete Wavelet Transform
            coeffs = pywt.wavedec(close, self.wavelet, level=self.level)

            # Aproximação (trend de longo prazo)
            approx = coeffs[0]
            # Replicar para ter mesmo tamanho do dataset original
            df['wavelet_trend'] = np.interp(
                np.arange(len(close)),
                np.linspace(0, len(close)-1, len(approx)),
                approx
            )

            # Detalhes em diferentes escalas
            for i, detail in enumerate(coeffs[1:], 1):
                # cD1 = high frequency (ruído rápido)
                # cD2 = medium frequency
                # cD3 = low frequency (oscilações lentas)
                df[f'wavelet_detail_{i}'] = np.interp(
                    np.arange(len(close)),
                    np.linspace(0, len(close)-1, len(detail)),
                    detail
                )

            # Features derivadas
            # Razão trend/noise (quando > 1, trend domina)
            df['wavelet_trend_strength'] = np.abs(df['wavelet_trend']) / (
                np.abs(df['wavelet_detail_1']) + 1e-8
            )

            # Energy em cada componente
            for i in range(1, self.level + 1):
                df[f'wavelet_energy_{i}'] = df[f'wavelet_detail_{i}'] ** 2

            # Total energy
            total_energy = sum(df[f'wavelet_energy_{i}'] for i in range(1, self.level + 1))

            # Entropy (medida de incerteza/aleatoriedade)
            df['wavelet_entropy'] = -sum(
                (df[f'wavelet_energy_{i}'] / (total_energy + 1e-8)) *
                np.log(df[f'wavelet_energy_{i}'] / (total_energy + 1e-8) + 1e-8)
                for i in range(1, self.level + 1)
            )

            n_features = 1 + self.level + 1 + self.level + 1  # trend + details + strength + energies + entropy
            logger.info(f"✓ {n_features} Wavelet features criadas")

        except Exception as e:
            logger.error(f"Erro ao criar Wavelet features: {e}")
            # Features dummy em caso de erro
            df['wavelet_trend'] = 0
            for i in range(1, self.level + 1):
                df[f'wavelet_detail_{i}'] = 0

        return df


class MarketRegimeDetector:
    """
    Detecta regime de mercado usando Gaussian Mixture Model (GMM).

    Regimes típicos:
    - Trending (alta volatilidade, tendência clara)
    - Ranging (baixa volatilidade, sideways)
    - Volatile (alta volatilidade, sem tendência)

    Estratégias diferentes funcionam melhor em regimes diferentes.
    """

    def __init__(self, n_regimes=3, lookback=100):
        """
        Args:
            n_regimes: Número de regimes a detectar
            lookback: Janela para calcular features de regime
        """
        self.n_regimes = n_regimes
        self.lookback = lookback
        self.gmm = None
        self.scaler = StandardScaler()

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detecta regime de mercado e cria features one-hot.
        """
        logger.info(f"Detectando Market Regimes (n={self.n_regimes})...")

        # Features para caracterizar regime
        regime_features = []

        # 1. Volatilidade (quanto mais volátil, mais incerto)
        if 'volatility_20' in df.columns:
            regime_features.append(df['volatility_20'].values)
        else:
            vol = df['close'].pct_change().rolling(20).std() * 100
            regime_features.append(vol.values)

        # 2. Trend strength
        if 'trend_20' in df.columns:
            regime_features.append(df['trend_20'].values)
        else:
            trend = (df['close'] - df['close'].shift(20)) / (df['close'].shift(20) + 1e-8) * 100
            regime_features.append(trend.values)

        # 3. Volume
        if 'volume_ma' in df.columns:
            vol_ratio = df['volume'] / (df['volume_ma'] + 1e-8)
            regime_features.append(vol_ratio.values)
        else:
            vol_ma = df['volume'].rolling(20).mean()
            vol_ratio = df['volume'] / (vol_ma + 1e-8)
            regime_features.append(vol_ratio.values)

        # Empilhar features
        X_regime = np.column_stack(regime_features)

        # Remover NaNs
        mask = ~np.isnan(X_regime).any(axis=1)
        X_regime_clean = X_regime[mask]

        if len(X_regime_clean) < self.lookback:
            logger.warning("Dados insuficientes para detectar regimes, usando regime único")
            df['market_regime'] = 0
            for i in range(self.n_regimes):
                df[f'regime_{i}'] = int(i == 0)
            return df

        # Normalizar
        X_regime_scaled = self.scaler.fit_transform(X_regime_clean)

        # Fit GMM
        self.gmm = GaussianMixture(n_components=self.n_regimes, random_state=42, max_iter=100)

        # Prever regimes
        regimes_clean = self.gmm.fit_predict(X_regime_scaled)

        # Mapear de volta para dataset completo
        regimes = np.full(len(df), -1)  # -1 para NaN
        regimes[mask] = regimes_clean

        # Forward fill NaNs
        df['market_regime'] = pd.Series(regimes).replace(-1, np.nan).fillna(method='ffill').fillna(0).astype(int)

        # One-hot encoding
        for i in range(self.n_regimes):
            df[f'regime_{i}'] = (df['market_regime'] == i).astype(int)

        # Tempo no regime atual (persistência)
        df['regime_duration'] = df.groupby((df['market_regime'] != df['market_regime'].shift()).cumsum()).cumcount() + 1

        # Estatísticas do regime
        logger.info("Distribuição de regimes:")
        for i in range(self.n_regimes):
            count = (df['market_regime'] == i).sum()
            pct = 100 * count / len(df)
            logger.info(f"  Regime {i}: {count} candles ({pct:.1f}%)")

        n_features = 1 + self.n_regimes + 1  # regime + one-hot + duration
        logger.info(f"✓ {n_features} Market Regime features criadas")

        return df


class MultiTimeframeFeatures:
    """
    Combina features de múltiplos timeframes.

    Estratégia:
    - 15m: Scalping signals (execução)
    - 1h: Trend intermediário (filtro)
    - 4h: Trend macro (contexto)

    Exemplo: Só fazer scalping LONG se 1h e 4h também estão em uptrend.
    """

    def __init__(self, base_timeframe='15m', higher_timeframes=['1h', '4h']):
        """
        Args:
            base_timeframe: Timeframe base (onde fazemos trades)
            higher_timeframes: Timeframes maiores para contexto
        """
        self.base_timeframe = base_timeframe
        self.higher_timeframes = higher_timeframes

    def create_features(
        self,
        df_base: pd.DataFrame,
        df_higher: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Adiciona features de timeframes maiores ao dataframe base.

        Args:
            df_base: DataFrame do timeframe base (15m)
            df_higher: Dict com DataFrames dos timeframes maiores {'1h': df_1h, '4h': df_4h}

        Returns:
            df_base com features adicionais dos timeframes maiores
        """
        logger.info(f"Criando Multi-Timeframe features...")
        logger.info(f"Base: {self.base_timeframe}, Higher: {self.higher_timeframes}")

        for tf in self.higher_timeframes:
            if tf not in df_higher:
                logger.warning(f"Timeframe {tf} não fornecido, pulando...")
                continue

            df_tf = df_higher[tf].copy()

            # Garantir que timestamp está em datetime
            if not pd.api.types.is_datetime64_any_dtype(df_tf['timestamp']):
                df_tf['timestamp'] = pd.to_datetime(df_tf['timestamp'])

            # Reindexar para alinhar com base timeframe (forward fill)
            df_tf_aligned = df_tf.set_index('timestamp').reindex(
                df_base['timestamp'],
                method='ffill'
            ).reset_index()

            # Features a trazer do timeframe maior
            key_features = []

            # Trend
            if 'trend_20' in df_tf_aligned.columns:
                key_features.append('trend_20')
            if 'trend_50' in df_tf_aligned.columns:
                key_features.append('trend_50')

            # Volatilidade
            if 'volatility_20' in df_tf_aligned.columns:
                key_features.append('volatility_20')

            # RSI
            if 'rsi' in df_tf_aligned.columns:
                key_features.append('rsi')

            # MACD
            if 'macd' in df_tf_aligned.columns:
                key_features.append('macd')
            if 'macd_signal' in df_tf_aligned.columns:
                key_features.append('macd_signal')

            # EMAs
            for ema_col in [c for c in df_tf_aligned.columns if c.startswith('ema_')]:
                key_features.append(ema_col)

            # Adicionar com prefixo do timeframe
            for feat in key_features:
                if feat in df_tf_aligned.columns:
                    df_base[f'{tf}_{feat}'] = df_tf_aligned[feat].values

            # Features específicas de alinhamento entre timeframes
            # Ex: 15m e 1h ambos em uptrend?
            if 'trend_20' in df_base.columns and f'{tf}_trend_20' in df_base.columns:
                df_base[f'{tf}_trend_alignment'] = (
                    np.sign(df_base['trend_20']) == np.sign(df_base[f'{tf}_trend_20'])
                ).astype(int)

            logger.info(f"  ✓ {len(key_features)} features de {tf} adicionadas")

        total_added = len([c for c in df_base.columns if any(tf in c for tf in self.higher_timeframes)])
        logger.info(f"✓ Total de {total_added} Multi-Timeframe features criadas")

        return df_base


# Factory function
def create_advanced_features(config: Dict) -> List:
    """
    Cria lista de feature engineers avançados baseado em config.

    Returns:
        Lista de objetos para criar features
    """
    feature_creators = []

    # Wavelet Transform
    if config.get('features', {}).get('wavelet', {}).get('enabled', False):
        wavelet_type = config['features']['wavelet'].get('wavelet_type', 'db4')
        level = config['features']['wavelet'].get('decomposition_level', 3)
        feature_creators.append(WaveletFeatures(wavelet=wavelet_type, level=level))
        logger.info(f"✓ Wavelet Transform habilitado ({wavelet_type}, level={level})")

    # Market Regime Detection
    if config.get('features', {}).get('market_regime', {}).get('enabled', False):
        n_regimes = config['features']['market_regime'].get('n_regimes', 3)
        lookback = config['features']['market_regime'].get('lookback', 100)
        feature_creators.append(MarketRegimeDetector(n_regimes=n_regimes, lookback=lookback))
        logger.info(f"✓ Market Regime Detection habilitado ({n_regimes} regimes)")

    return feature_creators
