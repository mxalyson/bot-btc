"""
Gerador de Dados Sintéticos Realistas para BTC 15m
Simula características reais do mercado crypto:
- Trending periods
- Mean reversion
- Volatility clustering
- Volume patterns
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from loguru import logger
import argparse

from core.utils import ensure_dir


class CryptoDataGenerator:
    """
    Gera dados sintéticos que simulam BTC em 15m.
    """

    def __init__(self, seed=42):
        np.random.seed(seed)
        self.seed = seed

    def generate(
        self,
        start_price=50000,
        n_candles=70000,  # ~2 anos de 15m
        volatility=0.02,
        trend_strength=0.0001,
        mean_reversion=0.05
    ) -> pd.DataFrame:
        """
        Gera dados OHLCV sintéticos.
        """
        logger.info(f"Gerando {n_candles:,} candles sintéticos...")
        logger.info(f"  Preço inicial: ${start_price:,.0f}")
        logger.info(f"  Volatilidade: {volatility}")
        logger.info(f"  Trend strength: {trend_strength}")

        # Timestamp
        start_time = datetime(2023, 1, 1)
        timestamps = [start_time + timedelta(minutes=15*i) for i in range(n_candles)]

        # Preço (GBM with mean reversion)
        prices = [start_price]

        for i in range(1, n_candles):
            # Regime switching (volatility clustering)
            if i % 500 == 0:
                volatility *= np.random.uniform(0.7, 1.3)

            # Mean reversion para preço médio
            mean_price = np.mean(prices[-100:]) if len(prices) >= 100 else start_price
            reversion = mean_reversion * (mean_price - prices[-1]) / prices[-1]

            # Trend component (random walk)
            trend = np.random.normal(trend_strength, volatility / 10)

            # Random shock
            shock = np.random.normal(0, volatility)

            # Novo preço
            change = trend + shock + reversion
            new_price = prices[-1] * (1 + change)

            # Limitar movimentos extremos
            new_price = np.clip(new_price, prices[-1] * 0.95, prices[-1] * 1.05)

            prices.append(new_price)

        prices = np.array(prices)

        # OHLC generation
        opens = prices[:]
        closes = prices[:]

        # Highs and lows (adicionar wicks realistas)
        highs = []
        lows = []

        for i, (o, c) in enumerate(zip(opens, closes)):
            # Wick range baseado em volatilidade
            wick_range = abs(o - c) * np.random.uniform(1.2, 3.0)

            if wick_range == 0:
                wick_range = o * volatility * np.random.uniform(0.5, 2.0)

            high = max(o, c) + abs(np.random.normal(0, wick_range / 2))
            low = min(o, c) - abs(np.random.normal(0, wick_range / 2))

            highs.append(high)
            lows.append(low)

        # Volume (com padrões realistas)
        base_volume = 1000000
        volumes = []

        for i in range(n_candles):
            # Volume maior em movimentos grandes
            price_change = abs(closes[i] - opens[i]) / opens[i]
            volume_multiplier = 1 + price_change * 10

            # Volume clustering
            if i > 0:
                volume_autocorr = volumes[-1] / base_volume
                volume_multiplier *= (0.5 + 0.5 * volume_autocorr)

            # Random component
            vol = base_volume * volume_multiplier * np.random.lognormal(0, 0.5)
            volumes.append(vol)

        # Criar DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        })

        # Adicionar alguns eventos extremos (breakouts, crashes)
        df = self._add_market_events(df)

        logger.info(f"✓ {len(df):,} candles gerados")
        logger.info(f"  Período: {df['timestamp'].min()} a {df['timestamp'].max()}")
        logger.info(f"  Preço inicial: ${df['close'].iloc[0]:,.2f}")
        logger.info(f"  Preço final: ${df['close'].iloc[-1]:,.2f}")
        logger.info(f"  Retorno total: {((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100:.2f}%")

        return df

    def _add_market_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adiciona eventos de mercado (crashes, rallies, consolidações).
        """
        n = len(df)

        # 5-10 eventos ao longo dos dados
        n_events = np.random.randint(5, 10)

        for _ in range(n_events):
            event_start = np.random.randint(1000, n - 1000)
            event_type = np.random.choice(['crash', 'rally', 'consolidation'])
            event_duration = np.random.randint(50, 200)

            if event_type == 'crash':
                # Queda abrupta
                for i in range(event_start, min(event_start + event_duration, n)):
                    df.loc[i, 'close'] *= 0.995
                    df.loc[i, 'low'] *= 0.99
                    df.loc[i, 'volume'] *= 1.5

            elif event_type == 'rally':
                # Subida forte
                for i in range(event_start, min(event_start + event_duration, n)):
                    df.loc[i, 'close'] *= 1.005
                    df.loc[i, 'high'] *= 1.01
                    df.loc[i, 'volume'] *= 1.3

            elif event_type == 'consolidation':
                # Lateralização (baixa volatilidade)
                mean_price = df.loc[event_start, 'close']
                for i in range(event_start, min(event_start + event_duration, n)):
                    df.loc[i, 'close'] = mean_price * (1 + np.random.normal(0, 0.001))
                    df.loc[i, 'volume'] *= 0.7

        # Recalcular OHLC após eventos
        df['open'] = df['close'].shift(1).fillna(df['close'].iloc[0])
        df['high'] = df[['open', 'close']].max(axis=1) * (1 + np.abs(np.random.normal(0, 0.002, len(df))))
        df['low'] = df[['open', 'close']].min(axis=1) * (1 - np.abs(np.random.normal(0, 0.002, len(df))))

        return df


def main():
    parser = argparse.ArgumentParser(description="Gerar dados sintéticos para testes")
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Símbolo')
    parser.add_argument('--timeframe', type=str, default='15m', help='Timeframe')
    parser.add_argument('--n-candles', type=int, default=70000, help='Número de candles (~2 anos)')
    parser.add_argument('--start-price', type=float, default=50000, help='Preço inicial')
    parser.add_argument('--output-dir', type=str, default='data/raw', help='Diretório de saída')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    logger.info("="*80)
    logger.info("GERADOR DE DADOS SINTÉTICOS")
    logger.info("="*80)

    # Gerar dados
    generator = CryptoDataGenerator(seed=args.seed)

    df = generator.generate(
        start_price=args.start_price,
        n_candles=args.n_candles,
        volatility=0.015,  # 1.5% volatility
        trend_strength=0.00005,  # Slight upward bias
        mean_reversion=0.03
    )

    # Salvar
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    filename = f"{args.symbol.lower()}_{args.timeframe}.csv"
    filepath = output_dir / filename

    df.to_csv(filepath, index=False)

    logger.info(f"✓ Dados salvos em: {filepath}")
    logger.info(f"  Tamanho: {filepath.stat().st_size / 1024 / 1024:.2f} MB")

    # Estatísticas
    logger.info("\nEstatísticas:")
    logger.info(f"  Candles: {len(df):,}")
    logger.info(f"  Período: {(df['timestamp'].max() - df['timestamp'].min()).days} dias")
    logger.info(f"  Preço min: ${df['low'].min():,.2f}")
    logger.info(f"  Preço max: ${df['high'].max():,.2f}")
    logger.info(f"  Volume médio: {df['volume'].mean():,.0f}")
    logger.info(f"  Volatilidade (std returns): {df['close'].pct_change().std():.4f}")

    logger.info("\n✅ Dados sintéticos gerados com sucesso!")


if __name__ == "__main__":
    main()
