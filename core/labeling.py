"""
Módulo para criação de labels de trading (LONG/SHORT/NONE).

Implementa o método Triple Barrier para labeling focado em scalping.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from loguru import logger


class TradingLabeler:
    """
    Classe para criar labels de trading usando Triple Barrier Method.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa o labeler.

        Args:
            config: Dicionário de configuração
        """
        self.config = config
        self.labeling_config = config['labeling']

    def create_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria labels de trading para o dataset.

        Args:
            df: DataFrame com OHLCV e ATR calculado

        Returns:
            DataFrame com colunas adicionais: target_class, target_return, barrier_hit
        """
        method = self.labeling_config.get('method', 'triple_barrier')

        if method == 'triple_barrier':
            return self._triple_barrier_labels(df)
        elif method == 'fixed_horizon':
            return self._fixed_horizon_labels(df)
        else:
            raise ValueError(f"Método de labeling desconhecido: {method}")

    def _triple_barrier_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Implementa o Triple Barrier Method para labeling.

        Para cada candle:
        - Define barreiras superior (profit target) e inferior (stop loss) baseadas em ATR
        - Observa os próximos N candles
        - Label LONG se atingir barreira superior primeiro
        - Label SHORT se atingir barreira inferior primeiro
        - Label NONE se não atingir nenhuma barreira ou movimento insuficiente

        IMPORTANTE: Não há look-ahead bias - apenas observamos o futuro para criar labels,
                   mas as features são calculadas apenas com dados passados.
        """
        logger.info("Criando labels usando Triple Barrier Method...")

        # Parâmetros
        forward_window = self.labeling_config.get('forward_window', 10)
        profit_atr = self.labeling_config.get('profit_target_atr', 1.5)
        stop_atr = self.labeling_config.get('stop_loss_atr', 1.0)
        min_move_atr = self.labeling_config.get('min_move_atr', 0.3)

        # Verificar se ATR existe
        if 'atr' not in df.columns:
            raise ValueError("Coluna 'atr' não encontrada. Execute feature engineering primeiro.")

        # Inicializar colunas de resultado
        df['target_class'] = 'NONE'
        df['target_return'] = 0.0
        df['barrier_hit'] = 'NONE'

        # Arrays numpy para performance
        close_prices = df['close'].values
        atr_values = df['atr'].values
        labels = np.array(['NONE'] * len(df), dtype=object)
        returns = np.zeros(len(df))
        barriers = np.array(['NONE'] * len(df), dtype=object)

        # Iterar por cada candle (exceto os últimos onde não há futuro suficiente)
        total = len(df) - forward_window
        progress_interval = max(total // 20, 1)

        for i in range(total):
            if i % progress_interval == 0:
                logger.debug(f"Processando labels: {i}/{total} ({100*i/total:.1f}%)")

            current_price = close_prices[i]
            current_atr = atr_values[i]

            # Pular se ATR é NaN
            if np.isnan(current_atr) or current_atr == 0:
                continue

            # Definir barreiras
            upper_barrier = current_price + (profit_atr * current_atr)
            lower_barrier = current_price - (stop_atr * current_atr)

            # Observar janela futura
            future_highs = df['high'].iloc[i+1:i+1+forward_window].values
            future_lows = df['low'].iloc[i+1:i+1+forward_window].values

            # Verificar qual barreira foi atingida primeiro
            upper_hit_idx = np.where(future_highs >= upper_barrier)[0]
            lower_hit_idx = np.where(future_lows <= lower_barrier)[0]

            # Determinar label
            if len(upper_hit_idx) > 0 and len(lower_hit_idx) > 0:
                # Ambas barreiras foram atingidas - ver qual foi primeiro
                if upper_hit_idx[0] < lower_hit_idx[0]:
                    # Upper barrier primeiro = LONG
                    labels[i] = 'LONG'
                    barriers[i] = 'UPPER'
                    returns[i] = (upper_barrier - current_price) / current_price
                else:
                    # Lower barrier primeiro = SHORT
                    labels[i] = 'SHORT'
                    barriers[i] = 'LOWER'
                    returns[i] = (current_price - lower_barrier) / current_price

            elif len(upper_hit_idx) > 0:
                # Apenas upper barrier atingida = LONG
                labels[i] = 'LONG'
                barriers[i] = 'UPPER'
                returns[i] = (upper_barrier - current_price) / current_price

            elif len(lower_hit_idx) > 0:
                # Apenas lower barrier atingida = SHORT
                labels[i] = 'SHORT'
                barriers[i] = 'LOWER'
                returns[i] = (current_price - lower_barrier) / current_price

            else:
                # Nenhuma barreira atingida
                # Verificar se houve movimento mínimo significativo
                max_price = future_highs.max()
                min_price = future_lows.min()

                max_move_up = (max_price - current_price) / current_atr
                max_move_down = (current_price - min_price) / current_atr

                if max_move_up > min_move_atr and max_move_up > max_move_down:
                    # Movimento ascendente significativo
                    labels[i] = 'LONG'
                    barriers[i] = 'TIME'
                    returns[i] = (max_price - current_price) / current_price
                elif max_move_down > min_move_atr and max_move_down > max_move_up:
                    # Movimento descendente significativo
                    labels[i] = 'SHORT'
                    barriers[i] = 'TIME'
                    returns[i] = (current_price - min_price) / current_price
                else:
                    # Movimento insuficiente = NONE
                    labels[i] = 'NONE'
                    barriers[i] = 'TIME'
                    returns[i] = 0.0

        # Atribuir arrays de volta ao DataFrame
        df['target_class'] = labels
        df['target_return'] = returns
        df['barrier_hit'] = barriers

        # Estatísticas
        label_counts = pd.Series(labels).value_counts()
        logger.info(f"✓ Labels criados:")
        for label, count in label_counts.items():
            pct = 100 * count / len(labels)
            logger.info(f"  {label}: {count} ({pct:.2f}%)")

        # Balanceamento (se configurado)
        if self.labeling_config.get('balance_classes', False):
            df = self._balance_classes(df)

        return df

    def _fixed_horizon_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Método alternativo: Fixed Horizon Labeling.

        Simplesmente observa o retorno após N candles e classifica como:
        - LONG se retorno > threshold
        - SHORT se retorno < -threshold
        - NONE caso contrário
        """
        logger.info("Criando labels usando Fixed Horizon Method...")

        forward_window = self.labeling_config.get('forward_window', 10)

        # Calcular retorno futuro
        df['future_return'] = df['close'].pct_change(forward_window).shift(-forward_window)

        # ATR para threshold dinâmico
        if 'atr' in df.columns:
            threshold = self.labeling_config.get('profit_target_atr', 1.0) * df['atr'] / df['close']
        else:
            # Threshold fixo se ATR não disponível
            threshold = 0.005  # 0.5%

        # Criar labels
        df['target_class'] = 'NONE'
        df.loc[df['future_return'] > threshold, 'target_class'] = 'LONG'
        df.loc[df['future_return'] < -threshold, 'target_class'] = 'SHORT'

        df['target_return'] = df['future_return'].abs()
        df['barrier_hit'] = 'HORIZON'

        # Remover coluna temporária
        df = df.drop('future_return', axis=1)

        # Estatísticas
        label_counts = df['target_class'].value_counts()
        logger.info(f"✓ Labels criados:")
        for label, count in label_counts.items():
            pct = 100 * count / len(df)
            logger.info(f"  {label}: {count} ({pct:.2f}%)")

        return df

    def _balance_classes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Balanceia classes usando undersampling da classe majoritária.

        Args:
            df: DataFrame com labels

        Returns:
            DataFrame balanceado
        """
        logger.info("Balanceando classes...")

        # Contar por classe
        label_counts = df['target_class'].value_counts()
        min_count = label_counts.min()

        logger.info(f"  Classe minoritária: {min_count} amostras")

        # Amostrar cada classe
        dfs = []
        for label in ['LONG', 'SHORT', 'NONE']:
            df_class = df[df['target_class'] == label]
            if len(df_class) > min_count:
                df_class = df_class.sample(n=min_count, random_state=42)
            dfs.append(df_class)

        # Concatenar e reordenar por timestamp
        df_balanced = pd.concat(dfs).sort_values('timestamp').reset_index(drop=True)

        logger.info(f"✓ Dataset balanceado: {len(df_balanced)} linhas")

        return df_balanced

    def split_temporal(
        self,
        df: pd.DataFrame,
        train_split: float = 0.7,
        val_split: float = 0.15,
        test_split: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Divide dataset temporalmente (sem embaralhamento).

        IMPORTANTE: Para séries temporais, não embaralhamos os dados.
                   Train vem primeiro, depois val, depois test.

        Args:
            df: DataFrame completo
            train_split: Proporção para treino
            val_split: Proporção para validação
            test_split: Proporção para teste

        Returns:
            Tupla (df_train, df_val, df_test)
        """
        if abs(train_split + val_split + test_split - 1.0) > 0.01:
            raise ValueError(f"Soma das divisões deve ser 1.0, obteve: {train_split + val_split + test_split}")

        n = len(df)
        train_end = int(n * train_split)
        val_end = int(n * (train_split + val_split))

        df_train = df.iloc[:train_end].copy()
        df_val = df.iloc[train_end:val_end].copy()
        df_test = df.iloc[val_end:].copy()

        logger.info(f"✓ Dataset dividido temporalmente:")
        logger.info(f"  Train: {len(df_train)} linhas ({df_train['timestamp'].min()} a {df_train['timestamp'].max()})")
        logger.info(f"  Val:   {len(df_val)} linhas ({df_val['timestamp'].min()} a {df_val['timestamp'].max()})")
        logger.info(f"  Test:  {len(df_test)} linhas ({df_test['timestamp'].min()} a {df_test['timestamp'].max()})")

        return df_train, df_val, df_test


if __name__ == "__main__":
    # Teste do módulo
    from core.utils import load_config, setup_logging
    from core.data_loader import BybitDataLoader
    from core.feature_engineering import FeatureEngineer

    config = load_config()
    setup_logging(config)

    # Carregar dados
    loader = BybitDataLoader(config)
    df = loader.download_ohlcv('BTCUSDT', '5m', lookback_days=30)

    # Criar features (necessário para ATR)
    engineer = FeatureEngineer(config)
    df = engineer.create_all_features(df)

    # Criar labels
    labeler = TradingLabeler(config)
    df = labeler.create_labels(df)

    print(f"\nShape final: {df.shape}")
    print(f"\nDistribuição de labels:\n{df['target_class'].value_counts()}")
    print(f"\nExemplo de dados rotulados:\n{df[['timestamp', 'close', 'atr', 'target_class', 'target_return']].head(20)}")
