"""
Triple Barrier Dinâmico
Ajusta barreiras baseado em volatilidade e regime de mercado
"""

import pandas as pd
import numpy as np
from loguru import logger


class DynamicTripleBarrier:
    """
    Triple Barrier com barreiras dinâmicas.
    
    Ajusta profit target e stop loss baseado em:
    - Volatilidade recente (ATR, std)
    - Regime de mercado (trending vs ranging)
    - Hora do dia (liquidez)
    """
    
    def __init__(self, config):
        self.config = config
        self.forward_window = config['labeling'].get('forward_window', 8)
        
        # Barreiras base
        self.profit_base = config['labeling'].get('profit_target_atr', 1.5)
        self.stop_base = config['labeling'].get('stop_loss_atr', 1.0)
        
        # Multiplicadores dinâmicos
        self.dynamic_profit = config['labeling'].get('profit_target_dynamic', True)
        self.dynamic_stop = config['labeling'].get('stop_loss_dynamic', True)
        
        self.profit_mult_low_vol = config['labeling'].get('profit_multiplier_low_vol', 1.2)
        self.profit_mult_high_vol = config['labeling'].get('profit_multiplier_high_vol', 2.0)
        
        self.stop_mult_low_vol = config['labeling'].get('stop_multiplier_low_vol', 0.8)
        self.stop_mult_high_vol = config['labeling'].get('stop_multiplier_high_vol', 1.5)
    
    def create_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria labels com barreiras dinâmicas.
        """
        logger.info("Criando labels com Triple Barrier DINÂMICO...")
        
        # Verificar features necessárias
        if 'atr' not in df.columns:
            raise ValueError("Coluna 'atr' não encontrada!")
        
        if 'volatility_20' not in df.columns:
            logger.warning("volatility_20 não encontrada, usando ATR como proxy")
            df['volatility_20'] = df['atr'] / df['close']
        
        # Calcular volatilidade normalizada (0-1)
        vol_median = df['volatility_20'].median()
        vol_high_threshold = vol_median * 1.5
        vol_low_threshold = vol_median * 0.7
        
        # Arrays
        close_prices = df['close'].values
        atr_values = df['atr'].values
        volatility = df['volatility_20'].values
        
        labels = np.array(['NONE'] * len(df), dtype=object)
        returns = np.zeros(len(df))
        barriers = np.array(['NONE'] * len(df), dtype=object)
        
        # Iterar
        total = len(df) - self.forward_window
        logger.info(f"Processando {total} candles...")
        
        for i in range(total):
            current_price = close_prices[i]
            current_atr = atr_values[i]
            current_vol = volatility[i]
            
            if np.isnan(current_atr) or current_atr == 0:
                continue
            
            # === AJUSTE DINÂMICO DAS BARREIRAS ===
            
            if self.dynamic_profit:
                # Baixa volatilidade → profit target menor (mercado não move muito)
                # Alta volatilidade → profit target maior (mercado move mais)
                if current_vol < vol_low_threshold:
                    profit_mult = self.profit_mult_low_vol
                elif current_vol > vol_high_threshold:
                    profit_mult = self.profit_mult_high_vol
                else:
                    # Interpolação linear entre low e high
                    ratio = (current_vol - vol_low_threshold) / (vol_high_threshold - vol_low_threshold)
                    profit_mult = self.profit_mult_low_vol + ratio * (self.profit_mult_high_vol - self.profit_mult_low_vol)
            else:
                profit_mult = 1.0
            
            if self.dynamic_stop:
                # Baixa volatilidade → stop mais apertado (menos ruído)
                # Alta volatilidade → stop mais largo (evitar stop prematuro)
                if current_vol < vol_low_threshold:
                    stop_mult = self.stop_mult_low_vol
                elif current_vol > vol_high_threshold:
                    stop_mult = self.stop_mult_high_vol
                else:
                    ratio = (current_vol - vol_low_threshold) / (vol_high_threshold - vol_low_threshold)
                    stop_mult = self.stop_mult_low_vol + ratio * (self.stop_mult_high_vol - self.stop_mult_low_vol)
            else:
                stop_mult = 1.0
            
            # Barreiras ajustadas
            profit_atr = self.profit_base * profit_mult
            stop_atr = self.stop_base * stop_mult
            
            upper_barrier = current_price + (profit_atr * current_atr)
            lower_barrier = current_price - (stop_atr * current_atr)
            
            # Observar futuro
            future_highs = df['high'].iloc[i+1:i+1+self.forward_window].values
            future_lows = df['low'].iloc[i+1:i+1+self.forward_window].values
            
            upper_hit_idx = np.where(future_highs >= upper_barrier)[0]
            lower_hit_idx = np.where(future_lows <= lower_barrier)[0]
            
            # Determinar label
            if len(upper_hit_idx) > 0 and len(lower_hit_idx) > 0:
                if upper_hit_idx[0] < lower_hit_idx[0]:
                    labels[i] = 'LONG'
                    barriers[i] = 'UPPER'
                    returns[i] = (upper_barrier - current_price) / current_price
                else:
                    labels[i] = 'SHORT'
                    barriers[i] = 'LOWER'
                    returns[i] = (current_price - lower_barrier) / current_price
            
            elif len(upper_hit_idx) > 0:
                labels[i] = 'LONG'
                barriers[i] = 'UPPER'
                returns[i] = (upper_barrier - current_price) / current_price
            
            elif len(lower_hit_idx) > 0:
                labels[i] = 'SHORT'
                barriers[i] = 'LOWER'
                returns[i] = (current_price - lower_barrier) / current_price
            
            else:
                # Timeout - usar preço final da janela
                final_price = df['close'].iloc[i + self.forward_window]
                price_change = (final_price - current_price) / current_price
                
                if price_change > 0.001:  # Subiu > 0.1%
                    labels[i] = 'LONG'
                    barriers[i] = 'TIME'
                    returns[i] = price_change
                elif price_change < -0.001:  # Caiu > 0.1%
                    labels[i] = 'SHORT'
                    barriers[i] = 'TIME'
                    returns[i] = abs(price_change)
                # else: permanece NONE
        
        # Aplicar no dataframe
        df['target_class'] = labels
        df['target_return'] = returns
        df['barrier_hit'] = barriers
        
        # Estatísticas
        logger.info("✓ Labels criados:")
        for label in ['LONG', 'SHORT', 'NONE']:
            count = (df['target_class'] == label).sum()
            pct = 100 * count / len(df)
            logger.info(f"  {label}: {count} ({pct:.2f}%)")
        
        return df
