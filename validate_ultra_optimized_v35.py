"""
VALIDAÇÃO V3.5 REFINED - Melhorias Conservadoras sobre V3
Baseado em análise V3 (90 dias: 49% WR, +69% ROI, 3.26 Sharpe)

Estratégia V3.5 REFINED:
- MANTER low_vol_bear ativo (contribui +8.83% ROI!)
- REDUZIR thresholds LEVEMENTE (5-10%, não 30%)
- AUMENTAR position sizing CONSERVADOR (+20-30%, não +100%)
- TRAILING STOP apenas em high_vol_bear (melhor regime)
- SEM filtros problemáticos (consolidação bloqueou 322 trades bons)
- Objetivo: 50-52% WR, +75-85% ROI, 3.5+ Sharpe
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
import argparse
import pickle
import yaml
from datetime import datetime, timedelta

from core.utils import load_config, setup_logging
from core.bybit_rest import BybitRESTClient
from core.data import DataManager
from core.features import FeatureStore

# Import feature engineering from train_ultra_scalper
from train_ultra_scalper import (
    create_microstructure_features,
    create_advanced_master_features,
    create_sequences
)

logger = None


class OptimizedUltraValidator:
    """Validação otimizada com filtros de regime e confiança adaptativa."""

    def __init__(self, config: dict, opt_config: dict, model_path: str):
        self.config = config
        self.opt_config = opt_config
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise ValueError(f"Model not found: {model_path}")

        # Load model ensemble
        with open(self.model_path, 'rb') as f:
            self.model_data = pickle.load(f)

        self.models = self.model_data['models']
        self.meta_learner = self.model_data['meta_learner']
        self.feature_names = self.model_data['feature_names']

        self.initial_capital = config.get('initial_capital', 10000)
        self.risk_per_trade = config.get('risk_per_trade_pct', 0.75) / 100

        # Optimization settings
        self.regime_filter = opt_config.get('regime_filter', {})
        self.confidence_filter = opt_config.get('confidence_filter', {})
        self.position_sizing = opt_config.get('position_sizing', {})
        self.stop_loss_config = opt_config.get('stop_loss', {})
        self.take_profit_config = opt_config.get('take_profit', {})
        self.trailing_stop_config = opt_config.get('trailing_stop', {})  # V4 NEW
        self.advanced_filters = opt_config.get('advanced_filters', {})   # V4 NEW

        # Statistics tracking
        self.blocked_trades_count = 0
        self.blocked_by_regime = {}
        self.blocked_by_confidence = 0
        self.blocked_by_volatility = 0    # V4 NEW
        self.blocked_by_consolidation = 0  # V4 NEW

    def detect_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detecta regime de mercado para cada candle."""

        regime_config = self.opt_config.get('regime_detection', {})
        vol_window = regime_config.get('volatility_window', 20)
        trend_window = regime_config.get('trend_window', 50)
        vol_quantiles = regime_config.get('vol_quantiles', {'low': 0.33, 'high': 0.67})

        # Calculate volatility
        df['volatility'] = df['close'].pct_change().rolling(vol_window).std()
        vol_low = df['volatility'].quantile(vol_quantiles['low'])
        vol_high = df['volatility'].quantile(vol_quantiles['high'])

        # Classify volatility regime
        df['vol_regime'] = 'medium'
        df.loc[df['volatility'] < vol_low, 'vol_regime'] = 'low_vol'
        df.loc[df['volatility'] > vol_high, 'vol_regime'] = 'high_vol'

        # Classify trend regime
        df['ma_trend'] = df['close'].rolling(trend_window).mean()
        df['trend_regime'] = 'bear'
        df.loc[df['close'] > df['ma_trend'], 'trend_regime'] = 'bull'

        # Combined regime
        df['regime'] = df['vol_regime'] + '_' + df['trend_regime']

        return df

    def is_regime_allowed(self, regime: str) -> bool:
        """Verifica se o regime está permitido."""
        if not self.regime_filter.get('enabled', True):
            return True

        regimes_config = self.regime_filter.get('regimes', {})

        if regime not in regimes_config:
            return True  # Allow unknown regimes by default

        return regimes_config[regime].get('enabled', True)

    def get_min_confidence_for_regime(self, regime: str) -> float:
        """Retorna confiança mínima para um regime."""
        if not self.regime_filter.get('enabled', True):
            return 0.10  # Default

        regimes_config = self.regime_filter.get('regimes', {})

        if regime not in regimes_config:
            return 0.10

        return regimes_config[regime].get('min_confidence', 0.10)

    def get_position_multiplier(self, regime: str, confidence: float) -> float:
        """Calcula multiplicador de posição baseado em regime e confiança."""

        # Regime multiplier
        regimes_config = self.regime_filter.get('regimes', {})
        regime_mult = 1.0
        if regime in regimes_config:
            regime_mult = regimes_config[regime].get('position_multiplier', 1.0)

        # Confidence multiplier
        conf_mult = 1.0
        if self.confidence_filter.get('enabled', True):
            tiers = self.confidence_filter.get('tiers', {})

            if confidence >= 0.70:
                conf_mult = tiers.get('ultra_high', {}).get('position_multiplier', 1.0)
            elif confidence >= 0.50:
                conf_mult = tiers.get('high', {}).get('position_multiplier', 0.8)
            elif confidence >= 0.30:
                conf_mult = tiers.get('medium', {}).get('position_multiplier', 0.6)
            else:
                conf_mult = tiers.get('low', {}).get('position_multiplier', 0.4)

        return regime_mult * conf_mult

    def get_stop_loss_multiplier(self, regime: str) -> float:
        """Retorna multiplicador de stop loss para um regime."""
        regimes_config = self.regime_filter.get('regimes', {})

        if regime not in regimes_config:
            return 1.0

        return regimes_config[regime].get('stop_atr_mult', 1.0)

    def get_take_profit_config(self, regime: str) -> dict:
        """Retorna configuração de take profit para um regime."""
        tp_regimes = self.take_profit_config.get('regimes', {})

        if regime not in tp_regimes:
            return {'tp_atr_mult': 1.5, 'risk_reward': 1.5}

        return tp_regimes[regime]

    def get_trailing_stop_config(self, regime: str) -> dict:
        """Retorna configuração de trailing stop para um regime. V4 NEW"""
        if not self.trailing_stop_config.get('enabled', False):
            return None

        ts_regimes = self.trailing_stop_config.get('regimes', {})

        if regime not in ts_regimes:
            return None

        return ts_regimes[regime]

    def check_extreme_volatility(self, current: pd.Series, atr: float) -> bool:
        """Verifica se a volatilidade está extrema (bloquear trade). V4 NEW"""
        extreme_vol_config = self.advanced_filters.get('extreme_volatility', {})

        if not extreme_vol_config.get('enabled', False):
            return False

        max_atr_mult = extreme_vol_config.get('max_atr_mult', 3.0)
        current_vol = current.get('volatility', 0)

        # Se volatilidade atual > max_atr_mult * ATR médio, é extrema
        if current_vol > max_atr_mult * atr:
            self.blocked_by_volatility += 1
            return True

        return False

    def check_consolidation(self, current: pd.Series, atr: float) -> bool:
        """Verifica se está em consolidação (range baixo = bloquear). V4 NEW"""
        consol_config = self.advanced_filters.get('consolidation', {})

        if not consol_config.get('enabled', False):
            return False

        min_range_atr = consol_config.get('min_range_atr', 0.5)
        price_range = current['high'] - current['low']

        # Se range < min_range_atr * ATR, é consolidação
        if price_range < min_range_atr * atr:
            self.blocked_by_consolidation += 1
            return True

        return False

    def get_ensemble_predictions(self, X: pd.DataFrame) -> np.ndarray:
        """Get ensemble predictions from all models."""

        base_predictions = {}

        # LightGBM
        if 'lightgbm' in self.models:
            base_predictions['lightgbm'] = self.models['lightgbm'].predict(X)

        # XGBoost
        if 'xgboost' in self.models:
            import xgboost as xgb
            dmatrix = xgb.DMatrix(X)
            base_predictions['xgboost'] = self.models['xgboost'].predict(dmatrix)

        # Transformer
        if 'transformer' in self.models and 'scaler' in self.models:
            import tensorflow as tf
            scaler = self.models['scaler']
            sequence_length = self.models['sequence_length']

            X_scaled = scaler.transform(X)
            X_seq, _ = create_sequences(X_scaled, np.zeros(len(X)), sequence_length)

            if len(X_seq) > 0:
                transformer_pred_full = np.zeros(len(X))
                transformer_pred = self.models['transformer'].predict(X_seq, verbose=0).flatten()
                transformer_pred_full[sequence_length:sequence_length+len(transformer_pred)] = transformer_pred
                transformer_pred_full[:sequence_length] = 0.5  # Neutral for first sequence_length
                base_predictions['transformer'] = transformer_pred_full

        # Meta-learner combination
        if len(base_predictions) > 0:
            X_meta = np.column_stack([pred for pred in base_predictions.values()])
            ensemble_prob = self.meta_learner.predict_proba(X_meta)[:, 1]
        else:
            ensemble_prob = np.array([0.5] * len(X))

        return ensemble_prob

    def backtest_optimized(self, df: pd.DataFrame) -> Dict:
        """Run backtest com filtros otimizados."""

        # Detect regimes
        df = self.detect_regime(df)

        # Get ensemble predictions
        X = df[self.feature_names].fillna(0)
        ml_probs = self.get_ensemble_predictions(X)

        df['ml_prob_up'] = ml_probs
        df['ml_prob_down'] = 1 - ml_probs
        df['ml_confidence'] = np.abs(ml_probs - 0.5) * 2

        # Generate signals with regime filters
        df['signal'] = 0
        df['blocked'] = False
        df['min_confidence_required'] = 0.10

        for i in range(len(df)):
            row = df.iloc[i]
            regime = row['regime']
            atr = row.get('atr', row['close'] * 0.01)

            # Check if regime is allowed
            if not self.is_regime_allowed(regime):
                df.iloc[i, df.columns.get_loc('blocked')] = True
                self.blocked_trades_count += 1
                self.blocked_by_regime[regime] = self.blocked_by_regime.get(regime, 0) + 1
                continue

            # V4 NEW: Check extreme volatility
            if self.check_extreme_volatility(row, atr):
                df.iloc[i, df.columns.get_loc('blocked')] = True
                self.blocked_trades_count += 1
                continue

            # V4 NEW: Check consolidation
            if self.check_consolidation(row, atr):
                df.iloc[i, df.columns.get_loc('blocked')] = True
                self.blocked_trades_count += 1
                continue

            # Get minimum confidence for this regime
            min_conf = self.get_min_confidence_for_regime(regime)
            df.iloc[i, df.columns.get_loc('min_confidence_required')] = min_conf

            # Check confidence
            if row['ml_confidence'] < min_conf:
                self.blocked_by_confidence += 1
                continue

            # Generate signal
            if row['ml_prob_up'] > 0.5:
                df.iloc[i, df.columns.get_loc('signal')] = 1
            elif row['ml_prob_down'] > 0.5:
                df.iloc[i, df.columns.get_loc('signal')] = -1

        # Simulate
        trades = self._simulate_optimized(df)

        # Stats
        stats = self._calculate_stats(trades, df)

        return stats

    def _simulate_optimized(self, df: pd.DataFrame) -> List[Dict]:
        """Simulate trading with optimized parameters."""
        trades = []
        position = None
        capital = self.initial_capital
        cooldown = 0

        for i in range(len(df)):
            current = df.iloc[i]

            if cooldown > 0:
                cooldown -= 1

            # Check exit
            if position:
                exit_reason = self._check_exit_optimized(position, current, i)
                if exit_reason:
                    trade = self._close_trade_optimized(position, current, exit_reason)
                    trades.append(trade)
                    capital += trade['pnl_amount']
                    position = None
                    cooldown = 4

            # Check entry
            if (not position and current['signal'] != 0 and not current['blocked']
                and cooldown == 0 and i < len(df) - 20):
                position = self._open_trade_optimized(current, capital, i)

        # Close final position
        if position:
            trade = self._close_trade_optimized(position, df.iloc[-1], 'end_of_data')
            trades.append(trade)

        return trades

    def _open_trade_optimized(self, current, capital, idx):
        """Open trade with optimized SL/TP and position sizing."""
        direction = 'long' if current['signal'] == 1 else 'short'
        price = current['close']
        atr = current.get('atr', price * 0.01)
        regime = current['regime']
        confidence = current['ml_confidence']

        # Get regime-specific parameters
        sl_mult = self.get_stop_loss_multiplier(regime)
        tp_config = self.get_take_profit_config(regime)
        tp_mult = tp_config.get('tp_atr_mult', 1.5)
        risk_reward = tp_config.get('risk_reward', 1.5)

        # Calculate SL/TP
        if direction == 'long':
            sl = price - (atr * sl_mult)
            tp1 = price + (atr * tp_mult)
            tp2 = price + (atr * tp_mult * risk_reward)
            tp3 = price + (atr * tp_mult * risk_reward * 1.5)
        else:
            sl = price + (atr * sl_mult)
            tp1 = price - (atr * tp_mult)
            tp2 = price - (atr * tp_mult * risk_reward)
            tp3 = price - (atr * tp_mult * risk_reward * 1.5)

        # Position sizing with multipliers
        sl_dist = abs((sl - price) / price)
        base_risk_amt = capital * self.risk_per_trade

        # Apply position multiplier
        pos_mult = self.get_position_multiplier(regime, confidence)
        risk_amt = base_risk_amt * pos_mult

        size = risk_amt / sl_dist if sl_dist > 0 else capital * 0.1
        size = min(size, capital * 0.95)

        # V4 NEW: Get trailing stop config
        ts_config = self.get_trailing_stop_config(regime)

        return {
            'entry_idx': idx,
            'entry_time': current.name,
            'entry_price': price,
            'direction': direction,
            'size': size,
            'stop_loss': sl,
            'tp1': tp1,
            'tp2': tp2,
            'tp3': tp3,
            'ml_confidence': confidence,
            'ml_prob_up': current['ml_prob_up'],
            'regime': regime,
            'position_multiplier': pos_mult,
            'atr': atr,
            'trailing_stop_config': ts_config,  # V4 NEW
            'peak_profit': 0,                    # V4 NEW: Track peak profit for trailing
            'trailing_active': False             # V4 NEW: Track if trailing is active
        }

    def _check_exit_optimized(self, position, current, idx):
        """Check exit conditions with V4 trailing stop support."""
        high = current['high']
        low = current['low']
        close = current['close']
        direction = position['direction']

        # V4 NEW: Update trailing stop
        ts_config = position.get('trailing_stop_config')
        if ts_config:
            # Calculate current profit in ATR units
            atr = position['atr']
            entry = position['entry_price']

            if direction == 'long':
                current_profit_atr = (close - entry) / atr
            else:
                current_profit_atr = (entry - close) / atr

            # Update peak profit
            if current_profit_atr > position['peak_profit']:
                position['peak_profit'] = current_profit_atr

            # Check if trailing should activate
            trigger_atr = ts_config.get('trigger_atr_mult', 1.5)
            if not position['trailing_active'] and position['peak_profit'] >= trigger_atr:
                position['trailing_active'] = True

            # Update stop loss if trailing is active
            if position['trailing_active']:
                trail_atr = ts_config.get('trail_atr_mult', 0.8)
                peak_price = entry + (position['peak_profit'] * atr) if direction == 'long' else entry - (position['peak_profit'] * atr)

                if direction == 'long':
                    new_sl = peak_price - (trail_atr * atr)
                    position['stop_loss'] = max(position['stop_loss'], new_sl)
                else:
                    new_sl = peak_price + (trail_atr * atr)
                    position['stop_loss'] = min(position['stop_loss'], new_sl)

        # Check exits
        if direction == 'long':
            if low <= position['stop_loss']:
                return 'trailing_stop' if position.get('trailing_active') else 'stop_loss'
            if high >= position['tp3']:
                return 'take_profit_3'
            if high >= position['tp2']:
                return 'take_profit_2'
            if high >= position['tp1']:
                return 'take_profit_1'
        else:
            if high >= position['stop_loss']:
                return 'trailing_stop' if position.get('trailing_active') else 'stop_loss'
            if low <= position['tp3']:
                return 'take_profit_3'
            if low <= position['tp2']:
                return 'take_profit_2'
            if low <= position['tp1']:
                return 'take_profit_1'

        # Time exit (48h = 192 candles 15m)
        if idx - position['entry_idx'] > 192:
            return 'time_exit'

        return None

    def _close_trade_optimized(self, position, current, reason):
        """Close trade."""
        if reason == 'stop_loss' or reason == 'trailing_stop':  # V4 NEW: trailing_stop
            exit_price = position['stop_loss']
        elif reason == 'take_profit_1':
            exit_price = position['tp1']
        elif reason == 'take_profit_2':
            exit_price = position['tp2']
        elif reason == 'take_profit_3':
            exit_price = position['tp3']
        elif reason == 'time_exit':
            exit_price = current['close']
        else:
            exit_price = current['close']

        direction = position['direction']

        if direction == 'long':
            pnl_pct = ((exit_price - position['entry_price']) / position['entry_price']) * 100
        else:
            pnl_pct = ((position['entry_price'] - exit_price) / position['entry_price']) * 100

        pnl_amount = position['size'] * (pnl_pct / 100)

        return {
            'entry_time': position['entry_time'],
            'exit_time': current.name,
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'direction': direction,
            'size': position['size'],
            'pnl_pct': pnl_pct,
            'pnl_amount': pnl_amount,
            'exit_reason': reason,
            'ml_confidence': position['ml_confidence'],
            'regime': position.get('regime', 'unknown'),
            'position_multiplier': position.get('position_multiplier', 1.0)
        }

    def _calculate_stats(self, trades: List[Dict], df: pd.DataFrame) -> Dict:
        """Calculate trading statistics."""
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'roi': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'profit_factor': 0,
                'trades': pd.DataFrame(),
                'blocked_trades': self.blocked_trades_count,
                'blocked_by_regime': self.blocked_by_regime
            }

        df_trades = pd.DataFrame(trades)

        wins = (df_trades['pnl_amount'] > 0).sum()
        losses = (df_trades['pnl_amount'] < 0).sum()
        win_rate = wins / len(trades) if len(trades) > 0 else 0

        total_pnl = df_trades['pnl_amount'].sum()
        roi = (total_pnl / self.initial_capital) * 100

        # Sharpe Ratio
        if len(df_trades) > 1:
            returns = df_trades['pnl_pct'].values
            sharpe = (returns.mean() / (returns.std() + 1e-8)) * np.sqrt(252)
        else:
            sharpe = 0

        # Max Drawdown
        cumulative = df_trades['pnl_amount'].cumsum() + self.initial_capital
        peak = cumulative.expanding().max()
        dd = ((cumulative - peak) / peak * 100)
        max_dd = dd.min()

        # Profit Factor
        gross_profit = df_trades[df_trades['pnl_amount'] > 0]['pnl_amount'].sum()
        gross_loss = abs(df_trades[df_trades['pnl_amount'] < 0]['pnl_amount'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        return {
            'total_trades': len(trades),
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'roi': roi,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'profit_factor': profit_factor,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'trades': df_trades,
            'blocked_trades': self.blocked_trades_count,
            'blocked_by_regime': self.blocked_by_regime
        }

    def walk_forward_validation(self, df: pd.DataFrame, n_splits=5) -> List[Dict]:
        """Walk-forward validation."""
        print(f"\n🚶 Walk-Forward Validation ({n_splits} splits)...")

        total_len = len(df)
        fold_size = total_len // (n_splits + 1)

        results = []

        for i in range(n_splits):
            start_idx = (i + 1) * fold_size
            end_idx = start_idx + fold_size

            if end_idx > total_len:
                break

            df_fold = df.iloc[start_idx:end_idx].copy()

            print(f"\n   Fold {i+1}/{n_splits}: {df_fold.index[0]} to {df_fold.index[-1]}")

            # Reset blocked counter for each fold
            self.blocked_trades_count = 0
            self.blocked_by_regime = {}

            stats = self.backtest_optimized(df_fold)
            stats['fold'] = i + 1
            stats['start_date'] = df_fold.index[0]
            stats['end_date'] = df_fold.index[-1]

            results.append(stats)

        return results

    def regime_based_analysis(self, df: pd.DataFrame) -> Dict:
        """Analisa performance em diferentes regimes."""
        print("\n📊 Regime-Based Analysis (OPTIMIZED)...")

        df = self.detect_regime(df)

        regime_results = {}

        for regime in df['regime'].unique():
            if pd.isna(regime):
                continue

            df_regime = df[df['regime'] == regime].copy()

            if len(df_regime) < 100:
                continue

            print(f"   Testing regime: {regime}...")

            # Reset blocked counter
            self.blocked_trades_count = 0
            self.blocked_by_regime = {}

            stats = self.backtest_optimized(df_regime)
            stats['regime'] = regime
            stats['samples'] = len(df_regime)

            regime_results[regime] = stats

        return regime_results

    def monte_carlo_simulation(self, trades: List[Dict], n_simulations=1000) -> Dict:
        """Monte Carlo simulation."""
        # Handle both list and DataFrame inputs
        if isinstance(trades, pd.DataFrame):
            if trades.empty:
                return {}
            df_trades = trades
        elif isinstance(trades, list):
            if not trades:
                return {}
            df_trades = pd.DataFrame(trades)
        else:
            return {}

        print(f"\n🎲 Monte Carlo Simulation ({n_simulations} runs)...")

        pnl_values = df_trades['pnl_amount'].values

        simulation_results = []

        for _ in range(n_simulations):
            # Randomize trade order
            shuffled_pnl = np.random.choice(pnl_values, size=len(pnl_values), replace=True)

            # Calculate cumulative
            cumsum = np.cumsum(shuffled_pnl)
            final_capital = self.initial_capital + cumsum[-1]
            roi = (cumsum[-1] / self.initial_capital) * 100

            # Max drawdown
            peak = np.maximum.accumulate(self.initial_capital + cumsum)
            dd = ((self.initial_capital + cumsum - peak) / peak * 100).min()

            simulation_results.append({
                'final_capital': final_capital,
                'roi': roi,
                'max_dd': dd
            })

        df_sim = pd.DataFrame(simulation_results)

        return {
            'mean_roi': df_sim['roi'].mean(),
            'median_roi': df_sim['roi'].median(),
            'std_roi': df_sim['roi'].std(),
            'best_roi': df_sim['roi'].max(),
            'worst_roi': df_sim['roi'].min(),
            'roi_5th_percentile': df_sim['roi'].quantile(0.05),
            'roi_95th_percentile': df_sim['roi'].quantile(0.95),
            'mean_dd': df_sim['max_dd'].mean(),
            'worst_dd': df_sim['max_dd'].min(),
            'prob_profit': (df_sim['roi'] > 0).mean() * 100
        }


def main():
    global logger

    parser = argparse.ArgumentParser(description='Validate Optimized Ultra Scalper')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Trading symbol')
    parser.add_argument('--days', type=int, default=365, help='Days of data')
    parser.add_argument('--model', type=str, default='ultra_scalper_btcusdt_365d.pkl', help='Model filename')
    parser.add_argument('--initial-capital', type=float, default=10000, help='Initial capital')

    args = parser.parse_args()

    # Load configs
    config = load_config('standard')
    config['initial_capital'] = args.initial_capital

    # Load V3.5 REFINED config
    opt_config_path = Path(__file__).parent / 'config_ultra_optimized_v35.yaml'
    with open(opt_config_path, 'r', encoding='utf-8') as f:
        opt_config = yaml.safe_load(f)

    # Setup logging
    from loguru import logger as loguru_logger
    logger = loguru_logger

    print("=" * 80)
    print("🚀 V3.5 REFINED ULTRA SCALPER VALIDATION")
    print("=" * 80)
    print(f"Symbol: {args.symbol}")
    print(f"Period: {args.days} days")
    print(f"Model:  {args.model}")
    print(f"Initial Capital: ${args.initial_capital:,.2f}")
    print()
    print("🎯 V3.5 REFINED OPTIMIZATIONS (CONSERVATIVE):")
    print(f"   Based on V3 results: 49% WR, +69% ROI, 3.26 Sharpe")
    print()
    print(f"   🔥 REGIME OPTIMIZATIONS:")
    print(f"   - Blocked: medium_bull, low_vol_bull")
    print(f"   - ✅ KEPT low_vol_bear ACTIVE (contributes +8.83% ROI!)")
    print(f"   - Reduced Thresholds (CONSERVATIVE):")
    print(f"     • high_vol_bear: 50% → 45% conf (60.6% WR, +51.82% ROI!)")
    print(f"     • medium_bear: 40% → 38% conf (50.0% WR, +18.95% ROI)")
    print(f"     • low_vol_bear: 45% → 43% conf (43.8% WR, +8.83% ROI)")
    print(f"     • high_vol_bull: 60% → 58% conf (44.6% WR, +20.44% ROI)")
    print()
    print(f"   💰 POSITION SIZING (CONSERVATIVE +20-30%):")
    print(f"   - high_vol_bear: 1.6x (vs 1.3x V3)")
    print(f"   - medium_bear: 1.8x (vs 1.5x V3)")
    print(f"   - Mode: {opt_config['position_sizing']['mode']}")
    print()
    print(f"   🎯 FEATURES:")
    print(f"   - Trailing Stop: {opt_config['trailing_stop']['enabled']} (only high_vol_bear)")
    print(f"   - Extreme Volatility Filter: {opt_config['advanced_filters']['extreme_volatility']['enabled']}")
    print(f"   - Consolidation Filter: {opt_config['advanced_filters']['consolidation']['enabled']}")
    print()
    print(f"   📈 TARGET: 50-52% WR, +75-85% ROI, 3.5+ Sharpe")
    print()

    # Download data
    print("📥 Downloading data...")
    rest_client = BybitRESTClient(
        api_key=config['bybit_api_key'],
        api_secret=config['bybit_api_secret'],
        testnet=config['bybit_testnet']
    )

    dm = DataManager(rest_client)
    df = dm.get_data(args.symbol, '15m', args.days, use_cache=False)

    if df.empty:
        print("❌ No data")
        return

    print(f"✅ Downloaded {len(df):,} candles\n")

    # Build features
    print("🔨 Building features...")
    fs = FeatureStore(config)
    df_features = fs.build_features(df, normalize=False)
    df_features = create_microstructure_features(df_features)
    df_features = create_advanced_master_features(df_features)
    print(f"✅ Features ready\n")

    # Load validator
    model_path = f"storage/models/{args.model}"

    try:
        validator = OptimizedUltraValidator(config, opt_config, model_path)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # 1. Walk-forward validation
    wf_results = validator.walk_forward_validation(df_features, n_splits=5)

    print("\n" + "=" * 80)
    print("📊 WALK-FORWARD RESULTS (V3.5 REFINED)")
    print("=" * 80)

    for r in wf_results:
        if r.get('total_trades', 0) > 0:
            print(f"\nFold {r['fold']}: {r['start_date']} to {r['end_date']}")
            print(f"   Trades: {r['total_trades']}  |  WR: {r['win_rate']*100:.1f}%  |  "
                  f"ROI: {r['roi']:+.2f}%  |  Sharpe: {r['sharpe_ratio']:.2f}  |  DD: {r['max_drawdown']:.1f}%")
            if r.get('blocked_trades', 0) > 0:
                print(f"   🚫 Blocked: {r['blocked_trades']} trades - {r.get('blocked_by_regime', {})}")

    # 2. Regime-based analysis
    regime_results = validator.regime_based_analysis(df_features)

    print("\n" + "=" * 80)
    print("📊 REGIME-BASED PERFORMANCE (V3.5 REFINED)")
    print("=" * 80)

    for regime, r in regime_results.items():
        status_emoji = "✅" if r.get('total_trades', 0) > 0 else "🚫"
        print(f"\n{status_emoji} {regime:20} ({r['samples']:,} samples)")
        if r.get('total_trades', 0) > 0:
            print(f"   Trades: {r['total_trades']}  |  WR: {r['win_rate']*100:.1f}%  |  ROI: {r['roi']:+.2f}%")
        else:
            print(f"   ⚠️  BLOCKED - {r.get('blocked_trades', 0)} trades blocked")

    # 3. Full backtest with Monte Carlo
    print("\n" + "=" * 80)
    print("📊 FULL BACKTEST + MONTE CARLO (V3.5 REFINED)")
    print("=" * 80)

    # Reset blocked counters
    validator.blocked_trades_count = 0
    validator.blocked_by_regime = {}
    validator.blocked_by_confidence = 0
    validator.blocked_by_volatility = 0      # V4 NEW
    validator.blocked_by_consolidation = 0   # V4 NEW

    full_stats = validator.backtest_optimized(df_features)

    if full_stats.get('total_trades', 0) > 0:
        print(f"\nBase Metrics:")
        print(f"   Trades: {full_stats['total_trades']}")
        print(f"   Win Rate: {full_stats['win_rate']*100:.1f}%")
        print(f"   ROI: {full_stats['roi']:+.2f}%")
        print(f"   Sharpe: {full_stats['sharpe_ratio']:.2f}")
        print(f"   Max DD: {full_stats['max_drawdown']:.1f}%")
        print(f"   Profit Factor: {full_stats['profit_factor']:.2f}")
        print(f"\n   🚫 Blocked Trades: {full_stats['blocked_trades']}")
        if full_stats.get('blocked_by_regime'):
            print(f"      By Regime: {full_stats['blocked_by_regime']}")
        if validator.blocked_by_confidence > 0:
            print(f"      By Confidence: {validator.blocked_by_confidence} trades")
        if validator.blocked_by_volatility > 0:
            print(f"      By Extreme Volatility: {validator.blocked_by_volatility} trades (V4 NEW)")
        if validator.blocked_by_consolidation > 0:
            print(f"      By Consolidation: {validator.blocked_by_consolidation} trades (V4 NEW)")

        # Monte Carlo
        mc_results = validator.monte_carlo_simulation(full_stats['trades'])

        if mc_results:
            print(f"\nMonte Carlo Simulation (1000 runs):")
            print(f"   Mean ROI:   {mc_results['mean_roi']:+.2f}%")
            print(f"   Median ROI: {mc_results['median_roi']:+.2f}%")
            print(f"   Std ROI:    {mc_results['std_roi']:.2f}%")
            print(f"   Best ROI:   {mc_results['best_roi']:+.2f}%")
            print(f"   Worst ROI:  {mc_results['worst_roi']:+.2f}%")
            print(f"   5th %ile:   {mc_results['roi_5th_percentile']:+.2f}%")
            print(f"   95th %ile:  {mc_results['roi_95th_percentile']:+.2f}%")
            print(f"   Prob Profit: {mc_results['prob_profit']:.1f}%")
            print(f"   Mean DD:    {mc_results['mean_dd']:.1f}%")
            print(f"   Worst DD:   {mc_results['worst_dd']:.1f}%")

    print("\n" + "=" * 80)
    print("✅ V3.5 REFINED VALIDATION COMPLETE")
    print("=" * 80)
    print("\n💡 V3.5 vs V3 Comparison:")
    print("   V3: 49% WR, +69% ROI, 3.26 Sharpe, 445 trades")
    print("   V3.5 Target: 50-52% WR, +75-85% ROI, 3.5+ Sharpe")
    print("=" * 80)


if __name__ == "__main__":
    main()
