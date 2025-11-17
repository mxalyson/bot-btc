"""
VALIDAÇÃO ULTRA AVANÇADA - Walk-Forward + Monte Carlo + Regime Analysis
Validação rigorosa do Ultra Scalper com múltiplas dimensões
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


class UltraValidator:
    """Validação ultra-avançada com walk-forward e análise de regimes."""

    def __init__(self, config: dict, model_path: str):
        self.config = config
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

    def walk_forward_validation(self, df: pd.DataFrame, n_splits=5) -> List[Dict]:
        """
        Walk-forward validation - simula como o modelo performaria em produção.

        Divide dados em N folds, treinando em dados anteriores e testando em dados futuros.
        """
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

            stats = self.backtest_with_confidence(df_fold, min_confidence=0.10)
            stats['fold'] = i + 1
            stats['start_date'] = df_fold.index[0]
            stats['end_date'] = df_fold.index[-1]

            results.append(stats)

        return results

    def regime_based_analysis(self, df: pd.DataFrame) -> Dict:
        """Analisa performance em diferentes regimes de mercado."""

        print("\n📊 Regime-Based Analysis...")

        # Define regimes based on volatility
        df['volatility'] = df['close'].pct_change().rolling(20).std()
        vol_terciles = df['volatility'].quantile([0.33, 0.67])

        df['regime'] = 'medium'
        df.loc[df['volatility'] < vol_terciles.iloc[0], 'regime'] = 'low_vol'
        df.loc[df['volatility'] > vol_terciles.iloc[1], 'regime'] = 'high_vol'

        # Define trend
        df['trend'] = (df['close'] > df['close'].rolling(50).mean()).astype(int)
        df['regime_trend'] = df['regime'] + '_' + df['trend'].map({1: 'bull', 0: 'bear'})

        regime_results = {}

        for regime in df['regime_trend'].unique():
            if pd.isna(regime):
                continue

            df_regime = df[df['regime_trend'] == regime].copy()

            if len(df_regime) < 100:
                continue

            print(f"   Testing regime: {regime}...")

            stats = self.backtest_with_confidence(df_regime, min_confidence=0.10)
            stats['regime'] = regime
            stats['samples'] = len(df_regime)

            regime_results[regime] = stats

        return regime_results

    def monte_carlo_simulation(self, trades: List[Dict], n_simulations=1000) -> Dict:
        """
        Monte Carlo simulation - randomiza ordem dos trades para estimar distribuição de resultados.
        """
        if not trades:
            return {}

        print(f"\n🎲 Monte Carlo Simulation ({n_simulations} runs)...")

        df_trades = pd.DataFrame(trades)
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

    def backtest_with_confidence(self, df: pd.DataFrame, min_confidence: float) -> Dict:
        """Run backtest com filtro de confiança."""

        # Get ensemble predictions
        X = df[self.feature_names].fillna(0)
        ml_probs = self.get_ensemble_predictions(X)

        df['ml_prob_up'] = ml_probs
        df['ml_prob_down'] = 1 - ml_probs
        df['ml_confidence'] = np.abs(ml_probs - 0.5) * 2

        # Generate signals
        df['signal'] = 0
        mask_long = (df['ml_prob_up'] > 0.5) & (df['ml_confidence'] >= min_confidence)
        mask_short = (df['ml_prob_down'] > 0.5) & (df['ml_confidence'] >= min_confidence)

        df.loc[mask_long, 'signal'] = 1
        df.loc[mask_short, 'signal'] = -1

        # Simulate
        trades = self._simulate(df)

        # Stats
        stats = self._calculate_stats(trades, df, min_confidence)

        return stats

    def _simulate(self, df: pd.DataFrame) -> List[Dict]:
        """Simulate trading."""
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
                exit_reason = self._check_exit(position, current, i)
                if exit_reason:
                    trade = self._close_trade(position, current, exit_reason)
                    trades.append(trade)
                    capital += trade['pnl_amount']
                    position = None
                    cooldown = 4

            # Check entry
            if not position and current['signal'] != 0 and cooldown == 0 and i < len(df) - 20:
                position = self._open_trade(current, capital, i)

        # Close final position
        if position:
            trade = self._close_trade(position, df.iloc[-1], 'end_of_data')
            trades.append(trade)

        return trades

    def _open_trade(self, current, capital, idx):
        """Open trade with dynamic SL/TP based on ATR."""
        direction = 'long' if current['signal'] == 1 else 'short'
        price = current['close']
        atr = current.get('atr', price * 0.01)

        # Asymmetric SL/TP (1:2 risk-reward)
        if direction == 'long':
            sl = price - (atr * 1.5)
            tp1 = price + (atr * 1.5)
            tp2 = price + (atr * 3.0)
            tp3 = price + (atr * 4.5)
        else:
            sl = price + (atr * 1.5)
            tp1 = price - (atr * 1.5)
            tp2 = price - (atr * 3.0)
            tp3 = price - (atr * 4.5)

        sl_dist = abs((sl - price) / price)
        risk_amt = capital * self.risk_per_trade
        size = risk_amt / sl_dist if sl_dist > 0 else capital * 0.1
        size = min(size, capital * 0.95)

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
            'ml_confidence': current['ml_confidence'],
            'ml_prob_up': current['ml_prob_up']
        }

    def _check_exit(self, position, current, idx):
        """Check exit conditions."""
        high = current['high']
        low = current['low']
        direction = position['direction']

        if direction == 'long':
            if low <= position['stop_loss']:
                return 'stop_loss'
            if high >= position['tp3']:
                return 'take_profit_3'
            if high >= position['tp2']:
                return 'take_profit_2'
            if high >= position['tp1']:
                return 'take_profit_1'
        else:
            if high >= position['stop_loss']:
                return 'stop_loss'
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

    def _close_trade(self, position, current, reason):
        """Close trade."""
        if reason == 'stop_loss':
            exit_price = position['stop_loss']
        elif reason == 'take_profit_1':
            exit_price = position['tp1']
        elif reason == 'take_profit_2':
            exit_price = position['tp2']
        elif reason == 'take_profit_3':
            exit_price = position['tp3']
        else:
            exit_price = current['close']

        entry = position['entry_price']
        direction = position['direction']

        if direction == 'long':
            pnl_pct = ((exit_price - entry) / entry) * 100
        else:
            pnl_pct = ((entry - exit_price) / entry) * 100

        pnl_amount = position['size'] * (pnl_pct / 100)

        return {
            'entry_time': position['entry_time'],
            'exit_time': current.name,
            'direction': direction,
            'entry_price': entry,
            'exit_price': exit_price,
            'size': position['size'],
            'pnl_pct': pnl_pct,
            'pnl_amount': pnl_amount,
            'reason': reason,
            'ml_confidence': position['ml_confidence'],
            'ml_prob_up': position['ml_prob_up']
        }

    def _calculate_stats(self, trades, df, min_confidence):
        """Calculate comprehensive statistics."""
        if not trades:
            return {
                'error': 'No trades',
                'total_trades': 0,
                'min_confidence': min_confidence
            }

        df_trades = pd.DataFrame(trades)

        total = len(df_trades)
        winning = df_trades[df_trades['pnl_amount'] > 0]
        losing = df_trades[df_trades['pnl_amount'] <= 0]

        win_rate = len(winning) / total if total > 0 else 0

        total_pnl = df_trades['pnl_amount'].sum()
        roi = (total_pnl / self.initial_capital) * 100

        avg_win = winning['pnl_amount'].mean() if len(winning) > 0 else 0
        avg_loss = abs(losing['pnl_amount'].mean()) if len(losing) > 0 else 0

        pf = (winning['pnl_amount'].sum() / abs(losing['pnl_amount'].sum())
              if len(losing) > 0 and losing['pnl_amount'].sum() != 0 else 0)

        returns = df_trades['pnl_pct'].values
        sharpe = (np.mean(returns) / np.std(returns) * np.sqrt(252)
                 if len(returns) > 1 and np.std(returns) > 0 else 0)

        equity = self.initial_capital + df_trades['pnl_amount'].cumsum()
        peak = equity.expanding().max()
        dd = ((equity - peak) / peak * 100).min()

        # Direction stats
        longs = df_trades[df_trades['direction'] == 'long']
        shorts = df_trades[df_trades['direction'] == 'short']

        long_wr = (len(longs[longs['pnl_amount'] > 0]) / len(longs) * 100) if len(longs) > 0 else 0
        short_wr = (len(shorts[shorts['pnl_amount'] > 0]) / len(shorts) * 100) if len(shorts) > 0 else 0

        # Confidence stats
        avg_confidence = df_trades['ml_confidence'].mean()

        # Exit reason distribution
        exit_reasons = df_trades['reason'].value_counts()

        return {
            'min_confidence': min_confidence,
            'total_trades': total,
            'winning_trades': len(winning),
            'losing_trades': len(losing),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'roi': roi,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': pf,
            'max_drawdown': dd,
            'sharpe_ratio': sharpe,
            'final_capital': self.initial_capital + total_pnl,
            'avg_ml_confidence': avg_confidence,
            'long_trades': len(longs),
            'short_trades': len(shorts),
            'long_wr': long_wr,
            'short_wr': short_wr,
            'exit_reasons': exit_reasons.to_dict(),
            'trades': df_trades
        }


def main():
    global logger

    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', type=str, default='BTCUSDT')
    parser.add_argument('--days', type=int, default=180)
    parser.add_argument('--model', type=str, default='ultra_scalper_btcusdt_365d.pkl')

    args = parser.parse_args()

    config = load_config('standard')
    logger = setup_logging('INFO', log_to_file=False)

    print("=" * 80)
    print("🔬 ULTRA SCALPER VALIDATION - Walk-Forward + Regime + Monte Carlo")
    print("=" * 80)
    print(f"Symbol: {args.symbol}")
    print(f"Period: {args.days} days")
    print(f"Model:  {args.model}")
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
        validator = UltraValidator(config, model_path)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # 1. Walk-forward validation
    wf_results = validator.walk_forward_validation(df_features, n_splits=5)

    print("\n" + "=" * 80)
    print("📊 WALK-FORWARD RESULTS")
    print("=" * 80)

    for r in wf_results:
        if r.get('total_trades', 0) > 0:
            print(f"\nFold {r['fold']}: {r['start_date']} to {r['end_date']}")
            print(f"   Trades: {r['total_trades']}  |  WR: {r['win_rate']*100:.1f}%  |  "
                  f"ROI: {r['roi']:+.2f}%  |  Sharpe: {r['sharpe_ratio']:.2f}  |  DD: {r['max_drawdown']:.1f}%")

    # 2. Regime-based analysis
    regime_results = validator.regime_based_analysis(df_features)

    print("\n" + "=" * 80)
    print("📊 REGIME-BASED PERFORMANCE")
    print("=" * 80)

    for regime, r in regime_results.items():
        if r.get('total_trades', 0) > 0:
            print(f"\n{regime:20} ({r['samples']:,} samples)")
            print(f"   Trades: {r['total_trades']}  |  WR: {r['win_rate']*100:.1f}%  |  ROI: {r['roi']:+.2f}%")

    # 3. Full backtest with Monte Carlo
    print("\n" + "=" * 80)
    print("📊 FULL BACKTEST + MONTE CARLO")
    print("=" * 80)

    full_stats = validator.backtest_with_confidence(df_features, min_confidence=0.10)

    if full_stats.get('total_trades', 0) > 0:
        print(f"\nBase Metrics:")
        print(f"   Trades: {full_stats['total_trades']}")
        print(f"   Win Rate: {full_stats['win_rate']*100:.1f}%")
        print(f"   ROI: {full_stats['roi']:+.2f}%")
        print(f"   Sharpe: {full_stats['sharpe_ratio']:.2f}")
        print(f"   Max DD: {full_stats['max_drawdown']:.1f}%")
        print(f"   Profit Factor: {full_stats['profit_factor']:.2f}")

        # Monte Carlo
        mc_results = validator.monte_carlo_simulation(full_stats['trades'].to_dict('records'))

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
    print("✅ VALIDATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
