"""
PAPER TRADING - Simulação de Trading em Tempo Real
Simula o bot operando ao vivo sem risco real
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import argparse
import pickle
import yaml
from datetime import datetime, timedelta
import time

from core.utils import load_config
from core.bybit_rest import BybitRESTClient
from core.data import DataManager
from core.features import FeatureStore

from train_ultra_scalper import (
    create_microstructure_features,
    create_advanced_master_features,
    create_sequences
)

from validate_optimized_ultra_scalper import OptimizedUltraValidator


class PaperTradingBot:
    """Bot de paper trading que simula operação em tempo real."""

    def __init__(self, validator, initial_capital: float = 10000):
        self.validator = validator
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.position: Optional[Dict] = None
        self.trades: List[Dict] = []
        self.candles_processed = 0

    def process_candle(self, candle: pd.Series, idx: int) -> Optional[str]:
        """Processa uma vela e toma decisões de trading."""

        # Check if we have a position
        if self.position:
            # Check exit conditions
            exit_reason = self._check_exit(candle)
            if exit_reason:
                self._close_position(candle, exit_reason)
                return f"CLOSED: {exit_reason}"

        # Check entry conditions if no position
        if not self.position:
            signal = self._get_signal(candle)
            if signal != 0:
                self._open_position(candle, signal, idx)
                return f"OPENED: {'LONG' if signal == 1 else 'SHORT'}"

        return None

    def _get_signal(self, candle: pd.Series) -> int:
        """Get trading signal for current candle."""

        # Get ML prediction
        regime = candle.get('regime', 'unknown')
        ml_prob_up = candle.get('ml_prob_up', 0.5)
        ml_prob_down = candle.get('ml_prob_down', 0.5)
        ml_confidence = candle.get('ml_confidence', 0)

        # Check if regime is allowed
        if not self.validator.is_regime_allowed(regime):
            return 0

        # Check confidence threshold
        min_conf = self.validator.get_min_confidence_for_regime(regime)
        if ml_confidence < min_conf:
            return 0

        # Generate signal
        if ml_prob_up > 0.5:
            return 1  # Long
        elif ml_prob_down > 0.5:
            return -1  # Short

        return 0

    def _open_position(self, candle: pd.Series, signal: int, idx: int):
        """Open a new position."""

        direction = 'long' if signal == 1 else 'short'
        price = candle['close']
        atr = candle.get('atr', price * 0.01)
        regime = candle.get('regime', 'unknown')
        confidence = candle.get('ml_confidence', 0)

        # Get regime-specific parameters
        sl_mult = self.validator.get_stop_loss_multiplier(regime)
        tp_config = self.validator.get_take_profit_config(regime)
        tp_mult = tp_config.get('tp_atr_mult', 1.5)

        # Calculate SL/TP
        if direction == 'long':
            sl = price - (atr * sl_mult)
            tp = price + (atr * tp_mult)
        else:
            sl = price + (atr * sl_mult)
            tp = price - (atr * tp_mult)

        # Position sizing
        pos_mult = self.validator.get_position_multiplier(regime, confidence)
        risk_amt = self.current_capital * 0.0075 * pos_mult  # 0.75% risk per trade
        sl_dist = abs((sl - price) / price)
        size = risk_amt / sl_dist if sl_dist > 0 else self.current_capital * 0.1
        size = min(size, self.current_capital * 0.95)

        self.position = {
            'entry_idx': idx,
            'entry_time': candle.name,
            'entry_price': price,
            'direction': direction,
            'size': size,
            'stop_loss': sl,
            'take_profit': tp,
            'regime': regime,
            'confidence': confidence,
            'atr': atr
        }

        print(f"\n🟢 OPENED {direction.upper()} @ ${price:,.2f}")
        print(f"   Regime: {regime} | Confidence: {confidence:.1%}")
        print(f"   SL: ${sl:,.2f} | TP: ${tp:,.2f}")
        print(f"   Size: ${size:,.2f} | Risk: ${risk_amt:,.2f}")

    def _check_exit(self, candle: pd.Series) -> Optional[str]:
        """Check if position should be closed."""

        if not self.position:
            return None

        high = candle['high']
        low = candle['low']
        direction = self.position['direction']

        if direction == 'long':
            if low <= self.position['stop_loss']:
                return 'stop_loss'
            if high >= self.position['take_profit']:
                return 'take_profit'
        else:
            if high >= self.position['stop_loss']:
                return 'stop_loss'
            if low <= self.position['take_profit']:
                return 'take_profit'

        return None

    def _close_position(self, candle: pd.Series, reason: str):
        """Close current position."""

        if reason == 'stop_loss':
            exit_price = self.position['stop_loss']
        elif reason == 'take_profit':
            exit_price = self.position['take_profit']
        else:
            exit_price = candle['close']

        direction = self.position['direction']

        if direction == 'long':
            pnl_pct = ((exit_price - self.position['entry_price']) / self.position['entry_price']) * 100
        else:
            pnl_pct = ((self.position['entry_price'] - exit_price) / self.position['entry_price']) * 100

        pnl_amount = self.position['size'] * (pnl_pct / 100)
        self.current_capital += pnl_amount

        trade = {
            'entry_time': self.position['entry_time'],
            'exit_time': candle.name,
            'entry_price': self.position['entry_price'],
            'exit_price': exit_price,
            'direction': direction,
            'size': self.position['size'],
            'pnl_pct': pnl_pct,
            'pnl_amount': pnl_amount,
            'exit_reason': reason,
            'regime': self.position['regime'],
            'confidence': self.position['confidence']
        }

        self.trades.append(trade)

        emoji = "🟢" if pnl_amount > 0 else "🔴"
        print(f"\n{emoji} CLOSED {direction.upper()} @ ${exit_price:,.2f}")
        print(f"   Reason: {reason}")
        print(f"   P&L: {pnl_pct:+.2f}% (${pnl_amount:+,.2f})")
        print(f"   Capital: ${self.current_capital:,.2f} ({((self.current_capital/self.initial_capital)-1)*100:+.2f}%)")
        print(f"   Total Trades: {len(self.trades)} | "
              f"Win Rate: {sum(1 for t in self.trades if t['pnl_amount'] > 0) / len(self.trades) * 100:.1f}%")

        self.position = None

    def get_stats(self) -> Dict:
        """Get trading statistics."""

        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'roi': 0,
                'total_pnl': 0
            }

        df_trades = pd.DataFrame(self.trades)
        wins = (df_trades['pnl_amount'] > 0).sum()
        win_rate = wins / len(self.trades)
        total_pnl = df_trades['pnl_amount'].sum()
        roi = (total_pnl / self.initial_capital) * 100

        return {
            'total_trades': len(self.trades),
            'wins': wins,
            'losses': len(self.trades) - wins,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'roi': roi,
            'current_capital': self.current_capital,
            'trades_df': df_trades
        }


def simulate_paper_trading(validator, df, initial_capital: float, speed: float = 1.0):
    """Simulate paper trading candle by candle."""

    bot = PaperTradingBot(validator, initial_capital)

    print("\n" + "=" * 80)
    print("🎮 INICIANDO PAPER TRADING")
    print("=" * 80)
    print(f"Capital Inicial: ${initial_capital:,.2f}")
    print(f"Total Candles: {len(df)}")
    print(f"Período: {df.index[0]} a {df.index[-1]}")
    print("=" * 80)

    # Process each candle
    for idx, (timestamp, candle) in enumerate(df.iterrows()):
        action = bot.process_candle(candle, idx)

        # Print progress every 100 candles
        if idx % 100 == 0:
            print(f"\n⏱️  Candle {idx}/{len(df)} | {timestamp} | Capital: ${bot.current_capital:,.2f}")

        # Simulate real-time delay (optional)
        if speed > 0:
            time.sleep(0.1 / speed)  # Scaled delay

    # Close final position if any
    if bot.position:
        bot._close_position(df.iloc[-1], 'end_of_simulation')

    # Final statistics
    stats = bot.get_stats()

    print("\n" + "=" * 80)
    print("📊 RESULTADOS FINAIS")
    print("=" * 80)
    print(f"Capital Inicial: ${initial_capital:,.2f}")
    print(f"Capital Final: ${stats['current_capital']:,.2f}")
    print(f"P&L Total: ${stats['total_pnl']:+,.2f}")
    print(f"ROI: {stats['roi']:+.2f}%")
    print(f"Total Trades: {stats['total_trades']}")
    print(f"Wins: {stats['wins']} | Losses: {stats['losses']}")
    print(f"Win Rate: {stats['win_rate']*100:.1f}%")

    if stats['total_trades'] > 0:
        print(f"\n📈 DETALHES DOS TRADES:")
        trades_df = stats['trades_df']

        print(f"   Melhor Trade: {trades_df['pnl_pct'].max():+.2f}%")
        print(f"   Pior Trade: {trades_df['pnl_pct'].min():+.2f}%")
        print(f"   Média P&L: {trades_df['pnl_pct'].mean():+.2f}%")

        # Performance by regime
        print(f"\n🎯 Performance por Regime:")
        regime_perf = trades_df.groupby('regime').agg({
            'pnl_pct': ['count', 'mean', lambda x: (x > 0).mean()]
        }).round(2)
        regime_perf.columns = ['Trades', 'Avg P&L%', 'Win Rate']
        print(regime_perf.to_string())

    return bot


def main():
    parser = argparse.ArgumentParser(description='Paper Trading - Simulação em Tempo Real')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Trading symbol')
    parser.add_argument('--days', type=int, default=30, help='Days to simulate')
    parser.add_argument('--initial-capital', type=float, default=10000, help='Initial capital')
    parser.add_argument('--speed', type=float, default=0, help='Simulation speed (0=instant, 1=1x, 10=10x)')
    parser.add_argument('--export-trades', action='store_true', help='Export trades to CSV')

    args = parser.parse_args()

    # Load configs
    config = load_config('standard')
    config['initial_capital'] = args.initial_capital

    opt_config_path = Path(__file__).parent / 'config_optimized.yaml'
    with open(opt_config_path, 'r', encoding='utf-8') as f:
        opt_config = yaml.safe_load(f)

    print("=" * 80)
    print("🎮 PAPER TRADING - SISTEMA V1 OTIMIZADO")
    print("=" * 80)
    print(f"Symbol: {args.symbol}")
    print(f"Period: {args.days} days")
    print(f"Initial Capital: ${args.initial_capital:,.2f}")
    print()

    # Download data
    print(f"📥 Downloading {args.days} days of data...")
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
    df = fs.build_features(df, normalize=False)
    df = create_microstructure_features(df)
    df = create_advanced_master_features(df)
    print(f"✅ Features ready\n")

    # Detect regimes
    print("🎯 Detecting regimes...")
    df = df.copy()
    df['volatility'] = df['close'].pct_change().rolling(20).std()
    vol_terciles = df['volatility'].quantile([0.33, 0.67])

    df['regime'] = 'medium'
    df.loc[df['volatility'] < vol_terciles.iloc[0], 'regime'] = 'low_vol'
    df.loc[df['volatility'] > vol_terciles.iloc[1], 'regime'] = 'high_vol'

    df['trend'] = (df['close'] > df['close'].rolling(50).mean()).astype(int)
    df['regime'] = df['regime'] + '_' + df['trend'].map({1: 'bull', 0: 'bear'})
    print(f"✅ Regimes detected\n")

    # Load model and get predictions
    print("🤖 Loading model and generating predictions...")
    model_path = "storage/models/ultra_scalper_btcusdt_365d.pkl"
    validator = OptimizedUltraValidator(config, opt_config, model_path)

    # Get ML predictions
    X = df[validator.feature_names].fillna(0)
    ml_probs = validator.get_ensemble_predictions(X)

    df['ml_prob_up'] = ml_probs
    df['ml_prob_down'] = 1 - ml_probs
    df['ml_confidence'] = np.abs(ml_probs - 0.5) * 2
    print(f"✅ Predictions ready\n")

    # Run paper trading simulation
    bot = simulate_paper_trading(validator, df, args.initial_capital, args.speed)

    # Export trades if requested
    if args.export_trades and bot.trades:
        filename = f"paper_trades_{args.symbol}_{args.days}d.csv"
        bot.get_stats()['trades_df'].to_csv(filename, index=False)
        print(f"\n💾 Trades exported to {filename}")

    print("\n" + "=" * 80)
    print("✅ PAPER TRADING COMPLETO")
    print("=" * 80)


if __name__ == "__main__":
    main()
