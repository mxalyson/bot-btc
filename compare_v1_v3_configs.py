"""
Comparison of V1 (Optimized) vs V3 (Hybrid) configurations
Shows how progressive confidence filtering differs
"""

import yaml
from pathlib import Path
import pandas as pd

def load_config(config_path):
    """Load YAML config."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def compare_regime_configs():
    """Compare regime configurations between V1 and V3."""

    v1_config = load_config('config_optimized.yaml')
    v3_config = load_config('config_ultra_optimized_v3.yaml')

    print("=" * 80)
    print("📊 V1 (OPTIMIZED) vs V3 (HYBRID) CONFIGURATION COMPARISON")
    print("=" * 80)
    print()

    print("🎯 REGIME FILTERING COMPARISON:")
    print("=" * 80)
    print(f"{'Regime':<20} {'V1 Min Conf':<15} {'V1 Pos Mult':<15} {'V3 Min Conf':<15} {'V3 Pos Mult':<15} {'Change'}")
    print("-" * 80)

    regimes = ['medium_bear', 'high_vol_bear', 'low_vol_bear', 'high_vol_bull', 'medium_bull', 'low_vol_bull']

    for regime in regimes:
        v1_regime = v1_config['regime_filter']['regimes'].get(regime, {})
        v3_regime = v3_config['regime_filter']['regimes'].get(regime, {})

        v1_enabled = v1_regime.get('enabled', False)
        v3_enabled = v3_regime.get('enabled', False)

        v1_conf = v1_regime.get('min_confidence', 0.0) if v1_enabled else 0.99
        v3_conf = v3_regime.get('min_confidence', 0.0) if v3_enabled else 0.99

        v1_mult = v1_regime.get('position_multiplier', 0.0) if v1_enabled else 0.0
        v3_mult = v3_regime.get('position_multiplier', 0.0) if v3_enabled else 0.0

        status = "🚫 BLOCKED" if not v3_enabled else ""

        if v1_enabled and v3_enabled:
            if v3_conf > v1_conf:
                status = f"⬆️ +{(v3_conf - v1_conf)*100:.0f}% conf"
            elif v3_conf < v1_conf:
                status = f"⬇️ -{(v1_conf - v3_conf)*100:.0f}% conf"
            else:
                status = "➡️ Same"

        v1_conf_str = f"{v1_conf*100:.0f}%" if v1_enabled else "BLOCKED"
        v3_conf_str = f"{v3_conf*100:.0f}%" if v3_enabled else "BLOCKED"
        v1_mult_str = f"{v1_mult:.1f}x" if v1_enabled else "0.0x"
        v3_mult_str = f"{v3_mult:.1f}x" if v3_enabled else "0.0x"

        print(f"{regime:<20} {v1_conf_str:<15} {v1_mult_str:<15} {v3_conf_str:<15} {v3_mult_str:<15} {status}")

    print()
    print("🎯 CONFIDENCE TIER MULTIPLIERS:")
    print("=" * 80)
    print(f"{'Tier':<20} {'V1 Multiplier':<20} {'V3 Multiplier':<20} {'Change'}")
    print("-" * 80)

    tiers = ['ultra_high', 'high', 'medium', 'low']

    for tier in tiers:
        v1_tier = v1_config['confidence_filter']['tiers'].get(tier, {})
        v3_tier = v3_config['confidence_filter']['tiers'].get(tier, {})

        v1_mult = v1_tier.get('position_multiplier', 0.0)
        v3_mult = v3_tier.get('position_multiplier', 0.0)

        change = ""
        if v3_mult > v1_mult:
            change = f"⬆️ +{(v3_mult - v1_mult):.1f}"
        elif v3_mult < v1_mult:
            change = f"⬇️ {(v3_mult - v1_mult):.1f}"
        else:
            change = "➡️ Same"

        print(f"{tier:<20} {v1_mult:.1f}x{'':<16} {v3_mult:.1f}x{'':<16} {change}")

    print()
    print("🎯 RISK MANAGEMENT COMPARISON:")
    print("=" * 80)

    v1_risk = v1_config['risk_management']
    v3_risk = v3_config['risk_management']

    params = [
        ('max_daily_trades', 'Max Daily Trades'),
        ('max_concurrent_trades', 'Max Concurrent'),
        ('max_daily_loss_pct', 'Max Daily Loss %'),
        ('max_drawdown_pct', 'Max Drawdown %'),
    ]

    for param, label in params:
        v1_val = v1_risk.get(param, 0)
        v3_val = v3_risk.get(param, 0)

        change = ""
        if v3_val > v1_val:
            change = f"⬆️ +{v3_val - v1_val}"
        elif v3_val < v1_val:
            change = f"⬇️ {v3_val - v1_val}"
        else:
            change = "➡️ Same"

        print(f"{label:<25} V1: {v1_val:<10} V3: {v3_val:<10} {change}")

    print()
    print("🎯 POSITION SIZING COMPARISON:")
    print("=" * 80)

    v1_pos = v1_config['position_sizing']
    v3_pos = v3_config['position_sizing']

    print(f"Mode:         V1: {v1_pos['mode']:<10} V3: {v3_pos['mode']}")
    print(f"Base Size:    V1: ${v1_pos['base_size_usd']:<9} V3: ${v3_pos['base_size_usd']}")
    print(f"Max Size:     V1: ${v1_pos['max_size_usd']:<9} V3: ${v3_pos['max_size_usd']:<9} ⬇️ -${v1_pos['max_size_usd'] - v3_pos['max_size_usd']}")

    print()
    print("=" * 80)
    print("📊 KEY DIFFERENCES SUMMARY:")
    print("=" * 80)
    print("""
V3 HYBRID IMPROVEMENTS over V1:
1. ✅ Progressive Confidence Filtering:
   - Weak regimes require HIGHER confidence to trade
   - medium_bear: 10% → 40% min confidence
   - high_vol_bear: 10% → 50% min confidence
   - high_vol_bull: 20% → 60% min confidence

2. ✅ Stricter Regime Blocking:
   - Blocks medium_bull (40.5% WR too low)
   - Blocks low_vol_bull (33.1% WR terrible)

3. ✅ More Conservative Risk:
   - Max daily trades: 60 → 40
   - Max concurrent: 3 → 2
   - Max daily loss: 5% → 3%
   - Max position size: $1000 → $600

4. ✅ Lower confidence tier multipliers:
   - Medium conf: 0.7x → 0.6x
   - Low conf: 0.5x → 0.3x

EXPECTED RESULTS:
- V1: 50.3% WR, +67% ROI (90 days), inconsistent walk-forward
- V3: 57-63% WR, +40-50% ROI (lower but consistent), positive walk-forward

TRADE-OFF:
V3 sacrifices ROI for consistency and win rate.
""")

if __name__ == "__main__":
    compare_regime_configs()
