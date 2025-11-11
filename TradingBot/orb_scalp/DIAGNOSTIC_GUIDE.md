# Diagnostic Guide

## Quick Start

### Step 1: Run in Diagnostic Mode

```bash
python run_backtest.py --config config_permissive.yaml --days 15 --symbols BTCUSD ETHUSD
```

This uses relaxed filters to verify the system works. You should see trades.

### Step 2: Analyze Gate Statistics

After running, check `results/gate_statistics.json` and `results/gate_tracking.csv` to see which filters are blocking trades.

### Step 3: Adjust and Re-run

Based on gate statistics, adjust parameters in `config_production.yaml` and re-run:

```bash
python run_backtest.py --config config_production.yaml --days 30 --symbols BTCUSD ETHUSD SOLUSD BNBUSD
```

## Understanding Gate Statistics

### ORB Gates

1. **Gate 1 (Basic)**: Trade hours, no gaps, warmup complete
2. **Gate 2 (OR Valid)**: Opening range exists (first 15 minutes of day)
3. **Gate 3 (Breakout)**: Price breaks above ORH or below ORL
4. **Gate 4 (Vol Spike)**: Volume >= 1.2× median (or adaptive threshold)
5. **Gate 5 (RSI)**: RSI >= 55 for longs, <= 45 for shorts
6. **Gate 6 (Retry)**: Not exceeded max retries per day

### Scalp Gates

1. **Gate 1 (Basic)**: Trade hours, no gaps, warmup complete
2. **Gate 2 (RV)**: Realized volatility above percentile floor
3. **Gate 3 (Daily Limit)**: Not exceeded max scalps per day
4. **Gate 4 (EMA Stack)**: EMA(5) > EMA(8) for longs (opposite for shorts)
5. **Gate 5 (RSI)**: RSI not exhausted (<= 75 for longs, >= 25 for shorts)
6. **Gate 6 (No Chase)**: Price move <= 0.6% from last signal

## Common Issues and Fixes

### Issue: No ORB signals

**Check:**
- `orb_gate2_fail` count (opening range missing)
- `orb_gate4_fail` count (volume spike too strict)
- `orb_gate5_fail` count (RSI filter too tight)

**Fix:**
- Lower `vol_spike_mult` (try 1.05-1.10)
- Relax RSI thresholds (try 52/48)
- Check if data has gaps in first 15 minutes of day

### Issue: No Scalp signals

**Check:**
- `scalp_gate2_fail` count (RV floor blocking)
- `scalp_gate4_fail` count (EMA stack not aligned)
- `scalp_gate6_fail` count (no-chase too tight)

**Fix:**
- Lower `rv_pctile_floor` (try 0-5)
- Increase `no_chase` (try 0.006-0.01)
- Check EMA calculation

### Issue: Signals generated but no trades

**Check:**
- Portfolio blocks in `portfolio_blocks.json`
- `MAX_CONCURRENT_POS`, `MAX_PER_SYMBOL_POS`, `MAX_NOTIONAL_EXPOSURE`

**Fix:**
- Increase position limits temporarily
- Check if kill switch is active
- Verify position sizing calculation

## Parameter Tuning Guide

### For More Trades (Lower Quality)

- `vol_spike_mult`: 1.05 → 1.10
- `rsi_long_min`: 52 → 50
- `rsi_short_max`: 48 → 50
- `rv_pctile_floor`: 5 → 0
- `no_chase`: 0.006 → 0.01
- `warmup_bars`: 100 → 50

### For Higher Quality (Fewer Trades)

- `vol_spike_mult`: 1.10 → 1.15
- `rsi_long_min`: 55 → 60
- `rsi_short_max`: 45 → 40
- `rv_pctile_floor`: 5 → 10
- `no_chase`: 0.006 → 0.004

## Expected Results

### Diagnostic Mode
- Should generate 10-50 trades over 15 days
- Win rate may be lower (40-50%)
- Goal: Verify system works

### Production Mode
- Should generate 5-20 trades over 15 days
- Win rate target: 45-55%
- Sharpe target: ≥ 1.3
- Max DD target: < 6%



