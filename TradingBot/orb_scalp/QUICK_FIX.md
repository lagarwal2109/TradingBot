# Quick Fix for PowerShell Command

## Correct PowerShell Command

PowerShell doesn't use `^` for line continuation. Use **one line** or **backtick** `` ` ``:

### Option 1: Single Line (Easiest)
```powershell
python run_backtest.py --config config_permissive.yaml --data-dir "C:\Users\KUSHAJ\Downloads\VSCode\VSCode\QHack\TradingBot\data2\data" --days 15 --symbols BTCUSD ETHUSD
```

### Option 2: Multi-line with Backtick
```powershell
python run_backtest.py --config config_permissive.yaml `
  --data-dir "C:\Users\KUSHAJ\Downloads\VSCode\VSCode\QHack\TradingBot\data2\data" `
  --days 15 --symbols BTCUSD ETHUSD
```

## Why Gate Statistics Are Empty

If `gate_statistics.json` is empty `{}`, it means:
- **No bars even passed the first gate** (trade_hours, gap_ok, warmup_ok)
- The system is failing before it can track anything

## Quick Diagnostic Steps

1. **Check if data is loading:**
   - Look for "Loading data for X symbols..." message
   - Should see "Computing features..." message

2. **Check trade hours:**
   - In diagnostic mode, trade hours should be 0-24 (all day)
   - If using production config, it's 8-20 UTC

3. **Check warmup:**
   - Diagnostic mode: 50 bars
   - Production: 100 bars
   - Need at least this many bars before any signals

4. **Run with verbose output:**
   - The script should print progress every 5%
   - If you see "Progress: X/Y" but 0 trades, gates are blocking

## Expected Output

After running, you should see:
- `gate_statistics.json` with counts like:
  ```json
  {
    "BTCUSD": {
      "orb_gate1_fail": 1000,
      "orb_gate2_fail": 500,
      "orb_gate3_fail": 200,
      ...
    }
  }
  ```

If it's still empty, the issue is before gate tracking starts (data loading, feature computation, or very early filters).



