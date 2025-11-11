# Implementation Plan for Risk Management Improvements

## Status: In Progress

### ✅ Completed:
1. Configuration updated with new parameters
2. Position dataclass updated with new fields
3. Risk controls tracking added

### 🔄 In Progress:
1. Position sizing (volatility-normalized with caps)
2. Exit logic (breakeven at +1R, chandelier at +2R, time stop)
3. Pyramiding (R-based laddering)
4. Entry gates (ADX>=20, histogram z-score, volatility band)
5. Partial profit taking
6. Daily loss limit and per-pair kill switch
7. Size-weighted diagnostics

### Key Changes Needed:

#### 1. Position Sizing (`_open_position`)
- Fix: `risk_amount = total_equity * self.config.risk_per_trade_pct` (remove /100, already decimal)
- Add per-pair risk cap check
- Add portfolio risk cap check

#### 2. Exit Logic (position update loop)
- Move to breakeven at +1R (not 0.3R)
- Chandelier trailing stop at +2R (HH/LL - 3*ATR)
- Time stop at 48 bars if not reached +1R
- Partial profit at +2R (take 30%, tighten stop to +0.5R)

#### 3. Pyramiding (replace current logic)
- Add only after prior unit locked at breakeven
- Spacing: every +1.0R from last add
- Diminishing sizes: [1.0, 0.7, 0.5] of initial risk
- Max 3 adds
- Block if ADX < 20 or dMACD < 0

#### 4. Entry Gates (`generate_signal`)
- ADX >= 20 (not 15)
- Histogram z-score >= 1.0
- Volatility band: 0.2% <= ATR/Price <= 2.5%
- MTF regime alignment

#### 5. Risk Controls
- Daily loss limit: stop new trades if equity down >2% from day's open
- Per-pair kill switch: 3 losses in a row → cooldown

#### 6. Diagnostics
- Size-weighted avg win/loss
- Worst 3 trades contribution
- R-distribution
- Win rate by regime/ADX bucket

