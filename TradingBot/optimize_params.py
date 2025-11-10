#!/usr/bin/env python3
"""Optimize backtest parameters for competition score."""

import subprocess
import json
from pathlib import Path

# Parameter combinations to test
tests = [
    # (window, momentum, max_position, min_sharpe, description)
    (48, 10, 0.35, 0.05, "Aggressive - short window"),
    (72, 12, 0.40, 0.05, "Balanced"),
    (96, 15, 0.35, 0.08, "Conservative - longer window"),
    (60, 10, 0.30, 0.10, "High quality signals"),
    (48, 8, 0.40, 0.03, "More trades"),
    (84, 12, 0.35, 0.06, "Medium-term"),
]

results = []

print("PARAMETER OPTIMIZATION FOR COMPETITION SCORE")
print("=" * 70)
print("Formula: 0.4×Sortino + 0.3×Sharpe + 0.3×Calmar\n")

for window, momentum, max_pos, min_sharpe, desc in tests:
    print(f"\nTesting: {desc}")
    print(f"  Window={window}, Momentum={momentum}, MaxPos={max_pos}, MinSharpe={min_sharpe}")
    
    try:
        # Run backtest
        cmd = [
            "python", "backtest.py",
            "--mode", "enhanced",
            "--window", str(window),
            "--momentum", str(momentum),
            "--max-position", str(max_pos),
            "--min-sharpe", str(min_sharpe)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        # Parse metrics from JSON
        metrics_file = Path("figures/backtest_metrics.json")
        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)
            
            score = metrics.get('competition_score', 0)
            ret = metrics.get('total_return', 0)
            sharpe = metrics.get('sharpe_ratio', 0)
            sortino = metrics.get('sortino_ratio', 0)
            calmar = metrics.get('calmar_ratio', 0)
            trades = metrics.get('n_trades', 0)
            
            print(f"  Result: Score={score:.3f}, Return={ret:.1f}%, Trades={trades}")
            print(f"          Sharpe={sharpe:.3f}, Sortino={sortino:.3f}, Calmar={calmar:.3f}")
            
            results.append({
                'description': desc,
                'window': window,
                'momentum': momentum,
                'max_position': max_pos,
                'min_sharpe': min_sharpe,
                'score': score,
                'return': ret,
                'sharpe': sharpe,
                'sortino': sortino,
                'calmar': calmar,
                'trades': trades
            })
        else:
            print("  ERROR: No metrics file generated")
            
    except Exception as e:
        print(f"  ERROR: {e}")

print("\n" + "=" * 70)
print("\nRESULTS SUMMARY (sorted by competition score):\n")

results.sort(key=lambda x: x['score'], reverse=True)

for i, r in enumerate(results[:5], 1):
    print(f"{i}. {r['description']}")
    print(f"   Score: {r['score']:.3f} | Return: {r['return']:.1f}% | Trades: {r['trades']}")
    print(f"   Window={r['window']}, Momentum={r['momentum']}, MaxPos={r['max_position']}, MinSharpe={r['min_sharpe']}")
    print()

if results:
    best = results[0]
    print(f"BEST PARAMETERS:")
    print(f"  python backtest.py --mode enhanced --window {best['window']} --momentum {best['momentum']} --max-position {best['max_position']} --min-sharpe {best['min_sharpe']}")
