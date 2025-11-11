#!/usr/bin/env python3
"""Generate a summary report comparing bot performance to buy-and-hold."""

import pandas as pd
import numpy as np
from pathlib import Path

def generate_summary():
    """Generate summary report from backtest results."""
    
    results_file = Path("backtest_enhanced_results.csv")
    trades_file = Path("backtest_enhanced_trades.csv")
    
    if not results_file.exists():
        print("No backtest results found. Run backtest_enhanced.py first.")
        return
    
    # Load results
    results = pd.read_csv(results_file)
    results["timestamp"] = pd.to_datetime(results["timestamp"])
    
    # Load trades
    trades = pd.read_csv(trades_file) if trades_file.exists() else pd.DataFrame()
    if len(trades) > 0:
        trades["time"] = pd.to_datetime(trades["time"])
    
    # Calculate metrics
    initial = results["equity"].iloc[0]
    final = results["equity"].iloc[-1]
    total_return = ((final - initial) / initial) * 100
    
    returns = results["equity"].pct_change().dropna()
    sharpe = (returns.mean() / returns.std() * np.sqrt(252 * 24 * 60)) if returns.std() > 0 else 0
    max_dd = ((results["equity"] / results["equity"].expanding().max()) - 1).min() * 100
    
    # Buy-and-hold comparison
    print(f"\n{'='*70}")
    print("ENHANCED BOT BACKTEST SUMMARY - 15 Days")
    print(f"{'='*70}\n")
    
    print(f"Initial Capital: ${initial:,.2f}")
    print(f"Final Value: ${final:,.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Sharpe Ratio: {sharpe:.3f}")
    print(f"Max Drawdown: {max_dd:.2f}%")
    print(f"Number of Trades: {len(trades)}")
    
    if len(trades) > 0:
        buy_trades = trades[trades["action"] == "BUY"]
        sell_trades = trades[trades["action"] == "SELL"]
        print(f"  - Buy Trades: {len(buy_trades)}")
        print(f"  - Sell Trades: {len(sell_trades)}")
        
        if len(sell_trades) > 0:
            winning = sell_trades[sell_trades.get("pnl", 0) > 0]
            win_rate = (len(winning) / len(sell_trades)) * 100 if len(sell_trades) > 0 else 0
            avg_pnl = sell_trades["pnl"].mean() if "pnl" in sell_trades else 0
            print(f"  - Win Rate: {win_rate:.1f}%")
            print(f"  - Avg P&L per trade: ${avg_pnl:.2f}")
    
    # Load BTC data for comparison
    btc_file = Path("data/BTCUSD.csv")
    if btc_file.exists():
        btc = pd.read_csv(btc_file)
        btc["timestamp"] = pd.to_datetime(btc["timestamp"], unit="ms")
        btc = btc.sort_values("timestamp")
        
        # Get prices at start and end of backtest period
        start_time = results["timestamp"].iloc[0]
        end_time = results["timestamp"].iloc[-1]
        
        btc_start = btc[btc["timestamp"] <= start_time]
        btc_end = btc[btc["timestamp"] <= end_time]
        
        if len(btc_start) > 0 and len(btc_end) > 0:
            btc_start_price = btc_start["price"].iloc[-1]
            btc_end_price = btc_end["price"].iloc[-1]
            btc_return = ((btc_end_price - btc_start_price) / btc_start_price) * 100
            
            print(f"\n{'='*70}")
            print("COMPARISON TO BUY-AND-HOLD BTC")
            print(f"{'='*70}\n")
            print(f"BTC Buy-and-Hold Return: {btc_return:.2f}%")
            print(f"Bot Return: {total_return:.2f}%")
            print(f"Difference: {total_return - btc_return:.2f}%")
            
            if total_return > btc_return:
                print(f"\n[+] Bot outperformed BTC by {total_return - btc_return:.2f}%")
            else:
                print(f"\n[-] Bot underperformed BTC by {btc_return - total_return:.2f}%")
    
    print(f"\n{'='*70}")
    print("KEY OBSERVATIONS")
    print(f"{'='*70}\n")
    
    if len(sell_trades) == 0 and len(buy_trades) > 0:
        print("[!] WARNING: Bot opened positions but never sold them!")
        print("   - This suggests sell signals aren't being generated")
        print("   - Or stop-loss/take-profit logic isn't working")
        print("   - Positions may have lost value while being held\n")
    
    max_positions = results["positions"].max()
    print(f"Maximum positions held: {max_positions}")
    if max_positions > 5:
        print("[!] Bot opened many positions - may be over-diversifying")
    
    print(f"\nPortfolio value trend:")
    print(f"  - Peak: ${results['equity'].max():,.2f}")
    print(f"  - Trough: ${results['equity'].min():,.2f}")
    print(f"  - Final: ${results['equity'].iloc[-1]:,.2f}")

if __name__ == "__main__":
    generate_summary()

