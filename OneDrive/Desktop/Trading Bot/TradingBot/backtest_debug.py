#!/usr/bin/env python3
"""Debug backtest to see what's happening."""

import pandas as pd
import numpy as np
from pathlib import Path

# Load data
data_dir = Path("data")
all_data = {}

for csv_file in data_dir.glob("*.csv"):
    if csv_file.stem == "state":
        continue
    pair = csv_file.stem
    df = pd.read_csv(csv_file)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("timestamp").sort_index()
    if len(df) > 48:
        all_data[pair] = df

print(f"Loaded {len(all_data)} pairs")

# Simulate simple strategy
initial_capital = 10000
cash = initial_capital
position = None
position_amount = 0
max_position_pct = 0.4

# Use timestamps from BTC
btc_timestamps = all_data["BTCUSD"].index.tolist()

print(f"\nSimulating {len(btc_timestamps)} periods...")
print(f"Starting capital: ${initial_capital:,.2f}\n")

trade_count = 0

for i in range(48, len(btc_timestamps), 1):  # Start after window, check every period
    timestamp = btc_timestamps[i]
    
    # Get current prices
    prices = {}
    sharpe_scores = {}
    
    for pair, df in all_data.items():
        hist = df[df.index <= timestamp]
        if len(hist) < 48:
            continue
            
        window = hist.tail(48)
        log_rets = np.log(window["price"] / window["price"].shift(1))
        
        mean_ret = log_rets.mean()
        std_ret = log_rets.std()
        sharpe = mean_ret / (std_ret + 1e-10) if std_ret > 0 else 0
        
        # Momentum
        if len(window) > 10:
            momentum = (window["price"].iloc[-1] - window["price"].iloc[-11]) / window["price"].iloc[-11]
        else:
            momentum = 0
            
        prices[pair] = float(window["price"].iloc[-1])
        
        if sharpe > 0 and momentum > 0:
            sharpe_scores[pair] = sharpe
    
    # Select best
    target = max(sharpe_scores, key=sharpe_scores.get) if sharpe_scores else None
    
    # Calculate portfolio value
    if position and position in prices:
        portfolio_value = cash + position_amount * prices[position]
    else:
        portfolio_value = cash
    
    # Trade logic
    if target != position:
        # Close position
        if position and position in prices:
            sell_value = position_amount * prices[position]
            fee = sell_value * 0.00155  # 0.155% total cost
            cash += sell_value - fee
            position_amount = 0
            position = None
            trade_count += 1
            print(f"[{i}] Sold at {timestamp.strftime('%Y-%m-%d %H:%M')}, portfolio: ${cash:,.2f}")
        
        # Open position
        if target and cash > 100:  # Minimum $100 to trade
            target_value = min(cash * max_position_pct, cash - 10)  # Keep $10 as buffer
            fee = target_value * 0.00155
            
            if cash >= target_value + fee:
                position_amount = target_value / prices[target]
                cash -= (target_value + fee)
                position = target
                trade_count += 1
                portfolio_value = cash + position_amount * prices[target]
                print(f"[{i}] Bought {target} at {timestamp.strftime('%Y-%m-%d %H:%M')}, portfolio: ${portfolio_value:,.2f}")

# Final value
if position and position in prices:
    final_value = cash + position_amount * prices[position]
else:
    final_value = cash

print(f"\n" + "="*50)
print(f"Total trades: {trade_count}")
print(f"Final value: ${final_value:,.2f}")
print(f"Return: {(final_value/initial_capital - 1)*100:.2f}%")
print(f"Final cash: ${cash:,.2f}")
if position:
    print(f"Position: {position_amount:.6f} {position} @ ${prices[position]:.2f}")
