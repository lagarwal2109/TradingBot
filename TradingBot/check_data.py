#!/usr/bin/env python3
"""Check data availability for backtesting."""

import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

data_dir = Path("data/data")
csv_files = list(data_dir.glob("*.csv"))
print(f"Found {len(csv_files)} CSV files in {data_dir}")

if csv_files:
    # Check a sample file
    sample_file = csv_files[0]
    df = pd.read_csv(sample_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    days = (df['timestamp'].max() - df['timestamp'].min()).days
    print(f"\nSample file: {sample_file.name}")
    print(f"  Records: {len(df)}")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  Total days: {days}")
    
    # Check how many files have at least 15 days
    files_with_15_days = 0
    for csv_file in csv_files:
        if csv_file.stem == "state":
            continue
        try:
            df = pd.read_csv(csv_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            if len(df) >= 100:
                days = (df['timestamp'].max() - df['timestamp'].min()).days
                if days >= 15:
                    files_with_15_days += 1
        except:
            pass
    
    print(f"\nFiles with >= 15 days of data: {files_with_15_days}")

