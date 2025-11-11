"""Diagnostic reporter for gate analysis."""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List
from collections import defaultdict


class DiagnosticReporter:
    """Reports gate statistics and filter analysis."""
    
    def __init__(self, output_dir: Path):
        """Initialize diagnostic reporter.
        
        Args:
            output_dir: Output directory
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def report_gate_statistics(
        self,
        gate_stats: Dict,
        gate_stats_df: pd.DataFrame = None
    ):
        """Report gate pass/fail statistics.
        
        Args:
            gate_stats: Dictionary of gate statistics from SignalEngine
            gate_stats_df: DataFrame with per-bar gate tracking
        """
        print("\n" + "="*60)
        print("GATE STATISTICS")
        print("="*60)
        
        if not gate_stats or all(len(stats) == 0 for stats in gate_stats.values()):
            print("\n⚠️  WARNING: No gate statistics collected!")
            print("This means no bars even reached the first gate.")
            print("Possible reasons:")
            print("  - All bars failing trade_hours filter")
            print("  - All bars failing gap_ok filter")
            print("  - All bars failing warmup_ok filter")
            print("  - Data format issues")
            print("\nCheck your data and config settings.")
        else:
            for symbol, stats in gate_stats.items():
                if stats:
                    print(f"\n{symbol}:")
                    for gate_name, count in sorted(stats.items()):
                        print(f"  {gate_name}: {count}")
        
        # Save to JSON
        filepath = self.output_dir / "gate_statistics.json"
        with open(filepath, 'w') as f:
            json.dump(gate_stats, f, indent=2, default=str)
        print(f"\nGate statistics saved to: {filepath}")
        
        # If we have per-bar gate tracking, analyze cumulative pass rates
        if gate_stats_df is not None and len(gate_stats_df) > 0:
            self._analyze_cumulative_gates(gate_stats_df)
    
    def _analyze_cumulative_gates(self, df: pd.DataFrame):
        """Analyze cumulative gate pass rates.
        
        Args:
            df: DataFrame with gate tracking columns
        """
        print("\n" + "="*60)
        print("CUMULATIVE GATE PASS RATES")
        print("="*60)
        
        # Count total bars
        total_bars = len(df)
        print(f"\nTotal bars analyzed: {total_bars}")
        
        # For ORB gates
        if 'gate1_basic' in df.columns:
            print("\nORB Gates:")
            gate1_pass = df['gate1_basic'].sum() if 'gate1_basic' in df.columns else 0
            print(f"  Gate 1 (Basic): {gate1_pass}/{total_bars} ({100*gate1_pass/total_bars:.1f}%)")
            
            if 'gate2_or_valid' in df.columns:
                gate2_pass = df['gate2_or_valid'].sum()
                print(f"  Gate 2 (OR Valid): {gate2_pass}/{total_bars} ({100*gate2_pass/total_bars:.1f}%)")
            
            if 'gate3_breakout' in df.columns:
                gate3_pass = df['gate3_breakout'].sum()
                print(f"  Gate 3 (Breakout): {gate3_pass}/{total_bars} ({100*gate3_pass/total_bars:.1f}%)")
            
            if 'gate4_vol_spike' in df.columns:
                gate4_pass = df['gate4_vol_spike'].sum()
                print(f"  Gate 4 (Vol Spike): {gate4_pass}/{total_bars} ({100*gate4_pass/total_bars:.1f}%)")
            
            if 'gate5_rsi' in df.columns:
                gate5_pass = df['gate5_rsi'].sum()
                print(f"  Gate 5 (RSI): {gate5_pass}/{total_bars} ({100*gate5_pass/total_bars:.1f}%)")
            
            if 'gate6_retry' in df.columns:
                gate6_pass = df['gate6_retry'].sum()
                print(f"  Gate 6 (Retry): {gate6_pass}/{total_bars} ({100*gate6_pass/total_bars:.1f}%)")
        
        # For Scalp gates
        if 'gate2_rv' in df.columns:
            print("\nScalp Gates:")
            gate1_pass = df['gate1_basic'].sum() if 'gate1_basic' in df.columns else 0
            print(f"  Gate 1 (Basic): {gate1_pass}/{total_bars} ({100*gate1_pass/total_bars:.1f}%)")
            
            gate2_pass = df['gate2_rv'].sum()
            print(f"  Gate 2 (RV): {gate2_pass}/{total_bars} ({100*gate2_pass/total_bars:.1f}%)")
            
            if 'gate3_daily_limit' in df.columns:
                gate3_pass = df['gate3_daily_limit'].sum()
                print(f"  Gate 3 (Daily Limit): {gate3_pass}/{total_bars} ({100*gate3_pass/total_bars:.1f}%)")
            
            if 'gate4_ema_stack' in df.columns:
                gate4_pass = df['gate4_ema_stack'].sum()
                print(f"  Gate 4 (EMA Stack): {gate4_pass}/{total_bars} ({100*gate4_pass/total_bars:.1f}%)")
            
            if 'gate5_rsi' in df.columns:
                gate5_pass = df['gate5_rsi'].sum()
                print(f"  Gate 5 (RSI): {gate5_pass}/{total_bars} ({100*gate5_pass/total_bars:.1f}%)")
            
            if 'gate6_no_chase' in df.columns:
                gate6_pass = df['gate6_no_chase'].sum()
                print(f"  Gate 6 (No Chase): {gate6_pass}/{total_bars} ({100*gate6_pass/total_bars:.1f}%)")
        
        # Save detailed gate tracking
        filepath = self.output_dir / "gate_tracking.csv"
        df.to_csv(filepath, index=False)
        print(f"\nDetailed gate tracking saved to: {filepath}")
    
    def report_portfolio_blocks(
        self,
        block_reasons: Dict[str, int]
    ):
        """Report portfolio-level blocking reasons.
        
        Args:
            block_reasons: Dictionary mapping reason to count
        """
        if not block_reasons:
            return
        
        print("\n" + "="*60)
        print("PORTFOLIO BLOCKING REASONS")
        print("="*60)
        
        for reason, count in sorted(block_reasons.items(), key=lambda x: x[1], reverse=True):
            print(f"  {reason}: {count}")
        
        filepath = self.output_dir / "portfolio_blocks.json"
        with open(filepath, 'w') as f:
            json.dump(block_reasons, f, indent=2)
        print(f"\nPortfolio blocks saved to: {filepath}")

