"""Reporter: saves results to CSV, JSON, and plots."""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
from datetime import datetime

from .portfolio import Trade


class Reporter:
    """Saves backtest results to files."""
    
    def __init__(self, output_dir: Path):
        """Initialize reporter.
        
        Args:
            output_dir: Output directory for results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_trades(self, trades: List[Trade], filename: str = "trades.csv"):
        """Save trades to CSV.
        
        Args:
            trades: List of Trade objects
            filename: Output filename
        """
        if not trades:
            return
        
        df = pd.DataFrame([
            {
                'symbol': t.symbol,
                'side': t.side,
                'signal_type': t.signal_type,
                'entry_dt': t.entry_dt,
                'exit_dt': t.exit_dt,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'quantity': t.quantity,
                'pnl_abs': t.pnl_abs,
                'pnl_pct': t.pnl_pct,
                'pnl_r': t.pnl_r,
                'fees': t.fees,
                'exit_reason': t.exit_reason,
                'risk_dollars': t.risk_dollars
            }
            for t in trades
        ])
        
        filepath = self.output_dir / filename
        df.to_csv(filepath, index=False)
        print(f"Trades saved to: {filepath}")
    
    def save_equity_curve(self, equity_curve: List, filename: str = "equity_curve.csv"):
        """Save equity curve to CSV.
        
        Args:
            equity_curve: List of (dt, equity) tuples
            filename: Output filename
        """
        if not equity_curve:
            return
        
        df = pd.DataFrame(equity_curve, columns=['dt', 'equity'])
        df = df.set_index('dt')
        
        filepath = self.output_dir / filename
        df.to_csv(filepath)
        print(f"Equity curve saved to: {filepath}")
    
    def save_metrics(self, metrics: Dict, filename: str = "metrics.json"):
        """Save metrics to JSON.
        
        Args:
            metrics: Dictionary of metrics
            filename: Output filename
        """
        # Convert numpy types to native Python types for JSON
        metrics_clean = {}
        for key, value in metrics.items():
            if isinstance(value, (np.integer, np.floating)):
                metrics_clean[key] = float(value)
            elif isinstance(value, dict):
                metrics_clean[key] = {k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                                     for k, v in value.items()}
            else:
                metrics_clean[key] = value
        
        metrics_clean['timestamp'] = datetime.now().isoformat()
        
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(metrics_clean, f, indent=2)
        print(f"Metrics saved to: {filepath}")
    
    def plot_equity_curve(
        self,
        equity_curve: List,
        metrics: Dict,
        filename: str = "equity_curve.png"
    ):
        """Plot equity curve.
        
        Args:
            equity_curve: List of (dt, equity) tuples
            metrics: Dictionary of metrics
            filename: Output filename
        """
        if not equity_curve:
            return
        
        df = pd.DataFrame(equity_curve, columns=['dt', 'equity'])
        df = df.set_index('dt')
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Equity curve
        axes[0].plot(df.index, df['equity'], linewidth=2, label='Equity')
        axes[0].axhline(y=metrics.get('initial_capital', 0), color='r', linestyle='--', label='Initial Capital')
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Equity ($)')
        axes[0].set_title(
            f"Equity Curve\n"
            f"Return: {metrics.get('total_return_pct', 0):.2f}% | "
            f"Sharpe: {metrics.get('sharpe_daily', 0):.2f} | "
            f"Max DD: {metrics.get('max_drawdown_pct', 0):.2f}%"
        )
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Drawdown
        df['cummax'] = df['equity'].cummax()
        df['drawdown'] = (df['equity'] - df['cummax']) / df['cummax'] * 100
        
        axes[1].fill_between(df.index, df['drawdown'], 0, alpha=0.3, color='red', label='Drawdown')
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Drawdown (%)')
        axes[1].set_title('Drawdown Chart')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=150)
        plt.close()
        print(f"Equity curve plot saved to: {filepath}")



