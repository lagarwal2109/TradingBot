"""CLI script for running MACD strategy backtests."""

import argparse
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Add parent directories to path
tradingbot2_path = Path(__file__).parent.parent
parent_bot_path = tradingbot2_path.parent
sys.path.insert(0, str(tradingbot2_path))
sys.path.insert(0, str(parent_bot_path))

from tradingbot2.bot.backtester import MACDBacktester
from tradingbot2.bot.config import get_config
from bot.datastore import DataStore


def main():
    """Run MACD strategy backtest."""
    parser = argparse.ArgumentParser(description="Run MACD strategy backtest")
    parser.add_argument(
        "--days",
        type=int,
        default=15,
        help="Number of days to backtest (default: 15)"
    )
    parser.add_argument(
        "--pairs",
        type=str,
        nargs="+",
        default=None,
        help="Trading pairs to backtest (default: all available)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="figures",
        help="Output directory for results (default: figures)"
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=10000.0,
        help="Initial capital (default: 10000)"
    )
    parser.add_argument(
        "--num-pairs",
        type=int,
        default=2,
        help="Number of pairs to test (default: 2, use 0 for all pairs)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config
    try:
        config = get_config()
    except ValueError as e:
        print(f"Error loading config: {e}")
        print("Make sure .env file exists in parent TradingBot directory")
        sys.exit(1)
    
    # Initialize backtester
    backtester = MACDBacktester(config=config, initial_capital=args.capital)
    
    # Get pairs
    if args.pairs:
        pairs = args.pairs
    else:
        # Get pairs with data - limit to 2 for quick testing
        datastore = DataStore(data_dir=config.data_dir)
        all_pairs = datastore.get_all_pairs_with_data()
        print(f"Found {len(all_pairs)} pairs with data")
        
        # Filter to pairs with at least some data
        valid_pairs = []
        for pair in all_pairs:
            df = datastore.read_minute_bars(pair)
            if len(df) >= 100:
                valid_pairs.append(pair)
        
        if len(valid_pairs) == 0:
            print(f"Error: No pairs have sufficient data (need at least 100 data points)")
            sys.exit(1)
        
        # Limit to specified number of pairs (0 = all pairs)
        if args.num_pairs == 0:
            pairs = valid_pairs
            print(f"Testing on all {len(pairs)} pairs: {', '.join(pairs[:10])}{'...' if len(pairs) > 10 else ''}")
        else:
            pairs = valid_pairs[:args.num_pairs]
            print(f"Testing on {len(pairs)} pairs: {', '.join(pairs)}")
    
    if args.pairs:
        print(f"Running backtest on {len(pairs)} pairs: {', '.join(pairs)}")
    print(f"Initial capital: ${args.capital:.2f}")
    print(f"Days: {args.days}")
    print()
    
    # Run backtest
    try:
        results = backtester.run_backtest(pairs=pairs, days=args.days)
    except Exception as e:
        print(f"Error during backtest: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Print results
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    print(f"Initial Capital: ${results['initial_capital']:.2f}")
    print(f"Final Equity: ${results['final_equity']:.2f}")
    print(f"Total Return: {results['total_return_pct']:.2f}%")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Winning Trades: {results['winning_trades']}")
    print(f"Losing Trades: {results['losing_trades']}")
    print(f"Win Rate: {results['win_rate_pct']:.2f}%")
    print(f"Total Fees: ${results['total_fees']:.2f}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Sortino Ratio: {results['sortino_ratio']:.2f}")
    print(f"Calmar Ratio: {results['calmar_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown_pct']:.2f}%")
    if results['avg_win'] != 0 or results['avg_loss'] != 0:
        print(f"Average Win: ${results['avg_win']:.2f}")
        print(f"Average Loss: ${results['avg_loss']:.2f}")
    print()
    print("DIAGNOSTICS:")
    print(f"  Mean 1R (dollars): ${results.get('mean_1r_dollars', 0):.2f}")
    print(f"  Median 1R (dollars): ${results.get('median_1r_dollars', 0):.2f}")
    print(f"  Size-weighted Avg Win: ${results.get('size_weighted_avg_win', 0):.2f}")
    print(f"  Size-weighted Avg Loss: ${results.get('size_weighted_avg_loss', 0):.2f}")
    print(f"  % Trades hitting +1R: {results.get('pct_hit_1r', 0):.1f}%")
    print(f"  % Trades hitting +2R: {results.get('pct_hit_2r', 0):.1f}%")
    print(f"  Flip Rate (reverse within 10 bars): {results.get('flip_rate', 0):.1f}%")
    print(f"  Avg Winner Holding: {results.get('avg_winner_holding_bars', 0):.1f} bars")
    print(f"  Avg Loser Holding: {results.get('avg_loser_holding_bars', 0):.1f} bars")
    print(f"  Worst 3 Trades Contribution: {results.get('worst_3_contribution_pct', 0):.1f}% of total P/L")
    print(f"  Median Realized R: {results.get('median_realized_r', 0):.2f}")
    if 'r_distribution' in results:
        print(f"  R-Distribution: {results['r_distribution']}")
    print("="*60)
    
    # Save trade log
    if len(results['trades']) > 0:
        trades_df = pd.DataFrame([
            {
                "pair": t.pair,
                "side": t.side,
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "amount": t.amount,
                "entry_value": t.entry_value,
                "exit_value": t.exit_value,
                "pnl": t.pnl,
                "pnl_pct": t.pnl_pct,
                "exit_reason": t.exit_reason,
                "fees": t.fees
            }
            for t in results['trades']
        ])
        
        trades_file = output_dir / "macd_backtest_trades.csv"
        trades_df.to_csv(trades_file, index=False)
        print(f"\nTrade log saved to: {trades_file}")
        print(f"Total trades: {len(trades_df)}")
    else:
        print("\nNo trades executed during backtest")
    
    # Save equity curve
    if len(results['equity_curve']) > 0:
        equity_df = pd.DataFrame(
            results['equity_curve'],
            columns=["timestamp", "equity"]
        )
        equity_df = equity_df.set_index("timestamp")
        
        equity_file = output_dir / "macd_backtest_equity.csv"
        equity_df.to_csv(equity_file)
        print(f"Equity curve saved to: {equity_file}")
        
        # Plot equity curve with metrics
        plt.figure(figsize=(14, 8))
        
        # Main equity curve
        plt.subplot(2, 1, 1)
        plt.plot(equity_df.index, equity_df["equity"], linewidth=2, label='Equity')
        plt.axhline(y=args.capital, color='r', linestyle='--', label='Initial Capital')
        plt.xlabel("Time")
        plt.ylabel("Equity ($)")
        plt.title(f"MACD Strategy Equity Curve\nTotal Return: {results['total_return_pct']:.2f}% | "
                  f"Sharpe: {results['sharpe_ratio']:.2f} | Sortino: {results['sortino_ratio']:.2f} | "
                  f"Calmar: {results['calmar_ratio']:.2f}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Metrics bar chart
        plt.subplot(2, 1, 2)
        metrics = ['Sharpe', 'Sortino', 'Calmar']
        values = [results['sharpe_ratio'], results['sortino_ratio'], results['calmar_ratio']]
        colors = ['blue', 'green', 'orange']
        bars = plt.bar(metrics, values, color=colors, alpha=0.7)
        plt.ylabel("Ratio")
        plt.title("Risk-Adjusted Return Ratios")
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        
        plot_file = output_dir / "macd_backtest_equity_curve.png"
        plt.savefig(plot_file, dpi=150)
        print(f"Equity curve plot saved to: {plot_file}")
        plt.close()
    
    # Save summary
    summary = {
        "initial_capital": results['initial_capital'],
        "final_equity": results['final_equity'],
        "total_return_pct": results['total_return_pct'],
        "total_trades": results['total_trades'],
        "winning_trades": results['winning_trades'],
        "losing_trades": results['losing_trades'],
        "win_rate_pct": results['win_rate_pct'],
        "total_fees": results['total_fees'],
        "sharpe_ratio": results['sharpe_ratio'],
        "sortino_ratio": results['sortino_ratio'],
        "calmar_ratio": results['calmar_ratio'],
        "max_drawdown_pct": results['max_drawdown_pct'],
        "avg_win": results['avg_win'],
        "avg_loss": results['avg_loss'],
        "pairs_tested": pairs,
        "days": args.days,
        "timestamp": datetime.now().isoformat()
    }
    
    import json
    summary_file = output_dir / "macd_backtest_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_file}")
    
    print("\nBacktest complete!")


if __name__ == "__main__":
    main()

