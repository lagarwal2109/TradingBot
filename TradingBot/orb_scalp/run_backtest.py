"""Main script to run ORB + Scalp backtest."""

import argparse
import sys
from pathlib import Path
import yaml

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orb_scalp.backtester_enhanced import BacktesterEnhanced
from orb_scalp.evaluator import Evaluator
from orb_scalp.reporter import Reporter
from orb_scalp.diagnostic_reporter import DiagnosticReporter


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to config YAML file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run ORB + Scalp backtest")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config YAML file (default: config.yaml)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/data",
        help="Directory with CSV data files relative to TradingBot root (default: data/data)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results (default: results)"
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=10000.0,
        help="Initial capital (default: 10000)"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Number of days to backtest (default: use all data)"
    )
    parser.add_argument(
        "--max-bars",
        type=int,
        default=None,
        help="Maximum number of bars to process (for quick testing, default: no limit)"
    )
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        default=None,
        help="Symbols to trade (default: from config)"
    )
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(__file__).parent / args.config
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    config = load_config(config_path)
    
    # Get symbols
    if args.symbols:
        symbols = args.symbols
    else:
        symbols = config.get('symbols', ['BTCUSD', 'ETHUSD', 'SOLUSD', 'BNBUSD'])
    
    print(f"Symbols: {symbols}")
    print(f"Initial capital: ${args.capital:.2f}")
    if args.days:
        print(f"Days: {args.days}")
    print()
    
    # Resolve data directory
    script_dir = Path(__file__).resolve().parent
    candidate_dirs = []
    
    # 1. Absolute path provided
    if Path(args.data_dir).is_absolute():
        candidate_dirs.append(Path(args.data_dir))
    else:
        # 2. Relative to script directory (inner TradingBot)
        candidate_dirs.append(script_dir / args.data_dir)
        # 3. Relative to parent (outer TradingBot root)
        candidate_dirs.append(script_dir.parent / args.data_dir)
        # 4. Relative to grandparent (workspace root)
        candidate_dirs.append(script_dir.parent.parent / args.data_dir)
        # 5. Relative to current working directory
        candidate_dirs.append(Path.cwd() / args.data_dir)
    
    data_dir = None
    for candidate in candidate_dirs:
        if candidate.exists():
            data_dir = candidate
            break
    
    if data_dir is None:
        print("Error: Data directory not found.")
        print("Tried the following locations:")
        for candidate in candidate_dirs:
            print(f"  - {candidate}")
        print("\nPlease specify the correct path with --data-dir")
        sys.exit(1)
    
    print(f"Using data directory: {data_dir}")
    
    # Check if diagnostic mode
    diagnostic_mode = config.get('diagnostic_mode', False)
    if diagnostic_mode:
        print("⚠️  DIAGNOSTIC MODE: Using relaxed filters for testing")
    
    backtester = BacktesterEnhanced(
        data_dir=data_dir,
        params=config,
        initial_capital=args.capital
    )
    
    # Run backtest
    try:
        results = backtester.run_backtest(
            symbols=symbols,
            days=args.days,
            max_bars=args.max_bars
        )
    except Exception as e:
        print(f"Error during backtest: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Compute metrics
    print("\nComputing metrics...")
    evaluator = Evaluator(results['trades'], results['equity_curve'])
    metrics = evaluator.compute_metrics()
    
    # Print results
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    print(f"Initial Capital: ${metrics['initial_capital']:.2f}")
    print(f"Final Equity: ${metrics['final_equity']:.2f}")
    print(f"Total Return: {metrics['total_return_pct']:.2f}%")
    print(f"Annualized Return: {metrics['annualized_return_pct']:.2f}%")
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Win Rate: {metrics['win_rate_pct']:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_daily']:.2f}")
    print(f"Sortino Ratio: {metrics['sortino_daily']:.2f}")
    print(f"Calmar Ratio: {metrics['calmar_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"Average R: {metrics['avg_r']:.2f}")
    print(f"Median R: {metrics['median_r']:.2f}")
    print(f"Trades/Day: {metrics['trades_per_day']:.2f}")
    print(f"ORB P&L: ${metrics['orb_pnl']:.2f}")
    print(f"Scalp P&L: ${metrics['scalp_pnl']:.2f}")
    print(f"Total Fees: ${metrics['total_fees']:.2f}")
    print("="*60)
    
    # Save results
    output_dir = Path(__file__).parent / args.output_dir
    reporter = Reporter(output_dir)
    
    reporter.save_trades(results['trades'])
    reporter.save_equity_curve(results['equity_curve'])
    reporter.save_metrics(metrics)
    reporter.plot_equity_curve(results['equity_curve'], metrics)
    
    # Diagnostic reporting
    if 'gate_stats' in results:
        diag_reporter = DiagnosticReporter(output_dir)
        diag_reporter.report_gate_statistics(
            results['gate_stats'],
            results.get('gate_stats_df')
        )
        diag_reporter.report_portfolio_blocks(
            results.get('portfolio_blocks', {})
        )
    
    print("\nBacktest complete!")


if __name__ == "__main__":
    main()

