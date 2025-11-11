"""
Multi-Asset Linear Momentum Trading Strategy.

Distributes capital evenly across assets whose linear momentum models
predict positive returns (after fees). Uses provided coefficients per asset.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from bot.roostoo_v3 import RoostooV3Client

# ---------------------------------------------------------------------------
# Model coefficients supplied by user
# ---------------------------------------------------------------------------

COEFFICIENTS: Dict[str, Dict[str, float]] = {
    "SHIB": {"intercept": 0.0, "close": 0.999213, "change": 0.000002, "accel": 0.0},
    "PEPE": {"intercept": 0.0, "close": 0.995480, "change": -0.0, "accel": 0.0},
    "FLOKI": {"intercept": 0.0, "close": 0.994872, "change": -0.000013, "accel": 0.0},
    "PUMP": {"intercept": 0.000027, "close": 0.993088, "change": -0.000242, "accel": -0.0},
    "LINEA": {"intercept": 0.000055, "close": 0.995467, "change": 0.002819, "accel": 0.0},
    "BMT": {"intercept": 0.000075, "close": 0.997834, "change": 0.001978, "accel": 0.0},
    "HEMI": {"intercept": 0.000327, "close": 0.990554, "change": 0.003792, "accel": 0.0},
    "PLUME": {"intercept": 0.000184, "close": 0.996370, "change": 0.008758, "accel": 0.0},
    "BIO": {"intercept": 0.000299, "close": 0.996173, "change": -0.003186, "accel": -0.0},
    "DOGE": {"intercept": 0.000359, "close": 0.997896, "change": 0.048417, "accel": 0.000001},
    "HBAR": {"intercept": 0.000935, "close": 0.994525, "change": 0.034989, "accel": 0.0},
    "SEI": {"intercept": 0.000356, "close": 0.997936, "change": 0.032477, "accel": 0.0},
    "POL": {"intercept": 0.000587, "close": 0.996676, "change": 0.000842, "accel": 0.000010},
    "XLM": {"intercept": 0.001244, "close": 0.995510, "change": 0.051012, "accel": 0.0},
    "ENA": {"intercept": 0.002780, "close": 0.991334, "change": 0.090865, "accel": 0.0},
    "CRV": {"intercept": 0.000706, "close": 0.998463, "change": 0.135038, "accel": -0.000002},
    "XPL": {"intercept": 0.000677, "close": 0.997656, "change": 0.055529, "accel": 0.000001},
    "SOMI": {"intercept": 0.000983, "close": 0.997356, "change": 0.020366, "accel": 0.0},
    "ONDO": {"intercept": 0.001252, "close": 0.998018, "change": 0.103188, "accel": 0.000001},
    "AVNT": {"intercept": 0.002451, "close": 0.995342, "change": 0.081259, "accel": 0.0},
    "WLD": {"intercept": 0.001080, "close": 0.998622, "change": 0.149263, "accel": -0.000004},
    "EIGEN": {"intercept": 0.003094, "close": 0.995994, "change": 0.156810, "accel": 0.000001},
    "XRP": {"intercept": 0.012575, "close": 0.994456, "change": 0.585774, "accel": 0.000001},
    "CAKE": {"intercept": 0.001820, "close": 0.999304, "change": 0.495942, "accel": 0.0},
    "PENDLE": {"intercept": 0.020088, "close": 0.992508, "change": 0.578506, "accel": -0.000005},
    "DOT": {"intercept": 0.002021, "close": 0.999384, "change": 0.799205, "accel": 0.000025},
    "APT": {"intercept": 0.005790, "close": 0.998110, "change": -0.068478, "accel": 0.000201},
    "LINK": {"intercept": 0.076413, "close": 0.994949, "change": 4.039716, "accel": -0.000009},
    "AVAX": {"intercept": 0.049694, "close": 0.997055, "change": 3.559542, "accel": -0.000001},
    "ICP": {"intercept": 0.013785, "close": 0.998225, "change": 0.828716, "accel": 0.000010},
    "UNI": {"intercept": 0.005265, "close": 0.999178, "change": -0.094452, "accel": 0.000208},
    "OMNI": {"intercept": -0.024282, "close": 1.002798, "change": -0.035417, "accel": -0.011436},
    "LTC": {"intercept": 0.080600, "close": 0.999187, "change": 34.389489, "accel": -0.000143},
    "AAVE": {"intercept": 1.179141, "close": 0.994104, "change": 43.899124, "accel": 0.000158},
    "BNB": {"intercept": 3.590551, "close": 0.996287, "change": 260.060321, "accel": 0.000072},
}

EXCLUDED_SYMBOLS = {"TRUMP", "BTC", "TAXO", "PAXG", "ZEC"}


def compute_performance_metrics(equity_series: pd.Series) -> Tuple[float, float, float]:
    """Compute Sharpe ratio, Calmar ratio, and CAGR from an equity curve."""
    equity = equity_series.astype(float)
    if len(equity) < 2:
        return np.nan, np.nan, np.nan

    returns = equity.pct_change().dropna()
    if returns.empty:
        return np.nan, np.nan, np.nan

    intervals = equity.index.to_series().diff().dropna()
    if not intervals.empty:
        seconds_per_period = intervals.dt.total_seconds().median()
    else:
        seconds_per_period = 300.0  # default to 5-minute data

    seconds_per_year = 365 * 24 * 60 * 60
    periods_per_year = seconds_per_year / seconds_per_period if seconds_per_period > 0 else 105120.0

    mean_return = returns.mean()
    std_return = returns.std(ddof=0)
    sharpe = np.nan
    if std_return > 0:
        sharpe = np.sqrt(periods_per_year) * mean_return / std_return

    duration_seconds = (equity.index[-1] - equity.index[0]).total_seconds()
    years = duration_seconds / seconds_per_year if duration_seconds > 0 else np.nan
    cagr = np.nan
    if years and years > 0:
        cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1

    running_max = equity.cummax()
    drawdowns = equity / running_max - 1
    max_drawdown = drawdowns.min()
    calmar = np.nan
    if cagr is not np.nan and max_drawdown < 0:
        calmar = cagr / abs(max_drawdown)

    return sharpe, calmar, cagr


@dataclass
class AssetState:
    symbol: str
    holdings: float = 0.0
    entry_price: Optional[float] = None
    last_trade_time: Optional[pd.Timestamp] = None


class MultiAssetMomentumTrader:
    def __init__(
        self,
        initial_capital: float = 50_000.0,
        transaction_fee: float = 0.001,
        data_directories: Optional[List[Path]] = None,
        min_notional: float = 10.0,
        min_trade_interval_minutes: int = 5,
        lookback_days: int = 15,
        api_client: Optional[RoostooV3Client] = None,
        start_date: Optional[pd.Timestamp] = None,
    ):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.transaction_fee = transaction_fee
        self.min_notional = min_notional
        self.min_trade_interval_minutes = min_trade_interval_minutes
        self.last_rebalance_time: Optional[pd.Timestamp] = None
        self.lookback_days = lookback_days
        self.api_client = api_client
        self.start_date = start_date

        self.data_dirs = data_directories or [
            Path("TradingBot/data/oct26_period"),
            Path("TradingBot/data/random_period"),
            Path("TradingBot/data/historical"),
            Path("TradingBot/data/alternative"),
            Path("data/oct26_period"),
            Path("data/random_period"),
            Path("data/historical"),
            Path("data/alternative"),
        ]

        self.symbols = sorted(
            symbol
            for symbol in COEFFICIENTS.keys()
            if symbol not in EXCLUDED_SYMBOLS
        )
        self.asset_states: Dict[str, AssetState] = {
            symbol: AssetState(symbol) for symbol in self.symbols
        }

        self.price_history: Dict[str, pd.DataFrame] = {}
        self.common_index: Optional[pd.DatetimeIndex] = None

        self.trades: List[Dict] = []
        self.equity_history: List[Dict] = []

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _fetch_from_api(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch historical data from Roostoo API for the last N days."""
        if not self.api_client or not self.lookback_days:
            return None
        
        try:
            # Convert symbol to pair format (e.g., "BTC" -> "BTC/USD")
            pair = f"{symbol}/USD"
            
            # Calculate time range
            if self.start_date is not None:
                start_time_dt = self.start_date.tz_localize("UTC") if self.start_date.tzinfo is None else self.start_date
                end_time_dt = start_time_dt + pd.Timedelta(days=self.lookback_days)
            else:
                end_time_dt = pd.Timestamp.utcnow().tz_localize("UTC")
                end_time_dt = end_time_dt + pd.Timedelta(minutes=5 - (end_time_dt.minute % 5 or 5))  # align to 5m
                start_time_dt = end_time_dt - pd.Timedelta(days=self.lookback_days)
            
            end_time = int(end_time_dt.timestamp() * 1000)
            start_time = int(start_time_dt.timestamp() * 1000)
            
            # Fetch klines
            klines = self.api_client.get_klines(
                pair=pair,
                interval="5m",
                start_time=start_time,
                end_time=end_time,
                limit=10000
            )
            
            if not klines:
                return None
            
            # Convert to DataFrame
            data = []
            for k in klines:
                data.append({
                    'open_time': pd.to_datetime(k['open_time'], unit='ms', utc=True),
                    'open': k['open'],
                    'high': k['high'],
                    'low': k['low'],
                    'close': k['close'],
                    'volume': k['volume']
                })
            
            df = pd.DataFrame(data)
            if df.empty:
                return None
            
            df.set_index('open_time', inplace=True)
            df.sort_index(inplace=True)
            
            return df
            
        except Exception as e:
            print(f"[WARN] Failed to fetch {symbol} from API: {e}")
            return None
    
    def _find_csv_path(self, symbol: str) -> Optional[Path]:
        possible_names = [f"{symbol}_5m.csv", f"{symbol}.csv"]
        for directory in self.data_dirs:
            for name in possible_names:
                path = directory / name
                if path.exists():
                    return path
        return None

    def _load_symbol(self, symbol: str) -> Optional[pd.DataFrame]:
        # Try to fetch from API first
        df = None
        if self.api_client:
            df = self._fetch_from_api(symbol)
        
        # Fallback to CSV if API fails
        if df is None or df.empty:
            path = self._find_csv_path(symbol)
            if path is None:
                print(f"[WARN] Data for {symbol} not found. Skipping.")
                return None
            df = pd.read_csv(path)

        # Normalise column names
        if "open_time" in df.columns:
            timestamp_col = "open_time"
        elif "timestamp" in df.columns:
            timestamp_col = "timestamp"
        else:
            raise ValueError(f"{symbol}: Missing timestamp column.")

        if "close" not in df.columns:
            # Some datasets use 'price'
            if "price" in df.columns:
                df["close"] = df["price"]
            else:
                raise ValueError(f"{symbol}: Missing close price column.")

        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df = df.sort_values(timestamp_col).set_index(timestamp_col)

        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["close"])

        # Momentum features (consistent with single-asset script)
        df["price_change_pct"] = df["close"].pct_change() * 100.0
        df["acceleration"] = df["price_change_pct"].diff()

        coeffs = COEFFICIENTS[symbol]
        df["predicted_price"] = (
            coeffs["intercept"]
            + coeffs["close"] * df["close"]
            + coeffs["change"] * df["price_change_pct"]
            + coeffs["accel"] * df["acceleration"]
        )

        # Use previous bar's prediction to decide current trade (avoid lookahead)
        df["predicted_price_for_trade"] = df["predicted_price"].shift(1)

        df = df.dropna(subset=["predicted_price_for_trade"])

        if self.start_date is not None:
            start_dt = self.start_date.tz_localize(df.index.tz) if df.index.tz is not None and self.start_date.tzinfo is None else self.start_date
            end_dt = start_dt + pd.Timedelta(days=self.lookback_days)
            df = df[(df.index >= start_dt) & (df.index < end_dt)]
        elif self.lookback_days and self.lookback_days > 0:
            cutoff = df.index.max() - pd.Timedelta(days=self.lookback_days)
            df = df[df.index >= cutoff]

        return df

    def load_all_data(self) -> None:
        for symbol in self.symbols:
            df = self._load_symbol(symbol)
            if df is not None and len(df) > 10:
                self.price_history[symbol] = df
            else:
                print(f"[WARN] Insufficient data for {symbol}. Skipping.")

        if not self.price_history:
            raise RuntimeError("No data loaded for any symbols.")

        # Align on common timestamps
        common_index = None
        for df in self.price_history.values():
            if common_index is None:
                common_index = df.index
            else:
                common_index = common_index.intersection(df.index)

        if common_index is None or len(common_index) == 0:
            raise RuntimeError("No overlapping timestamps across assets.")

        # For consistency use sorted index
        self.common_index = pd.DatetimeIndex(sorted(common_index))
        print(f"[OK] Loaded data for {len(self.price_history)} symbols.")
        print(f"[OK] Common timeline length: {len(self.common_index)} bars.")

    # ------------------------------------------------------------------
    # Trading helpers
    # ------------------------------------------------------------------

    def _get_equity(self, timestamp: pd.Timestamp) -> float:
        equity = self.cash
        for symbol, state in self.asset_states.items():
            if state.holdings <= 0:
                continue
            price = self.price_history[symbol].loc[timestamp, "close"]
            equity += state.holdings * price
        return equity

    def _rebalance(self, timestamp: pd.Timestamp) -> None:
        # Check if enough time has passed since last rebalance
        if self.last_rebalance_time is not None:
            time_since_last = timestamp - self.last_rebalance_time
            minutes_since_last = time_since_last.total_seconds() / 60.0
            if minutes_since_last < self.min_trade_interval_minutes:
                # Skip rebalancing, just record equity
                self.equity_history.append(
                    {
                        "timestamp": timestamp,
                        "equity": self._get_equity(timestamp),
                        "cash": self.cash,
                    }
                )
                return
        
        # Compute expected returns (based on previous prediction)
        candidates: List[Tuple[str, float, float]] = []
        price_snapshot: Dict[str, float] = {}

        for symbol, df in self.price_history.items():
            price = df.loc[timestamp, "close"]
            predicted = df.loc[timestamp, "predicted_price_for_trade"]
            expected_return = (predicted - price) / price
            price_snapshot[symbol] = price

            if expected_return > (self.transaction_fee * 2):
                candidates.append((symbol, price, expected_return))

        equity = self._get_equity(timestamp)
        if not candidates:
            # Liquidate everything (if holdings remain)
            for symbol, state in self.asset_states.items():
                if state.holdings <= 0:
                    continue
                self._sell(symbol, price_snapshot[symbol], state.holdings, timestamp, reason="Rebalance (no candidates)")
            self.equity_history.append(
                {
                    "timestamp": timestamp,
                    "equity": self._get_equity(timestamp),
                    "cash": self.cash,
                }
            )
            return

        # Determine allocation weights proportional to expected return
        total_expected = sum(max(c[2], 0) for c in candidates)
        if total_expected <= 0:
            total_expected = len(candidates)
            weights = {symbol: 1.0 / len(candidates) for symbol, _, _ in candidates}
        else:
            weights = {symbol: max(exp_ret, 0) / total_expected for symbol, _, exp_ret in candidates}

        # First ensure non-candidate holdings are sold
        candidate_symbols = {c[0] for c in candidates}
        for symbol, state in self.asset_states.items():
            if state.holdings > 0 and symbol not in candidate_symbols:
                self._sell(symbol, price_snapshot[symbol], state.holdings, timestamp, reason="Rebalance (remove asset)")

        # Rebalance candidates
        for symbol, price, _ in candidates:
            state = self.asset_states[symbol]
            current_value = state.holdings * price
            target_value = equity * weights.get(symbol, 0.0)
            delta_value = target_value - current_value

            if abs(delta_value) < self.min_notional:
                continue

            if delta_value > 0:
                # Need to buy
                qty = delta_value / price
                self._buy(symbol, price, qty, timestamp)
            else:
                qty = min(state.holdings, abs(delta_value) / price)
                if qty > 0:
                    self._sell(symbol, price, qty, timestamp, reason="Rebalance (trim)")

        self.last_rebalance_time = timestamp  # Update last rebalance time
        
        self.equity_history.append(
            {
                "timestamp": timestamp,
                "equity": self._get_equity(timestamp),
                "cash": self.cash,
            }
        )

    def _buy(self, symbol: str, price: float, quantity: float, timestamp: pd.Timestamp) -> None:
        if quantity <= 0:
            return

        notional = quantity * price
        if notional < self.min_notional:
            return

        fee = notional * self.transaction_fee
        total_cost = notional + fee

        if total_cost > self.cash:
            # Scale down to available cash
            quantity = max((self.cash / (1 + self.transaction_fee)) / price, 0.0)
            if quantity * price < self.min_notional:
                return
            notional = quantity * price
            fee = notional * self.transaction_fee
            total_cost = notional + fee

        self.cash -= total_cost
        state = self.asset_states[symbol]
        state.holdings += quantity
        state.entry_price = price  # latest entry price
        state.last_trade_time = timestamp

        self.trades.append(
            {
                "timestamp": timestamp,
                "symbol": symbol,
                "side": "BUY",
                "price": price,
                "quantity": quantity,
                "notional": notional,
                "fee": fee,
                "cash_after": self.cash,
                "holdings_after": state.holdings,
            }
        )

    def _sell(
        self,
        symbol: str,
        price: float,
        quantity: float,
        timestamp: pd.Timestamp,
        reason: str = "Sell",
    ) -> None:
        state = self.asset_states[symbol]
        if quantity <= 0 or state.holdings <= 0:
            return

        quantity = min(quantity, state.holdings)
        notional = quantity * price
        if notional < self.min_notional:
            return

        fee = notional * self.transaction_fee
        proceeds = notional - fee

        self.cash += proceeds
        state.holdings -= quantity
        state.last_trade_time = timestamp

        pnl = 0.0
        entry_price = state.entry_price or price
        pnl = (price - entry_price) * quantity - fee
        pnl_pct = ((price - entry_price) / entry_price) * 100 if entry_price else 0.0

        if state.holdings <= 0:
            state.entry_price = None

        self.trades.append(
            {
                "timestamp": timestamp,
                "symbol": symbol,
                "side": "SELL",
                "price": price,
                "quantity": quantity,
                "notional": notional,
                "fee": fee,
                "cash_after": self.cash,
                "holdings_after": state.holdings,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "reason": reason,
            }
        )

    # ------------------------------------------------------------------
    # Backtest & reporting
    # ------------------------------------------------------------------

    def backtest(self) -> pd.DataFrame:
        if self.common_index is None:
            self.load_all_data()

        assert self.common_index is not None
        
        # Reset state for backtest
        self.cash = self.initial_capital
        self.last_rebalance_time = None
        for state in self.asset_states.values():
            state.holdings = 0.0
            state.entry_price = None
            state.last_trade_time = None
        self.trades = []
        self.equity_history = []

        for timestamp in self.common_index:
            self._rebalance(timestamp)

        equity_df = pd.DataFrame(self.equity_history).set_index("timestamp")
        return equity_df

    def print_summary(self, equity_df: pd.DataFrame) -> None:
        final_equity = equity_df["equity"].iloc[-1]
        total_return = (final_equity / self.initial_capital - 1) * 100

        total_trades = len(self.trades)
        buys = sum(1 for t in self.trades if t["side"] == "BUY")
        sells = total_trades - buys

        realized_pnls = [t.get("pnl", 0.0) for t in self.trades if t["side"] == "SELL"]
        total_realized_pnl = sum(realized_pnls)
        fees_paid = sum(t["fee"] for t in self.trades)

        winners = [p for p in realized_pnls if p > 0]
        losers = [p for p in realized_pnls if p < 0]
        win_rate = (len(winners) / len(realized_pnls) * 100) if realized_pnls else 0.0
        
        # Calculate total profit and total loss
        total_profit = sum(winners) if winners else 0.0
        total_loss = sum(losers) if losers else 0.0

        print("\n" + "=" * 60)
        print("MULTI-ASSET MOMENTUM SUMMARY")
        print("=" * 60)
        print(f"Initial Capital:      ${self.initial_capital:,.2f}")
        print(f"Final Equity:          ${final_equity:,.2f}")
        print(f"Total Return:          {total_return:+.2f}%")
        print(f"Realized P&L:          ${total_realized_pnl:,.2f}")
        print(f"Total Fees Paid:       ${fees_paid:,.2f}")
        print(f"\nTotal Trades:          {total_trades} ({buys} buys, {sells} sells)")
        print(f"Win Rate:              {win_rate:.1f}%")
        if winners:
            print(f"Winning Trades:        {len(winners)}")
            print(f"Total Profit:          ${total_profit:,.2f}")
            print(f"Average Win:           ${np.mean(winners):,.2f}")
        if losers:
            print(f"Losing Trades:         {len(losers)}")
            print(f"Total Loss:            ${total_loss:,.2f}")
            print(f"Average Loss:          ${np.mean(losers):,.2f}")
        print("=" * 60)

        sharpe, calmar, cagr = compute_performance_metrics(equity_df["equity"])
        print("Performance Ratios:")
        if np.isnan(sharpe):
            print("  Sharpe Ratio:        n/a")
        else:
            print(f"  Sharpe Ratio:        {sharpe:.3f}")
        if np.isnan(calmar):
            print("  Calmar Ratio:        n/a")
        else:
            print(f"  Calmar Ratio:        {calmar:.3f}")
        if np.isnan(cagr):
            print("  CAGR:                n/a")
        else:
            print(f"  CAGR:                {cagr*100:+.2f}%")
        print("=" * 60)

    def plot_equity(self, equity_df: pd.DataFrame, save_path: Optional[Path] = None) -> None:
        plt.figure(figsize=(12, 6))
        plt.plot(equity_df.index, equity_df["equity"], label="Portfolio Equity", linewidth=2)
        plt.axhline(
            y=self.initial_capital,
            color="gray",
            linestyle="--",
            label="Initial Capital",
        )
        plt.title("Multi-Asset Momentum Portfolio Equity Curve", fontsize=14, fontweight="bold")
        plt.xlabel("Time")
        plt.ylabel("Equity (USD)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"[OK] Equity curve saved to {save_path}")
        plt.show()

    def save_outputs(self, equity_df: pd.DataFrame, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)

        equity_path = directory / "multi_asset_equity.csv"
        trades_path = directory / "multi_asset_trades.json"

        equity_df.to_csv(equity_path)
        with open(trades_path, "w") as f:
            json.dump(
                [
                    {k: (str(v) if isinstance(v, pd.Timestamp) else v) for k, v in trade.items()}
                    for trade in self.trades
                ],
                f,
                indent=2,
            )

        print(f"[OK] Saved equity history to {equity_path}")
        print(f"[OK] Saved trades to {trades_path}")


def main():
    print("\n" + "="*60)
    print("MULTI-ASSET LINEAR MOMENTUM TRADING")
    print("="*60)
    
    # Initialize API client
    client = None
    try:
        client = RoostooV3Client()
        print("[OK] Roostoo API client initialized")
    except Exception as e:
        print(f"[WARN] Could not initialize API client: {e}")
        print("[WARN] Will attempt to use CSV data as fallback")
    
    # Ask user for lookback window
    print("\n" + "="*60)
    print("DATA SOURCE SETTINGS")
    print("="*60)
    print("How many days back should the simulation cover? (default=15)")
    print("The bot will attempt to fetch real data from Roostoo API.")
    try:
        lookback_input = input("\nEnter lookback window in days (default=15): ").strip()
        lookback_days = int(lookback_input) if lookback_input else 15
        if lookback_days < 1:
            print("[WARN] Lookback days must be at least 1. Using 15 days.")
            lookback_days = 15
    except (ValueError, KeyboardInterrupt, EOFError):
        lookback_days = 15
        print("\nUsing default: 15 days")

    print("\nWould you like to analyse a specific historical window?")
    print("Leave blank to use the most recent period.")
    start_input = input("Enter custom start date (YYYY-MM-DD) or press Enter: ").strip()
    custom_start: Optional[pd.Timestamp] = None
    if start_input:
        try:
            custom_start = pd.to_datetime(start_input)
            print(f"[OK] Using custom start date: {custom_start.date()}")
        except ValueError:
            print("[WARN] Could not parse start date. Using most recent period.")
            custom_start = None

    # Ask user for trading frequency
    print("\nTRADING FREQUENCY SETTINGS")
    print("How often should the bot rebalance the portfolio?")
    print("(This is the minimum time between rebalancing)")
    print("\nExamples:")
    print("  - 5 minutes: Standard setting (default)")
    print("  - 10 minutes: Less frequent trading")
    print("  - 15 minutes: Conservative approach")
    print("  - 1 minute: More frequent (higher transaction costs)")
    
    try:
        freq_input = input("\nEnter minimum rebalance interval in minutes (default=5): ").strip()
        min_trade_interval = int(freq_input) if freq_input else 5
        if min_trade_interval < 1:
            print("[WARN] Minimum interval must be at least 1 minute. Using 1 minute.")
            min_trade_interval = 1
    except (ValueError, KeyboardInterrupt, EOFError):
        min_trade_interval = 5
        print("\nUsing default: 5 minutes")
    
    print(f"\n[OK] Rebalancing frequency set to: {min_trade_interval} minutes")
    
    trader = MultiAssetMomentumTrader(
        initial_capital=50_000.0, 
        transaction_fee=0.001,
        min_trade_interval_minutes=min_trade_interval,
        lookback_days=lookback_days,
        api_client=client,
        start_date=custom_start
    )
    trader.load_all_data()

    print(f"\nActive symbols: {', '.join(trader.price_history.keys())}")

    equity_df = trader.backtest()
    trader.print_summary(equity_df)

    figures_dir = Path("TradingBot/figures")
    trader.plot_equity(equity_df, save_path=figures_dir / "multi_asset_momentum_equity.png")
    trader.save_outputs(equity_df, figures_dir)


if __name__ == "__main__":
    main()


