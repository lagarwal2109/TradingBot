"""Utility functions for the trading bot."""

from typing import Dict


def to_roostoo_pair(pair: str) -> str:
    """Convert pair format to Roostoo format.
    
    Args:
        pair: Pair like "BTCUSD" or "BTC/USD"
        
    Returns:
        Roostoo format: "BTC/USD"
    """
    if "/" in pair:
        return pair
    
    # Assume USD quote
    if pair.endswith("USD"):
        base = pair[:-3]
        return f"{base}/USD"
    
    return pair


def from_roostoo_pair(pair: str) -> str:
    """Convert Roostoo pair format to internal format.
    
    Args:
        pair: Roostoo format like "BTC/USD"
        
    Returns:
        Internal format: "BTCUSD"
    """
    return pair.replace("/", "")


def normalize_asset(symbol: str) -> str:
    """Extract base asset from various formats.
    
    Args:
        symbol: "BTC", "BTCUSD", "BTC/USD", "BTCUSDT"
        
    Returns:
        Base asset: "BTC"
    """
    symbol = symbol.replace("/", "")
    
    if symbol.endswith("USDT"):
        return symbol[:-4]
    elif symbol.endswith("USD"):
        return symbol[:-3]
    
    return symbol


def filter_zero_balances(balances: Dict[str, float], threshold: float = 0.0001) -> Dict[str, float]:
    """Filter out zero or near-zero balances.
    
    Args:
        balances: Dictionary of asset -> balance
        threshold: Minimum balance to include (default: 0.0001)
        
    Returns:
        Filtered dictionary with only non-zero balances
    """
    return {asset: amount for asset, amount in balances.items() if amount >= threshold}