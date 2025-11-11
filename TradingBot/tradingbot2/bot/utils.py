"""Utility functions for MACD trading bot."""

import sys
from pathlib import Path

# Add parent TradingBot to path to import shared modules
parent_bot_path = Path(__file__).parent.parent.parent
if str(parent_bot_path) not in sys.path:
    sys.path.insert(0, str(parent_bot_path))

# Import shared modules
try:
    from bot.roostoo import RoostooClient
    from bot.datastore import DataStore
except ImportError:
    # Fallback if imports fail
    RoostooClient = None
    DataStore = None


