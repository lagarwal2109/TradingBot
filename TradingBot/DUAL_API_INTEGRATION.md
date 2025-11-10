# Dual API Integration - Horus + Roostoo

The trading bot now uses **two APIs** working together:

## API Roles

### 🔵 Horus API (Market Data)
- **Purpose**: Real-time and historical market data
- **Endpoints Used**:
  - `get_ticker()` - Real-time price and volume data
  - `get_klines()` - Historical candlestick data
  - `get_exchange_info()` - Market information
  - `get_orderbook()` - Order book depth
  - `get_recent_trades()` - Recent trade history

### 🟢 Roostoo API (Trading Execution)
- **Purpose**: Order execution and account management
- **Endpoints Used**:
  - `balance()` - Account balances
  - `place_order()` - Submit buy/sell orders
  - `cancel_order()` - Cancel pending orders
  - `query_order()` - Check order status
  - `get_open_orders()` - List active orders

## Why Both APIs?

This separation provides several advantages:

1. **Specialized Services**: Each API focuses on what it does best
2. **Data Quality**: Horus provides comprehensive market data
3. **Trading Reliability**: Roostoo handles execution efficiently
4. **Redundancy**: If one API has issues, the other continues working
5. **Compliance**: Separates market data from trading operations

## Configuration

### Environment Variables (.env)

```bash
# Roostoo API (Trading Execution)
ROOSTOO_API_KEY=your_roostoo_api_key
ROOSTOO_API_SECRET=your_roostoo_api_secret
ROOSTOO_BASE_URL=https://api.roostoo.com

# Horus API (Market Data)
HORUS_API_KEY=1e7afb8bcb9ed5899b0cbeec8adead96f18d5620a2530c45b0ad5a54b5e9e6ad
HORUS_BASE_URL=https://api.horus.com
USE_HORUS_DATA=true
```

## How It Works

### Data Collection Flow
```
Horus API → get_ticker() → Ticker Data
                         ↓
                    DataStore → CSV Files
                         ↓
              Signal Generator → Trading Signals
```

### Trading Execution Flow
```
Trading Signal → Risk Manager → Position Size
                              ↓
               Roostoo API → place_order()
                              ↓
                         Order Executed
```

### Complete Cycle
```
1. Horus: Get market data (prices, volumes)
2. Bot: Analyze data, generate signals
3. Bot: Calculate position size
4. Roostoo: Get current balance
5. Roostoo: Execute trade
6. Bot: Track position
7. Horus: Monitor price for stops
8. Roostoo: Close position when needed
```

## Code Integration

### Enhanced Trading Engine

```python
# Initialize both clients
trading_client = RoostooClient()  # For orders
market_client = HorusClient(api_key=horus_key)  # For data

# Create engine with both
engine = EnhancedTradingEngine(
    trading_client=trading_client,
    market_data_client=market_client,
    datastore=datastore,
    risk_manager=risk_manager
)
```

### Usage in Strategy

```python
# Get market data from Horus
tickers = self.market_client.get_ticker()
prices = {t.symbol: t.price for t in tickers}

# Execute trade on Roostoo
balance = self.trading_client.balance()
order = self.trading_client.place_order(
    pair="BTCUSD",
    side="buy",
    type="market",
    amount=0.01
)
```

## API Mapping

### Symbol Naming
- **Horus**: Uses `symbol` field (e.g., "BTCUSDT")
- **Roostoo**: Uses `pair` field (e.g., "BTCUSD")
- **Bot**: Normalizes to common format

### Data Fields
| Data Type | Horus Field | Roostoo Field | Bot Uses |
|-----------|-------------|---------------|----------|
| Price | `price` | `price` | ✓ |
| Volume | `volume` | `volume_24h` | ✓ |
| Bid | `bid` | `bid` | ✓ |
| Ask | `ask` | `ask` | ✓ |

## Error Handling

The bot handles API failures gracefully:

1. **Horus Failure**: Continues with last known data
2. **Roostoo Failure**: Delays trades, retries with backoff
3. **Both Fail**: Activates kill switch, closes positions

## Testing the Integration

### 1. Test Horus Connection
```python
python -c "from bot.horus import HorusClient; from bot.config import get_config; c = get_config(); h = HorusClient(c.horus_api_key); print(h.ping())"
```

### 2. Test Roostoo Connection
```python
python -c "from bot.roostoo import RoostooClient; r = RoostooClient(); print(r.server_time())"
```

### 3. Test Data Collection
```bash
# Collect one minute of data
python run.py --mode collect --once
```

### 4. Test Trading
```bash
# Run one trading cycle
python run.py --mode enhanced --once
```

## Monitoring

Watch logs for API usage:

```bash
tail -f logs/bot.log | grep -E "Horus|Roostoo"
```

You should see:
- `"Initialized Roostoo (trading) and Horus (market data) clients"`
- `"Collected data for X pairs from Horus API"`
- `"Placed buy order"` (from Roostoo)

## Benefits for Competition

1. **Better Data**: Horus provides more comprehensive market data
2. **Faster Execution**: Specialized trading API
3. **Reliability**: Redundancy across two services
4. **Scalability**: Can add more data sources easily

## Troubleshooting

### Horus API Errors
- Check `HORUS_API_KEY` in .env
- Verify Horus API is accessible
- Check rate limits

### Roostoo API Errors
- Check `ROOSTOO_API_KEY` and `ROOSTOO_API_SECRET`
- Verify signature generation
- Check order parameters

### Symbol Mismatches
- Ensure symbol naming is consistent
- Check exchange_info mapping
- Verify pair exists on both platforms
