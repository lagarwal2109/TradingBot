"""Roostoo API v3 client implementation (correct version)."""

import time
import hmac
import hashlib
import requests
from typing import Dict, List, Optional, Any, Union
from urllib.parse import urlencode
from pydantic import BaseModel
from bot.config import get_config


class OrderResponse(BaseModel):
    """Order response model."""
    order_id: int
    pair: str
    side: str
    type: str
    status: str
    price: float
    quantity: float
    filled_quantity: float
    filled_aver_price: float
    

class TickerData(BaseModel):
    """Ticker data model."""
    pair: str
    last_price: float
    max_bid: float
    min_ask: float
    change: float
    coin_trade_value: float
    unit_trade_value: float


class ExchangeInfo(BaseModel):
    """Exchange pair information."""
    pair: str
    coin: str
    coin_full_name: str
    unit: str
    unit_full_name: str
    can_trade: bool
    price_precision: int
    amount_precision: int
    mini_order: float
    
    @property
    def is_tradable(self) -> bool:
        """Check if pair is tradable."""
        return self.can_trade


class RoostooV3Client:
    """Client for Roostoo API v3 (correct implementation)."""
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, base_url: Optional[str] = None):
        """Initialize the Roostoo v3 client."""
        config = get_config()
        self.api_key = api_key or config.api_key
        self.api_secret = api_secret or config.api_secret
        self.base_url = base_url or config.base_url
        
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "roostoo-sharpe-bot/1.0.0"
        })
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.2  # 200ms between requests (increased to avoid 429 errors)
    
    def _get_timestamp(self) -> str:
        """Get current timestamp as 13-digit string."""
        return str(int(time.time() * 1000))
    
    def _sign_request(self, params: Dict[str, Any]) -> str:
        """Create HMAC-SHA256 signature."""
        # Sort parameters and create query string
        sorted_keys = sorted(params.keys())
        total_params = "&".join(f"{k}={params[k]}" for k in sorted_keys)
        
        # Create signature
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            total_params.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        
        return signature, total_params
    
    def _request(self, method: str, endpoint: str, signed: bool = False, max_retries: int = 3, **kwargs) -> Dict[str, Any]:
        """Make HTTP request to the API with retry logic for rate limiting."""
        url = f"{self.base_url}{endpoint}"
        
        # Retry logic for rate limiting (429 errors)
        for attempt in range(max_retries):
            # Rate limiting - wait between requests
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.min_request_interval:
                time.sleep(self.min_request_interval - time_since_last)
            
            try:
                request_kwargs = kwargs.copy()
                
                if signed:
                    # Add timestamp
                    params = request_kwargs.get("params", {})
                    if "data" in request_kwargs:
                        params.update(request_kwargs["data"])
                        request_kwargs.pop("data")
                    
                    params["timestamp"] = self._get_timestamp()
                    
                    # Create signature
                    signature, total_params = self._sign_request(params)
                    
                    # Set headers
                    headers = request_kwargs.get("headers", {})
                    headers.update({
                        "RST-API-KEY": self.api_key,
                        "MSG-SIGNATURE": signature
                    })
                    
                    if method.upper() == "POST":
                        headers["Content-Type"] = "application/x-www-form-urlencoded"
                        request_kwargs["data"] = total_params
                        request_kwargs["headers"] = headers
                    else:
                        request_kwargs["params"] = params
                        request_kwargs["headers"] = headers
                
                # Make request
                response = self.session.request(method, url, **request_kwargs)
                
                # Handle 429 (Too Many Requests) with exponential backoff
                if response.status_code == 429:
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) * 0.5  # Exponential backoff: 0.5s, 1s, 2s
                        time.sleep(wait_time)
                        continue
                    else:
                        response.raise_for_status()
                
                response.raise_for_status()
                
                # Update last request time for rate limiting
                self.last_request_time = time.time()
                
                result = response.json()
                
                # Check for API-level errors (but ignore "no order matched" and similar)
                if not result.get("Success", True):
                    err_msg = result.get('ErrMsg', 'Unknown error')
                    # Don't raise for expected "no results" messages
                    if err_msg not in ["no order matched", "no pending order under this account", "no order canceled"]:
                        raise Exception(f"API Error: {err_msg}")
                
                return result
                    
            except requests.exceptions.HTTPError as e:
                if e.response and e.response.status_code == 429:
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) * 0.5
                        time.sleep(wait_time)
                        continue
                raise
            except Exception as e:
                if attempt < max_retries - 1 and "429" in str(e):
                    wait_time = (2 ** attempt) * 0.5
                    time.sleep(wait_time)
                    continue
                raise
        
        # Should not reach here, but just in case
        raise Exception("Max retries exceeded")
    
    # Public endpoints
    
    def server_time(self) -> int:
        """Get server time."""
        response = self._request("GET", "/v3/serverTime")
        return response.get("ServerTime", int(time.time() * 1000))
    
    def exchange_info(self) -> List[ExchangeInfo]:
        """Get exchange information."""
        response = self._request("GET", "/v3/exchangeInfo")
        
        trade_pairs = response.get("TradePairs", {})
        
        infos = []
        for pair_name, pair_data in trade_pairs.items():
            infos.append(ExchangeInfo(
                pair=pair_name,
                coin=pair_data["Coin"],
                coin_full_name=pair_data["CoinFullName"],
                unit=pair_data["Unit"],
                unit_full_name=pair_data["UnitFullName"],
                can_trade=pair_data["CanTrade"],
                price_precision=pair_data["PricePrecision"],
                amount_precision=pair_data["AmountPrecision"],
                mini_order=pair_data["MiniOrder"]
            ))
        
        return infos
    
    def ticker(self, pair: Optional[str] = None) -> Union[TickerData, List[TickerData]]:
        """Get ticker data.
        
        Args:
            pair: Trading pair with slash (e.g., "BTC/USD"). If None, returns all.
        """
        params = {"timestamp": self._get_timestamp()}
        if pair:
            params["pair"] = pair
            
        response = self._request("GET", "/v3/ticker", params=params)
        
        data = response.get("Data", {})
        
        if pair:
            if pair in data:
                ticker_data = data[pair]
                return TickerData(
                    pair=pair,
                    last_price=ticker_data["LastPrice"],
                    max_bid=ticker_data["MaxBid"],
                    min_ask=ticker_data["MinAsk"],
                    change=ticker_data["Change"],
                    coin_trade_value=ticker_data["CoinTradeValue"],
                    unit_trade_value=ticker_data["UnitTradeValue"]
                )
        else:
            tickers = []
            for pair_name, ticker_data in data.items():
                tickers.append(TickerData(
                    pair=pair_name,
                    last_price=ticker_data["LastPrice"],
                    max_bid=ticker_data["MaxBid"],
                    min_ask=ticker_data["MinAsk"],
                    change=ticker_data["Change"],
                    coin_trade_value=ticker_data["CoinTradeValue"],
                    unit_trade_value=ticker_data["UnitTradeValue"]
                ))
            return tickers
    
    # Signed endpoints
    
    def balance(self) -> Dict[str, float]:
        """Get account balances."""
        response = self._request("GET", "/v3/balance", signed=True)
        
        # API returns SpotWallet, not Wallet
        wallet = response.get("SpotWallet", {})
        balances = {}
        
        for asset, balance_data in wallet.items():
            if isinstance(balance_data, dict):
                balances[asset] = float(balance_data.get("Free", 0))
            else:
                balances[asset] = float(balance_data)
        
        return balances
    
    def place_order(
        self,
        pair: str,
        side: str,
        type: str,
        quantity: float,
        price: Optional[float] = None
    ) -> OrderResponse:
        """Place a new order.
        
        Args:
            pair: Trading pair with slash (e.g., "BTC/USD")
            side: "BUY" or "SELL"
            type: "MARKET" or "LIMIT"
            quantity: Order amount
            price: Price for LIMIT orders
        """
        params = {
            "pair": pair,
            "side": side.upper(),
            "type": type.upper(),
            "quantity": str(quantity)
        }
        
        if type.upper() == "LIMIT" and price is not None:
            params["price"] = str(price)
        
        response = self._request("POST", "/v3/place_order", signed=True, data=params)
        
        order_detail = response.get("OrderDetail", {})
        
        return OrderResponse(
            order_id=order_detail["OrderID"],
            pair=order_detail["Pair"],
            side=order_detail["Side"],
            type=order_detail["Type"],
            status=order_detail["Status"],
            price=order_detail.get("Price", 0),
            quantity=order_detail["Quantity"],
            filled_quantity=order_detail["FilledQuantity"],
            filled_aver_price=order_detail["FilledAverPrice"]
        )
    
    def cancel_order(self, order_id: Optional[int] = None, pair: Optional[str] = None) -> List[int]:
        """Cancel order(s).
        
        Args:
            order_id: Specific order ID to cancel
            pair: Cancel all pending orders for this pair
            
        Returns:
            List of cancelled order IDs
        """
        params = {}
        if order_id:
            params["order_id"] = str(order_id)
        elif pair:
            params["pair"] = pair
            
        response = self._request("POST", "/v3/cancel_order", signed=True, data=params)
        
        return response.get("CanceledList", [])
    
    def query_order(
        self,
        order_id: Optional[int] = None,
        pair: Optional[str] = None,
        pending_only: bool = False
    ) -> List[OrderResponse]:
        """Query orders.
        
        Args:
            order_id: Specific order ID
            pair: Filter by pair
            pending_only: Only return pending orders
        """
        params = {}
        if order_id:
            params["order_id"] = str(order_id)
        elif pair:
            params["pair"] = pair
            if pending_only:
                params["pending_only"] = "TRUE"
                
        response = self._request("POST", "/v3/query_order", signed=True, data=params)
        
        orders = response.get("OrderMatched", [])
        
        return [OrderResponse(
            order_id=o["OrderID"],
            pair=o["Pair"],
            side=o["Side"],
            type=o["Type"],
            status=o["Status"],
            price=o.get("Price", 0),
            quantity=o["Quantity"],
            filled_quantity=o["FilledQuantity"],
            filled_aver_price=o["FilledAverPrice"]
        ) for o in orders]
    
    def get_open_orders(self, pair: Optional[str] = None) -> List[OrderResponse]:
        """Get open orders."""
        return self.query_order(pair=pair, pending_only=True)
