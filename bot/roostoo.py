"""Roostoo API client implementation."""

import time
import hmac
import hashlib
import json
from typing import Dict, List, Optional, Any, Union
from urllib.parse import urlencode
import requests
from pydantic import BaseModel
from bot.config import get_config


class OrderResponse(BaseModel):
    """Order response model."""
    id: str
    pair: str
    side: str
    type: str
    price: Optional[float]
    amount: Optional[float]
    status: str
    created_at: int


class TickerData(BaseModel):
    """Ticker data model."""
    pair: str
    price: float
    volume_24h: float
    change_24h: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    timestamp: int


class ExchangeInfo(BaseModel):
    """Exchange pair information."""
    pair: str
    base_currency: str
    quote_currency: str
    price_precision: int
    amount_precision: int
    mini_order: float
    status: str
    
    @property
    def is_tradable(self) -> bool:
        """Check if pair is tradable."""
        return self.status.lower() == "active"


class RoostooClient:
    """Client for interacting with the Roostoo API."""
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, base_url: Optional[str] = None):
        """Initialize the Roostoo client."""
        config = get_config()
        self.api_key = api_key or config.api_key
        self.api_secret = api_secret or config.api_secret
        self.base_url = base_url or config.base_url
        
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "roostoo-sharpe-bot/0.1.0"
        })
    
    def _get_timestamp(self) -> int:
        """Get current timestamp in milliseconds (13 digits)."""
        return int(time.time() * 1000)
    
    def _sign_request(self, params: Dict[str, Any]) -> str:
        """Create HMAC-SHA256 signature for signed requests."""
        # Sort parameters by key and create query string
        sorted_params = sorted(params.items())
        query_string = urlencode(sorted_params)
        
        # Create signature
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def _request(self, method: str, endpoint: str, signed: bool = False, **kwargs) -> Dict[str, Any]:
        """Make HTTP request to the API."""
        url = f"{self.base_url}{endpoint}"
        
        if signed:
            # Add timestamp to parameters
            timestamp = self._get_timestamp()
            
            # Get params from either params or json kwargs
            params = kwargs.get("params", {})
            if "json" in kwargs:
                params.update(kwargs["json"])
                kwargs.pop("json")
            
            params["timestamp"] = timestamp
            
            # Create signature
            signature = self._sign_request(params)
            
            # Add headers
            headers = kwargs.get("headers", {})
            headers.update({
                "X-API-KEY": self.api_key,
                "X-SIGNATURE": signature
            })
            kwargs["headers"] = headers
            
            # Ensure params are in the request
            if method.upper() == "GET":
                kwargs["params"] = params
            else:
                kwargs["json"] = params
        
        # Make request
        response = self.session.request(method, url, **kwargs)
        
        # Raise for HTTP errors
        response.raise_for_status()
        
        return response.json()
    
    # Public endpoints (no authentication required)
    
    def server_time(self) -> int:
        """Get server time in milliseconds."""
        response = self._request("GET", "/api/v1/time")
        return response["serverTime"]
    
    def exchange_info(self) -> List[ExchangeInfo]:
        """Get exchange information for all trading pairs."""
        response = self._request("GET", "/api/v1/exchangeInfo")
        return [ExchangeInfo(**info) for info in response["symbols"]]
    
    def ticker(self, pair: Optional[str] = None) -> Union[TickerData, List[TickerData]]:
        """Get ticker data for one or all pairs.
        
        Args:
            pair: Trading pair (e.g., "BTCUSD"). If None, returns all pairs.
        
        Returns:
            Single TickerData if pair specified, list of TickerData otherwise.
        """
        if pair:
            endpoint = f"/api/v1/ticker/{pair}"
            response = self._request("GET", endpoint)
            return TickerData(**response)
        else:
            endpoint = "/api/v1/ticker"
            response = self._request("GET", endpoint)
            return [TickerData(**ticker) for ticker in response]
    
    # Private endpoints (authentication required)
    
    def balance(self) -> Dict[str, float]:
        """Get account balances (SIGNED endpoint)."""
        response = self._request("GET", "/api/v1/account/balance", signed=True)
        return {asset["asset"]: float(asset["free"]) for asset in response["balances"]}
    
    def place_order(
        self,
        pair: str,
        side: str,
        type: str,
        price: Optional[float] = None,
        amount: Optional[float] = None
    ) -> OrderResponse:
        """Place a new order (SIGNED endpoint).
        
        Args:
            pair: Trading pair (e.g., "BTCUSD")
            side: "buy" or "sell"
            type: "market" or "limit"
            price: Price for limit orders
            amount: Order amount
        
        Returns:
            Order response with order details.
        """
        params = {
            "symbol": pair,
            "side": side.upper(),
            "type": type.upper(),
        }
        
        if amount is not None:
            params["quantity"] = amount
            
        if type.upper() == "LIMIT" and price is not None:
            params["price"] = price
        
        response = self._request("POST", "/api/v1/order", signed=True, json=params)
        return OrderResponse(**response)
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an order (SIGNED endpoint).
        
        Args:
            order_id: Order ID to cancel
        
        Returns:
            Cancellation response.
        """
        params = {"orderId": order_id}
        return self._request("DELETE", "/api/v1/order", signed=True, json=params)
    
    def query_order(self, order_id: str) -> OrderResponse:
        """Query order status (SIGNED endpoint).
        
        Args:
            order_id: Order ID to query
        
        Returns:
            Order details.
        """
        params = {"orderId": order_id}
        response = self._request("GET", "/api/v1/order", signed=True, params=params)
        return OrderResponse(**response)
    
    def get_open_orders(self, pair: Optional[str] = None) -> List[OrderResponse]:
        """Get all open orders (SIGNED endpoint).
        
        Args:
            pair: Optional trading pair filter
        
        Returns:
            List of open orders.
        """
        params = {}
        if pair:
            params["symbol"] = pair
            
        response = self._request("GET", "/api/v1/openOrders", signed=True, params=params)
        return [OrderResponse(**order) for order in response]
