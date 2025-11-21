#!/usr/bin/env python3
"""Paper trading mode for volatility expansion strategy - simulates trades without executing real orders."""

import sys
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pydantic import BaseModel

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from bot.config import get_config
from bot.datastore import DataStore
from bot.signals_volatility_expansion import VolatilityExpansionSignalGenerator, VolatilityExpansionSignal
from bot.position_manager import PositionManager, ManagedPosition
from bot.risk import RiskManager
from bot.roostoo_v3 import RoostooV3Client, ExchangeInfo, TickerData, OrderResponse
from bot.engine_volatility import VolatilityExpansionEngine
from bot.utils import to_roostoo_pair, from_roostoo_pair


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('paper_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SimulatedOrder(BaseModel):
    """Simulated order for paper trading."""
    order_id: int
    pair: str
    side: str
    type: str
    status: str = "NEW"
    price: float
    quantity: float
    filled_quantity: float = 0.0
    filled_aver_price: float = 0.0
    created_at: float = time.time()


class PaperTradingClient:
    """Mock Roostoo client that simulates trades without executing real orders."""
    
    def __init__(self, real_client: RoostooV3Client):
        """Initialize paper trading client with real client for market data."""
        self.real_client = real_client
        self.order_counter = 1000
        self.orders: Dict[int, SimulatedOrder] = {}
        self.balance: Dict[str, float] = {"USD": 10000.0}  # Starting balance
        self.positions: Dict[str, float] = {}  # Asset -> amount
        self.trade_history: List[Dict[str, Any]] = []
        
    def ticker(self, pair: Optional[str] = None) -> List[TickerData]:
        """Get ticker data from real client."""
        return self.real_client.ticker(pair)
    
    def exchange_info(self) -> List[ExchangeInfo]:
        """Get exchange info from real client."""
        return self.real_client.exchange_info()
    
    def balance(self) -> Dict[str, float]:
        """Get simulated balance."""
        return self.balance.copy()
    
    def server_time(self) -> int:
        """Get server time from real client."""
        return self.real_client.server_time()
    
    def place_order(
        self,
        pair: str,
        side: str,
        type: str,
        quantity: float,
        price: Optional[float] = None
    ) -> OrderResponse:
        """Simulate placing an order."""
        # Get current price
        ticker_data = self.ticker(pair)
        if not ticker_data:
            raise ValueError(f"No ticker data for {pair}")
        
        current_price = ticker_data[0].last_price
        
        # Create simulated order
        order_id = self.order_counter
        self.order_counter += 1
        
        order_price = price if price else current_price
        
        simulated_order = SimulatedOrder(
            order_id=order_id,
            pair=pair,
            side=side,
            type=type,
            price=order_price,
            quantity=quantity
        )
        
        # For market orders, fill immediately
        if type.lower() == "market":
            simulated_order.status = "FILLED"
            simulated_order.filled_quantity = quantity
            simulated_order.filled_aver_price = current_price
            
            # Update balance and positions
            self._execute_fill(simulated_order, current_price)
        else:
            # For limit orders, check if price is favorable
            if side.lower() == "buy":
                if current_price <= order_price:
                    simulated_order.status = "FILLED"
                    simulated_order.filled_quantity = quantity
                    simulated_order.filled_aver_price = current_price
                    self._execute_fill(simulated_order, current_price)
                else:
                    simulated_order.status = "NEW"
            else:  # sell
                if current_price >= order_price:
                    simulated_order.status = "FILLED"
                    simulated_order.filled_quantity = quantity
                    simulated_order.filled_aver_price = current_price
                    self._execute_fill(simulated_order, current_price)
                else:
                    simulated_order.status = "NEW"
        
        self.orders[order_id] = simulated_order
        
        # Convert to OrderResponse
        return OrderResponse(
            order_id=order_id,
            pair=pair,
            side=side,
            type=type,
            status=simulated_order.status,
            price=order_price,
            quantity=quantity,
            filled_quantity=simulated_order.filled_quantity,
            filled_aver_price=simulated_order.filled_aver_price
        )
    
    def _execute_fill(self, order: SimulatedOrder, fill_price: float):
        """Execute order fill and update balance/positions."""
        pair = order.pair
        asset = pair.replace("USD", "")
        
        if order.side.lower() == "buy":
            # Buy: deduct USD, add asset
            cost = order.filled_quantity * fill_price
            fee = cost * 0.001  # 0.1% fee
            
            if self.balance.get("USD", 0) < (cost + fee):
                logger.warning(f"Insufficient USD for order {order.order_id}: need ${cost + fee:.2f}, have ${self.balance.get('USD', 0):.2f}")
                order.status = "CANCELLED"
                return
            
            self.balance["USD"] = self.balance.get("USD", 0) - cost - fee
            self.positions[asset] = self.positions.get(asset, 0) + order.filled_quantity
            
            self.trade_history.append({
                "timestamp": datetime.now(),
                "order_id": order.order_id,
                "pair": pair,
                "side": "buy",
                "quantity": order.filled_quantity,
                "price": fill_price,
                "cost": cost,
                "fee": fee
            })
            
            logger.info(f"PAPER TRADE: Bought {order.filled_quantity} {asset} at ${fill_price:.4f} (cost: ${cost:.2f}, fee: ${fee:.2f})")
        
        else:  # sell
            # Sell: deduct asset, add USD
            if self.positions.get(asset, 0) < order.filled_quantity:
                logger.warning(f"Insufficient {asset} for order {order.order_id}: need {order.filled_quantity}, have {self.positions.get(asset, 0)}")
                order.status = "CANCELLED"
                return
            
            proceeds = order.filled_quantity * fill_price
            fee = proceeds * 0.001  # 0.1% fee
            
            self.positions[asset] = self.positions.get(asset, 0) - order.filled_quantity
            if self.positions[asset] <= 0:
                del self.positions[asset]
            
            self.balance["USD"] = self.balance.get("USD", 0) + proceeds - fee
            
            self.trade_history.append({
                "timestamp": datetime.now(),
                "order_id": order.order_id,
                "pair": pair,
                "side": "sell",
                "quantity": order.filled_quantity,
                "price": fill_price,
                "proceeds": proceeds,
                "fee": fee
            })
            
            logger.info(f"PAPER TRADE: Sold {order.filled_quantity} {asset} at ${fill_price:.4f} (proceeds: ${proceeds:.2f}, fee: ${fee:.2f})")
    
    def query_order(self, order_id: int) -> List[OrderResponse]:
        """Query order status."""
        if order_id not in self.orders:
            return []
        
        order = self.orders[order_id]
        
        # For limit orders, check if they should be filled now
        if order.status == "NEW" and order.type.lower() == "limit":
            ticker_data = self.ticker(order.pair)
            if ticker_data:
                current_price = ticker_data[0].last_price
                
                if order.side.lower() == "buy" and current_price <= order.price:
                    order.status = "FILLED"
                    order.filled_quantity = order.quantity
                    order.filled_aver_price = current_price
                    self._execute_fill(order, current_price)
                elif order.side.lower() == "sell" and current_price >= order.price:
                    order.status = "FILLED"
                    order.filled_quantity = order.quantity
                    order.filled_aver_price = current_price
                    self._execute_fill(order, current_price)
        
        return [OrderResponse(
            order_id=order.order_id,
            pair=order.pair,
            side=order.side,
            type=order.type,
            status=order.status,
            price=order.price,
            quantity=order.quantity,
            filled_quantity=order.filled_quantity,
            filled_aver_price=order.filled_aver_price
        )]
    
    def cancel_order(self, order_id: int) -> bool:
        """Cancel an order."""
        if order_id in self.orders:
            if self.orders[order_id].status == "NEW":
                self.orders[order_id].status = "CANCELLED"
                return True
        return False
    
    def get_total_equity(self, prices: Dict[str, float]) -> float:
        """Calculate total equity in USD."""
        total = self.balance.get("USD", 0.0)
        for asset, amount in self.positions.items():
            pair = f"{asset}USD"
            if pair in prices:
                total += amount * prices[pair]
        return total


class PaperTradingEngine(VolatilityExpansionEngine):
    """Paper trading engine that extends VolatilityExpansionEngine with simulation."""
    
    def __init__(
        self,
        paper_client: PaperTradingClient,
        client: PaperTradingClient,
        datastore: DataStore,
        signal_generator: VolatilityExpansionSignalGenerator,
        risk_manager: RiskManager,
        position_manager: PositionManager
    ):
        """Initialize paper trading engine."""
        super().__init__(
            client=client,
            datastore=datastore,
            signal_generator=signal_generator,
            risk_manager=risk_manager,
            position_manager=position_manager
        )
        self.paper_client = paper_client
        self.start_equity = 10000.0
        self.start_time = datetime.now()
    
    def get_total_equity(self, balance: Dict[str, float], prices: Dict[str, float]) -> float:
        """Get total equity from paper client."""
        return self.paper_client.get_total_equity(prices)
    
    def print_status(self, cycle_num: int):
        """Print current status."""
        ticker_data = self.client.ticker()
        prices = {from_roostoo_pair(t.pair): t.last_price for t in ticker_data}
        
        total_equity = self.get_total_equity(self.client.balance(), prices)
        positions = self.position_manager.get_positions()
        
        # Update position prices
        self.position_manager.update_prices(prices)
        
        # Calculate utilization
        position_value = sum(
            pos.total_amount * prices.get(pair, 0) 
            for pair, pos in positions.items() 
            if pair in prices
        )
        utilization = (position_value / total_equity * 100) if total_equity > 0 else 0.0
        
        # Calculate P&L
        total_pnl = total_equity - self.start_equity
        total_pnl_pct = (total_pnl / self.start_equity * 100) if self.start_equity > 0 else 0.0
        
        elapsed = datetime.now() - self.start_time
        
        print(f"\n{'='*80}")
        print(f"CYCLE #{cycle_num} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")
        print(f"Total Equity: ${total_equity:.2f} (Started: ${self.start_equity:.2f})")
        print(f"Total P&L: ${total_pnl:+.2f} ({total_pnl_pct:+.2f}%)")
        print(f"Cash: ${self.client.balance().get('USD', 0):.2f}")
        print(f"Utilization: {utilization:.1f}% ({len(positions)}/{self.config.max_positions} positions)")
        print(f"Elapsed: {elapsed}")
        print(f"\nPositions:")
        for pair, pos in positions.items():
            if pair in prices:
                current_price = prices[pair]
                pos.update_pnl(current_price)
                print(f"  {pair}: {pos.total_amount:.6f} @ ${pos.average_entry_price:.4f} avg | "
                      f"Current: ${current_price:.4f} | "
                      f"P&L: ${pos.unrealized_pnl:+.2f} ({pos.unrealized_pnl_pct:+.2f}%)")
        print(f"{'='*80}\n")


def run_paper_trading(duration_hours: int = 24):
    """Run paper trading for specified duration."""
    logger.info(f"Starting paper trading for {duration_hours} hours")
    
    # Initialize components
    config = get_config()
    real_client = RoostooV3Client()
    paper_client = PaperTradingClient(real_client)
    
    datastore = DataStore()
    signal_generator = VolatilityExpansionSignalGenerator(config)
    risk_manager = RiskManager(max_position_pct=config.max_position_pct)
    position_manager = PositionManager(config)
    
    engine = PaperTradingEngine(
        paper_client=paper_client,
        client=paper_client,  # Pass paper_client as both paper_client and client
        datastore=datastore,
        signal_generator=signal_generator,
        risk_manager=risk_manager,
        position_manager=position_manager
    )
    
    # Update exchange info
    engine.update_exchange_info()
    
    # Run for specified duration
    end_time = datetime.now() + timedelta(hours=duration_hours)
    cycle_num = 0
    
    logger.info(f"Paper trading will run until {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        while datetime.now() < end_time:
            cycle_num += 1
            
            # Run cycle
            results = engine.run_cycle()
            
            # Print status every cycle
            engine.print_status(cycle_num)
            
            # Log results
            logger.info(f"Cycle {cycle_num}: {results.get('signals_generated', 0)} signals, "
                       f"{results.get('positions_opened', 0)} positions opened, "
                       f"{results.get('exits_executed', 0)} exits")
            
            if results.get("error"):
                logger.error(f"Cycle {cycle_num} error: {results['error']}")
            
            # Sleep until next minute
            now = time.time()
            sleep_time = 60 - (now % 60)
            if sleep_time < 0:
                sleep_time = 0
            
            # Check if we should continue
            if datetime.now() >= end_time:
                break
            
            time.sleep(sleep_time)
    
    except KeyboardInterrupt:
        logger.info("Paper trading stopped by user")
    
    # Final status
    logger.info("Paper trading completed")
    engine.print_status(cycle_num)
    
    # Print summary
    ticker_data = paper_client.ticker()
    prices = {from_roostoo_pair(t.pair): t.last_price for t in ticker_data}
    final_equity = paper_client.get_total_equity(prices)
    total_pnl = final_equity - engine.start_equity
    total_pnl_pct = (total_pnl / engine.start_equity * 100) if engine.start_equity > 0 else 0.0
    
    print(f"\n{'='*80}")
    print(f"PAPER TRADING SUMMARY")
    print(f"{'='*80}")
    print(f"Duration: {duration_hours} hours")
    print(f"Total Cycles: {cycle_num}")
    print(f"Starting Equity: ${engine.start_equity:.2f}")
    print(f"Final Equity: ${final_equity:.2f}")
    print(f"Total P&L: ${total_pnl:+.2f} ({total_pnl_pct:+.2f}%)")
    print(f"Total Trades: {len(paper_client.trade_history)}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Paper trading for volatility expansion strategy")
    parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="Duration in hours (default: 24)"
    )
    
    args = parser.parse_args()
    
    run_paper_trading(duration_hours=args.hours)

