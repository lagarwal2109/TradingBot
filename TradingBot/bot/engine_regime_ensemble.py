"""Regime-adaptive ensemble trading engine."""

import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd

from bot.config import get_config
from bot.roostoo_v3 import RoostooV3Client, ExchangeInfo
from bot.datastore import DataStore
from bot.risk import RiskManager
from bot.portfolio_manager import RegimeAdaptivePortfolioManager, PositionInfo
from bot.models.feature_engineering import FeatureEngineer
from bot.models.regime_detection import GMMRegimeDetector, HMMTrendDetector, RegimeFusion
from bot.models.ensemble_model import StackedEnsemble
from bot.models.model_storage import ModelStorage
from bot.models.model_scheduler import ModelScheduler
from bot.utils import to_roostoo_pair, from_roostoo_pair

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Track open position details."""
    symbol: str
    entry_price: float
    amount: float
    stop_loss: float
    take_profit: float
    confidence: float
    regime: str
    trailing_stop_active: bool = False
    highest_price: float = 0.0
    entry_time: int = 0


class RegimeEnsembleTradingEngine:
    """Trading engine with regime-adaptive ensemble strategy."""
    
    def __init__(
        self,
        trading_client: RoostooV3Client,
        datastore: DataStore,
        risk_manager: RiskManager
    ):
        """Initialize regime ensemble trading engine.
        
        Args:
            trading_client: Roostoo API client
            datastore: Data storage instance
            risk_manager: Risk management instance
        """
        self.trading_client = trading_client
        self.datastore = datastore
        self.risk_manager = risk_manager
        self.config = get_config()
        
        # Initialize components
        self.model_storage = ModelStorage()
        self.portfolio_manager = RegimeAdaptivePortfolioManager()
        self.feature_engineer = FeatureEngineer()
        self.model_scheduler = ModelScheduler()
        
        # Regime detectors
        self.gmm_detector = None
        self.hmm_detector = None
        self.regime_fusion = RegimeFusion()
        
        # Ensemble model
        self.ensemble = None
        
        # Cache
        self.exchange_info_map: Dict[str, ExchangeInfo] = {}
        self.positions: Dict[str, Position] = {}
        
        # Alias for compatibility
        self.client = self.trading_client
        self.mode = "regime_ensemble"
        
        # Load models
        self.load_models()
    
    def load_models(self) -> None:
        """Load pre-trained models from storage."""
        try:
            # Try to load ensemble (use first available pair's models as default)
            pairs = self.datastore.get_all_pairs_with_data()
            if pairs:
                pair_name = pairs[0].replace("USD", "")
                
                # Load GMM
                try:
                    self.gmm_detector = GMMRegimeDetector(model_storage=self.model_storage)
                    self.gmm_detector.load(name=f"gmm_regime_{pair_name}")
                    logger.info(f"Loaded GMM detector for {pair_name}")
                except Exception as e:
                    logger.warning(f"Could not load GMM detector: {e}")
                    self.gmm_detector = GMMRegimeDetector(model_storage=self.model_storage)
                
                # Load HMM
                try:
                    self.hmm_detector = HMMTrendDetector(model_storage=self.model_storage)
                    self.hmm_detector.load(name=f"hmm_trend_{pair_name}")
                    logger.info(f"Loaded HMM detector for {pair_name}")
                except Exception as e:
                    logger.warning(f"Could not load HMM detector: {e}")
                    self.hmm_detector = HMMTrendDetector(model_storage=self.model_storage)
            
            # Load ensemble
            try:
                self.ensemble = StackedEnsemble([], model_storage=self.model_storage)
                self.ensemble.load(name="stacked_ensemble")
                logger.info("Loaded stacked ensemble model")
            except Exception as e:
                logger.warning(f"Could not load ensemble model: {e}. Models need to be trained first.")
                self.ensemble = None
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def update_exchange_info(self) -> None:
        """Update cached exchange information."""
        try:
            exchange_info_list = self.trading_client.exchange_info()
            self.exchange_info_map = {info.pair: info for info in exchange_info_list}
            self.datastore.mark_exchange_info_updated()
            logger.info(f"Updated exchange info for {len(self.exchange_info_map)} pairs")
        except Exception as e:
            logger.error(f"Failed to update exchange info: {e}")
    
    def get_total_equity(self, balance: Dict[str, float], prices: Dict[str, float]) -> float:
        """Calculate total equity in USD."""
        total = balance.get("USD", 0.0)
        for asset, amount in balance.items():
            if asset != "USD" and amount > 0:
                pair = f"{asset}USD"
                if pair in prices:
                    total += amount * prices[pair]
        return total
    
    def detect_regime(self, pair: str) -> Tuple[Dict[str, float], str]:
        """Detect current regime for a pair.
        
        Args:
            pair: Trading pair
        
        Returns:
            Tuple of (regime_probabilities, dominant_regime)
        """
        try:
            # Get minute data for GMM
            df_minute = self.datastore.read_minute_bars(pair, limit=1440)
            
            # Get 4h data for HMM
            df_4h = self.datastore.read_aggregated_bars(pair, interval="4h", limit=100)
            
            # Get GMM probabilities
            if self.gmm_detector and not df_minute.empty:
                gmm_proba = self.gmm_detector.predict_proba(df_minute)
            else:
                gmm_proba = {"calm": 0.5, "volatile": 0.5}
            
            # Get HMM probabilities
            if self.hmm_detector and not df_4h.empty:
                hmm_proba = self.hmm_detector.predict_proba(df_4h)
            else:
                hmm_proba = {"bearish": 0.33, "neutral": 0.33, "bullish": 0.34}
            
            # Fuse regimes
            combined_proba = self.regime_fusion.fuse_regimes(gmm_proba, hmm_proba)
            gmm_regime = self.gmm_detector.get_dominant_regime(gmm_proba) if self.gmm_detector else "calm"
            hmm_regime = self.hmm_detector.get_dominant_trend(hmm_proba) if self.hmm_detector else "neutral"
            dominant_regime = self.regime_fusion.get_combined_regime(gmm_regime, hmm_regime)
            
            return combined_proba, dominant_regime
            
        except Exception as e:
            logger.error(f"Error detecting regime for {pair}: {e}")
            return {"calm_bullish": 0.5}, "calm_bullish"
    
    def generate_predictions(self, pairs_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, float]]:
        """Generate ensemble predictions for all pairs.
        
        Args:
            pairs_data: Dictionary mapping pairs to DataFrames
        
        Returns:
            Dictionary mapping pairs to prediction dicts with 'confidence', 'probability', etc.
        """
        if self.ensemble is None:
            logger.warning("Ensemble model not loaded, returning empty predictions")
            return {}
        
        predictions = {}
        
        for pair, df in pairs_data.items():
            try:
                # Detect regime
                regime_proba, regime = self.detect_regime(pair)
                
                # Create features
                feature_matrix, _ = self.feature_engineer.create_feature_matrix(
                    {pair: df},
                    regime_detector=None  # Regime already incorporated
                )
                
                if feature_matrix.empty:
                    continue
                
                # Get features (exclude pair column)
                feature_cols = [col for col in feature_matrix.columns if col != "pair"]
                X = feature_matrix[feature_cols].values
                
                # Handle NaN values
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Get prediction
                proba = self.ensemble.predict_proba(X)
                confidence = self.ensemble.get_confidence_score(proba)[0]
                
                # Probability of price going up
                prob_up = proba[0, 1] if proba.shape[1] == 2 else proba[0, 0]
                
                predictions[pair] = {
                    "confidence": float(confidence),
                    "probability": float(prob_up),
                    "regime": regime,
                    "regime_proba": regime_proba
                }
                
            except Exception as e:
                logger.error(f"Error generating prediction for {pair}: {e}")
                continue
        
        return predictions
    
    def update_trailing_stops(self, prices: Dict[str, float]) -> List[str]:
        """Update trailing stops and return positions to close."""
        positions_to_close = []
        
        for symbol, position in self.positions.items():
            pair = f"{symbol}USD"
            if pair not in prices:
                continue
            
            current_price = prices[pair]
            
            # Update highest price
            if current_price > position.highest_price:
                position.highest_price = current_price
                if current_price > position.entry_price * 1.02:  # 2% profit
                    position.trailing_stop_active = True
            
            # Check trailing stop
            if position.trailing_stop_active:
                trailing_stop = position.highest_price * (1 - self.config.trailing_stop_pct)
                if current_price <= trailing_stop:
                    logger.info(f"Trailing stop hit for {symbol}")
                    positions_to_close.append(symbol)
                    continue
            
            # Check fixed stops
            if current_price <= position.stop_loss:
                logger.info(f"Stop loss hit for {symbol}")
                positions_to_close.append(symbol)
                continue
            
            if current_price >= position.take_profit:
                logger.info(f"Take profit hit for {symbol}")
                positions_to_close.append(symbol)
                continue
        
        return positions_to_close
    
    def close_position(self, symbol: str) -> bool:
        """Close a position."""
        try:
            balance = self.trading_client.balance()
            if symbol in balance and balance[symbol] > 0:
                pair = f"{symbol}USD"
                roostoo_pair = to_roostoo_pair(pair)
                
                amount = balance[symbol]
                exchange_info = self.exchange_info_map.get(roostoo_pair)
                if exchange_info:
                    amount = self.risk_manager.round_to_precision(amount, exchange_info.amount_precision)
                
                order = self.trading_client.place_order(
                    pair=roostoo_pair,
                    side="sell",
                    type="market",
                    quantity=amount
                )
                
                if symbol in self.positions:
                    del self.positions[symbol]
                
                logger.info(f"Closed position: {symbol}")
                return True
        except Exception as e:
            logger.error(f"Error closing position {symbol}: {e}")
        
        return False
    
    def execute_trade(self, pair: str, signal: Dict[str, float], balance: Dict[str, float], prices: Dict[str, float]) -> bool:
        """Execute a trade based on signal."""
        try:
            base_currency = pair.replace("USD", "")
            roostoo_pair = to_roostoo_pair(pair)
            
            if roostoo_pair not in self.exchange_info_map:
                logger.error(f"No exchange info for {roostoo_pair}")
                return False
            
            exchange_info = self.exchange_info_map[roostoo_pair]
            total_equity = self.get_total_equity(balance, prices)
            current_price = prices[pair]
            
            # Calculate position size
            regime = signal.get("regime", "calm_bullish")
            confidence = signal.get("confidence", 0.5)
            
            # Get volatility from recent data
            df = self.datastore.read_minute_bars(pair, limit=60)
            if not df.empty:
                returns = np.log(df["price"] / df["price"].shift(1))
                volatility = returns.std() if len(returns) > 1 else 0.02
            else:
                volatility = 0.02
            
            position_value = self.portfolio_manager.calculate_position_size(
                base_size=self.config.max_position_pct,
                regime=regime,
                confidence=confidence,
                volatility=volatility,
                total_equity=total_equity
            )
            
            if position_value <= 0:
                return False
            
            amount = position_value / current_price
            amount = self.risk_manager.round_to_precision(amount, exchange_info.amount_precision)
            
            # Calculate stops
            stop_loss, take_profit = self.portfolio_manager.calculate_adaptive_stops(
                entry_price=current_price,
                regime=regime,
                confidence=confidence,
                volatility=volatility
            )
            
            # Validate order
            is_valid, error_msg = self.risk_manager.validate_order(
                amount=amount,
                price=current_price,
                side="buy",
                balance=balance,
                exchange_info=exchange_info
            )
            
            if not is_valid:
                logger.warning(f"Order validation failed: {error_msg}")
                return False
            
            # Place order
            order = self.trading_client.place_order(
                pair=roostoo_pair,
                side="buy",
                type="market",
                quantity=amount
            )
            
            # Track position
            self.positions[base_currency] = Position(
                symbol=base_currency,
                entry_price=current_price,
                amount=amount,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=confidence,
                regime=regime,
                highest_price=current_price,
                entry_time=int(time.time() * 1000)
            )
            
            logger.info(f"Opened position: {amount:.6f} {base_currency} at {current_price:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Error executing trade for {pair}: {e}")
            return False
    
    def check_kill_switch(self) -> bool:
        """Check if kill switch should be activated."""
        if self.datastore.state.consecutive_errors >= self.config.max_consecutive_errors:
            logger.critical("Kill switch activated")
            # Close all positions
            for symbol in list(self.positions.keys()):
                self.close_position(symbol)
            return True
        return False
    
    def run_cycle(self) -> Dict[str, Any]:
        """Run a complete trading cycle."""
        cycle_start = time.time()
        results = {
            "timestamp": int(cycle_start * 1000),
            "success": False,
            "positions": list(self.positions.keys()),
            "error": None
        }
        
        try:
            # Check kill switch
            if self.check_kill_switch():
                results["error"] = "Kill switch activated"
                return results
            
            # Update exchange info if needed
            if self.datastore.should_update_exchange_info() or not self.exchange_info_map:
                self.update_exchange_info()
            
            # Get market data
            roostoo_tickers = self.trading_client.ticker()
            ticker_list = [
                {
                    "pair": from_roostoo_pair(t.pair),
                    "price": t.last_price,
                    "volume_24h": t.coin_trade_value,
                    "bid": t.max_bid,
                    "ask": t.min_ask
                }
                for t in roostoo_tickers
            ]
            prices = {from_roostoo_pair(t.pair): t.last_price for t in roostoo_tickers}
            
            # Collect minute bars
            self.datastore.collect_minute_bars(ticker_list)
            
            # Check existing positions
            positions_to_close = self.update_trailing_stops(prices)
            for symbol in positions_to_close:
                if self.close_position(symbol):
                    self.datastore.record_trade(None)
                    time.sleep(0.5)
            
            # Generate predictions if ensemble is loaded
            if self.ensemble is not None:
                # Prepare data for predictions
                pairs_data = {}
                for ticker in ticker_list:
                    pair = ticker["pair"]
                    df = self.datastore.read_minute_bars(pair, limit=500)
                    if not df.empty:
                        pairs_data[pair] = df
                
                # Get predictions
                predictions = self.generate_predictions(pairs_data)
                
                # Filter by confidence threshold
                min_confidence = self.config.min_prediction_confidence
                filtered_signals = {
                    pair: pred for pair, pred in predictions.items()
                    if pred.get("confidence", 0) >= min_confidence and pred.get("probability", 0) > 0.5
                }
                
                # Rank signals
                ranked_signals = self.portfolio_manager.rank_signals(filtered_signals)
                
                # Get balance
                balance = self.trading_client.balance()
                total_equity = self.get_total_equity(balance, prices)
                
                # Check portfolio allocation
                current_positions_info = {}
                for symbol, pos in self.positions.items():
                    pair = f"{symbol}USD"
                    if pair in prices:
                        current_positions_info[symbol] = PositionInfo(
                            symbol=symbol,
                            entry_price=pos.entry_price,
                            amount=pos.amount,
                            current_value=pos.amount * prices[pair],
                            confidence=pos.confidence,
                            regime=pos.regime
                        )
                
                allocated_value, allocated_pct, can_add = self.portfolio_manager.check_portfolio_allocation(
                    current_positions_info,
                    total_equity
                )
                
                # Execute trades if we can add positions
                if can_add and ranked_signals:
                    for pair, score, signal in ranked_signals[:3]:  # Top 3 signals
                        base_currency = pair.replace("USD", "")
                        
                        # Skip if we already have this position
                        if base_currency in self.positions:
                            continue
                        
                        # Check if we still have capacity
                        if len(self.positions) >= self.config.max_simultaneous_positions:
                            break
                        
                        if not self.datastore.check_trade_allowed():
                            break
                        
                        # Execute trade
                        if self.execute_trade(pair, signal, balance, prices):
                            self.datastore.record_trade(base_currency)
                            self.datastore.reset_error_count()
                            time.sleep(1)  # Rate limiting
                            break  # One trade per cycle
            
            # Update model retraining schedule
            retrain_schedule = self.model_scheduler.schedule_retraining()
            if retrain_schedule.get("gmm") or retrain_schedule.get("hmm") or retrain_schedule.get("ensemble"):
                logger.info(f"Models need retraining: {retrain_schedule}")
                # Retraining would be done in a separate process/script
            
            results["success"] = True
            
            # Get final state
            balance = self.trading_client.balance()
            results["total_equity"] = self.get_total_equity(balance, prices)
            results["balances"] = {k: v for k, v in balance.items() if v > 0}
            results["open_positions"] = len(self.positions)
            
        except Exception as e:
            logger.error(f"Trading cycle failed: {e}")
            results["error"] = str(e)
            self.datastore.increment_error_count()
        
        cycle_time = time.time() - cycle_start
        results["cycle_time"] = cycle_time
        logger.info(f"Cycle completed in {cycle_time:.2f}s")
        
        return results
    
    def cancel_all_orders(self) -> None:
        """Cancel all open orders."""
        try:
            open_orders = self.trading_client.get_open_orders()
            for order in open_orders:
                try:
                    self.trading_client.cancel_order(order.order_id)
                    logger.info(f"Cancelled order {order.order_id}")
                except Exception as e:
                    logger.error(f"Failed to cancel order {order.order_id}: {e}")
        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")

