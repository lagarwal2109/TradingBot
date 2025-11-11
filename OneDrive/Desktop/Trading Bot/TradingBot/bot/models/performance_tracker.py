"""Performance tracking and model drift detection."""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class PerformanceTracker:
    """Tracks model performance and detects drift."""
    
    def __init__(self, tracking_file: Optional[Path] = None):
        """Initialize performance tracker.
        
        Args:
            tracking_file: Path to tracking data file
        """
        if tracking_file is None:
            tracking_file = Path(__file__).parent.parent.parent / "models" / "performance_tracking.json"
        
        self.tracking_file = Path(tracking_file)
        self.tracking_file.parent.mkdir(exist_ok=True)
        
        # Load existing tracking data
        self.tracking_data = self._load_tracking_data()
    
    def _load_tracking_data(self) -> Dict:
        """Load tracking data from file."""
        if self.tracking_file.exists():
            try:
                with open(self.tracking_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading tracking data: {e}")
        
        return {
            "predictions": [],
            "regime_detections": [],
            "trades": [],
            "model_performance": {}
        }
    
    def _save_tracking_data(self) -> None:
        """Save tracking data to file."""
        try:
            with open(self.tracking_file, "w") as f:
                json.dump(self.tracking_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving tracking data: {e}")
    
    def record_prediction(
        self,
        pair: str,
        prediction: float,
        confidence: float,
        actual: Optional[float] = None
    ) -> None:
        """Record a model prediction.
        
        Args:
            pair: Trading pair
            prediction: Predicted probability
            confidence: Prediction confidence
            actual: Actual outcome (if known)
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "pair": pair,
            "prediction": prediction,
            "confidence": confidence,
            "actual": actual
        }
        
        self.tracking_data["predictions"].append(entry)
        
        # Keep only last 1000 predictions
        if len(self.tracking_data["predictions"]) > 1000:
            self.tracking_data["predictions"] = self.tracking_data["predictions"][-1000:]
        
        self._save_tracking_data()
    
    def record_regime_detection(
        self,
        pair: str,
        regime: str,
        probabilities: Dict[str, float]
    ) -> None:
        """Record regime detection.
        
        Args:
            pair: Trading pair
            regime: Detected regime
            probabilities: Regime probabilities
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "pair": pair,
            "regime": regime,
            "probabilities": probabilities
        }
        
        self.tracking_data["regime_detections"].append(entry)
        
        # Keep only last 500 detections
        if len(self.tracking_data["regime_detections"]) > 500:
            self.tracking_data["regime_detections"] = self.tracking_data["regime_detections"][-500:]
        
        self._save_tracking_data()
    
    def record_trade(
        self,
        pair: str,
        action: str,
        price: float,
        amount: float,
        regime: str,
        confidence: float,
        pnl: Optional[float] = None
    ) -> None:
        """Record a trade.
        
        Args:
            pair: Trading pair
            action: "open" or "close"
            price: Trade price
            amount: Trade amount
            regime: Market regime
            confidence: Signal confidence
            pnl: Profit/loss (for closed trades)
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "pair": pair,
            "action": action,
            "price": price,
            "amount": amount,
            "regime": regime,
            "confidence": confidence,
            "pnl": pnl
        }
        
        self.tracking_data["trades"].append(entry)
        self._save_tracking_data()
    
    def calculate_prediction_accuracy(self, window_hours: int = 24) -> Dict[str, float]:
        """Calculate prediction accuracy over recent window.
        
        Args:
            window_hours: Time window in hours
        
        Returns:
            Dictionary with accuracy metrics
        """
        cutoff_time = datetime.now() - timedelta(hours=window_hours)
        
        recent_predictions = [
            p for p in self.tracking_data["predictions"]
            if datetime.fromisoformat(p["timestamp"]) >= cutoff_time and p.get("actual") is not None
        ]
        
        if not recent_predictions:
            return {"accuracy": 0.0, "count": 0}
        
        correct = sum(1 for p in recent_predictions if (p["prediction"] > 0.5) == (p["actual"] > 0))
        total = len(recent_predictions)
        
        return {
            "accuracy": correct / total if total > 0 else 0.0,
            "count": total
        }
    
    def detect_model_drift(self, window_hours: int = 24, threshold: float = 0.1) -> bool:
        """Detect if model performance has drifted.
        
        Args:
            window_hours: Time window to check
            threshold: Accuracy drop threshold
        
        Returns:
            True if drift detected
        """
        recent_accuracy = self.calculate_prediction_accuracy(window_hours)
        
        # Compare with longer-term accuracy
        long_term_accuracy = self.calculate_prediction_accuracy(window_hours * 7)
        
        if recent_accuracy["count"] < 10 or long_term_accuracy["count"] < 10:
            return False
        
        accuracy_drop = long_term_accuracy["accuracy"] - recent_accuracy["accuracy"]
        
        if accuracy_drop > threshold:
            logger.warning(f"Model drift detected: accuracy dropped by {accuracy_drop:.2%}")
            return True
        
        return False
    
    def get_regime_distribution(self, window_hours: int = 24) -> Dict[str, int]:
        """Get distribution of detected regimes.
        
        Args:
            window_hours: Time window
        
        Returns:
            Dictionary mapping regime to count
        """
        cutoff_time = datetime.now() - timedelta(hours=window_hours)
        
        recent_detections = [
            d for d in self.tracking_data["regime_detections"]
            if datetime.fromisoformat(d["timestamp"]) >= cutoff_time
        ]
        
        distribution = {}
        for detection in recent_detections:
            regime = detection["regime"]
            distribution[regime] = distribution.get(regime, 0) + 1
        
        return distribution
    
    def get_trade_performance_by_regime(self) -> Dict[str, Dict[str, float]]:
        """Get trade performance statistics by regime.
        
        Returns:
            Dictionary mapping regime to performance metrics
        """
        closed_trades = [
            t for t in self.tracking_data["trades"]
            if t["action"] == "close" and t.get("pnl") is not None
        ]
        
        performance_by_regime = {}
        
        for trade in closed_trades:
            regime = trade["regime"]
            if regime not in performance_by_regime:
                performance_by_regime[regime] = {
                    "total_trades": 0,
                    "winning_trades": 0,
                    "total_pnl": 0.0,
                    "avg_pnl": 0.0
                }
            
            perf = performance_by_regime[regime]
            perf["total_trades"] += 1
            perf["total_pnl"] += trade["pnl"]
            
            if trade["pnl"] > 0:
                perf["winning_trades"] += 1
        
        # Calculate averages
        for regime, perf in performance_by_regime.items():
            if perf["total_trades"] > 0:
                perf["avg_pnl"] = perf["total_pnl"] / perf["total_trades"]
                perf["win_rate"] = perf["winning_trades"] / perf["total_trades"]
        
        return performance_by_regime

