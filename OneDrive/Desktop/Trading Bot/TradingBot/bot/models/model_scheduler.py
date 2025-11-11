"""Model retraining scheduler for periodic updates."""

import time
from datetime import datetime, timedelta
from typing import Dict, Optional
from pathlib import Path
import json
import logging

from bot.config import get_config

logger = logging.getLogger(__name__)


class ModelScheduler:
    """Tracks and schedules periodic model retraining."""
    
    def __init__(self, schedule_file: Optional[Path] = None):
        """Initialize model scheduler.
        
        Args:
            schedule_file: Path to schedule state file
        """
        self.config = get_config()
        
        if schedule_file is None:
            schedule_file = Path(__file__).parent.parent.parent / "models" / "schedule.json"
        
        self.schedule_file = Path(schedule_file)
        self.schedule_file.parent.mkdir(exist_ok=True)
        
        # Load schedule state
        self.schedule = self._load_schedule()
    
    def _load_schedule(self) -> Dict[str, int]:
        """Load schedule state from file.
        
        Returns:
            Dictionary mapping model type to last training timestamp
        """
        if self.schedule_file.exists():
            try:
                with open(self.schedule_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading schedule: {e}")
        
        return {}
    
    def _save_schedule(self) -> None:
        """Save schedule state to file."""
        try:
            with open(self.schedule_file, "w") as f:
                json.dump(self.schedule, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving schedule: {e}")
    
    def mark_trained(self, model_type: str) -> None:
        """Mark a model as trained.
        
        Args:
            model_type: Model type ("gmm", "hmm", "ensemble")
        """
        self.schedule[model_type] = int(time.time())
        self._save_schedule()
        logger.info(f"Marked {model_type} as trained at {datetime.fromtimestamp(self.schedule[model_type])}")
    
    def get_last_training_time(self, model_type: str) -> Optional[datetime]:
        """Get last training time for a model type.
        
        Args:
            model_type: Model type ("gmm", "hmm", "ensemble")
        
        Returns:
            Last training datetime, or None if never trained
        """
        timestamp = self.schedule.get(model_type)
        if timestamp:
            return datetime.fromtimestamp(timestamp)
        return None
    
    def should_retrain_gmm(self) -> bool:
        """Check if GMM should be retrained (hourly).
        
        Returns:
            True if retraining is needed
        """
        last_training = self.get_last_training_time("gmm")
        
        if last_training is None:
            return True
        
        hours_since = (datetime.now() - last_training).total_seconds() / 3600
        retrain_hours = getattr(self.config, 'regime_gmm_retrain_hours', 1)
        
        return hours_since >= retrain_hours
    
    def should_retrain_hmm(self) -> bool:
        """Check if HMM should be retrained (weekly).
        
        Returns:
            True if retraining is needed
        """
        last_training = self.get_last_training_time("hmm")
        
        if last_training is None:
            return True
        
        days_since = (datetime.now() - last_training).days
        retrain_days = getattr(self.config, 'regime_hmm_retrain_days', 7)
        
        return days_since >= retrain_days
    
    def should_retrain_ensemble(self) -> bool:
        """Check if ensemble should be retrained (weekly).
        
        Returns:
            True if retraining is needed
        """
        last_training = self.get_last_training_time("ensemble")
        
        if last_training is None:
            return True
        
        days_since = (datetime.now() - last_training).days
        retrain_days = getattr(self.config, 'ensemble_retrain_days', 7)
        
        return days_since >= retrain_days
    
    def schedule_retraining(self) -> Dict[str, bool]:
        """Check which models need retraining.
        
        Returns:
            Dictionary mapping model type to whether retraining is needed
        """
        return {
            "gmm": self.should_retrain_gmm(),
            "hmm": self.should_retrain_hmm(),
            "ensemble": self.should_retrain_ensemble()
        }
    
    def get_time_until_retrain(self, model_type: str) -> Optional[timedelta]:
        """Get time until next retraining is needed.
        
        Args:
            model_type: Model type ("gmm", "hmm", "ensemble")
        
        Returns:
            Time delta until retraining, or None if retraining is needed now
        """
        if model_type == "gmm":
            if self.should_retrain_gmm():
                return None
            last_training = self.get_last_training_time("gmm")
            if last_training:
                retrain_hours = getattr(self.config, 'regime_gmm_retrain_hours', 1)
                next_training = last_training + timedelta(hours=retrain_hours)
                return next_training - datetime.now()
        
        elif model_type == "hmm":
            if self.should_retrain_hmm():
                return None
            last_training = self.get_last_training_time("hmm")
            if last_training:
                retrain_days = getattr(self.config, 'regime_hmm_retrain_days', 7)
                next_training = last_training + timedelta(days=retrain_days)
                return next_training - datetime.now()
        
        elif model_type == "ensemble":
            if self.should_retrain_ensemble():
                return None
            last_training = self.get_last_training_time("ensemble")
            if last_training:
                retrain_days = getattr(self.config, 'ensemble_retrain_days', 7)
                next_training = last_training + timedelta(days=retrain_days)
                return next_training - datetime.now()
        
        return None

