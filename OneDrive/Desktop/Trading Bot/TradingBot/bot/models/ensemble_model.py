"""Ensemble models for price movement prediction."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

from sklearn.ensemble import RandomForestClassifier

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except ImportError:
    tf = None
    keras = None

from bot.models.model_storage import ModelStorage

logger = logging.getLogger(__name__)


class BasePredictor:
    """Base class for prediction models."""
    
    def __init__(self, model_storage: Optional[ModelStorage] = None):
        """Initialize base predictor.
        
        Args:
            model_storage: Optional ModelStorage instance
        """
        self.model = None
        self.model_storage = model_storage or ModelStorage()
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model.
        
        Args:
            X: Feature matrix
            y: Target labels
        """
        raise NotImplementedError
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            X: Feature matrix
        
        Returns:
            Probability array (n_samples, n_classes)
        """
        raise NotImplementedError
    
    def save(self, name: str, version: Optional[str] = None) -> str:
        """Save model to storage.
        
        Args:
            name: Model name
            version: Optional version string
        
        Returns:
            Version string
        """
        metadata = {"is_fitted": self.is_fitted}
        return self.model_storage.save_model(self.model, name, version, metadata)
    
    def load(self, name: str, version: Optional[str] = None) -> None:
        """Load model from storage.
        
        Args:
            name: Model name
            version: Optional version string
        """
        self.model = self.model_storage.load_model(name, version)
        metadata = self.model_storage.load_metadata(name, version)
        self.is_fitted = metadata.get("is_fitted", False)


class XGBoostPredictor(BasePredictor):
    """XGBoost binary classifier for price movement prediction."""
    
    def __init__(self, model_storage: Optional[ModelStorage] = None, **kwargs):
        """Initialize XGBoost predictor.
        
        Args:
            model_storage: Optional ModelStorage instance
            **kwargs: XGBoost parameters
        """
        super().__init__(model_storage)
        
        if xgb is None:
            raise ImportError("xgboost is required but not installed")
        
        default_params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42
        }
        default_params.update(kwargs)
        
        self.model = xgb.XGBClassifier(**default_params)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train XGBoost model.
        
        Args:
            X: Feature matrix
            y: Target labels (binary: 0 or 1)
        """
        self.model.fit(X, y)
        self.is_fitted = True
        logger.info(f"XGBoost model fitted on {len(X)} samples")
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities.
        
        Args:
            X: Feature matrix
        
        Returns:
            Probability array (n_samples, 2)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        proba = self.model.predict_proba(X)
        return proba


class LightGBMPredictor(BasePredictor):
    """LightGBM binary classifier for price movement prediction."""
    
    def __init__(self, model_storage: Optional[ModelStorage] = None, **kwargs):
        """Initialize LightGBM predictor.
        
        Args:
            model_storage: Optional ModelStorage instance
            **kwargs: LightGBM parameters
        """
        super().__init__(model_storage)
        
        if lgb is None:
            raise ImportError("lightgbm is required but not installed")
        
        default_params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "random_state": 42
        }
        default_params.update(kwargs)
        
        self.model = lgb.LGBMClassifier(**default_params)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train LightGBM model.
        
        Args:
            X: Feature matrix
            y: Target labels (binary: 0 or 1)
        """
        self.model.fit(X, y)
        self.is_fitted = True
        logger.info(f"LightGBM model fitted on {len(X)} samples")
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities.
        
        Args:
            X: Feature matrix
        
        Returns:
            Probability array (n_samples, 2)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        proba = self.model.predict_proba(X)
        return proba


class RandomForestPredictor(BasePredictor):
    """Random Forest binary classifier for price movement prediction."""
    
    def __init__(self, model_storage: Optional[ModelStorage] = None, **kwargs):
        """Initialize Random Forest predictor.
        
        Args:
            model_storage: Optional ModelStorage instance
            **kwargs: Random Forest parameters
        """
        super().__init__(model_storage)
        
        default_params = {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "random_state": 42,
            "n_jobs": -1
        }
        default_params.update(kwargs)
        
        self.model = RandomForestClassifier(**default_params)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train Random Forest model.
        
        Args:
            X: Feature matrix
            y: Target labels (binary: 0 or 1)
        """
        self.model.fit(X, y)
        self.is_fitted = True
        logger.info(f"Random Forest model fitted on {len(X)} samples")
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities.
        
        Args:
            X: Feature matrix
        
        Returns:
            Probability array (n_samples, 2)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        proba = self.model.predict_proba(X)
        return proba


class LSTMPredictor(BasePredictor):
    """LSTM model for sequential price movement prediction."""
    
    def __init__(
        self,
        sequence_length: int = 60,
        model_storage: Optional[ModelStorage] = None,
        **kwargs
    ):
        """Initialize LSTM predictor.
        
        Args:
            sequence_length: Length of input sequences
            model_storage: Optional ModelStorage instance
            **kwargs: Additional parameters
        """
        super().__init__(model_storage)
        
        if tf is None or keras is None:
            raise ImportError("tensorflow is required but not installed")
        
        self.sequence_length = sequence_length
        self.n_features = None
        self.model = None
    
    def _build_model(self, n_features: int) -> keras.Model:
        """Build LSTM model architecture.
        
        Args:
            n_features: Number of input features
        
        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            layers.LSTM(64, return_sequences=True, input_shape=(self.sequence_length, n_features)),
            layers.Dropout(0.2),
            layers.LSTM(32, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 50, batch_size: int = 32, validation_split: float = 0.2) -> None:
        """Train LSTM model.
        
        Args:
            X: Sequential feature matrix (n_samples, sequence_length, n_features)
            y: Target labels (binary: 0 or 1)
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Validation split ratio
        """
        if len(X.shape) != 3:
            raise ValueError(f"LSTM expects 3D input (n_samples, sequence_length, n_features), got shape {X.shape}")
        
        n_features = X.shape[2]
        
        if self.model is None or self.n_features != n_features:
            self.model = self._build_model(n_features)
            self.n_features = n_features
        
        # Train model
        self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=0
        )
        
        self.is_fitted = True
        logger.info(f"LSTM model fitted on {len(X)} sequences")
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities.
        
        Args:
            X: Sequential feature matrix (n_samples, sequence_length, n_features)
        
        Returns:
            Probability array (n_samples, 2) - [prob_class_0, prob_class_1]
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        if len(X.shape) != 3:
            raise ValueError(f"LSTM expects 3D input, got shape {X.shape}")
        
        # Predict probability of class 1
        prob_class_1 = self.model.predict(X, verbose=0).flatten()
        prob_class_0 = 1 - prob_class_1
        
        # Stack into (n_samples, 2) format
        proba = np.column_stack([prob_class_0, prob_class_1])
        return proba


class StackedEnsemble:
    """Two-stage stacked ensemble with base models and meta-learner."""
    
    def __init__(
        self,
        base_models: List[BasePredictor],
        meta_model: Optional[BasePredictor] = None,
        cv_folds: int = 5,
        model_storage: Optional[ModelStorage] = None
    ):
        """Initialize stacked ensemble.
        
        Args:
            base_models: List of base model predictors
            meta_model: Meta-learner (default: XGBoost)
            cv_folds: Number of cross-validation folds
            model_storage: Optional ModelStorage instance
        """
        self.base_models = base_models
        self.cv_folds = cv_folds
        self.model_storage = model_storage or ModelStorage()
        
        # Default meta-model is XGBoost
        if meta_model is None:
            if xgb is None:
                raise ImportError("XGBoost required for meta-learner")
            meta_model = XGBoostPredictor(model_storage=model_storage)
        
        self.meta_model = meta_model
        self.is_fitted = False
    
    def fit_base_models(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Train base models with cross-validation and generate meta-features.
        
        Args:
            X_train: Training feature matrix
            y_train: Training labels
        
        Returns:
            Meta-features matrix (n_samples, n_base_models)
        """
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        meta_features = np.zeros((len(X_train), len(self.base_models)))
        
        logger.info(f"Training {len(self.base_models)} base models with {self.cv_folds}-fold CV...")
        
        for i, base_model in enumerate(self.base_models):
            logger.info(f"Training base model {i+1}/{len(self.base_models)}: {type(base_model).__name__}")
            
            # Get out-of-fold predictions
            oof_predictions = cross_val_predict(
                base_model.model if hasattr(base_model, 'model') else base_model,
                X_train,
                y_train,
                cv=kf,
                method='predict_proba',
                n_jobs=-1
            )
            
            # Use probability of class 1 as meta-feature
            if oof_predictions.shape[1] == 2:
                meta_features[:, i] = oof_predictions[:, 1]
            else:
                meta_features[:, i] = oof_predictions.flatten()
            
            # Also train on full dataset for final predictions
            base_model.fit(X_train, y_train)
        
        logger.info("Base models trained")
        return meta_features
    
    def fit_meta_model(self, meta_X: np.ndarray, y_train: np.ndarray) -> None:
        """Train meta-learner on base model predictions.
        
        Args:
            meta_X: Meta-features from base models
            y_train: Training labels
        """
        logger.info("Training meta-learner...")
        self.meta_model.fit(meta_X, y_train)
        self.is_fitted = True
        logger.info("Meta-learner trained")
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the complete stacked ensemble.
        
        Args:
            X_train: Training feature matrix
            y_train: Training labels
        """
        # Train base models and get meta-features
        meta_features = self.fit_base_models(X_train, y_train)
        
        # Train meta-model
        self.fit_meta_model(meta_features, y_train)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities using two-stage ensemble.
        
        Args:
            X: Feature matrix
        
        Returns:
            Probability array (n_samples, 2)
        """
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted")
        
        # Get base model predictions
        base_predictions = []
        for base_model in self.base_models:
            proba = base_model.predict_proba(X)
            if proba.shape[1] == 2:
                base_predictions.append(proba[:, 1])  # Probability of class 1
            else:
                base_predictions.append(proba.flatten())
        
        # Stack into meta-features
        meta_features = np.column_stack(base_predictions)
        
        # Get meta-model prediction
        final_proba = self.meta_model.predict_proba(meta_features)
        return final_proba
    
    def get_confidence_score(self, proba: np.ndarray) -> np.ndarray:
        """Extract confidence score from probabilities.
        
        Args:
            proba: Probability array (n_samples, 2)
        
        Returns:
            Confidence scores (n_samples,) - distance from 0.5
        """
        # Confidence is how far from 0.5 (uncertainty)
        prob_class_1 = proba[:, 1] if proba.shape[1] == 2 else proba.flatten()
        confidence = np.abs(prob_class_1 - 0.5) * 2  # Scale to [0, 1]
        return confidence
    
    def save(self, name: str = "stacked_ensemble", version: Optional[str] = None) -> str:
        """Save ensemble to storage.
        
        Args:
            name: Model name
            version: Optional version string
        
        Returns:
            Version string
        """
        # Save each base model
        base_model_names = []
        for i, base_model in enumerate(self.base_models):
            base_name = f"{name}_base_{i}_{type(base_model).__name__}"
            base_model.save(base_name, version)
            base_model_names.append(base_name)
        
        # Save meta-model
        meta_name = f"{name}_meta"
        self.meta_model.save(meta_name, version)
        
        # Save ensemble metadata
        metadata = {
            "is_fitted": self.is_fitted,
            "n_base_models": len(self.base_models),
            "base_model_names": base_model_names,
            "meta_model_name": meta_name,
            "cv_folds": self.cv_folds
        }
        
        return self.model_storage.save_model(self, name, version, metadata)
    
    def load(self, name: str = "stacked_ensemble", version: Optional[str] = None) -> None:
        """Load ensemble from storage.
        
        Args:
            name: Model name
            version: Optional version string
        """
        metadata = self.model_storage.load_metadata(name, version)
        
        # Load base models
        base_model_names = metadata.get("base_model_names", [])
        self.base_models = []
        
        for base_name in base_model_names:
            # Determine model type from name
            if "XGBoost" in base_name:
                base_model = XGBoostPredictor(self.model_storage)
            elif "LightGBM" in base_name:
                base_model = LightGBMPredictor(self.model_storage)
            elif "RandomForest" in base_name:
                base_model = RandomForestPredictor(self.model_storage)
            elif "LSTM" in base_name:
                base_model = LSTMPredictor(model_storage=self.model_storage)
            else:
                logger.warning(f"Unknown base model type: {base_name}")
                continue
            
            base_model.load(base_name, version)
            self.base_models.append(base_model)
        
        # Load meta-model
        meta_name = metadata.get("meta_model_name", f"{name}_meta")
        self.meta_model = XGBoostPredictor(self.model_storage)
        self.meta_model.load(meta_name, version)
        
        self.is_fitted = metadata.get("is_fitted", False)

