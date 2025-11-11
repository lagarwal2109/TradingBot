"""Overfit Risk Score calculator and aggregator."""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .overfitting_tests import DiagnosticResult


@dataclass
class OverfitRiskScore:
    """Overfit Risk Score with component breakdown."""
    total_score: float  # 0-100, higher = more risk
    risk_level: str  # "low", "medium", "high", "critical"
    component_scores: Dict[str, float]
    details: Dict[str, Any]


class RiskScorer:
    """Calculate Overfit Risk Score from diagnostic results."""
    
    # Component weights
    WEIGHTS = {
        "random_label": 20,
        "pbo": 20,
        "dsr": 15,
        "cost_sensitivity": 10,
        "parameter_flatness": 10,
        "regime_consistency": 10,
        "adversarial_auc": 10,
        "bootstrap_ci": 5
    }
    
    def __init__(self):
        """Initialize risk scorer."""
        pass
    
    def score_random_label(self, result: DiagnosticResult) -> float:
        """Score random label test (0-100, higher = more risk).
        
        Args:
            result: Random label test result
            
        Returns:
            Risk score component
        """
        if not result.passed:
            # High risk if permuted labels still work
            return 80.0
        
        # Score based on Sharpe delta
        # If delta is small, high risk
        sharpe_delta = result.details.get("sharpe_delta", 0) if result.details else 0
        
        if sharpe_delta < 0.5:
            return 70.0
        elif sharpe_delta < 1.0:
            return 40.0
        else:
            return 10.0
    
    def score_pbo(self, result: DiagnosticResult) -> float:
        """Score PBO test (0-100, higher = more risk).
        
        Args:
            result: PBO test result
            
        Returns:
            Risk score component
        """
        if not result.passed:
            pbo = result.score
            # PBO > 0.2 is high risk
            if pbo > 0.5:
                return 90.0
            elif pbo > 0.3:
                return 70.0
            else:
                return 50.0
        
        # Low risk if PBO < 0.2
        return 10.0
    
    def score_dsr(self, result: DiagnosticResult) -> float:
        """Score DSR test (0-100, higher = more risk).
        
        Args:
            result: DSR test result
            
        Returns:
            Risk score component
        """
        if not result.passed:
            p_value = result.details.get("p_value", 1.0) if result.details else 1.0
            dsr = result.score
            
            if p_value > 0.1 or dsr < 0:
                return 80.0
            elif p_value > 0.05:
                return 60.0
            else:
                return 40.0
        
        # Low risk if significant
        return 5.0
    
    def score_cost_sensitivity(self, result: DiagnosticResult) -> float:
        """Score cost sensitivity (0-100, higher = more risk).
        
        Args:
            result: Cost sensitivity test result
            
        Returns:
            Risk score component
        """
        if not result.passed:
            slope = abs(result.score)
            # Very steep slope = high risk
            if slope > 0.2:
                return 90.0
            elif slope > 0.1:
                return 60.0
            else:
                return 30.0
        
        return 10.0
    
    def score_parameter_flatness(self, result: DiagnosticResult) -> float:
        """Score parameter flatness (0-100, higher = more risk).
        
        Args:
            result: Parameter stability test result
            
        Returns:
            Risk score component
        """
        if not result.passed:
            flatness = result.score
            # Low flatness = needle peak = high risk
            if flatness < 0.05:
                return 85.0
            elif flatness < 0.1:
                return 60.0
            else:
                return 30.0
        
        return 10.0
    
    def score_regime_consistency(self, result: DiagnosticResult) -> float:
        """Score regime consistency (0-100, higher = more risk).
        
        Args:
            result: Regime robustness test result
            
        Returns:
            Risk score component
        """
        if not result.passed:
            consistency = result.score
            # Low consistency = high risk
            if consistency < 0.5:
                return 70.0
            elif consistency < 0.7:
                return 50.0
            else:
                return 20.0
        
        return 10.0
    
    def score_adversarial_auc(self, result: DiagnosticResult) -> float:
        """Score adversarial validation (0-100, higher = more risk).
        
        Args:
            result: Adversarial validation result
            
        Returns:
            Risk score component
        """
        if not result.passed:
            auc = result.score
            # High AUC = can distinguish train/test = high risk
            if auc > 0.8:
                return 90.0
            elif auc > 0.7:
                return 70.0
            elif auc > 0.65:
                return 50.0
            else:
                return 20.0
        
        return 10.0
    
    def score_bootstrap_ci(self, result: DiagnosticResult) -> float:
        """Score bootstrap CI consistency (0-100, higher = more risk).
        
        Args:
            result: Block bootstrap result
            
        Returns:
            Risk score component
        """
        if not result.passed:
            # If observed Sharpe outside CI, high risk
            return 80.0
        
        return 10.0
    
    def calculate_risk_score(
        self,
        diagnostic_results: Dict[str, DiagnosticResult]
    ) -> OverfitRiskScore:
        """Calculate overall Overfit Risk Score.
        
        Args:
            diagnostic_results: Dictionary of diagnostic test results
            
        Returns:
            OverfitRiskScore object
        """
        component_scores = {}
        total_weighted_score = 0.0
        total_weight = 0.0
        
        # Score each component
        scorers = {
            "random_label": self.score_random_label,
            "pbo": self.score_pbo,
            "dsr": self.score_dsr,
            "cost_sensitivity": self.score_cost_sensitivity,
            "parameter_stability": self.score_parameter_flatness,
            "regime_robustness": self.score_regime_consistency,
            "adversarial_validation": self.score_adversarial_auc,
            "block_bootstrap": self.score_bootstrap_ci
        }
        
        for test_name, scorer in scorers.items():
            if test_name in diagnostic_results:
                result = diagnostic_results[test_name]
                score = scorer(result)
                component_scores[test_name] = score
                
                weight = self.WEIGHTS.get(test_name, 10)
                total_weighted_score += score * weight
                total_weight += weight
        
        # Calculate weighted average
        if total_weight > 0:
            total_score = total_weighted_score / total_weight
        else:
            total_score = 50.0  # Neutral if no tests
        
        # Determine risk level
        if total_score >= 60:
            risk_level = "high"
        elif total_score >= 40:
            risk_level = "medium"
        elif total_score >= 20:
            risk_level = "low"
        else:
            risk_level = "very_low"
        
        # Check for critical failures
        critical_tests = ["random_label", "future_shift"]
        for test_name in critical_tests:
            if test_name in diagnostic_results:
                result = diagnostic_results[test_name]
                if not result.passed and result.score < 0:
                    risk_level = "critical"
                    break
        
        return OverfitRiskScore(
            total_score=total_score,
            risk_level=risk_level,
            component_scores=component_scores,
            details={
                "total_weight": total_weight,
                "n_tests": len(component_scores)
            }
        )


