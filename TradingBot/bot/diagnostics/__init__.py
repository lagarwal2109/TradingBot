"""Diagnostics module for overfitting detection and strategy validation."""

from .diagnostics import DiagnosticsSuite
from .data_integrity import DataIntegrityChecker
from .time_series_splitter import TimeSeriesSplitter
from .backtester import CostAwareBacktester, BacktestMetrics
from .overfitting_tests import OverfittingDiagnostics, DiagnosticResult
from .risk_scorer import RiskScorer, OverfitRiskScore
from .reporter import DiagnosticsReporter
from .tuner import NestedWFOTuner, TuningResult

__all__ = [
    "DiagnosticsSuite",
    "DataIntegrityChecker",
    "TimeSeriesSplitter",
    "CostAwareBacktester",
    "BacktestMetrics",
    "OverfittingDiagnostics",
    "DiagnosticResult",
    "RiskScorer",
    "OverfitRiskScore",
    "DiagnosticsReporter",
    "NestedWFOTuner",
    "TuningResult",
]

