"""HTML report generator for overfitting diagnostics."""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
import numpy as np

from .risk_scorer import OverfitRiskScore
from .overfitting_tests import DiagnosticResult
from .data_integrity import IntegrityCheckResult


class DiagnosticsReporter:
    """Generate HTML reports for diagnostic results."""
    
    def __init__(self, output_dir: Path):
        """Initialize reporter.
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_html_report(
        self,
        diagnostic_results: Dict[str, Any],
        risk_score: OverfitRiskScore,
        equity_curves: Optional[Dict[str, pd.DataFrame]] = None
    ) -> Path:
        """Generate comprehensive HTML report.
        
        Args:
            diagnostic_results: Dictionary with all diagnostic results
            risk_score: OverfitRiskScore object
            equity_curves: Optional dictionary of equity curves
            
        Returns:
            Path to generated HTML file
        """
        html_content = self._generate_html_content(
            diagnostic_results, risk_score, equity_curves
        )
        
        output_path = self.output_dir / f"overfitting_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        output_path.write_text(html_content, encoding='utf-8')
        
        return output_path
    
    def _generate_html_content(
        self,
        diagnostic_results: Dict[str, Any],
        risk_score: OverfitRiskScore,
        equity_curves: Optional[Dict[str, pd.DataFrame]] = None
    ) -> str:
        """Generate HTML content string."""
        
        # Risk score banner
        risk_color = {
            "very_low": "#28a745",
            "low": "#6c757d",
            "medium": "#ffc107",
            "high": "#fd7e14",
            "critical": "#dc3545"
        }.get(risk_score.risk_level, "#6c757d")
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Overfitting Diagnostics Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #007bff;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
        }}
        .risk-banner {{
            background-color: {risk_color};
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            margin: 20px 0;
            font-size: 24px;
            font-weight: bold;
        }}
        .score {{
            font-size: 48px;
            margin: 10px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #007bff;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .pass {{
            color: #28a745;
            font-weight: bold;
        }}
        .fail {{
            color: #dc3545;
            font-weight: bold;
        }}
        .component-score {{
            display: inline-block;
            padding: 5px 10px;
            border-radius: 4px;
            font-weight: bold;
        }}
        .score-high {{
            background-color: #dc3545;
            color: white;
        }}
        .score-medium {{
            background-color: #ffc107;
            color: black;
        }}
        .score-low {{
            background-color: #28a745;
            color: white;
        }}
        .details {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 4px;
            margin: 10px 0;
            font-family: monospace;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Overfitting Diagnostics Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="risk-banner">
            <div>Overfit Risk Score</div>
            <div class="score">{risk_score.total_score:.1f}/100</div>
            <div>Risk Level: {risk_score.risk_level.upper().replace('_', ' ')}</div>
        </div>
        
        <h2>Component Scores</h2>
        <table>
            <tr>
                <th>Component</th>
                <th>Score</th>
                <th>Weight</th>
                <th>Status</th>
            </tr>
"""
        
        # Component scores
        weights = {
            "random_label": 20,
            "pbo": 20,
            "dsr": 15,
            "cost_sensitivity": 10,
            "parameter_stability": 10,
            "regime_robustness": 10,
            "adversarial_validation": 10,
            "block_bootstrap": 5
        }
        
        for component, score in risk_score.component_scores.items():
            weight = weights.get(component, 10)
            score_class = "score-high" if score >= 60 else "score-medium" if score >= 40 else "score-low"
            html += f"""
            <tr>
                <td>{component.replace('_', ' ').title()}</td>
                <td><span class="component-score {score_class}">{score:.1f}</span></td>
                <td>{weight}</td>
                <td>{'✓' if score < 40 else '✗'}</td>
            </tr>
"""
        
        html += """
        </table>
        
        <h2>Diagnostic Test Results</h2>
        <table>
            <tr>
                <th>Test</th>
                <th>Status</th>
                <th>Score</th>
                <th>Message</th>
            </tr>
"""
        
        # Diagnostic results
        diagnostics = diagnostic_results.get("diagnostics", {})
        for test_name, result in diagnostics.items():
            if isinstance(result, DiagnosticResult):
                status_class = "pass" if result.passed else "fail"
                status_text = "PASS" if result.passed else "FAIL"
                html += f"""
            <tr>
                <td>{result.test_name.replace('_', ' ').title()}</td>
                <td class="{status_class}">{status_text}</td>
                <td>{result.score:.3f}</td>
                <td>{result.message}</td>
            </tr>
"""
                if result.details:
                    html += f"""
            <tr>
                <td colspan="4">
                    <div class="details">
                        {json.dumps(result.details, indent=2, default=str)}
                    </div>
                </td>
            </tr>
"""
        
        html += """
        </table>
        
        <h2>Data Integrity Checks</h2>
        <table>
            <tr>
                <th>Check</th>
                <th>Status</th>
                <th>Severity</th>
                <th>Message</th>
            </tr>
"""
        
        # Integrity checks
        integrity = diagnostic_results.get("integrity", {})
        for check_name, result in integrity.items():
            if isinstance(result, IntegrityCheckResult):
                status_class = "pass" if result.passed else "fail"
                status_text = "PASS" if result.passed else "FAIL"
                html += f"""
            <tr>
                <td>{check_name.replace('_', ' ').title()}</td>
                <td class="{status_class}">{status_text}</td>
                <td>{result.severity.upper()}</td>
                <td>{result.message}</td>
            </tr>
"""
        
        html += """
        </table>
        
        <h2>Summary</h2>
        <p>
"""
        
        # Summary
        n_passed = sum(1 for r in diagnostics.values() if isinstance(r, DiagnosticResult) and r.passed)
        n_total = len([r for r in diagnostics.values() if isinstance(r, DiagnosticResult)])
        
        html += f"""
            Total Tests: {n_total}<br>
            Passed: {n_passed}<br>
            Failed: {n_total - n_passed}<br>
            <br>
            <strong>Recommendation:</strong> 
"""
        
        if risk_score.total_score >= 60:
            html += "HIGH RISK - Strategy likely overfit. Do not deploy without significant changes."
        elif risk_score.total_score >= 40:
            html += "MEDIUM RISK - Strategy may be overfit. Review failed tests and consider additional validation."
        else:
            html += "LOW RISK - Strategy appears robust. Proceed with caution and monitor performance."
        
        html += """
        </p>
    </div>
</body>
</html>
"""
        
        return html
    
    def save_json_report(
        self,
        diagnostic_results: Dict[str, Any],
        risk_score: OverfitRiskScore
    ) -> Path:
        """Save results as JSON.
        
        Args:
            diagnostic_results: Dictionary with all diagnostic results
            risk_score: OverfitRiskScore object
            
        Returns:
            Path to JSON file
        """
        # Convert to serializable format
        report_data = {
            "risk_score": {
                "total_score": risk_score.total_score,
                "risk_level": risk_score.risk_level,
                "component_scores": risk_score.component_scores,
                "details": risk_score.details
            },
            "diagnostics": {},
            "integrity": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Convert diagnostic results
        diagnostics = diagnostic_results.get("diagnostics", {})
        for test_name, result in diagnostics.items():
            if isinstance(result, DiagnosticResult):
                report_data["diagnostics"][test_name] = {
                    "test_name": result.test_name,
                    "passed": result.passed,
                    "score": result.score,
                    "message": result.message,
                    "details": result.details
                }
        
        # Convert integrity results
        integrity = diagnostic_results.get("integrity", {})
        for check_name, result in integrity.items():
            if isinstance(result, IntegrityCheckResult):
                report_data["integrity"][check_name] = {
                    "passed": result.passed,
                    "message": result.message,
                    "severity": result.severity,
                    "details": result.details
                }
        
        output_path = self.output_dir / f"overfitting_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_path.write_text(json.dumps(report_data, indent=2, default=str), encoding='utf-8')
        
        return output_path

