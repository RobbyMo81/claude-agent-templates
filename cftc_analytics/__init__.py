"""
CFTC Analytics Tool

A comprehensive data analytics tool for CFTC Commitments of Traders (COT) data.

This package provides:
- Data fetching from CFTC's Socrata API
- Analytics and indicators (net positioning, sentiment, trends)
- Visualization capabilities
- Integration with multi-agent system
"""

from cftc_analytics.data.client import CFTCClient
from cftc_analytics.analytics.engine import CFTCAnalytics
from cftc_analytics.data.models import ReportType, TraderCategory

__version__ = "0.1.0"

__all__ = [
    "CFTCClient",
    "CFTCAnalytics",
    "ReportType",
    "TraderCategory",
]
