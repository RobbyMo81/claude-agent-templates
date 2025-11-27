"""CFTC Data Agent for COT analysis within multi-agent system."""

from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import json

from agents.base.agent import Agent, AgentCapability, Task
from cftc_analytics.data.client import CFTCClient
from cftc_analytics.analytics.engine import CFTCAnalytics
from cftc_analytics.data.models import ReportType


class CFTCAgent(Agent):
    """
    Specialized agent for CFTC Commitments of Traders data analysis.

    Capabilities:
    - Fetch COT data from CFTC API
    - Analyze trader positioning and sentiment
    - Detect market extremes and position shifts
    - Generate analytics reports
    """

    def __init__(self, name: str = "CFTCAgent", app_token: Optional[str] = None):
        """
        Initialize CFTC agent.

        Args:
            name: Agent name
            app_token: Optional Socrata API app token
        """
        capabilities = {
            AgentCapability.CODE_ANALYSIS,  # DATA_ANALYSIS not in base enum, using CODE_ANALYSIS
            AgentCapability.DOCUMENTATION,  # REPORTING not in base enum, using DOCUMENTATION
        }

        super().__init__(name=name, capabilities=capabilities)

        self.client = CFTCClient(app_token=app_token)
        self.analytics = CFTCAnalytics(client=self.client)

    async def _execute_task_logic(self, task: Task) -> Any:
        """
        Execute CFTC analysis task.

        Task data should contain:
        - task_name: Name of the operation (fetch_commodity, analyze_commodity, etc.)
        - Other parameters specific to the task
        """
        task_name = task.data.get("task_name", "")
        return await self.execute_task_by_name(task_name, task.data)

    async def execute_task_by_name(self, task_name: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a CFTC data analysis task.

        Supported tasks:
        - fetch_commodity: Fetch COT data for a commodity
        - analyze_commodity: Perform comprehensive analysis
        - compare_traders: Compare trader category positioning
        - find_extremes: Find extreme positioning periods
        - detect_shifts: Detect major position shifts
        - generate_report: Generate text report
        - list_commodities: Get available commodities

        Args:
            task_name: Name of the task to execute
            context: Task context and parameters

        Returns:
            Task execution result
        """
        try:

            if task_name == "fetch_commodity":
                return await self._fetch_commodity(context)
            elif task_name == "analyze_commodity":
                return await self._analyze_commodity(context)
            elif task_name == "compare_traders":
                return await self._compare_traders(context)
            elif task_name == "find_extremes":
                return await self._find_extremes(context)
            elif task_name == "detect_shifts":
                return await self._detect_shifts(context)
            elif task_name == "generate_report":
                return await self._generate_report(context)
            elif task_name == "list_commodities":
                return await self._list_commodities(context)
            else:
                return {
                    "success": False,
                    "error": f"Unknown task: {task_name}",
                }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    async def _fetch_commodity(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch COT data for a commodity."""
        commodity = context.get("commodity", "")
        report_type = context.get("report_type", ReportType.DISAGGREGATED_FUTURES)
        weeks = context.get("weeks", 52)

        if isinstance(report_type, str):
            report_type = ReportType[report_type.upper()]

        df = self.client.fetch_commodity(commodity, report_type, weeks)

        return {
            "success": True,
            "task": "fetch_commodity",
            "commodity": commodity,
            "records": len(df),
            "data": df.to_dict(orient="records")[:10],  # Return first 10 for preview
            "message": f"Fetched {len(df)} records for {commodity}",
        }

    async def _analyze_commodity(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive analysis on a commodity."""
        commodity = context.get("commodity", "")
        report_type = context.get("report_type", ReportType.DISAGGREGATED_FUTURES)
        weeks = context.get("weeks", 52)

        if isinstance(report_type, str):
            report_type = ReportType[report_type.upper()]

        analysis = self.analytics.analyze_commodity(commodity, report_type, weeks)

        # Convert DataFrames to serializable format
        serialized_analysis = {}
        for key, value in analysis.items():
            if hasattr(value, "to_dict"):
                serialized_analysis[key] = value.to_dict(orient="records")
            elif isinstance(value, dict):
                serialized_analysis[key] = {
                    k: v.to_dict(orient="records") if hasattr(v, "to_dict") else v
                    for k, v in value.items()
                }
            else:
                serialized_analysis[key] = value

        return {
            "success": True,
            "task": "analyze_commodity",
            "commodity": commodity,
            "analysis": serialized_analysis,
            "message": f"Analysis complete for {commodity}",
        }

    async def _compare_traders(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Compare trader category positioning."""
        commodity = context.get("commodity", "")
        report_type = context.get("report_type", ReportType.DISAGGREGATED_FUTURES)
        weeks = context.get("weeks", 52)

        if isinstance(report_type, str):
            report_type = ReportType[report_type.upper()]

        comparison = self.analytics.compare_traders(commodity, report_type, weeks)

        return {
            "success": True,
            "task": "compare_traders",
            "commodity": commodity,
            "comparison": comparison.to_dict(orient="records"),
            "message": f"Trader comparison complete for {commodity}",
        }

    async def _find_extremes(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Find extreme positioning periods."""
        commodity = context.get("commodity", "")
        report_type = context.get("report_type", ReportType.DISAGGREGATED_FUTURES)
        weeks = context.get("weeks", 104)
        threshold = context.get("threshold", 90)

        if isinstance(report_type, str):
            report_type = ReportType[report_type.upper()]

        extremes = self.analytics.find_extremes(commodity, report_type, weeks, threshold)

        serialized_extremes = {
            category: df.to_dict(orient="records") for category, df in extremes.items()
        }

        return {
            "success": True,
            "task": "find_extremes",
            "commodity": commodity,
            "extremes": serialized_extremes,
            "message": f"Found extreme periods for {commodity}",
        }

    async def _detect_shifts(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Detect major position shifts."""
        commodity = context.get("commodity", "")
        report_type = context.get("report_type", ReportType.DISAGGREGATED_FUTURES)
        weeks = context.get("weeks", 26)
        threshold_pct = context.get("threshold_pct", 15.0)

        if isinstance(report_type, str):
            report_type = ReportType[report_type.upper()]

        shifts = self.analytics.detect_major_shifts(commodity, report_type, weeks, threshold_pct)

        serialized_shifts = {
            category: df.to_dict(orient="records") for category, df in shifts.items()
        }

        return {
            "success": True,
            "task": "detect_shifts",
            "commodity": commodity,
            "shifts": serialized_shifts,
            "message": f"Detected position shifts for {commodity}",
        }

    async def _generate_report(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate text report."""
        commodity = context.get("commodity", "")
        report_type = context.get("report_type", ReportType.DISAGGREGATED_FUTURES)
        weeks = context.get("weeks", 52)

        if isinstance(report_type, str):
            report_type = ReportType[report_type.upper()]

        report = self.analytics.generate_report(commodity, report_type, weeks)

        return {
            "success": True,
            "task": "generate_report",
            "commodity": commodity,
            "report": report,
            "message": f"Report generated for {commodity}",
        }

    async def _list_commodities(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """List available commodities."""
        report_type = context.get("report_type", ReportType.DISAGGREGATED_FUTURES)

        if isinstance(report_type, str):
            report_type = ReportType[report_type.upper()]

        commodities = self.client.get_available_commodities(report_type)

        return {
            "success": True,
            "task": "list_commodities",
            "commodities": commodities,
            "count": len(commodities),
            "message": f"Found {len(commodities)} commodities",
        }
