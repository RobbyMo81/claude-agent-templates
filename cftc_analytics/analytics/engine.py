"""Main analytics engine for CFTC COT data."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from cftc_analytics.data.client import CFTCClient
from cftc_analytics.data.models import ReportType
from cftc_analytics.analytics.indicators import (
    calculate_net_positions,
    calculate_sentiment_index,
    calculate_extremes,
    detect_position_changes,
    calculate_cot_index,
    calculate_open_interest_trend,
    calculate_hedger_speculator_ratio,
    calculate_trader_participation,
    identify_divergences,
)


class CFTCAnalytics:
    """Main analytics engine for CFTC COT data."""

    def __init__(self, client: Optional[CFTCClient] = None):
        """
        Initialize analytics engine.

        Args:
            client: Optional CFTCClient instance. If not provided, creates new one.
        """
        self.client = client or CFTCClient()

    def analyze_commodity(
        self,
        commodity: str,
        report_type: ReportType = ReportType.DISAGGREGATED_FUTURES,
        weeks: int = 52,
    ) -> Dict[str, pd.DataFrame]:
        """
        Perform comprehensive analysis on a commodity.

        Args:
            commodity: Commodity name
            report_type: Type of COT report
            weeks: Number of weeks of historical data

        Returns:
            Dictionary containing various analysis results
        """
        # Fetch data
        df = self.client.fetch_commodity(commodity, report_type, weeks)

        if df.empty:
            return {"error": "No data available for this commodity"}

        # Ensure report_date is datetime
        if "report_date_as_yyyy_mm_dd" in df.columns:
            df["report_date"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"])
        elif "report_date_as_mm_dd_yyyy" in df.columns:
            df["report_date"] = pd.to_datetime(df["report_date_as_mm_dd_yyyy"])

        df = df.sort_values("report_date")

        # Determine trader categories based on report type
        categories = self._get_trader_categories(report_type)

        results = {
            "raw_data": df,
            "summary": self._generate_summary(df),
            "trader_analysis": {},
            "open_interest": calculate_open_interest_trend(df),
            "participation": calculate_trader_participation(df),
        }

        # Analyze each trader category
        for category_name, category_prefix in categories.items():
            results["trader_analysis"][category_name] = self._analyze_trader_category(
                df, category_prefix
            )

        # Add hedger/speculator ratio if applicable
        if report_type in [ReportType.LEGACY_FUTURES, ReportType.LEGACY_COMBINED]:
            results["hedger_speculator_ratio"] = calculate_hedger_speculator_ratio(df)

        return results

    def compare_traders(
        self,
        commodity: str,
        report_type: ReportType = ReportType.DISAGGREGATED_FUTURES,
        weeks: int = 52,
    ) -> pd.DataFrame:
        """
        Compare positioning across different trader categories.

        Args:
            commodity: Commodity name
            report_type: Type of COT report
            weeks: Number of weeks of historical data

        Returns:
            DataFrame with comparative analysis
        """
        df = self.client.fetch_commodity(commodity, report_type, weeks)

        if df.empty:
            return pd.DataFrame()

        # Ensure report_date is datetime
        if "report_date_as_yyyy_mm_dd" in df.columns:
            df["report_date"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"])

        df = df.sort_values("report_date")

        categories = self._get_trader_categories(report_type)
        comparison = pd.DataFrame({"report_date": df["report_date"]})

        for category_name, category_prefix in categories.items():
            net_pos = calculate_net_positions(df, category_prefix)
            sentiment = calculate_sentiment_index(df, category_prefix)

            comparison[f"{category_name}_net"] = net_pos
            comparison[f"{category_name}_sentiment"] = sentiment

        return comparison

    def find_extremes(
        self,
        commodity: str,
        report_type: ReportType = ReportType.DISAGGREGATED_FUTURES,
        weeks: int = 104,
        threshold: float = 90,
    ) -> Dict[str, pd.DataFrame]:
        """
        Find periods of extreme positioning.

        Args:
            commodity: Commodity name
            report_type: Type of COT report
            weeks: Number of weeks of historical data
            threshold: Percentile threshold for extremes

        Returns:
            Dictionary of extreme periods by trader category
        """
        df = self.client.fetch_commodity(commodity, report_type, weeks)

        if df.empty:
            return {}

        if "report_date_as_yyyy_mm_dd" in df.columns:
            df["report_date"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"])

        df = df.sort_values("report_date")

        categories = self._get_trader_categories(report_type)
        extremes = {}

        for category_name, category_prefix in categories.items():
            bullish, bearish = calculate_extremes(df, category_prefix, 52, threshold)

            net_pos = calculate_net_positions(df, category_prefix)
            sentiment = calculate_sentiment_index(df, category_prefix)

            extremes_df = pd.DataFrame(
                {
                    "report_date": df["report_date"].reset_index(drop=True),
                    "bullish_extreme": bullish.reset_index(drop=True),
                    "bearish_extreme": bearish.reset_index(drop=True),
                    "net_position": net_pos.reset_index(drop=True),
                    "sentiment": sentiment.reset_index(drop=True),
                }
            )

            # Filter to only extreme periods
            extremes[category_name] = extremes_df[extremes_df["bullish_extreme"] | extremes_df["bearish_extreme"]]

        return extremes

    def detect_major_shifts(
        self,
        commodity: str,
        report_type: ReportType = ReportType.DISAGGREGATED_FUTURES,
        weeks: int = 26,
        threshold_pct: float = 15.0,
    ) -> Dict[str, pd.DataFrame]:
        """
        Detect major position shifts.

        Args:
            commodity: Commodity name
            report_type: Type of COT report
            weeks: Number of weeks of historical data
            threshold_pct: Percentage threshold for significant change

        Returns:
            Dictionary of major shifts by trader category
        """
        df = self.client.fetch_commodity(commodity, report_type, weeks)

        if df.empty:
            return {}

        if "report_date_as_yyyy_mm_dd" in df.columns:
            df["report_date"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"])

        df = df.sort_values("report_date")

        categories = self._get_trader_categories(report_type)
        shifts = {}

        for category_name, category_prefix in categories.items():
            changes = detect_position_changes(df, category_prefix, threshold_pct)
            changes["report_date"] = df["report_date"].values

            # Filter to only significant changes
            significant = changes[
                changes["significant_increase"] | changes["significant_decrease"]
            ]
            shifts[category_name] = significant

        return shifts

    def generate_report(
        self,
        commodity: str,
        report_type: ReportType = ReportType.DISAGGREGATED_FUTURES,
        weeks: int = 52,
    ) -> str:
        """
        Generate a text report summarizing COT analysis.

        Args:
            commodity: Commodity name
            report_type: Type of COT report
            weeks: Number of weeks of historical data

        Returns:
            Formatted text report
        """
        analysis = self.analyze_commodity(commodity, report_type, weeks)

        if "error" in analysis:
            return f"Error: {analysis['error']}"

        report = []
        report.append(f"CFTC Commitments of Traders Analysis")
        report.append(f"{'=' * 60}")
        report.append(f"Commodity: {commodity}")
        report.append(f"Report Type: {report_type.value}")
        report.append(f"Analysis Period: {weeks} weeks")
        report.append(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
        report.append("")

        # Summary
        summary = analysis["summary"]
        report.append("Summary")
        report.append("-" * 60)
        report.append(f"Total Records: {summary['total_records']}")
        report.append(f"Date Range: {summary['start_date']} to {summary['end_date']}")
        report.append(f"Latest Open Interest: {summary['latest_open_interest']:,.0f}")
        report.append("")

        # Trader Analysis
        report.append("Trader Category Analysis")
        report.append("-" * 60)

        for category_name, category_data in analysis["trader_analysis"].items():
            if category_data.empty:
                continue

            latest = category_data.iloc[-1]
            report.append(f"\n{category_name.upper()}")
            report.append(
                f"  Latest Net Position: {latest.get('net_position', 0):,.0f} contracts"
            )
            report.append(f"  Sentiment Index: {latest.get('sentiment', 0):.1f}/100")
            report.append(f"  COT Index: {latest.get('cot_index', 0):.1f}/100")

            if latest.get("bullish_extreme", False):
                report.append("  ⚠️  BULLISH EXTREME detected")
            if latest.get("bearish_extreme", False):
                report.append("  ⚠️  BEARISH EXTREME detected")

        # Open Interest Trend
        if not analysis["open_interest"].empty:
            oi_latest = analysis["open_interest"].iloc[-1]
            report.append("\nOpen Interest Trend")
            report.append("-" * 60)
            report.append(f"Current: {oi_latest.get('open_interest', 0):,.0f}")
            report.append(f"4-Week MA: {oi_latest.get('oi_ma', 0):,.0f}")
            report.append(f"Weekly Change: {oi_latest.get('oi_pct_change', 0):.2f}%")

            if oi_latest.get("trend_increasing", False):
                report.append("Trend: ↑ INCREASING")
            elif oi_latest.get("trend_decreasing", False):
                report.append("Trend: ↓ DECREASING")

        return "\n".join(report)

    def _get_trader_categories(self, report_type: ReportType) -> Dict[str, str]:
        """Get trader category prefixes for a report type."""
        if report_type in [ReportType.LEGACY_FUTURES, ReportType.LEGACY_COMBINED]:
            return {
                "commercial": "comm",
                "non_commercial": "noncomm",
                "non_reportable": "nonrept",
            }
        elif report_type in [ReportType.DISAGGREGATED_FUTURES, ReportType.DISAGGREGATED_COMBINED]:
            return {
                "producer_merchant": "prod_merc",
                "swap_dealer": "swap",
                "managed_money": "m_money",
                "other_reportable": "other_rept",
                "non_reportable": "nonrept",
            }
        elif report_type in [ReportType.TFF_FUTURES, ReportType.TFF_COMBINED]:
            return {
                "dealer_intermediary": "dealer",
                "asset_manager": "asset_mgr",
                "leveraged_funds": "lev_money",
                "other_reportable": "other_rept",
                "non_reportable": "nonrept",
            }
        else:
            return {}

    def _analyze_trader_category(self, df: pd.DataFrame, category_prefix: str) -> pd.DataFrame:
        """Perform analysis for a specific trader category."""
        result = pd.DataFrame()

        if not df.empty:
            result["report_date"] = df["report_date"]
            result["net_position"] = calculate_net_positions(df, category_prefix)
            result["sentiment"] = calculate_sentiment_index(df, category_prefix)
            result["cot_index"] = calculate_cot_index(df, category_prefix)

            bullish, bearish = calculate_extremes(df, category_prefix)
            result["bullish_extreme"] = bullish
            result["bearish_extreme"] = bearish

            changes = detect_position_changes(df, category_prefix)
            result["position_change"] = changes["change"]
            result["position_pct_change"] = changes["pct_change"]

        return result

    def _generate_summary(self, df: pd.DataFrame) -> Dict:
        """Generate summary statistics."""
        summary = {
            "total_records": len(df),
            "start_date": df["report_date"].min().strftime("%Y-%m-%d")
            if not df.empty
            else "N/A",
            "end_date": df["report_date"].max().strftime("%Y-%m-%d") if not df.empty else "N/A",
            "latest_open_interest": pd.to_numeric(
                df["open_interest_all"].iloc[-1], errors='coerce'
            )
            if not df.empty and "open_interest_all" in df.columns
            else 0,
        }
        return summary
