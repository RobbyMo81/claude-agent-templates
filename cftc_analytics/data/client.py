"""CFTC data client for fetching COT reports from Socrata API."""

import requests
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import pandas as pd
from urllib.parse import urlencode

from cftc_analytics.data.models import ReportType, COTRecord, REPORT_RESOURCE_IDS


class CFTCClient:
    """Client for fetching CFTC Commitments of Traders data."""

    BASE_URL = "https://publicreporting.cftc.gov/resource"
    DEFAULT_LIMIT = 50000  # Maximum records per request

    def __init__(self, app_token: Optional[str] = None):
        """
        Initialize CFTC client.

        Args:
            app_token: Optional Socrata app token for higher rate limits
        """
        self.app_token = app_token
        self.session = requests.Session()
        if app_token:
            self.session.headers.update({"X-App-Token": app_token})

    def _build_url(self, report_type: ReportType, params: Optional[Dict[str, Any]] = None) -> str:
        """Build API URL with query parameters."""
        resource_id = REPORT_RESOURCE_IDS[report_type]
        url = f"{self.BASE_URL}/{resource_id}.json"

        if params:
            url += f"?{urlencode(params, safe='$():>=<')}"

        return url

    def fetch_latest(
        self,
        report_type: ReportType = ReportType.DISAGGREGATED_FUTURES,
        commodity: Optional[str] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch latest COT data.

        Args:
            report_type: Type of COT report to fetch
            commodity: Optional commodity name filter
            limit: Maximum number of records to fetch

        Returns:
            DataFrame containing COT data
        """
        params = {"$limit": limit, "$order": "report_date_as_yyyy_mm_dd DESC"}

        if commodity:
            params["$where"] = f"UPPER(market_and_exchange_names) LIKE '%{commodity.upper()}%'"

        url = self._build_url(report_type, params)
        response = self.session.get(url, timeout=30)
        response.raise_for_status()

        data = response.json()
        return pd.DataFrame(data)

    def fetch_by_date_range(
        self,
        report_type: ReportType,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        commodity: Optional[str] = None,
        limit: int = 50000,
    ) -> pd.DataFrame:
        """
        Fetch COT data for a specific date range.

        Args:
            report_type: Type of COT report to fetch
            start_date: Start date for data retrieval
            end_date: End date for data retrieval (default: today)
            commodity: Optional commodity name filter
            limit: Maximum number of records to fetch

        Returns:
            DataFrame containing COT data
        """
        if end_date is None:
            end_date = datetime.now()

        start_str = start_date.strftime("%Y-%m-%dT00:00:00.000")
        end_str = end_date.strftime("%Y-%m-%dT23:59:59.999")

        where_clauses = [f"report_date_as_yyyy_mm_dd BETWEEN '{start_str}' AND '{end_str}'"]

        if commodity:
            where_clauses.append(f"UPPER(market_and_exchange_names) LIKE '%{commodity.upper()}%'")

        params = {
            "$where": " AND ".join(where_clauses),
            "$limit": limit,
            "$order": "report_date_as_yyyy_mm_dd ASC",
        }

        url = self._build_url(report_type, params)
        response = self.session.get(url, timeout=30)
        response.raise_for_status()

        data = response.json()
        return pd.DataFrame(data)

    def fetch_commodity(
        self,
        commodity: str,
        report_type: ReportType = ReportType.DISAGGREGATED_FUTURES,
        weeks: int = 52,
    ) -> pd.DataFrame:
        """
        Fetch COT data for a specific commodity.

        Args:
            commodity: Commodity name (e.g., "GOLD", "CRUDE OIL", "BITCOIN")
            report_type: Type of COT report to fetch
            weeks: Number of weeks of historical data to fetch

        Returns:
            DataFrame containing COT data for the commodity
        """
        start_date = datetime.now() - timedelta(weeks=weeks)
        return self.fetch_by_date_range(
            report_type=report_type,
            start_date=start_date,
            commodity=commodity,
        )

    def get_available_commodities(
        self, report_type: ReportType = ReportType.DISAGGREGATED_FUTURES
    ) -> List[str]:
        """
        Get list of available commodities in the specified report.

        Args:
            report_type: Type of COT report

        Returns:
            List of commodity names
        """
        params = {
            "$select": "DISTINCT market_and_exchange_names",
            "$order": "market_and_exchange_names ASC",
            "$limit": 1000,
        }

        url = self._build_url(report_type, params)
        response = self.session.get(url, timeout=30)
        response.raise_for_status()

        data = response.json()
        return [item["market_and_exchange_names"] for item in data]

    def fetch_multiple_commodities(
        self,
        commodities: List[str],
        report_type: ReportType = ReportType.DISAGGREGATED_FUTURES,
        weeks: int = 52,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch COT data for multiple commodities.

        Args:
            commodities: List of commodity names
            report_type: Type of COT report to fetch
            weeks: Number of weeks of historical data to fetch

        Returns:
            Dictionary mapping commodity names to DataFrames
        """
        result = {}
        for commodity in commodities:
            try:
                result[commodity] = self.fetch_commodity(commodity, report_type, weeks)
            except Exception as e:
                print(f"Error fetching {commodity}: {e}")
                result[commodity] = pd.DataFrame()

        return result

    def fetch_records(
        self,
        report_type: ReportType,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        commodity: Optional[str] = None,
    ) -> List[COTRecord]:
        """
        Fetch COT data as structured COTRecord objects.

        Args:
            report_type: Type of COT report to fetch
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            commodity: Optional commodity name filter

        Returns:
            List of COTRecord objects
        """
        df = self.fetch_by_date_range(report_type, start_date, end_date, commodity)
        records = []

        for _, row in df.iterrows():
            try:
                record = COTRecord.from_dict(row.to_dict(), report_type)
                records.append(record)
            except Exception as e:
                print(f"Error parsing record: {e}")
                continue

        return records
