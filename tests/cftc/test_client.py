"""Tests for CFTC data client."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import pandas as pd

from cftc_analytics.data.client import CFTCClient
from cftc_analytics.data.models import ReportType


@pytest.fixture
def client():
    """Create CFTC client for testing."""
    return CFTCClient()


@pytest.fixture
def mock_api_response():
    """Mock API response data."""
    return [
        {
            "report_date_as_yyyy_mm_dd": "2024-01-09T00:00:00.000",
            "market_and_exchange_names": "GOLD - COMMODITY EXCHANGE INC.",
            "commodity_code": "088581",
            "open_interest_all": "500000",
            "m_money_positions_long_all": "200000",
            "m_money_positions_short_all": "50000",
            "prod_merc_positions_long_all": "100000",
            "prod_merc_positions_short_all": "180000",
            "swap_positions_long_all": "80000",
            "swap_positions_short_all": "150000",
            "cftc_contract_market_code": "001612",
            "cftc_market_code": "001",
        },
        {
            "report_date_as_yyyy_mm_dd": "2024-01-02T00:00:00.000",
            "market_and_exchange_names": "GOLD - COMMODITY EXCHANGE INC.",
            "commodity_code": "088581",
            "open_interest_all": "480000",
            "m_money_positions_long_all": "190000",
            "m_money_positions_short_all": "55000",
            "prod_merc_positions_long_all": "95000",
            "prod_merc_positions_short_all": "175000",
            "swap_positions_long_all": "85000",
            "swap_positions_short_all": "145000",
            "cftc_contract_market_code": "001612",
            "cftc_market_code": "001",
        },
    ]


class TestCFTCClient:
    """Test CFTC client functionality."""

    def test_client_initialization(self):
        """Test client initialization."""
        client = CFTCClient()
        assert client.app_token is None
        assert client.session is not None

        client_with_token = CFTCClient(app_token="test_token")
        assert client_with_token.app_token == "test_token"
        assert "X-App-Token" in client_with_token.session.headers

    def test_build_url(self, client):
        """Test URL building."""
        url = client._build_url(ReportType.DISAGGREGATED_FUTURES)
        assert "publicreporting.cftc.gov" in url
        assert "72hh-3qpy" in url  # Disaggregated futures resource ID
        assert url.endswith(".json")

    def test_build_url_with_params(self, client):
        """Test URL building with parameters."""
        params = {"$limit": 1000, "$order": "report_date_as_yyyy_mm_dd DESC"}
        url = client._build_url(ReportType.DISAGGREGATED_FUTURES, params)
        assert "$limit=1000" in url
        assert "$order=report_date_as_yyyy_mm_dd" in url

    @patch("cftc_analytics.data.client.requests.Session.get")
    def test_fetch_latest(self, mock_get, client, mock_api_response):
        """Test fetching latest data."""
        mock_get.return_value.json.return_value = mock_api_response
        mock_get.return_value.raise_for_status = Mock()

        df = client.fetch_latest(
            report_type=ReportType.DISAGGREGATED_FUTURES,
            commodity="GOLD",
            limit=100
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "market_and_exchange_names" in df.columns
        mock_get.assert_called_once()

    @patch("cftc_analytics.data.client.requests.Session.get")
    def test_fetch_by_date_range(self, mock_get, client, mock_api_response):
        """Test fetching data by date range."""
        mock_get.return_value.json.return_value = mock_api_response
        mock_get.return_value.raise_for_status = Mock()

        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)

        df = client.fetch_by_date_range(
            report_type=ReportType.DISAGGREGATED_FUTURES,
            start_date=start_date,
            end_date=end_date
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        mock_get.assert_called_once()

        # Verify date range in URL
        call_args = mock_get.call_args[0][0]
        assert "2024-01-01" in call_args
        assert "2024-01-31" in call_args

    @patch("cftc_analytics.data.client.requests.Session.get")
    def test_fetch_commodity(self, mock_get, client, mock_api_response):
        """Test fetching commodity data."""
        mock_get.return_value.json.return_value = mock_api_response
        mock_get.return_value.raise_for_status = Mock()

        df = client.fetch_commodity(
            commodity="GOLD",
            report_type=ReportType.DISAGGREGATED_FUTURES,
            weeks=52
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        mock_get.assert_called_once()

    @patch("cftc_analytics.data.client.requests.Session.get")
    def test_get_available_commodities(self, mock_get, client):
        """Test getting available commodities."""
        mock_response = [
            {"market_and_exchange_names": "GOLD - COMMODITY EXCHANGE INC."},
            {"market_and_exchange_names": "SILVER - COMMODITY EXCHANGE INC."},
            {"market_and_exchange_names": "CRUDE OIL - NEW YORK MERCANTILE EXCHANGE"},
        ]

        mock_get.return_value.json.return_value = mock_response
        mock_get.return_value.raise_for_status = Mock()

        commodities = client.get_available_commodities(ReportType.DISAGGREGATED_FUTURES)

        assert isinstance(commodities, list)
        assert len(commodities) == 3
        assert "GOLD - COMMODITY EXCHANGE INC." in commodities
        mock_get.assert_called_once()

    @patch("cftc_analytics.data.client.requests.Session.get")
    def test_fetch_multiple_commodities(self, mock_get, client, mock_api_response):
        """Test fetching multiple commodities."""
        mock_get.return_value.json.return_value = mock_api_response
        mock_get.return_value.raise_for_status = Mock()

        commodities = ["GOLD", "SILVER"]
        result = client.fetch_multiple_commodities(
            commodities=commodities,
            report_type=ReportType.DISAGGREGATED_FUTURES,
            weeks=26
        )

        assert isinstance(result, dict)
        assert len(result) == 2
        assert "GOLD" in result
        assert "SILVER" in result
        assert isinstance(result["GOLD"], pd.DataFrame)

    @patch("cftc_analytics.data.client.requests.Session.get")
    def test_api_error_handling(self, mock_get, client):
        """Test API error handling."""
        mock_get.return_value.raise_for_status.side_effect = Exception("API Error")

        with pytest.raises(Exception):
            client.fetch_latest(ReportType.DISAGGREGATED_FUTURES)


@pytest.mark.integration
class TestCFTCClientIntegration:
    """Integration tests for CFTC client (requires network access)."""

    @pytest.mark.skip(reason="Requires network access and may be rate-limited")
    def test_real_api_call(self):
        """Test real API call to CFTC."""
        client = CFTCClient()

        df = client.fetch_latest(
            report_type=ReportType.DISAGGREGATED_FUTURES,
            limit=10
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "market_and_exchange_names" in df.columns

    @pytest.mark.skip(reason="Requires network access and may be rate-limited")
    def test_fetch_real_commodity(self):
        """Test fetching real commodity data."""
        client = CFTCClient()

        df = client.fetch_commodity(
            commodity="GOLD",
            report_type=ReportType.DISAGGREGATED_FUTURES,
            weeks=4
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
