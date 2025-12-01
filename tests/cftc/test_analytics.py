"""Tests for CFTC analytics engine."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from cftc_analytics.analytics.engine import CFTCAnalytics
from cftc_analytics.analytics.indicators import (
    calculate_net_positions,
    calculate_sentiment_index,
    calculate_extremes,
    detect_position_changes,
    calculate_cot_index,
    calculate_open_interest_trend,
)
from cftc_analytics.data.models import ReportType


@pytest.fixture
def sample_cot_data():
    """Create sample COT data for testing."""
    dates = pd.date_range(start="2024-01-01", periods=52, freq="W")

    data = {
        "report_date": dates,
        "report_date_as_yyyy_mm_dd": dates,
        "market_and_exchange_names": ["GOLD - COMMODITY EXCHANGE INC."] * 52,
        "open_interest_all": np.random.randint(400000, 600000, 52),
        "m_money_positions_long_all": np.random.randint(150000, 250000, 52),
        "m_money_positions_short_all": np.random.randint(40000, 80000, 52),
        "prod_merc_positions_long_all": np.random.randint(80000, 120000, 52),
        "prod_merc_positions_short_all": np.random.randint(150000, 200000, 52),
        "swap_positions_long_all": np.random.randint(70000, 100000, 52),
        "swap_positions_short_all": np.random.randint(130000, 170000, 52),
        "cftc_contract_market_code": ["001612"] * 52,
        "cftc_market_code": ["001"] * 52,
    }

    return pd.DataFrame(data)


@pytest.fixture
def analytics_engine():
    """Create analytics engine for testing."""
    return CFTCAnalytics()


class TestIndicators:
    """Test analytics indicators."""

    def test_calculate_net_positions(self, sample_cot_data):
        """Test net position calculation."""
        net_pos = calculate_net_positions(sample_cot_data, "m_money")

        assert isinstance(net_pos, pd.Series)
        assert len(net_pos) == len(sample_cot_data)
        assert not net_pos.isna().all()

        # Net position should be long - short
        expected = (
            sample_cot_data["m_money_positions_long_all"]
            - sample_cot_data["m_money_positions_short_all"]
        )
        pd.testing.assert_series_equal(net_pos, expected, check_names=False)

    def test_calculate_net_positions_missing_columns(self):
        """Test net position with missing columns."""
        df = pd.DataFrame({"other_column": [1, 2, 3]})
        net_pos = calculate_net_positions(df, "m_money")

        assert isinstance(net_pos, pd.Series)
        assert len(net_pos) == 0

    def test_calculate_sentiment_index(self, sample_cot_data):
        """Test sentiment index calculation."""
        sentiment = calculate_sentiment_index(sample_cot_data, "m_money")

        assert isinstance(sentiment, pd.Series)
        assert len(sentiment) == len(sample_cot_data)
        assert (sentiment >= 0).all()
        assert (sentiment <= 100).all()

    def test_calculate_extremes(self, sample_cot_data):
        """Test extreme detection."""
        bullish, bearish = calculate_extremes(sample_cot_data, "m_money", threshold=90)

        assert isinstance(bullish, pd.Series)
        assert isinstance(bearish, pd.Series)
        assert len(bullish) == len(sample_cot_data)
        assert len(bearish) == len(sample_cot_data)
        assert bullish.dtype == bool
        assert bearish.dtype == bool

    def test_detect_position_changes(self, sample_cot_data):
        """Test position change detection."""
        changes = detect_position_changes(sample_cot_data, "m_money", threshold_pct=10.0)

        assert isinstance(changes, pd.DataFrame)
        assert len(changes) == len(sample_cot_data)
        assert "net_position" in changes.columns
        assert "change" in changes.columns
        assert "pct_change" in changes.columns
        assert "significant_increase" in changes.columns
        assert "significant_decrease" in changes.columns

    def test_calculate_cot_index(self, sample_cot_data):
        """Test COT index calculation."""
        cot_index = calculate_cot_index(sample_cot_data, "m_money", smoothing=3)

        assert isinstance(cot_index, pd.Series)
        assert len(cot_index) == len(sample_cot_data)
        assert (cot_index >= 0).all()
        assert (cot_index <= 100).all()

    def test_calculate_open_interest_trend(self, sample_cot_data):
        """Test open interest trend calculation."""
        oi_trend = calculate_open_interest_trend(sample_cot_data)

        assert isinstance(oi_trend, pd.DataFrame)
        assert len(oi_trend) == len(sample_cot_data)
        assert "open_interest" in oi_trend.columns
        assert "oi_ma" in oi_trend.columns
        assert "oi_change" in oi_trend.columns
        assert "oi_pct_change" in oi_trend.columns
        assert "trend_increasing" in oi_trend.columns
        assert "trend_decreasing" in oi_trend.columns


class TestCFTCAnalytics:
    """Test CFTC analytics engine."""

    @patch("cftc_analytics.analytics.engine.CFTCClient.fetch_commodity")
    def test_analyze_commodity(self, mock_fetch, analytics_engine, sample_cot_data):
        """Test commodity analysis."""
        mock_fetch.return_value = sample_cot_data

        result = analytics_engine.analyze_commodity(
            commodity="GOLD",
            report_type=ReportType.DISAGGREGATED_FUTURES,
            weeks=52
        )

        assert isinstance(result, dict)
        assert "raw_data" in result
        assert "summary" in result
        assert "trader_analysis" in result
        assert "open_interest" in result
        assert "participation" in result

        # Check summary
        summary = result["summary"]
        assert summary["total_records"] == 52
        assert "start_date" in summary
        assert "end_date" in summary

        # Check trader analysis
        trader_analysis = result["trader_analysis"]
        assert "managed_money" in trader_analysis
        assert "producer_merchant" in trader_analysis
        assert "swap_dealer" in trader_analysis

    @patch("cftc_analytics.analytics.engine.CFTCClient.fetch_commodity")
    def test_analyze_commodity_empty_data(self, mock_fetch, analytics_engine):
        """Test analysis with empty data."""
        mock_fetch.return_value = pd.DataFrame()

        result = analytics_engine.analyze_commodity(
            commodity="UNKNOWN",
            report_type=ReportType.DISAGGREGATED_FUTURES,
            weeks=52
        )

        assert isinstance(result, dict)
        assert "error" in result

    @patch("cftc_analytics.analytics.engine.CFTCClient.fetch_commodity")
    def test_compare_traders(self, mock_fetch, analytics_engine, sample_cot_data):
        """Test trader comparison."""
        mock_fetch.return_value = sample_cot_data

        comparison = analytics_engine.compare_traders(
            commodity="GOLD",
            report_type=ReportType.DISAGGREGATED_FUTURES,
            weeks=52
        )

        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 52
        assert "report_date" in comparison.columns
        assert "managed_money_net" in comparison.columns
        assert "managed_money_sentiment" in comparison.columns

    @patch("cftc_analytics.analytics.engine.CFTCClient.fetch_commodity")
    def test_find_extremes(self, mock_fetch, analytics_engine, sample_cot_data):
        """Test finding extremes."""
        mock_fetch.return_value = sample_cot_data

        extremes = analytics_engine.find_extremes(
            commodity="GOLD",
            report_type=ReportType.DISAGGREGATED_FUTURES,
            weeks=104,
            threshold=90
        )

        assert isinstance(extremes, dict)
        assert "managed_money" in extremes
        assert "producer_merchant" in extremes

        for category, df in extremes.items():
            assert isinstance(df, pd.DataFrame)
            if not df.empty:
                assert "report_date" in df.columns
                assert "bullish_extreme" in df.columns
                assert "bearish_extreme" in df.columns

    @patch("cftc_analytics.analytics.engine.CFTCClient.fetch_commodity")
    def test_detect_major_shifts(self, mock_fetch, analytics_engine, sample_cot_data):
        """Test detecting major shifts."""
        mock_fetch.return_value = sample_cot_data

        shifts = analytics_engine.detect_major_shifts(
            commodity="GOLD",
            report_type=ReportType.DISAGGREGATED_FUTURES,
            weeks=26,
            threshold_pct=15.0
        )

        assert isinstance(shifts, dict)
        assert "managed_money" in shifts

        for category, df in shifts.items():
            assert isinstance(df, pd.DataFrame)
            if not df.empty:
                assert "report_date" in df.columns
                assert "position_change" in df.columns or "change" in df.columns

    @patch("cftc_analytics.analytics.engine.CFTCClient.fetch_commodity")
    def test_generate_report(self, mock_fetch, analytics_engine, sample_cot_data):
        """Test report generation."""
        mock_fetch.return_value = sample_cot_data

        report = analytics_engine.generate_report(
            commodity="GOLD",
            report_type=ReportType.DISAGGREGATED_FUTURES,
            weeks=52
        )

        assert isinstance(report, str)
        assert "GOLD" in report
        assert "Summary" in report
        assert "Trader Category Analysis" in report
        assert len(report) > 0

    def test_get_trader_categories(self, analytics_engine):
        """Test trader category mapping."""
        # Disaggregated report
        categories = analytics_engine._get_trader_categories(ReportType.DISAGGREGATED_FUTURES)
        assert "managed_money" in categories
        assert "producer_merchant" in categories
        assert "swap_dealer" in categories

        # Legacy report
        categories = analytics_engine._get_trader_categories(ReportType.LEGACY_FUTURES)
        assert "commercial" in categories
        assert "non_commercial" in categories

        # TFF report
        categories = analytics_engine._get_trader_categories(ReportType.TFF_FUTURES)
        assert "dealer_intermediary" in categories
        assert "asset_manager" in categories
        assert "leveraged_funds" in categories


@pytest.mark.integration
class TestAnalyticsIntegration:
    """Integration tests for analytics engine."""

    @pytest.mark.skip(reason="Requires network access")
    def test_real_commodity_analysis(self):
        """Test analysis with real data."""
        analytics = CFTCAnalytics()

        result = analytics.analyze_commodity(
            commodity="GOLD",
            report_type=ReportType.DISAGGREGATED_FUTURES,
            weeks=4
        )

        assert isinstance(result, dict)
        assert "summary" in result
        assert result["summary"]["total_records"] > 0
