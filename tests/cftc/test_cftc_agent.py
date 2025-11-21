"""Tests for CFTC specialized agent."""

import pytest
import asyncio
from unittest.mock import Mock, patch
import pandas as pd

from agents.specialized.cftc_agent import CFTCAgent
from agents.base.agent import AgentCapability
from cftc_analytics.data.models import ReportType


@pytest.fixture
def cftc_agent():
    """Create CFTC agent for testing."""
    return CFTCAgent(name="TestCFTCAgent")


@pytest.fixture
def sample_commodity_data():
    """Create sample commodity data."""
    return pd.DataFrame({
        "report_date_as_yyyy_mm_dd": ["2024-01-09", "2024-01-02"],
        "market_and_exchange_names": ["GOLD"] * 2,
        "open_interest_all": [500000, 480000],
        "m_money_positions_long_all": [200000, 190000],
        "m_money_positions_short_all": [50000, 55000],
    })


class TestCFTCAgent:
    """Test CFTC agent functionality."""

    def test_agent_initialization(self, cftc_agent):
        """Test agent initialization."""
        assert cftc_agent.name == "TestCFTCAgent"
        assert AgentCapability.CODE_ANALYSIS in cftc_agent.capabilities
        assert AgentCapability.DOCUMENTATION in cftc_agent.capabilities
        assert cftc_agent.client is not None
        assert cftc_agent.analytics is not None

    @pytest.mark.asyncio
    @patch("cftc_analytics.data.client.CFTCClient.fetch_commodity")
    async def test_fetch_commodity_task(self, mock_fetch, cftc_agent, sample_commodity_data):
        """Test fetch commodity task."""
        mock_fetch.return_value = sample_commodity_data

        result = await cftc_agent.execute_task_by_name(
            task_name="fetch_commodity",
            context={
                "commodity": "GOLD",
                "report_type": "DISAGGREGATED_FUTURES",
                "weeks": 52
            }
        )

        assert result["success"] is True
        assert result["task"] == "fetch_commodity"
        assert result["commodity"] == "GOLD"
        assert result["records"] == 2
        assert "data" in result

    @pytest.mark.asyncio
    @patch("cftc_analytics.analytics.engine.CFTCAnalytics.analyze_commodity")
    async def test_analyze_commodity_task(self, mock_analyze, cftc_agent):
        """Test analyze commodity task."""
        mock_analyze.return_value = {
            "summary": {"total_records": 52, "start_date": "2024-01-01", "end_date": "2024-12-31"},
            "trader_analysis": {},
            "open_interest": pd.DataFrame(),
        }

        result = await cftc_agent.execute_task_by_name(
            task_name="analyze_commodity",
            context={
                "commodity": "GOLD",
                "report_type": "DISAGGREGATED_FUTURES",
                "weeks": 52
            }
        )

        assert result["success"] is True
        assert result["task"] == "analyze_commodity"
        assert result["commodity"] == "GOLD"
        assert "analysis" in result

    @pytest.mark.asyncio
    @patch("cftc_analytics.analytics.engine.CFTCAnalytics.compare_traders")
    async def test_compare_traders_task(self, mock_compare, cftc_agent):
        """Test compare traders task."""
        mock_compare.return_value = pd.DataFrame({
            "report_date": ["2024-01-01"],
            "managed_money_net": [150000],
            "managed_money_sentiment": [75.5],
        })

        result = await cftc_agent.execute_task_by_name(
            task_name="compare_traders",
            context={
                "commodity": "GOLD",
                "report_type": "DISAGGREGATED_FUTURES",
                "weeks": 52
            }
        )

        assert result["success"] is True
        assert result["task"] == "compare_traders"
        assert "comparison" in result

    @pytest.mark.asyncio
    @patch("cftc_analytics.analytics.engine.CFTCAnalytics.find_extremes")
    async def test_find_extremes_task(self, mock_extremes, cftc_agent):
        """Test find extremes task."""
        mock_extremes.return_value = {
            "managed_money": pd.DataFrame({
                "report_date": ["2024-01-01"],
                "bullish_extreme": [True],
            })
        }

        result = await cftc_agent.execute_task_by_name(
            task_name="find_extremes",
            context={
                "commodity": "GOLD",
                "report_type": "DISAGGREGATED_FUTURES",
                "weeks": 104,
                "threshold": 90
            }
        )

        assert result["success"] is True
        assert result["task"] == "find_extremes"
        assert "extremes" in result

    @pytest.mark.asyncio
    @patch("cftc_analytics.analytics.engine.CFTCAnalytics.detect_major_shifts")
    async def test_detect_shifts_task(self, mock_shifts, cftc_agent):
        """Test detect shifts task."""
        mock_shifts.return_value = {
            "managed_money": pd.DataFrame({
                "report_date": ["2024-01-01"],
                "position_change": [25000],
            })
        }

        result = await cftc_agent.execute_task_by_name(
            task_name="detect_shifts",
            context={
                "commodity": "GOLD",
                "report_type": "DISAGGREGATED_FUTURES",
                "weeks": 26,
                "threshold_pct": 15.0
            }
        )

        assert result["success"] is True
        assert result["task"] == "detect_shifts"
        assert "shifts" in result

    @pytest.mark.asyncio
    @patch("cftc_analytics.analytics.engine.CFTCAnalytics.generate_report")
    async def test_generate_report_task(self, mock_report, cftc_agent):
        """Test generate report task."""
        mock_report.return_value = "CFTC Report for GOLD\n===================="

        result = await cftc_agent.execute_task_by_name(
            task_name="generate_report",
            context={
                "commodity": "GOLD",
                "report_type": "DISAGGREGATED_FUTURES",
                "weeks": 52
            }
        )

        assert result["success"] is True
        assert result["task"] == "generate_report"
        assert "report" in result
        assert "GOLD" in result["report"]

    @pytest.mark.asyncio
    @patch("cftc_analytics.data.client.CFTCClient.get_available_commodities")
    async def test_list_commodities_task(self, mock_list, cftc_agent):
        """Test list commodities task."""
        mock_list.return_value = ["GOLD", "SILVER", "COPPER"]

        result = await cftc_agent.execute_task_by_name(
            task_name="list_commodities",
            context={
                "report_type": "DISAGGREGATED_FUTURES"
            }
        )

        assert result["success"] is True
        assert result["task"] == "list_commodities"
        assert "commodities" in result
        assert result["count"] == 3
        assert "GOLD" in result["commodities"]

    @pytest.mark.asyncio
    async def test_unknown_task(self, cftc_agent):
        """Test handling of unknown task."""
        result = await cftc_agent.execute_task_by_name(
            task_name="unknown_task",
            context={}
        )

        assert result["success"] is False
        assert "error" in result
        assert "Unknown task" in result["error"]

    @pytest.mark.asyncio
    @patch("cftc_analytics.data.client.CFTCClient.fetch_commodity")
    async def test_task_error_handling(self, mock_fetch, cftc_agent):
        """Test task error handling."""
        mock_fetch.side_effect = Exception("API Error")

        result = await cftc_agent.execute_task_by_name(
            task_name="fetch_commodity",
            context={
                "commodity": "GOLD",
                "report_type": "DISAGGREGATED_FUTURES",
                "weeks": 52
            }
        )

        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_report_type_string_conversion(self, cftc_agent):
        """Test conversion of report type from string."""
        with patch("cftc_analytics.data.client.CFTCClient.fetch_commodity") as mock_fetch:
            mock_fetch.return_value = pd.DataFrame()

            result = await cftc_agent.execute_task_by_name(
                task_name="fetch_commodity",
                context={
                    "commodity": "GOLD",
                    "report_type": "DISAGGREGATED_FUTURES",  # String
                    "weeks": 52
                }
            )

            # Should convert string to ReportType enum
            mock_fetch.assert_called_once()
            call_args = mock_fetch.call_args[0]  # Positional args
            # Check that second positional argument (report_type) is converted correctly
            assert call_args[1] == ReportType.DISAGGREGATED_FUTURES
