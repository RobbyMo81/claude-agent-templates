"""Data models for CFTC COT reports."""

from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


class ReportType(Enum):
    """COT report types available from CFTC."""

    LEGACY_FUTURES = "legacy_futures"
    LEGACY_COMBINED = "legacy_combined"
    DISAGGREGATED_FUTURES = "disaggregated_futures"
    DISAGGREGATED_COMBINED = "disaggregated_combined"
    TFF_FUTURES = "tff_futures"
    TFF_COMBINED = "tff_combined"
    SUPPLEMENTAL_COMBINED = "supplemental_combined"


# Socrata resource identifiers for each report type
REPORT_RESOURCE_IDS = {
    ReportType.LEGACY_FUTURES: "6dca-aqww",
    ReportType.LEGACY_COMBINED: "jun7-fc8e",
    ReportType.DISAGGREGATED_FUTURES: "72hh-3qpy",
    ReportType.DISAGGREGATED_COMBINED: "kh3c-gbw2",
    ReportType.TFF_FUTURES: "gpe5-46if",
    ReportType.TFF_COMBINED: "dw8z-x6ih",
    ReportType.SUPPLEMENTAL_COMBINED: "aca9-w7k8",
}


class TraderCategory(Enum):
    """Trader categories in COT reports."""

    # Legacy reports
    COMMERCIAL = "commercial"
    NON_COMMERCIAL = "non_commercial"
    NON_REPORTABLE = "non_reportable"

    # Disaggregated reports
    PRODUCER_MERCHANT = "producer_merchant"
    SWAP_DEALER = "swap_dealer"
    MANAGED_MONEY = "managed_money"
    OTHER_REPORTABLE = "other_reportable"

    # TFF reports
    DEALER_INTERMEDIARY = "dealer_intermediary"
    ASSET_MANAGER = "asset_manager"
    LEVERAGED_FUNDS = "leveraged_funds"


@dataclass
class COTRecord:
    """Represents a single COT report record."""

    report_date: datetime
    commodity_name: str
    commodity_code: str
    market_and_exchange_names: str

    # Open interest
    open_interest_all: int

    # Positions by category (will vary by report type)
    positions: dict

    # Additional metadata
    cftc_contract_market_code: str
    cftc_market_code: str
    cftc_region_code: Optional[str] = None
    cftc_commodity_code: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict, report_type: ReportType) -> "COTRecord":
        """Create COTRecord from API response dictionary."""
        # Parse report date
        report_date = datetime.fromisoformat(
            data.get("report_date_as_yyyy_mm_dd", data.get("report_date_as_mm_dd_yyyy", ""))
        )

        # Extract common fields
        record = cls(
            report_date=report_date,
            commodity_name=data.get("commodity", data.get("market_and_exchange_names", "")),
            commodity_code=data.get("commodity_code", ""),
            market_and_exchange_names=data.get("market_and_exchange_names", ""),
            open_interest_all=int(data.get("open_interest_all", 0)),
            positions={},
            cftc_contract_market_code=data.get("cftc_contract_market_code", ""),
            cftc_market_code=data.get("cftc_market_code", ""),
            cftc_region_code=data.get("cftc_region_code"),
            cftc_commodity_code=data.get("cftc_commodity_code"),
        )

        # Parse positions based on report type
        if report_type in [ReportType.LEGACY_FUTURES, ReportType.LEGACY_COMBINED]:
            record.positions = cls._parse_legacy_positions(data)
        elif report_type in [ReportType.DISAGGREGATED_FUTURES, ReportType.DISAGGREGATED_COMBINED]:
            record.positions = cls._parse_disaggregated_positions(data)
        elif report_type in [ReportType.TFF_FUTURES, ReportType.TFF_COMBINED]:
            record.positions = cls._parse_tff_positions(data)

        return record

    @staticmethod
    def _parse_legacy_positions(data: dict) -> dict:
        """Parse positions from legacy report."""
        return {
            "commercial_long": int(data.get("comm_positions_long_all", 0)),
            "commercial_short": int(data.get("comm_positions_short_all", 0)),
            "non_commercial_long": int(data.get("noncomm_positions_long_all", 0)),
            "non_commercial_short": int(data.get("noncomm_positions_short_all", 0)),
            "non_commercial_spreading": int(data.get("noncomm_positions_spread_all", 0)),
            "non_reportable_long": int(data.get("nonrept_positions_long_all", 0)),
            "non_reportable_short": int(data.get("nonrept_positions_short_all", 0)),
        }

    @staticmethod
    def _parse_disaggregated_positions(data: dict) -> dict:
        """Parse positions from disaggregated report."""
        return {
            "producer_merchant_long": int(data.get("prod_merc_positions_long_all", 0)),
            "producer_merchant_short": int(data.get("prod_merc_positions_short_all", 0)),
            "swap_dealer_long": int(data.get("swap_positions_long_all", 0)),
            "swap_dealer_short": int(data.get("swap_positions_short_all", 0)),
            "swap_dealer_spreading": int(data.get("swap_positions_spread_all", 0)),
            "managed_money_long": int(data.get("m_money_positions_long_all", 0)),
            "managed_money_short": int(data.get("m_money_positions_short_all", 0)),
            "managed_money_spreading": int(data.get("m_money_positions_spread_all", 0)),
            "other_reportable_long": int(data.get("other_rept_positions_long_all", 0)),
            "other_reportable_short": int(data.get("other_rept_positions_short_all", 0)),
            "other_reportable_spreading": int(data.get("other_rept_positions_spread_all", 0)),
            "non_reportable_long": int(data.get("nonrept_positions_long_all", 0)),
            "non_reportable_short": int(data.get("nonrept_positions_short_all", 0)),
        }

    @staticmethod
    def _parse_tff_positions(data: dict) -> dict:
        """Parse positions from TFF (Traders in Financial Futures) report."""
        return {
            "dealer_intermediary_long": int(data.get("dealer_positions_long_all", 0)),
            "dealer_intermediary_short": int(data.get("dealer_positions_short_all", 0)),
            "dealer_intermediary_spreading": int(data.get("dealer_positions_spread_all", 0)),
            "asset_manager_long": int(data.get("asset_mgr_positions_long_all", 0)),
            "asset_manager_short": int(data.get("asset_mgr_positions_short_all", 0)),
            "asset_manager_spreading": int(data.get("asset_mgr_positions_spread_all", 0)),
            "leveraged_funds_long": int(data.get("lev_money_positions_long_all", 0)),
            "leveraged_funds_short": int(data.get("lev_money_positions_short_all", 0)),
            "leveraged_funds_spreading": int(data.get("lev_money_positions_spread_all", 0)),
            "other_reportable_long": int(data.get("other_rept_positions_long_all", 0)),
            "other_reportable_short": int(data.get("other_rept_positions_short_all", 0)),
            "other_reportable_spreading": int(data.get("other_rept_positions_spread_all", 0)),
            "non_reportable_long": int(data.get("nonrept_positions_long_all", 0)),
            "non_reportable_short": int(data.get("nonrept_positions_short_all", 0)),
        }
