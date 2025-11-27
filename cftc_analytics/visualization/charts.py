"""Visualization and charting for CFTC COT data."""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from cftc_analytics.analytics.indicators import (
    calculate_net_positions,
    calculate_sentiment_index,
    calculate_open_interest_trend,
)


class CFTCCharts:
    """Charting utilities for CFTC COT data."""

    def __init__(self, figsize: Tuple[int, int] = (14, 10)):
        """
        Initialize chart generator.

        Args:
            figsize: Default figure size (width, height)
        """
        self.figsize = figsize
        plt.style.use("seaborn-v0_8-darkgrid")

    def plot_net_positions(
        self,
        df: pd.DataFrame,
        categories: Dict[str, str],
        title: str = "Net Positions",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot net positions for multiple trader categories.

        Args:
            df: DataFrame with COT data
            categories: Dict mapping category names to column prefixes
            title: Chart title
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Ensure date column
        date_col = self._get_date_column(df)
        dates = pd.to_datetime(df[date_col])

        for category_name, category_prefix in categories.items():
            net_pos = calculate_net_positions(df, category_prefix)
            ax.plot(dates, net_pos, label=category_name, linewidth=2)

        ax.axhline(y=0, color="black", linestyle="--", alpha=0.3)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Net Position (Contracts)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.xticks(rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_sentiment(
        self,
        df: pd.DataFrame,
        categories: Dict[str, str],
        title: str = "Sentiment Index",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot sentiment index for multiple trader categories.

        Args:
            df: DataFrame with COT data
            categories: Dict mapping category names to column prefixes
            title: Chart title
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        date_col = self._get_date_column(df)
        dates = pd.to_datetime(df[date_col])

        for category_name, category_prefix in categories.items():
            sentiment = calculate_sentiment_index(df, category_prefix)
            ax.plot(dates, sentiment, label=category_name, linewidth=2)

        # Add extreme zones
        ax.axhspan(0, 10, alpha=0.1, color="red", label="Bearish Extreme")
        ax.axhspan(90, 100, alpha=0.1, color="green", label="Bullish Extreme")
        ax.axhline(y=50, color="black", linestyle="--", alpha=0.3)

        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Sentiment Index (0-100)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_ylim(0, 100)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.xticks(rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_open_interest(
        self,
        df: pd.DataFrame,
        title: str = "Open Interest",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot open interest and trend.

        Args:
            df: DataFrame with COT data
            title: Chart title
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        oi_data = calculate_open_interest_trend(df)

        if oi_data.empty:
            return fig

        date_col = self._get_date_column(df)
        dates = pd.to_datetime(df[date_col])

        ax.plot(dates, oi_data["open_interest"], label="Open Interest", linewidth=2, color="blue")
        ax.plot(dates, oi_data["oi_ma"], label="4-Week MA", linewidth=2, color="orange", linestyle="--")

        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Open Interest (Contracts)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.xticks(rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_comprehensive(
        self,
        df: pd.DataFrame,
        categories: Dict[str, str],
        commodity_name: str = "Commodity",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Create comprehensive multi-panel chart.

        Args:
            df: DataFrame with COT data
            categories: Dict mapping category names to column prefixes
            commodity_name: Name of commodity for title
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 1, hspace=0.3)

        date_col = self._get_date_column(df)
        dates = pd.to_datetime(df[date_col])

        # Panel 1: Net Positions
        ax1 = fig.add_subplot(gs[0, 0])
        for category_name, category_prefix in categories.items():
            net_pos = calculate_net_positions(df, category_prefix)
            ax1.plot(dates, net_pos, label=category_name, linewidth=2)

        ax1.axhline(y=0, color="black", linestyle="--", alpha=0.3)
        ax1.set_ylabel("Net Position", fontsize=11)
        ax1.set_title(f"{commodity_name} - Net Positions", fontsize=13, fontweight="bold")
        ax1.legend(loc="best", fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        # Panel 2: Sentiment Index
        ax2 = fig.add_subplot(gs[1, 0])
        for category_name, category_prefix in categories.items():
            sentiment = calculate_sentiment_index(df, category_prefix)
            ax2.plot(dates, sentiment, label=category_name, linewidth=2)

        ax2.axhspan(0, 10, alpha=0.1, color="red")
        ax2.axhspan(90, 100, alpha=0.1, color="green")
        ax2.axhline(y=50, color="black", linestyle="--", alpha=0.3)
        ax2.set_ylabel("Sentiment (0-100)", fontsize=11)
        ax2.set_title("Sentiment Index", fontsize=13, fontweight="bold")
        ax2.set_ylim(0, 100)
        ax2.legend(loc="best", fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        # Panel 3: Open Interest
        ax3 = fig.add_subplot(gs[2, 0])
        oi_data = calculate_open_interest_trend(df)

        if not oi_data.empty:
            ax3.plot(dates, oi_data["open_interest"], label="Open Interest", linewidth=2, color="blue")
            ax3.plot(dates, oi_data["oi_ma"], label="4-Week MA", linewidth=2, color="orange", linestyle="--")

        ax3.set_xlabel("Date", fontsize=11)
        ax3.set_ylabel("Open Interest", fontsize=11)
        ax3.set_title("Open Interest Trend", fontsize=13, fontweight="bold")
        ax3.legend(loc="best", fontsize=9)
        ax3.grid(True, alpha=0.3)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_comparison(
        self,
        data_dict: Dict[str, pd.DataFrame],
        category_prefix: str,
        title: str = "Commodity Comparison",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Compare net positions across multiple commodities.

        Args:
            data_dict: Dictionary mapping commodity names to DataFrames
            category_prefix: Category prefix to compare
            title: Chart title
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        for commodity_name, df in data_dict.items():
            if df.empty:
                continue

            date_col = self._get_date_column(df)
            dates = pd.to_datetime(df[date_col])

            net_pos = calculate_net_positions(df, category_prefix)

            # Normalize to percentage change from start
            if len(net_pos) > 0:
                normalized = ((net_pos - net_pos.iloc[0]) / (abs(net_pos.iloc[0]) + 1)) * 100
                ax.plot(dates, normalized, label=commodity_name, linewidth=2)

        ax.axhline(y=0, color="black", linestyle="--", alpha=0.3)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("% Change from Start", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.xticks(rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def _get_date_column(self, df: pd.DataFrame) -> str:
        """Get the appropriate date column from DataFrame."""
        if "report_date" in df.columns:
            return "report_date"
        elif "report_date_as_yyyy_mm_dd" in df.columns:
            return "report_date_as_yyyy_mm_dd"
        elif "report_date_as_mm_dd_yyyy" in df.columns:
            return "report_date_as_mm_dd_yyyy"
        else:
            raise ValueError("No date column found in DataFrame")
