"""Technical indicators and analytics functions for COT data."""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional


def calculate_net_positions(df: pd.DataFrame, category_prefix: str) -> pd.Series:
    """
    Calculate net positions (long - short) for a trader category.

    Args:
        df: DataFrame containing COT data
        category_prefix: Prefix for position columns (e.g., 'comm', 'noncomm', 'm_money')

    Returns:
        Series containing net positions
    """
    long_col = f"{category_prefix}_positions_long_all"
    short_col = f"{category_prefix}_positions_short_all"

    if long_col in df.columns and short_col in df.columns:
        return pd.to_numeric(df[long_col], errors='coerce') - pd.to_numeric(
            df[short_col], errors='coerce'
        )
    else:
        return pd.Series(dtype=float)


def calculate_sentiment_index(
    df: pd.DataFrame, category_prefix: str, lookback_period: int = 52
) -> pd.Series:
    """
    Calculate sentiment index (0-100) based on net position percentile.

    Args:
        df: DataFrame containing COT data
        category_prefix: Prefix for position columns
        lookback_period: Number of periods for percentile calculation

    Returns:
        Series containing sentiment index (0-100)
    """
    net_positions = calculate_net_positions(df, category_prefix)

    # Calculate rolling percentile
    sentiment = net_positions.rolling(window=lookback_period, min_periods=1).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) > 0 else 50
    )

    return sentiment


def calculate_extremes(
    df: pd.DataFrame, category_prefix: str, lookback_period: int = 52, threshold: float = 90
) -> Tuple[pd.Series, pd.Series]:
    """
    Identify extreme bullish and bearish positioning.

    Args:
        df: DataFrame containing COT data
        category_prefix: Prefix for position columns
        lookback_period: Number of periods for extremes calculation
        threshold: Percentile threshold for extremes (default: 90)

    Returns:
        Tuple of (bullish_extremes, bearish_extremes) as boolean Series
    """
    sentiment = calculate_sentiment_index(df, category_prefix, lookback_period)

    bullish_extremes = sentiment >= threshold
    bearish_extremes = sentiment <= (100 - threshold)

    return bullish_extremes, bearish_extremes


def detect_position_changes(
    df: pd.DataFrame, category_prefix: str, threshold_pct: float = 10.0
) -> pd.DataFrame:
    """
    Detect significant position changes week-over-week.

    Args:
        df: DataFrame containing COT data
        category_prefix: Prefix for position columns
        threshold_pct: Percentage threshold for significant change

    Returns:
        DataFrame with change detection columns
    """
    net_positions = calculate_net_positions(df, category_prefix)

    # Calculate week-over-week changes
    change = net_positions.diff()
    pct_change = net_positions.pct_change() * 100

    # Detect significant changes
    significant_increase = pct_change >= threshold_pct
    significant_decrease = pct_change <= -threshold_pct

    result = pd.DataFrame(
        {
            "net_position": net_positions,
            "change": change,
            "pct_change": pct_change,
            "significant_increase": significant_increase,
            "significant_decrease": significant_decrease,
        }
    )

    return result


def calculate_cot_index(df: pd.DataFrame, category_prefix: str, smoothing: int = 3) -> pd.Series:
    """
    Calculate COT Index using moving average smoothing.

    The COT Index normalizes net positions to a 0-100 scale using historical data.

    Args:
        df: DataFrame containing COT data
        category_prefix: Prefix for position columns
        smoothing: Number of periods for moving average smoothing

    Returns:
        Series containing COT Index
    """
    net_positions = calculate_net_positions(df, category_prefix)

    # Calculate rolling min and max
    rolling_min = net_positions.rolling(window=52, min_periods=1).min()
    rolling_max = net_positions.rolling(window=52, min_periods=1).max()

    # Normalize to 0-100
    cot_index = ((net_positions - rolling_min) / (rolling_max - rolling_min + 1)) * 100

    # Apply smoothing
    if smoothing > 1:
        cot_index = cot_index.rolling(window=smoothing, min_periods=1).mean()

    return cot_index


def calculate_open_interest_trend(df: pd.DataFrame, window: int = 4) -> pd.DataFrame:
    """
    Calculate open interest trends.

    Args:
        df: DataFrame containing COT data with 'open_interest_all' column
        window: Window size for trend calculation

    Returns:
        DataFrame with open interest trend analysis
    """
    if "open_interest_all" not in df.columns:
        return pd.DataFrame()

    oi = pd.to_numeric(df["open_interest_all"], errors='coerce')

    # Calculate moving average
    oi_ma = oi.rolling(window=window, min_periods=1).mean()

    # Calculate trend
    oi_change = oi.diff()
    oi_pct_change = oi.pct_change() * 100

    # Trend direction
    increasing = oi > oi_ma
    decreasing = oi < oi_ma

    result = pd.DataFrame(
        {
            "open_interest": oi,
            "oi_ma": oi_ma,
            "oi_change": oi_change,
            "oi_pct_change": oi_pct_change,
            "trend_increasing": increasing,
            "trend_decreasing": decreasing,
        }
    )

    return result


def calculate_hedger_speculator_ratio(df: pd.DataFrame) -> pd.Series:
    """
    Calculate the ratio of hedger (commercial) to speculator (non-commercial) net positions.

    Args:
        df: DataFrame containing legacy COT data

    Returns:
        Series containing hedger/speculator ratio
    """
    hedger_net = calculate_net_positions(df, "comm")
    speculator_net = calculate_net_positions(df, "noncomm")

    # Avoid division by zero
    ratio = hedger_net / (speculator_net.abs() + 1)

    return ratio


def calculate_trader_participation(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Calculate participation metrics for different trader categories.

    Args:
        df: DataFrame containing COT data

    Returns:
        Dictionary of participation metrics
    """
    total_oi = pd.to_numeric(df.get("open_interest_all", 0), errors='coerce')

    metrics = {}

    # Try to calculate for different trader categories
    categories = [
        ("comm", "commercial"),
        ("noncomm", "non_commercial"),
        ("m_money", "managed_money"),
        ("prod_merc", "producer_merchant"),
        ("swap", "swap_dealer"),
        ("lev_money", "leveraged_funds"),
        ("dealer", "dealer_intermediary"),
        ("asset_mgr", "asset_manager"),
    ]

    for prefix, name in categories:
        long_col = f"{prefix}_positions_long_all"
        short_col = f"{prefix}_positions_short_all"

        if long_col in df.columns and short_col in df.columns:
            total_positions = pd.to_numeric(df[long_col], errors='coerce') + pd.to_numeric(
                df[short_col], errors='coerce'
            )
            participation = (total_positions / (total_oi + 1)) * 100
            metrics[f"{name}_participation"] = participation

    return metrics


def identify_divergences(
    df: pd.DataFrame,
    category_prefix: str,
    price_col: Optional[str] = None,
    window: int = 13,
) -> pd.DataFrame:
    """
    Identify divergences between COT positions and price (if available).

    Args:
        df: DataFrame containing COT data
        category_prefix: Prefix for position columns
        price_col: Optional column name containing price data
        window: Window for trend comparison

    Returns:
        DataFrame with divergence signals
    """
    net_positions = calculate_net_positions(df, category_prefix)

    # Calculate position trend
    position_trend = net_positions.diff(window)

    result = pd.DataFrame(
        {
            "net_position": net_positions,
            "position_trend": position_trend,
            "position_increasing": position_trend > 0,
            "position_decreasing": position_trend < 0,
        }
    )

    # If price data available, calculate divergences
    if price_col and price_col in df.columns:
        price = pd.to_numeric(df[price_col], errors='coerce')
        price_trend = price.diff(window)

        result["price_trend"] = price_trend
        result["price_increasing"] = price_trend > 0
        result["price_decreasing"] = price_trend < 0

        # Bullish divergence: price down, positions up
        result["bullish_divergence"] = (result["price_decreasing"]) & (
            result["position_increasing"]
        )

        # Bearish divergence: price up, positions down
        result["bearish_divergence"] = (result["price_increasing"]) & (
            result["position_decreasing"]
        )

    return result
