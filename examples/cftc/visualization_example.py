"""
Visualization example for CFTC Analytics Tool.

This script demonstrates how to create charts and visualizations from COT data.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cftc_analytics import CFTCClient, ReportType
from cftc_analytics.visualization import CFTCCharts
import matplotlib.pyplot as plt


def main():
    """Visualization example."""
    print("CFTC Analytics Tool - Visualization Example")
    print("=" * 60)

    # Initialize client and chart generator
    client = CFTCClient()
    charts = CFTCCharts(figsize=(14, 10))

    # Fetch data for a commodity
    commodity = "CRUDE OIL"
    print(f"\nFetching {commodity} data...")

    df = client.fetch_commodity(
        commodity=commodity,
        report_type=ReportType.DISAGGREGATED_FUTURES,
        weeks=52
    )

    if df.empty:
        print(f"No data found for {commodity}")
        return

    print(f"Retrieved {len(df)} records")

    # Define trader categories for disaggregated report
    categories = {
        "Managed Money": "m_money",
        "Swap Dealer": "swap",
        "Producer/Merchant": "prod_merc",
    }

    # Create visualizations
    print("\nGenerating charts...")

    # 1. Net Positions Chart
    print("  Creating net positions chart...")
    fig1 = charts.plot_net_positions(
        df=df,
        categories=categories,
        title=f"{commodity} - Net Positions by Trader Category",
        save_path="cftc_net_positions.png"
    )
    print("  Saved: cftc_net_positions.png")

    # 2. Sentiment Index Chart
    print("  Creating sentiment index chart...")
    fig2 = charts.plot_sentiment(
        df=df,
        categories=categories,
        title=f"{commodity} - Sentiment Index by Trader Category",
        save_path="cftc_sentiment.png"
    )
    print("  Saved: cftc_sentiment.png")

    # 3. Open Interest Chart
    print("  Creating open interest chart...")
    fig3 = charts.plot_open_interest(
        df=df,
        title=f"{commodity} - Open Interest Trend",
        save_path="cftc_open_interest.png"
    )
    print("  Saved: cftc_open_interest.png")

    # 4. Comprehensive Multi-Panel Chart
    print("  Creating comprehensive chart...")
    fig4 = charts.plot_comprehensive(
        df=df,
        categories=categories,
        commodity_name=commodity,
        save_path="cftc_comprehensive.png"
    )
    print("  Saved: cftc_comprehensive.png")

    # 5. Multi-Commodity Comparison
    print("\n  Creating multi-commodity comparison...")
    commodities_to_compare = ["GOLD", "SILVER", "COPPER"]
    data_dict = {}

    for comm in commodities_to_compare:
        try:
            data = client.fetch_commodity(comm, ReportType.DISAGGREGATED_FUTURES, weeks=26)
            if not data.empty:
                data_dict[comm] = data
                print(f"    Fetched {comm}: {len(data)} records")
        except Exception as e:
            print(f"    Error fetching {comm}: {e}")

    if data_dict:
        fig5 = charts.plot_comparison(
            data_dict=data_dict,
            category_prefix="m_money",  # Compare Managed Money positions
            title="Precious Metals - Managed Money Net Positions Comparison",
            save_path="cftc_commodity_comparison.png"
        )
        print("  Saved: cftc_commodity_comparison.png")

    print("\n✅ Visualization example complete!")
    print("\nGenerated files:")
    print("  - cftc_net_positions.png")
    print("  - cftc_sentiment.png")
    print("  - cftc_open_interest.png")
    print("  - cftc_comprehensive.png")
    print("  - cftc_commodity_comparison.png")

    # Optionally display charts
    # plt.show()


if __name__ == "__main__":
    main()
