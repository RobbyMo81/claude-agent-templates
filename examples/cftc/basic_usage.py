"""
Basic usage example for CFTC Analytics Tool.

This script demonstrates how to fetch and analyze COT data for a single commodity.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cftc_analytics import CFTCClient, CFTCAnalytics, ReportType
from datetime import datetime, timedelta


def main():
    """Basic usage example."""
    print("CFTC Analytics Tool - Basic Usage Example")
    print("=" * 60)

    # Initialize client and analytics engine
    client = CFTCClient()
    analytics = CFTCAnalytics(client=client)

    # Example 1: Fetch latest data for a commodity
    print("\n1. Fetching latest GOLD data...")
    commodity = "GOLD"
    weeks = 52  # 1 year of data

    df = client.fetch_commodity(
        commodity=commodity,
        report_type=ReportType.DISAGGREGATED_FUTURES,
        weeks=weeks
    )

    print(f"   Retrieved {len(df)} records")
    print(f"   Columns: {', '.join(df.columns[:5])}...")

    # Example 2: Generate comprehensive analysis
    print("\n2. Performing comprehensive analysis...")
    analysis = analytics.analyze_commodity(
        commodity=commodity,
        report_type=ReportType.DISAGGREGATED_FUTURES,
        weeks=weeks
    )

    summary = analysis['summary']
    print(f"   Date range: {summary['start_date']} to {summary['end_date']}")
    print(f"   Total records: {summary['total_records']}")
    print(f"   Latest open interest: {summary['latest_open_interest']:,.0f}")

    # Example 3: Examine trader analysis
    print("\n3. Trader Category Analysis:")
    for category, data in analysis['trader_analysis'].items():
        if not data.empty:
            latest = data.iloc[-1]
            print(f"\n   {category.upper()}:")
            print(f"     Net Position: {latest.get('net_position', 0):,.0f} contracts")
            print(f"     Sentiment: {latest.get('sentiment', 0):.1f}/100")
            print(f"     COT Index: {latest.get('cot_index', 0):.1f}/100")

            if latest.get('bullish_extreme', False):
                print(f"     Status: ⚠️  BULLISH EXTREME")
            elif latest.get('bearish_extreme', False):
                print(f"     Status: ⚠️  BEARISH EXTREME")

    # Example 4: Find extreme positioning periods
    print("\n4. Finding extreme positioning periods...")
    extremes = analytics.find_extremes(
        commodity=commodity,
        report_type=ReportType.DISAGGREGATED_FUTURES,
        weeks=104,  # 2 years
        threshold=90
    )

    for category, extreme_data in extremes.items():
        if not extreme_data.empty:
            print(f"   {category}: {len(extreme_data)} extreme periods detected")

    # Example 5: Detect major position shifts
    print("\n5. Detecting major position shifts (>15%)...")
    shifts = analytics.detect_major_shifts(
        commodity=commodity,
        report_type=ReportType.DISAGGREGATED_FUTURES,
        weeks=26,
        threshold_pct=15.0
    )

    for category, shift_data in shifts.items():
        if not shift_data.empty:
            print(f"   {category}: {len(shift_data)} significant shifts detected")
            if len(shift_data) > 0:
                latest_shift = shift_data.iloc[-1]
                print(f"     Latest: {latest_shift.get('pct_change', 0):.1f}% change")

    # Example 6: Generate text report
    print("\n6. Generating comprehensive report...")
    report = analytics.generate_report(
        commodity=commodity,
        report_type=ReportType.DISAGGREGATED_FUTURES,
        weeks=52
    )

    print("\n" + "=" * 60)
    print(report)
    print("=" * 60)

    # Example 7: Compare multiple trader categories
    print("\n7. Comparing trader categories...")
    comparison = analytics.compare_traders(
        commodity=commodity,
        report_type=ReportType.DISAGGREGATED_FUTURES,
        weeks=26
    )

    if not comparison.empty:
        print(f"   Comparison data: {len(comparison)} records")
        print(f"   Latest date: {comparison.iloc[-1].get('report_date', 'N/A')}")

    print("\n✅ Basic usage example complete!")


if __name__ == "__main__":
    main()
