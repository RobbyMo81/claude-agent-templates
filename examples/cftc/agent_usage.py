"""
Agent usage example for CFTC Analytics Tool.

This script demonstrates how to use the CFTC agent within the multi-agent framework.
"""

import sys
from pathlib import Path
import asyncio

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.base.coordinator import Coordinator
from agents.specialized.cftc_agent import CFTCAgent
from cftc_analytics.data.models import ReportType


async def main():
    """Agent usage example."""
    print("CFTC Analytics Tool - Agent Usage Example")
    print("=" * 60)

    # Create coordinator and register CFTC agent
    coordinator = Coordinator()
    cftc_agent = CFTCAgent(name="CFTCDataAnalyst")

    coordinator.register_agent(cftc_agent)
    print(f"\nRegistered agent: {cftc_agent.name}")
    print(f"Capabilities: {[cap.value for cap in cftc_agent.capabilities]}")

    # Example 1: List available commodities
    print("\n1. Listing available commodities...")
    result = await cftc_agent.execute_task(
        task_name="list_commodities",
        context={
            "report_type": "DISAGGREGATED_FUTURES"
        }
    )

    if result["success"]:
        print(f"   Found {result['count']} commodities")
        print(f"   Sample: {result['commodities'][:5]}")
    else:
        print(f"   Error: {result.get('error', 'Unknown error')}")

    # Example 2: Fetch commodity data
    print("\n2. Fetching BITCOIN data...")
    result = await cftc_agent.execute_task(
        task_name="fetch_commodity",
        context={
            "commodity": "BITCOIN",
            "report_type": "DISAGGREGATED_FUTURES",
            "weeks": 26
        }
    )

    if result["success"]:
        print(f"   {result['message']}")
        print(f"   Records: {result['records']}")
    else:
        print(f"   Error: {result.get('error', 'Unknown error')}")

    # Example 3: Analyze commodity
    print("\n3. Analyzing GOLD...")
    result = await cftc_agent.execute_task(
        task_name="analyze_commodity",
        context={
            "commodity": "GOLD",
            "report_type": "DISAGGREGATED_FUTURES",
            "weeks": 52
        }
    )

    if result["success"]:
        print(f"   {result['message']}")
        analysis = result['analysis']
        summary = analysis['summary']
        print(f"   Date range: {summary['start_date']} to {summary['end_date']}")
        print(f"   Records: {summary['total_records']}")
    else:
        print(f"   Error: {result.get('error', 'Unknown error')}")

    # Example 4: Find extremes
    print("\n4. Finding extreme positioning in CRUDE OIL...")
    result = await cftc_agent.execute_task(
        task_name="find_extremes",
        context={
            "commodity": "CRUDE OIL",
            "report_type": "DISAGGREGATED_FUTURES",
            "weeks": 104,
            "threshold": 90
        }
    )

    if result["success"]:
        print(f"   {result['message']}")
        extremes = result['extremes']
        for category, periods in extremes.items():
            if periods:
                print(f"   {category}: {len(periods)} extreme periods")
    else:
        print(f"   Error: {result.get('error', 'Unknown error')}")

    # Example 5: Detect position shifts
    print("\n5. Detecting position shifts in NATURAL GAS...")
    result = await cftc_agent.execute_task(
        task_name="detect_shifts",
        context={
            "commodity": "NATURAL GAS",
            "report_type": "DISAGGREGATED_FUTURES",
            "weeks": 26,
            "threshold_pct": 15.0
        }
    )

    if result["success"]:
        print(f"   {result['message']}")
        shifts = result['shifts']
        for category, shift_periods in shifts.items():
            if shift_periods:
                print(f"   {category}: {len(shift_periods)} significant shifts")
    else:
        print(f"   Error: {result.get('error', 'Unknown error')}")

    # Example 6: Generate comprehensive report
    print("\n6. Generating report for S&P 500...")
    result = await cftc_agent.execute_task(
        task_name="generate_report",
        context={
            "commodity": "S&P 500",
            "report_type": "DISAGGREGATED_FUTURES",
            "weeks": 52
        }
    )

    if result["success"]:
        print("\n" + "=" * 60)
        print(result['report'])
        print("=" * 60)
    else:
        print(f"   Error: {result.get('error', 'Unknown error')}")

    # Example 7: Compare traders
    print("\n7. Comparing trader categories in CORN...")
    result = await cftc_agent.execute_task(
        task_name="compare_traders",
        context={
            "commodity": "CORN",
            "report_type": "DISAGGREGATED_FUTURES",
            "weeks": 52
        }
    )

    if result["success"]:
        print(f"   {result['message']}")
        comparison = result['comparison']
        print(f"   Comparison records: {len(comparison)}")
    else:
        print(f"   Error: {result.get('error', 'Unknown error')}")

    print("\n✅ Agent usage example complete!")


if __name__ == "__main__":
    asyncio.run(main())
