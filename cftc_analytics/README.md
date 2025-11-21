# CFTC Analytics Tool

A comprehensive data analytics tool for CFTC (Commodity Futures Trading Commission) Commitments of Traders (COT) reports.

## Overview

The CFTC Analytics Tool provides a Python-based framework for fetching, analyzing, and visualizing COT data from the CFTC's public reporting system. It integrates seamlessly with the multi-agent framework and offers powerful analytics capabilities for market positioning analysis.

## Features

### Data Fetching
- **Socrata API Integration**: Direct access to CFTC's public data via Socrata Open Data API
- **Multiple Report Types**: Support for all COT report formats:
  - Legacy (Futures-only and Combined)
  - Disaggregated (Futures-only and Combined)
  - Traders in Financial Futures (TFF)
  - Supplemental
- **Flexible Queries**: Fetch data by commodity, date range, or custom filters
- **Batch Operations**: Retrieve data for multiple commodities simultaneously

### Analytics Engine
- **Net Position Analysis**: Calculate and track net positions for all trader categories
- **Sentiment Indicators**: Generate sentiment indexes (0-100 scale) based on historical positioning
- **Extreme Detection**: Identify periods of extreme bullish/bearish positioning
- **Position Change Detection**: Detect significant week-over-week position shifts
- **COT Index**: Normalized positioning indicators with customizable smoothing
- **Open Interest Trends**: Track and analyze open interest changes
- **Trader Comparisons**: Compare positioning across different trader categories
- **Divergence Analysis**: Identify divergences between positions and price (when available)

### Visualization
- **Net Position Charts**: Visualize trader positioning over time
- **Sentiment Heatmaps**: Track sentiment extremes across periods
- **Open Interest Trends**: Monitor market participation changes
- **Multi-Panel Dashboards**: Comprehensive views with multiple metrics
- **Commodity Comparisons**: Compare positioning across different markets
- **Export Capabilities**: Save charts as high-resolution PNG files

### Agent Integration
- **Specialized CFTC Agent**: Purpose-built agent for COT data analysis
- **Multi-Agent Coordination**: Works within the existing multi-agent framework
- **Async Task Execution**: Non-blocking data retrieval and analysis
- **Workflow Support**: Integrates with predefined workflows

## Installation

```bash
# Install dependencies
pip install -e .

# Or install with visualization support
pip install -e ".[viz]"
```

### Dependencies
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `requests` - HTTP client for API access
- `matplotlib` - Visualization (optional)

## Quick Start

### Basic Usage

```python
from cftc_analytics import CFTCClient, CFTCAnalytics, ReportType

# Initialize client
client = CFTCClient()

# Fetch commodity data
df = client.fetch_commodity(
    commodity="GOLD",
    report_type=ReportType.DISAGGREGATED_FUTURES,
    weeks=52  # 1 year of data
)

# Perform analysis
analytics = CFTCAnalytics(client=client)
analysis = analytics.analyze_commodity(
    commodity="GOLD",
    report_type=ReportType.DISAGGREGATED_FUTURES,
    weeks=52
)

# Generate report
report = analytics.generate_report(
    commodity="GOLD",
    report_type=ReportType.DISAGGREGATED_FUTURES,
    weeks=52
)
print(report)
```

### Visualization

```python
from cftc_analytics.visualization import CFTCCharts

# Create chart generator
charts = CFTCCharts(figsize=(14, 10))

# Define trader categories
categories = {
    "Managed Money": "m_money",
    "Swap Dealer": "swap",
    "Producer/Merchant": "prod_merc",
}

# Generate comprehensive chart
fig = charts.plot_comprehensive(
    df=df,
    categories=categories,
    commodity_name="GOLD",
    save_path="gold_analysis.png"
)
```

### Agent Usage

```python
import asyncio
from agents.specialized.cftc_agent import CFTCAgent

async def analyze_markets():
    # Create CFTC agent
    agent = CFTCAgent(name="MarketAnalyst")

    # Analyze commodity
    result = await agent.execute_task(
        task_name="analyze_commodity",
        context={
            "commodity": "CRUDE OIL",
            "report_type": "DISAGGREGATED_FUTURES",
            "weeks": 52
        }
    )

    return result

# Run analysis
result = asyncio.run(analyze_markets())
```

## Available Report Types

### Legacy Reports
- **LEGACY_FUTURES**: Traditional COT report, futures-only
- **LEGACY_COMBINED**: Traditional COT report, futures and options

Trader Categories:
- Commercial (Hedgers)
- Non-Commercial (Speculators)
- Non-Reportable (Small Traders)

### Disaggregated Reports
- **DISAGGREGATED_FUTURES**: Disaggregated report, futures-only
- **DISAGGREGATED_COMBINED**: Disaggregated report, futures and options

Trader Categories:
- Producer/Merchant/Processor/User
- Swap Dealers
- Managed Money (Hedge Funds, CTAs)
- Other Reportables
- Non-Reportable

### Traders in Financial Futures (TFF)
- **TFF_FUTURES**: TFF report, futures-only
- **TFF_COMBINED**: TFF report, futures and options

Trader Categories:
- Dealer/Intermediary
- Asset Manager/Institutional
- Leveraged Funds
- Other Reportables
- Non-Reportable

## Analytics Functions

### Net Position Calculation
```python
from cftc_analytics.analytics.indicators import calculate_net_positions

net_pos = calculate_net_positions(df, category_prefix="m_money")
```

### Sentiment Index
```python
from cftc_analytics.analytics.indicators import calculate_sentiment_index

sentiment = calculate_sentiment_index(df, category_prefix="m_money", lookback_period=52)
```

### Extreme Detection
```python
from cftc_analytics.analytics.indicators import calculate_extremes

bullish, bearish = calculate_extremes(df, category_prefix="m_money", threshold=90)
```

### Position Change Detection
```python
from cftc_analytics.analytics.indicators import detect_position_changes

changes = detect_position_changes(df, category_prefix="m_money", threshold_pct=10.0)
```

## API Reference

### CFTCClient

Main client for fetching COT data from CFTC's Socrata API.

**Methods:**
- `fetch_latest(report_type, commodity=None, limit=1000)` - Fetch latest reports
- `fetch_by_date_range(report_type, start_date, end_date, commodity=None)` - Fetch by date range
- `fetch_commodity(commodity, report_type, weeks=52)` - Fetch commodity data
- `get_available_commodities(report_type)` - List available commodities
- `fetch_multiple_commodities(commodities, report_type, weeks)` - Batch fetch

### CFTCAnalytics

Analytics engine for COT data analysis.

**Methods:**
- `analyze_commodity(commodity, report_type, weeks)` - Comprehensive analysis
- `compare_traders(commodity, report_type, weeks)` - Compare trader categories
- `find_extremes(commodity, report_type, weeks, threshold)` - Find extreme positioning
- `detect_major_shifts(commodity, report_type, weeks, threshold_pct)` - Detect position shifts
- `generate_report(commodity, report_type, weeks)` - Generate text report

### CFTCCharts

Visualization utilities for COT data.

**Methods:**
- `plot_net_positions(df, categories, title, save_path)` - Net position chart
- `plot_sentiment(df, categories, title, save_path)` - Sentiment index chart
- `plot_open_interest(df, title, save_path)` - Open interest trend
- `plot_comprehensive(df, categories, commodity_name, save_path)` - Multi-panel chart
- `plot_comparison(data_dict, category_prefix, title, save_path)` - Commodity comparison

### CFTCAgent

Specialized agent for CFTC data analysis within multi-agent framework.

**Supported Tasks:**
- `fetch_commodity` - Fetch COT data
- `analyze_commodity` - Perform analysis
- `compare_traders` - Compare trader positioning
- `find_extremes` - Find extreme periods
- `detect_shifts` - Detect position shifts
- `generate_report` - Generate text report
- `list_commodities` - List available commodities

## Examples

See the `examples/cftc/` directory for complete examples:

1. **basic_usage.py** - Basic data fetching and analysis
2. **visualization_example.py** - Chart generation and visualization
3. **agent_usage.py** - Integration with multi-agent framework

## Data Sources

All data is sourced from the CFTC's public reporting system:
- **Base URL**: https://publicreporting.cftc.gov
- **API Type**: Socrata Open Data API (SODA)
- **Update Frequency**: Weekly (Tuesday data released Friday afternoon)
- **Historical Data**: Available from 1986 (varies by report type)

## Rate Limits

The CFTC's Socrata API has the following limits:
- **Without App Token**: 1,000 requests per rolling 24-hour period
- **With App Token**: 10,000 requests per rolling 24-hour period

To use an app token:
```python
client = CFTCClient(app_token="YOUR_APP_TOKEN")
```

Get a free app token at: https://dev.socrata.com/foundry/publicreporting.cftc.gov

## Testing

Run tests with pytest:

```bash
# Run all tests
pytest tests/cftc/

# Run specific test file
pytest tests/cftc/test_client.py

# Run with coverage
pytest tests/cftc/ --cov=cftc_analytics --cov-report=html

# Run integration tests (requires network)
pytest tests/cftc/ -m integration
```

## Architecture

```
cftc_analytics/
├── data/
│   ├── client.py       # API client for data fetching
│   └── models.py       # Data models and enums
├── analytics/
│   ├── engine.py       # Main analytics engine
│   └── indicators.py   # Technical indicators and calculations
└── visualization/
    └── charts.py       # Charting and visualization tools
```

## Integration with Multi-Agent System

The CFTC Analytics Tool integrates seamlessly with the existing multi-agent framework:

```python
from agents.base.coordinator import Coordinator
from agents.specialized.cftc_agent import CFTCAgent

# Create coordinator
coordinator = Coordinator()

# Register CFTC agent
cftc_agent = CFTCAgent(name="MarketDataAnalyst")
coordinator.register_agent(cftc_agent)

# Execute analysis task
result = await coordinator.execute_task(
    agent=cftc_agent,
    task_name="analyze_commodity",
    context={"commodity": "GOLD", "weeks": 52}
)
```

## Common Use Cases

### Market Sentiment Analysis
Track how different trader groups (commercials, speculators, managed money) are positioned in a market to gauge sentiment and potential turning points.

### Extreme Positioning Detection
Identify when markets reach extreme positioning levels, which often precede reversals or significant moves.

### Multi-Market Analysis
Compare positioning across related commodities (e.g., gold, silver, copper) to identify relative value opportunities.

### Trend Confirmation
Use open interest trends alongside price trends to confirm or question the strength of market moves.

### Divergence Signals
Detect divergences between trader positioning and price action for potential trading signals.

## Contributing

When adding new features:
1. Add tests in `tests/cftc/`
2. Update documentation
3. Run the test suite
4. Follow existing code style

## License

This tool is part of the claude-agent-templates project.

## Support

For issues or questions:
1. Check the examples in `examples/cftc/`
2. Review the test files for usage patterns
3. Consult the CFTC's official documentation: https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm

## Acknowledgments

- Data provided by the Commodity Futures Trading Commission (CFTC)
- API powered by Socrata Open Data
- Built on the multi-agent framework architecture
