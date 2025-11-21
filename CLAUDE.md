# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a Python-based multi-agent system for AL (Application Language) development with Microsoft Dynamics 365 Business Central. The system implements self-organizing agents that coordinate to execute complex workflows simultaneously.

## Project Structure

```
claude-agent-templates/
├── agents/                    # Multi-agent system implementation
│   ├── base/                 # Core agent framework
│   │   ├── agent.py         # Base agent class with self-organization
│   │   ├── coordinator.py   # Workflow coordination and orchestration
│   │   └── workflow.py      # Workflow definitions and management
│   └── specialized/          # Specialized agents
│       ├── code_agent.py    # AL code generation and analysis
│       ├── test_agent.py    # Test generation and execution
│       ├── schema_agent.py  # Database schema management
│       ├── api_agent.py     # API integration
│       ├── deployment_agent.py  # App deployment and compilation
│       ├── documentation_agent.py  # Documentation generation
│       └── cftc_agent.py    # CFTC COT data analysis
├── cftc_analytics/           # CFTC Analytics Tool
│   ├── data/                # Data fetching and models
│   │   ├── client.py        # CFTC API client
│   │   └── models.py        # Data models and enums
│   ├── analytics/           # Analytics engine
│   │   ├── engine.py        # Main analytics engine
│   │   └── indicators.py    # Technical indicators
│   ├── visualization/       # Charting and visualization
│   │   └── charts.py        # Chart generation
│   └── README.md            # CFTC tool documentation
├── examples/                 # Usage examples
│   └── cftc/                # CFTC analytics examples
│       ├── basic_usage.py   # Basic usage example
│       ├── visualization_example.py  # Charting examples
│       └── agent_usage.py   # Agent integration example
├── tests/                    # Comprehensive test suite
│   ├── conftest.py          # Test fixtures and configuration
│   ├── test_agents.py       # Unit tests for individual agents
│   ├── test_coordination.py # Agent coordination and self-organization tests
│   ├── test_workflows.py    # Workflow execution tests
│   ├── test_concurrent_workflows.py  # Concurrent workflow tests
│   └── cftc/                # CFTC analytics tests
│       ├── test_client.py   # Client tests
│       ├── test_analytics.py # Analytics engine tests
│       └── test_cftc_agent.py # CFTC agent tests
├── pyproject.toml           # Python project configuration
└── CLAUDE.md                # This file
```

## Architecture

### Multi-Agent System

The system implements a **self-organizing multi-agent architecture** where:

1. **Agents** are autonomous entities with specific capabilities
2. **Workflows** define tasks that need to be completed
3. **Coordinator** facilitates agent discovery and workflow execution
4. **Self-Organization** allows agents to coordinate without central control

### Agent Capabilities

Each agent has specific capabilities:

- **CodeAgent**: `CODE_GENERATION`, `CODE_ANALYSIS`
- **TestAgent**: `TESTING`
- **SchemaAgent**: `SCHEMA_MANAGEMENT`, `CODE_GENERATION`, `CODE_ANALYSIS`
- **APIAgent**: `API_INTEGRATION`, `CODE_GENERATION`, `CODE_ANALYSIS`
- **DeploymentAgent**: `DEPLOYMENT`
- **DocumentationAgent**: `DOCUMENTATION`
- **CFTCAgent**: `DATA_ANALYSIS`, `REPORTING`

### Workflows

Pre-defined workflows for common AL/Business Central development tasks:

- **FeatureWorkflow**: Code → Tests + Documentation (parallel)
- **SchemaWorkflow**: Analyze → Generate → Test → Deploy (sequential)
- **APIWorkflow**: Design → Code → Tests + Docs (parallel)
- **DeploymentWorkflow**: Analyze → Test → Build → Docs + Deploy

### Coordination Mechanisms

- **Peer-to-peer messaging**: Agents communicate directly
- **Capability broadcasting**: Agents announce their abilities
- **Task offers**: Agents self-select tasks based on confidence
- **Status notifications**: Agents notify peers of task progress

## Common Development Tasks

### Running Tests

```bash
# Install dependencies
pip install -e .

# Install with visualization support (for CFTC charts)
pip install -e ".[viz]"

# Run all tests
pytest

# Run specific test categories
pytest -m unit              # Unit tests only
pytest -m integration       # Integration tests only
pytest -m workflow          # Workflow tests only
pytest -m concurrent        # Concurrent workflow tests only
pytest -m cftc              # CFTC analytics tests only

# Run with coverage
pytest --cov=agents --cov=cftc_analytics --cov-report=html

# Run specific test file
pytest tests/test_concurrent_workflows.py
pytest tests/cftc/test_client.py

# Run with verbose output
pytest -v

# Run with test duration info
pytest --durations=10
```

### Development Commands

```bash
# Format code
black agents/ cftc_analytics/ tests/ examples/

# Lint code
ruff check agents/ cftc_analytics/ tests/ examples/

# Type checking
mypy agents/ cftc_analytics/

# Run all quality checks
black agents/ cftc_analytics/ tests/ examples/ && ruff check agents/ cftc_analytics/ tests/ examples/ && mypy agents/ cftc_analytics/
```

### Using the Agent System

```python
from agents.base.coordinator import Coordinator
from agents.specialized import (
    CodeAgent, TestAgent, SchemaAgent,
    APIAgent, DeploymentAgent, DocumentationAgent
)
from agents.base.workflow import FeatureWorkflow

# Create coordinator
coordinator = Coordinator()

# Register agents
coordinator.register_agent(CodeAgent())
coordinator.register_agent(TestAgent())
coordinator.register_agent(DocumentationAgent())

# Execute a workflow
workflow = FeatureWorkflow("MyFeature", {
    "name": "CustomerPortal",
    "object_type": "table",
    "fields": ["PortalID", "CustomerNo"]
})

result = await coordinator.execute_workflow(workflow)

# Execute multiple workflows concurrently
workflows = [
    FeatureWorkflow("Feature1", spec1),
    FeatureWorkflow("Feature2", spec2),
]
results = await coordinator.execute_concurrent_workflows(workflows)
```

## Testing Strategy

### Test Levels

1. **Unit Tests** (`test_agents.py`): Test individual agent functionality
2. **Coordination Tests** (`test_coordination.py`): Test agent communication and self-organization
3. **Workflow Tests** (`test_workflows.py`): Test workflow execution and task dependencies
4. **Concurrent Tests** (`test_concurrent_workflows.py`): Test simultaneous workflow execution

### Key Test Scenarios

- Agent self-organization and capability matching
- Peer-to-peer communication
- Task dependency resolution
- Parallel task execution
- Multiple concurrent workflows
- Agent load distribution
- Error handling and recovery
- Real-world development scenarios

## AL/Business Central Integration

While this is a Python-based orchestration system, it's designed for AL development:

- Agents generate AL code (tables, pages, codeunits, APIs)
- Schema agents manage table extensions
- Test agents create AL test codeunits
- Deployment agents handle `.app` compilation and publishing
- Documentation agents create markdown documentation

### Gitignored AL Artifacts

- `.vscode/` - VS Code configuration
- `.alcache/` - AL compiler cache
- `.alpackages/` - Symbol packages
- `.snapshots/` - Schema snapshots
- `*.app` - Compiled extension files
- `rad.json` - RAD configuration
- `*.g.xlf` - Translation files
- `*.flf` - License files

## CFTC Analytics Tool

The repository includes a comprehensive data analytics tool for CFTC Commitments of Traders (COT) reports.

### Features

- **Data Fetching**: Fetch COT data from CFTC's Socrata API
- **Analytics Engine**: Calculate net positions, sentiment indexes, detect extremes
- **Visualization**: Generate charts and comprehensive dashboards
- **Agent Integration**: CFTCAgent for use in multi-agent workflows

### Quick Start

```python
from cftc_analytics import CFTCClient, CFTCAnalytics, ReportType

# Initialize
client = CFTCClient()
analytics = CFTCAnalytics(client=client)

# Analyze commodity
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

### Examples

See `examples/cftc/` for complete usage examples:
- `basic_usage.py` - Data fetching and analysis
- `visualization_example.py` - Chart generation
- `agent_usage.py` - Multi-agent integration

### Documentation

Complete documentation available in `cftc_analytics/README.md`

### Testing CFTC Tool

```bash
# Run CFTC tests
pytest tests/cftc/ -v

# Run with coverage
pytest tests/cftc/ --cov=cftc_analytics --cov-report=html

# Run integration tests (requires network)
pytest tests/cftc/ -m integration
```

## Git Workflow

- Main development branch: `claude/init-project-01A73ZUkYevJRXcgmUrzMddz`
- Feature branches: `claude/*`
- Commit agent and test changes together
- Run tests before committing
