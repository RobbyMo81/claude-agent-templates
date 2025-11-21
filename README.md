# Claude Agent Templates

A Python-based multi-agent system for AL (Application Language) development with Microsoft Dynamics 365 Business Central. Agents self-organize to execute complex workflows simultaneously.

## Features

- **Self-Organizing Agents**: Agents coordinate autonomously based on capabilities
- **Concurrent Workflow Execution**: Multiple workflows run simultaneously with intelligent task distribution
- **AL/Business Central Focus**: Specialized agents for AL development tasks
- **Comprehensive Testing**: Full test suite including concurrent workflow tests
- **Flexible Architecture**: Easy to extend with new agents and workflows

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/RobbyMo81/claude-agent-templates.git
cd claude-agent-templates

# Install dependencies
pip install -e .

# Install development dependencies (optional)
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit              # Unit tests
pytest -m integration       # Integration tests
pytest -m workflow          # Workflow tests
pytest -m concurrent        # Concurrent workflow tests

# Run with coverage
pytest --cov=agents --cov-report=html
```

## Architecture

### Multi-Agent System

The system implements a self-organizing multi-agent architecture:

```
┌─────────────────────────────────────────────────┐
│              Coordinator                         │
│  - Facilitates agent discovery                  │
│  - Manages workflow execution                   │
│  - Tracks system status                         │
└─────────────────────────────────────────────────┘
                      │
        ┌─────────────┴─────────────┐
        ▼                           ▼
┌──────────────┐           ┌──────────────┐
│   Workflows  │           │    Agents    │
│              │           │              │
│ - Feature    │◄──────────┤ - Code       │
│ - Schema     │           │ - Test       │
│ - API        │           │ - Schema     │
│ - Deployment │           │ - API        │
└──────────────┘           │ - Deployment │
                           │ - Docs       │
                           └──────────────┘
```

### Specialized Agents

1. **CodeAgent**: Generates and analyzes AL code
2. **TestAgent**: Creates and runs AL tests
3. **SchemaAgent**: Manages database schema and table extensions
4. **APIAgent**: Handles API integration
5. **DeploymentAgent**: Compiles and deploys apps
6. **DocumentationAgent**: Generates documentation

### Pre-defined Workflows

- **FeatureWorkflow**: Complete feature development (code → tests + docs)
- **SchemaWorkflow**: Schema migration (analyze → generate → test → deploy)
- **APIWorkflow**: API integration (design → code → tests + docs)
- **DeploymentWorkflow**: Full deployment cycle

## Usage Example

```python
import asyncio
from agents.base.coordinator import Coordinator
from agents.specialized import (
    CodeAgent, TestAgent, DocumentationAgent
)
from agents.base.workflow import FeatureWorkflow

async def main():
    # Create coordinator
    coordinator = Coordinator()

    # Register agents
    coordinator.register_agent(CodeAgent())
    coordinator.register_agent(TestAgent())
    coordinator.register_agent(DocumentationAgent())

    # Define feature specification
    feature_spec = {
        "name": "CustomerPortal",
        "object_type": "table",
        "fields": ["PortalID", "CustomerNo", "AccessLevel"]
    }

    # Execute workflow
    workflow = FeatureWorkflow("CustomerPortal", feature_spec)
    result = await coordinator.execute_workflow(workflow)

    print(f"Workflow Status: {result.status}")
    print(f"Tasks Completed: {len(result.tasks_completed)}")
    print(f"Execution Time: {result.execution_time:.2f}s")

asyncio.run(main())
```

### Concurrent Workflows

```python
# Execute multiple workflows simultaneously
workflows = [
    FeatureWorkflow("Authentication", auth_spec),
    FeatureWorkflow("Reporting", report_spec),
    APIWorkflow(api_spec),
]

results = await coordinator.execute_concurrent_workflows(workflows)

for result in results:
    print(f"{result.workflow_id}: {result.status}")
```

## Testing

### Test Coverage

The test suite includes:

- **Unit Tests**: Individual agent functionality
- **Coordination Tests**: Agent communication and self-organization
- **Workflow Tests**: Task dependency resolution and execution
- **Concurrent Tests**: Simultaneous workflow execution and load distribution

### Running Specific Tests

```bash
# Test individual agents
pytest tests/test_agents.py -v

# Test coordination
pytest tests/test_coordination.py -v

# Test workflows
pytest tests/test_workflows.py -v

# Test concurrent execution (most comprehensive)
pytest tests/test_concurrent_workflows.py -v
```

### Test Scenarios

The concurrent workflow tests cover:

- Multiple workflows running simultaneously
- Agent load distribution
- Workflow isolation
- High-load scenarios (10+ concurrent workflows)
- Real-world development scenarios
- Error handling and recovery

## Development

### Code Quality

```bash
# Format code
black agents/ tests/

# Lint
ruff check agents/ tests/

# Type check
mypy agents/

# All checks
black agents/ tests/ && ruff check agents/ tests/ && mypy agents/
```

### Adding New Agents

1. Create a new agent class inheriting from `Agent`
2. Define capabilities in `__init__`
3. Implement `_execute_task_logic` method
4. Register in `agents/specialized/__init__.py`
5. Add tests in `tests/test_agents.py`

### Adding New Workflows

1. Create workflow class inheriting from `Workflow`
2. Define tasks with dependencies in `__init__`
3. Add to `agents/base/workflow.py`
4. Add tests in `tests/test_workflows.py`

## Documentation

See [CLAUDE.md](CLAUDE.md) for detailed architecture and development guidance.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please ensure:

1. All tests pass: `pytest`
2. Code is formatted: `black agents/ tests/`
3. No linting errors: `ruff check agents/ tests/`
4. Type checking passes: `mypy agents/`
