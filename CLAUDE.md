# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a **hybrid multi-agent system** for AL/Business Central development that combines **autonomous agent coordination** with **human-in-the-loop oversight**. The system is designed to graduate from supervised execution (Phase 1) to full autonomy (Phase 4) through decision learning.

**Current Status**: Phase 1 (Hybrid Mode) - 67/71 tests passing (94%)

## Architecture: Vision-Centric Development

The core innovation is that **every agent operates from a single source of truth** (`ProjectVision` in `agents/base/vision.py`) that evolves through the development lifecycle:

```
USER INTENT → SCOPING → REQUIREMENTS → ARCHITECTURE → DESIGN → CONSTRUCTION → TESTING → DEPLOYMENT
                ↓           ↓              ↓             ↓            ↓            ↓
           [HUMAN GATE] [HUMAN GATE]  [HUMAN GATE]  [HUMAN GATE] [HUMAN GATE] [HUMAN GATE]
```

### Key Components

**Vision System** (`agents/base/vision.py`):
- `ProjectVision`: Single source of truth containing creator intent, refined vision, requirements, architecture, design, and all decisions
- `LifecyclePhase`: Enum defining phases (SCOPING → REQUIREMENTS → ARCHITECTURE → DESIGN → CONSTRUCTION → TESTING → DEPLOYMENT)
- `ClarifyingQuestion`: Questions asked by agents to refine the vision

**Decision System** (`agents/base/decision.py`):
- `DecisionMode`: AUTO_EXECUTE (≥90% confidence), PROPOSE_APPROVE (≥70%), HUMAN_DECIDE (<70%)
- `ConfidenceScore`: Agent self-assessment that determines decision routing
- `DecisionLog`: Tracks all decisions for learning patterns
- `HumanInterface`: Abstract interface with `CLIHumanInterface` implementation

**Lifecycle Management** (`agents/base/lifecycle.py`):
- `PhaseGate`: Human approval checkpoint between phases
- `GateStatus`: APPROVED, REWORK_NEEDED, REJECTED, INCOMPLETE
- `HybridLifecycleWorkflow`: Base class for workflows with gate integration

**Agent Coordination** (`agents/base/agent.py`, `agents/base/coordinator.py`):
- Peer-to-peer messaging without central control
- Capability-based task selection
- Self-organization around shared vision

## Specialized Agents

Located in `agents/specialized/`:

- **TeamAssistantAgent**: Asks clarifying questions, refines vision (VISION_FACILITATION, REQUIREMENTS_ANALYSIS)
- **SchemaAgent**: Designs table extensions (SCHEMA_MANAGEMENT, CODE_GENERATION, CODE_ANALYSIS)
- **CodeAgent**: Generates AL code (CODE_GENERATION, CODE_ANALYSIS)
- **TestAgent**: Creates test codeunits (TESTING)
- **APIAgent**: Designs APIs (API_INTEGRATION, CODE_GENERATION, CODE_ANALYSIS)
- **DeploymentAgent**: Builds and deploys .app files (DEPLOYMENT)
- **DocumentationAgent**: Generates markdown docs (DOCUMENTATION)
- **CFTCAgent**: CFTC data analysis (DATA_ANALYSIS, REPORTING)

## Common Commands

### Testing

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit              # Individual agent tests
pytest -m integration       # Agent coordination tests
pytest -m workflow          # Lifecycle execution tests
pytest -m concurrent        # Multiple workflows
pytest -m cftc              # CFTC analytics tests

# Run specific test file
pytest tests/test_table_extension_workflow.py -v

# With coverage
pytest --cov=agents --cov-report=html

# With verbose output
pytest -v

# Show test durations
pytest --durations=10
```

### Code Quality

```bash
# Format code
black agents/ tests/ examples/

# Lint
ruff check agents/ tests/ examples/

# Type checking
mypy agents/

# All quality checks
black agents/ tests/ examples/ && ruff check agents/ tests/ examples/ && mypy agents/
```

### Running Examples

```bash
# Interactive table extension demo
python examples/table_extension_demo.py

# CFTC analytics examples
python examples/cftc/basic_usage.py
python examples/cftc/visualization_example.py
python examples/cftc/agent_usage.py
```

## Development Workflow

### Concrete Workflow: Table Extension

The `TableExtensionWorkflow` (`agents/workflows/table_extension.py`) demonstrates the complete lifecycle:

```python
from agents.base.coordinator import Coordinator
from agents.base.decision import CLIHumanInterface
from agents.workflows.table_extension import TableExtensionWorkflow
from agents.specialized import (
    TeamAssistantAgent, CodeAgent,
    SchemaAgent, TestAgent
)

# Setup
coordinator = Coordinator()
coordinator.register_agent(TeamAssistantAgent())
coordinator.register_agent(CodeAgent())
coordinator.register_agent(SchemaAgent())
coordinator.register_agent(TestAgent())

# Create workflow
workflow = TableExtensionWorkflow(
    "Add Email and Phone fields to Customer table"
)

# Execute with human oversight
human = CLIHumanInterface()
result = await workflow.execute_with_gates(
    coordinator=coordinator,
    human=human
)
```

**What Happens**:
1. **Scoping Phase**: TeamAssistant asks AL-specific questions ("What BC version?", "Data classification?")
2. **Vision Gate**: Human reviews refined vision and approves/requests rework
3. **Requirements Phase**: Extract functional/non-functional requirements
4. **Requirements Gate**: Human approves requirements
5. **Architecture Phase**: SchemaAgent designs table extension structure
6. **Architecture Gate**: Human approves design
7. **Design Phase**: Define field IDs, types, properties
8. **Design Gate**: Human reviews specs
9. **Construction Phase**: CodeAgent generates AL code
10. **Construction Gate**: Human reviews code
11. **Testing Phase**: TestAgent creates test codeunit
12. **Testing Gate**: Human reviews tests

### Creating Custom Agents

```python
from agents.base.agent import Agent, AgentCapability, Task

class MyAgent(Agent):
    def __init__(self):
        super().__init__(
            name="MyAgent",
            capabilities={AgentCapability.CODE_GENERATION}
        )

    async def _execute_task_logic(self, task: Task):
        # Agent-specific logic
        return {"status": "success", "result": "..."}
```

### Creating Custom Workflows

```python
from agents.base.lifecycle import HybridLifecycleWorkflow

class MyWorkflow(HybridLifecycleWorkflow):
    def __init__(self, intent: str):
        super().__init__("MyWorkflow", intent)

    async def execute_phase(self, phase, coordinator):
        # Phase-specific logic using coordinator
        return PhaseExecutionResult(
            phase=phase,
            success=True,
            outputs={...}
        )
```

## Key Architectural Patterns

### 1. Vision Context Flow

Agents query the vision for relevant context at each phase:

```python
context = vision.get_context_for_phase(LifecyclePhase.CONSTRUCTION)
# Returns all requirements, architecture decisions, and design specs
```

### 2. Decision Learning

Every decision becomes training data:

```python
decision_log.log(
    agent_proposal="Use Text[80] for Email field",
    human_decision="Use Text[100] for Email field",
    confidence=0.85
)

# After patterns emerge:
analyzer.suggest_autonomy_upgrade(
    task_type="email_field_sizing",
    mode="auto_execute"  # Ready for automation
)
```

### 3. Confidence-Based Routing

Agents self-assess and route decisions accordingly:

```python
confidence = agent.assess_confidence(task)

if confidence.overall >= 0.9:
    # Execute automatically, notify human
    result = await agent.execute(task)
    await human.notify(f"Executed: {task.name}")
elif confidence.overall >= 0.7:
    # Propose for quick approval
    approved = await human.quick_approve(agent.proposal)
else:
    # Human decides, agent assists
    decision = await human.choose(
        prompt=f"Multiple options for {task.name}",
        options=agent.generate_alternatives()
    )
```

### 4. Phase Gates as Quality Checkpoints

Gates ensure quality before phase transitions:

```python
gate = PhaseGate(
    from_phase=LifecyclePhase.ARCHITECTURE,
    to_phase=LifecyclePhase.CONSTRUCTION,
    required_artifacts=["architecture", "data_model"],
    approval_criteria=[
        "Architecture supports all requirements",
        "Performance considerations addressed"
    ]
)

decision = await gate.review(vision, outputs, human)
# Returns: APPROVED | REWORK_NEEDED | REJECTED
```

## Test Organization

Tests are organized by concern:

- `test_agents.py`: Unit tests for individual agent functionality
- `test_coordination.py`: Peer-to-peer communication and self-organization
- `test_workflows.py`: Workflow execution and task dependencies
- `test_concurrent_workflows.py`: Simultaneous workflow execution
- `test_table_extension_workflow.py`: Complete hybrid workflow lifecycle

**Important**: Tests use `pytest-asyncio` with `asyncio_mode = "auto"` in `pyproject.toml`. All async tests must use `async def test_*`.

## Roadmap to Autonomy

The system is designed to graduate through four phases:

**Phase 1: Hybrid Mode** ✅ (Current)
- Agents propose, humans approve
- All decisions logged
- 67/71 tests passing

**Phase 2: Selective Automation** (Next)
- Automate high-confidence patterns
- Target: 30% automated, 95% accuracy
- Requires: 1000+ logged decisions

**Phase 3: Expanded Autonomy**
- Most patterns automated
- Target: 60% automated, <10% intervention

**Phase 4: Full Autonomy**
- Complete vision-to-code pipeline
- Target: 90% automated, human-level quality

## CFTC Analytics Tool

The repository includes a data analytics tool for CFTC Commitments of Traders reports (`cftc_analytics/`):

- **Data Fetching**: CFTC Socrata API client
- **Analytics**: Net positions, sentiment indexes, extremes detection
- **Visualization**: Charts and dashboards (requires `pip install -e ".[viz]"`)
- **Agent Integration**: CFTCAgent for multi-agent workflows

See `cftc_analytics/README.md` for complete documentation.

## AL/Business Central Context

While this is a Python orchestration system, it generates AL code for Dynamics 365 Business Central:

- Agents generate AL tables, pages, codeunits, APIs
- SchemaAgent manages table extensions
- TestAgent creates AL test codeunits
- DeploymentAgent handles `.app` compilation

## Installation

```bash
# Basic installation
pip install -e .

# With development tools
pip install -e ".[dev]"

# With visualization support
pip install -e ".[viz]"

# All dependencies
pip install -e ".[all]"
```

## Git Workflow

- Main branch: `main`
- Feature branches: `claude/*`
- Commit related changes together (agent + tests)
- Run tests before committing
