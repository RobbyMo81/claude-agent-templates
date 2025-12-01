# Claude Agent Templates - Hybrid Multi-Agent System

A vision-centric, self-organizing multi-agent system for AL/Business Central development with **human-in-the-loop oversight** and a clear path to **full autonomy**.

> **Status**: Hybrid Mode (Phase 1) - Human oversight with decision learning
> **Goal**: Graduate to autonomous operation when readiness criteria are met

---

## 🎯 The Vision

Create an autonomous engineering team that can take a project from concept to deployment through **self-organization around a shared vision**, with the ability to **learn from human oversight** and gradually increase autonomy.

### The Journey: Hybrid → Autonomous

```
Phase 1: HYBRID MODE ✅ (Current)
├─ Agents propose solutions
├─ Humans approve at phase gates
├─ All decisions logged for learning
└─ Test: 67/71 passing (94%)

Phase 2: SELECTIVE AUTOMATION (Next)
├─ Automate high-confidence patterns
├─ Human approval for novel cases
└─ Target: 30% automated, 95% accuracy

Phase 3: EXPANDED AUTONOMY
├─ Most patterns automated
├─ Human handles exceptions only
└─ Target: 60% automated, <10% intervention

Phase 4: FULL AUTONOMY
├─ Complete vision-to-code pipeline
├─ Human oversight optional
└─ Target: 90% automated, human-level quality
```

---

## 🏗️ Architecture

### Core Innovation: Vision-Centric Development

Every agent operates from a **single source of truth** (`ProjectVision`) that evolves through the development lifecycle:

```
USER INTENT: "Add Email and Phone fields to Customer table"
      ↓
┌─────────────────────────────────────┐
│       PROJECT VISION                │
│  - Original Intent                  │
│  - Refined Vision (after questions) │
│  - Requirements                     │
│  - Architecture                     │
│  - Design                           │
│  - All Decisions Made               │
└─────────────────────────────────────┘
      ↓
┌─────────┐   ┌──────────┐   ┌────────┐   ┌──────┐
│ SCOPING │──▶│  REQMTS  │──▶│  ARCH  │──▶│ CODE │
└─────────┘   └──────────┘   └────────┘   └──────┘
     │              │             │            │
[HUMAN GATE]  [HUMAN GATE]  [HUMAN GATE] [HUMAN GATE]
```

### Self-Organizing Agents

Agents coordinate **peer-to-peer** without central control:
- Broadcast capabilities
- Self-select tasks based on confidence
- Notify peers of progress
- Learn from human corrections

---

## 📦 What We Built

### 1. Foundation (`agents/base/`)

#### **Vision System** (`vision.py`)
```python
class ProjectVision:
    """Single source of truth for the project"""
    creator_intent: str           # Original request
    refined_vision: str           # After clarification
    requirements: Requirements    # Extracted requirements
    architecture: Architecture    # System design
    design: Design               # Detailed specs
    decisions: List[Decision]    # All decisions made
```

**Lifecycle Phases**: Scoping → Requirements → Architecture → Design → Construction → Testing → Deployment

#### **Decision Tracking** (`decision.py`)
```python
class DecisionLog:
    """Logs every decision for learning"""

    def log(self, agent_proposal, human_decision, confidence):
        # Track for pattern analysis

    def get_override_rate(self) -> float:
        # How often do humans override agents?

    def get_confidence_calibration(self) -> float:
        # When confident, are agents actually correct?
```

**Confidence-Based Routing**:
- `>= 0.9`: Auto-execute with notification
- `>= 0.7`: Propose, wait for quick approval
- `< 0.7`: Human decides, agent assists

#### **Phase Gates** (`lifecycle.py`)
```python
class PhaseGate:
    """Human approval checkpoint between phases"""

    async def review(self, vision, outputs, human):
        # Present outputs for approval
        # Return: APPROVED | REWORK_NEEDED | REJECTED
```

**Standard Gates**:
1. Vision Gate (Scoping → Requirements)
2. Requirements Gate (Requirements → Architecture)
3. Architecture Gate (Architecture → Design)
4. Design Gate (Design → Construction)
5. Construction Gate (Construction → Testing)
6. Testing Gate (Testing → Deployment)

### 2. Specialized Agents (`agents/specialized/`)

| Agent | Capabilities | Purpose |
|-------|-------------|---------|
| **TeamAssistantAgent** | Vision Facilitation, Requirements Analysis | Asks clarifying questions, refines vision |
| **CodeAgent** | Code Generation, Analysis | Generates AL code |
| **SchemaAgent** | Schema Management | Designs table extensions |
| **TestAgent** | Testing | Creates test codeunits |
| **APIAgent** | API Integration | Designs BC APIs |
| **DeploymentAgent** | Deployment | Builds and deploys .app files |
| **DocumentationAgent** | Documentation | Generates markdown docs |

### 3. Concrete Workflow (`agents/workflows/`)

#### **TableExtensionWorkflow** - Proof of Concept

Takes a simple request through the full lifecycle with human oversight:

```python
workflow = TableExtensionWorkflow(
    "Add Email and Phone fields to Customer table"
)

result = await workflow.execute_with_gates(
    coordinator=coordinator,
    human=human,  # CLI or Web interface
)
```

**What Happens**:
1. **Scoping**: TeamAssistant asks AL-specific questions
   - "What BC version?"
   - "Data classification?"
   - "Visible on which pages?"
2. **[Vision Gate]**: Human reviews refined vision
3. **Requirements**: Extract functional/non-functional requirements
4. **[Requirements Gate]**: Human approves requirements
5. **Architecture**: SchemaAgent designs table extension structure
6. **[Architecture Gate]**: Human approves design
7. **Design**: Define field IDs, types, properties
8. **[Design Gate]**: Human reviews specs
9. **Construction**: CodeAgent generates AL code
10. **[Construction Gate]**: Human reviews code
11. **Testing**: TestAgent creates test codeunit
12. **[Testing Gate]**: Human reviews tests

---

## 🚀 Quick Start

### Run the Demo

```bash
cd /home/user/claude-agent-templates
python3 examples/table_extension_demo.py
```

**Interactive Experience**:
- Type your intent or press Enter for demo
- TeamAssistant asks clarifying questions
- Approve at each phase gate (type 'a')
- Watch agents coordinate to generate code

### Programmatic Usage

```python
import asyncio
from agents.base.coordinator import Coordinator
from agents.workflows.table_extension import TableExtensionWorkflow
from agents.specialized import (
    TeamAssistantAgent, CodeAgent,
    SchemaAgent, TestAgent
)
from agents.base.decision import CLIHumanInterface

async def main():
    # Setup
    coordinator = Coordinator()
    coordinator.register_agent(TeamAssistantAgent())
    coordinator.register_agent(CodeAgent())
    coordinator.register_agent(SchemaAgent())
    coordinator.register_agent(TestAgent())

    # Create workflow
    workflow = TableExtensionWorkflow(
        "Add loyalty points to Customer table"
    )

    # Execute with human in the loop
    human = CLIHumanInterface()
    result = await workflow.execute_with_gates(
        coordinator, human
    )

    print(f"Status: {result.current_phase}")
    print(result.format_readable())

asyncio.run(main())
```

---

## 📊 Test Coverage

### Overall System
```bash
pytest
# 67/71 tests passing (94%)
```

**Test Categories**:
- ✅ Unit tests (agent functionality)
- ✅ Coordination tests (self-organization)
- ✅ Workflow tests (lifecycle execution)
- ✅ Concurrent tests (multiple workflows)
- ✅ Hybrid workflow tests (table extension)

### Table Extension Workflow
```bash
pytest tests/test_table_extension_workflow.py -v
# 15/17 tests passing (88%)
```

**Coverage**:
- TeamAssistantAgent question generation
- Vision refinement
- Requirements extraction
- All 6 lifecycle phases
- Agent coordination
- Vision context flow

### Run Tests

```bash
# All tests
pytest

# Specific categories
pytest -m unit
pytest -m workflow
pytest -m concurrent

# With coverage
pytest --cov=agents --cov-report=html

# Specific workflow
pytest tests/test_table_extension_workflow.py -v
```

---

## 🎓 Key Innovations

### 1. Vision as Version Control for Intent

Like Git tracks code changes, `ProjectVision` tracks intent evolution:

```python
vision.add_decision(
    question="Should Email be validated?",
    decision="Yes, use standard email validation",
    made_by="human",
    confidence=1.0
)

# Later, agents can query:
context = vision.get_context_for_phase(LifecyclePhase.CONSTRUCTION)
# Returns all relevant decisions and requirements
```

### 2. Learning from Human Oversight

Every decision becomes training data:

```python
# Agent proposes
agent_proposal = "Use Text[80] for Email field"

# Human decides
human_decision = "Use Text[100] for Email field"

# System learns
decision_log.log(
    agent_proposal=agent_proposal,
    human_decision=human_decision,
    confidence=0.85
)

# After 20 similar decisions with 0 overrides:
analyzer.suggest_autonomy_upgrade(
    task_type="email_field_sizing",
    mode="auto_execute"  # Ready for automation!
)
```

### 3. Confidence-Based Decision Routing

Agents self-assess confidence and route accordingly:

```python
confidence = agent.assess_confidence(task)

if confidence.overall >= 0.9:
    # High confidence - execute automatically
    result = await agent.execute(task)
    await human.notify(f"Executed: {task.name}")

elif confidence.overall >= 0.7:
    # Medium confidence - propose for approval
    approved = await human.quick_approve(
        f"Proposal: {agent.proposal}\nApprove?"
    )

else:
    # Low confidence - human decides
    decision = await human.choose(
        prompt=f"Multiple options for {task.name}",
        options=agent.generate_alternatives()
    )
```

### 4. Phase Gates as Quality Checkpoints

Human reviews at critical transitions ensure quality:

```python
gate = PhaseGate(
    from_phase=LifecyclePhase.ARCHITECTURE,
    to_phase=LifecyclePhase.CONSTRUCTION,
    required_artifacts=["architecture", "data_model"],
    approval_criteria=[
        "Architecture supports all requirements",
        "Performance considerations addressed",
        "Security approach defined"
    ]
)

decision = await gate.review(vision, outputs, human)
# Returns: APPROVED | REWORK_NEEDED | REJECTED
```

---

## 📈 Roadmap to Autonomy

### Graduation Criteria

```python
@dataclass
class AutonomyReadiness:
    vision_alignment_score: float      # >= 0.95 required
    architecture_correctness: float    # >= 0.90 required
    code_quality_score: float          # >= 0.85 required
    confidence_accuracy: float         # >= 0.90 required
    false_confidence_rate: float       # < 0.05 required
    critical_failures: int             # 0 required

    @property
    def ready_for_autonomy(self) -> bool:
        # All criteria must be met
```

### Milestones

**Phase 1: Hybrid Mode** ✅ (Complete)
- Decision logging: ✅ Operational
- Pattern analysis: ✅ Available
- Human gates: ✅ Working
- Test coverage: ✅ 94%

**Phase 2: Selective Automation** (Next)
- Complete 10+ real projects
- Collect 1000+ decisions
- Identify automation candidates
- **Target**: 30% automated, 95% accuracy

**Phase 3: Expanded Autonomy**
- Automate proven patterns
- Human handles exceptions
- **Target**: 60% automated, <10% intervention

**Phase 4: Full Autonomy**
- Most decisions autonomous
- Self-correction enabled
- **Target**: 90% automated, human-level quality

---

## 🔧 Development

### Project Structure

```
claude-agent-templates/
├── agents/
│   ├── base/                  # Core framework
│   │   ├── agent.py          # Base agent class
│   │   ├── coordinator.py    # Agent coordination
│   │   ├── workflow.py       # Workflow base
│   │   ├── vision.py         # ProjectVision system ✨
│   │   ├── decision.py       # Decision tracking ✨
│   │   └── lifecycle.py      # Phase gates ✨
│   ├── specialized/           # Domain agents
│   │   ├── team_assistant_agent.py  ✨
│   │   ├── code_agent.py
│   │   ├── schema_agent.py
│   │   ├── test_agent.py
│   │   ├── api_agent.py
│   │   ├── deployment_agent.py
│   │   └── documentation_agent.py
│   └── workflows/             # Concrete workflows
│       └── table_extension.py ✨
├── tests/
│   ├── test_agents.py
│   ├── test_coordination.py
│   ├── test_workflows.py
│   ├── test_concurrent_workflows.py
│   └── test_table_extension_workflow.py ✨
├── examples/
│   └── table_extension_demo.py ✨
├── pyproject.toml
├── CLAUDE.md
└── README.md

✨ = New in hybrid architecture
```

### Adding Custom Agents

```python
from agents.base.agent import Agent, AgentCapability, Task

class MyAgent(Agent):
    def __init__(self):
        super().__init__(
            name="MyAgent",
            capabilities={AgentCapability.CODE_GENERATION}
        )

    async def _execute_task_logic(self, task: Task):
        # Your logic here
        return {"status": "success"}
```

### Creating Custom Workflows

```python
from agents.base.lifecycle import HybridLifecycleWorkflow

class MyWorkflow(HybridLifecycleWorkflow):
    def __init__(self, intent: str):
        super().__init__("MyWorkflow", intent)

    async def execute_phase(self, phase, coordinator):
        # Phase-specific logic
        return PhaseExecutionResult(
            phase=phase,
            success=True,
            outputs={...}
        )
```

---

## 📚 Documentation

- **CLAUDE.md**: Detailed architecture and development guide
- **agents/base/vision.py**: Vision system implementation
- **agents/base/decision.py**: Decision tracking and learning
- **agents/base/lifecycle.py**: Phase gate system
- **examples/table_extension_demo.py**: Usage example

---

## 🤝 Contributing

This is a research project exploring autonomous software development. Areas for contribution:

1. **New Agents**: Specialized agents for different domains
2. **Workflow Patterns**: Additional development workflows
3. **Learning Algorithms**: Improved pattern recognition
4. **Human Interfaces**: Web UI, IDE plugins
5. **AL/BC Templates**: More Business Central patterns
6. **Autonomy Research**: Better confidence calibration

---

## 📄 License

MIT License - see LICENSE file for details

---

## 🙏 Acknowledgments

This project explores concepts from:
- Multi-agent systems and swarm intelligence
- Human-in-the-loop machine learning
- Software development lifecycle automation
- AL/Business Central best practices

---

## 📞 Contact

- Repository: `RobbyMo81/claude-agent-templates`
- Branch: `claude/start-new-project-018nWZgmB51DegGoCBF7JY4r`

---

**Built with the vision of creating truly autonomous software development teams that learn from human expertise and gradually achieve independence.** 🚀

*"The goal is not to replace human judgment, but to amplify it—and eventually, to internalize it."*
