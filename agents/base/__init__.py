"""Base agent framework components."""

from agents.base.agent import Agent, AgentCapability, AgentStatus, Task, AgentMessage
from agents.base.coordinator import Coordinator
from agents.base.workflow import (
    Workflow,
    WorkflowStatus,
    WorkflowResult,
    FeatureWorkflow,
    SchemaWorkflow,
    APIWorkflow,
    DeploymentWorkflow,
)
from agents.base.vision import (
    ProjectVision,
    LifecyclePhase,
    Requirements,
    Architecture,
    Design,
    Decision,
    ClarifyingQuestion,
    QuestionCategory,
    HumanIntervention,
)
from agents.base.decision import (
    DecisionMode,
    ConfidenceScore,
    AgentProposal,
    DecisionContext,
    DecisionRecord,
    DecisionLog,
    LearningInsight,
    LearningAnalyzer,
    HumanInterface,
    CLIHumanInterface,
)
from agents.base.lifecycle import (
    PhaseGate,
    GateDecision,
    GateStatus,
    PhaseArtifact,
    PhaseExecutionResult,
    HybridLifecycleWorkflow,
    WorkflowAborted,
)

__all__ = [
    # Core agent
    "Agent",
    "AgentCapability",
    "AgentStatus",
    "Task",
    "AgentMessage",
    # Coordination
    "Coordinator",
    # Workflows
    "Workflow",
    "WorkflowStatus",
    "WorkflowResult",
    "FeatureWorkflow",
    "SchemaWorkflow",
    "APIWorkflow",
    "DeploymentWorkflow",
    # Vision
    "ProjectVision",
    "LifecyclePhase",
    "Requirements",
    "Architecture",
    "Design",
    "Decision",
    "ClarifyingQuestion",
    "QuestionCategory",
    "HumanIntervention",
    # Decision support
    "DecisionMode",
    "ConfidenceScore",
    "AgentProposal",
    "DecisionContext",
    "DecisionRecord",
    "DecisionLog",
    "LearningInsight",
    "LearningAnalyzer",
    "HumanInterface",
    "CLIHumanInterface",
    # Lifecycle
    "PhaseGate",
    "GateDecision",
    "GateStatus",
    "PhaseArtifact",
    "PhaseExecutionResult",
    "HybridLifecycleWorkflow",
    "WorkflowAborted",
]
