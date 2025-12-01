"""
Project Vision and Requirements - Single Source of Truth for Projects.

This module defines the vision-centric architecture where all agents
operate from a shared understanding of the project goals.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class LifecyclePhase(Enum):
    """Phases in the software development lifecycle."""

    SCOPING = "scoping"
    REQUIREMENTS = "requirements"
    ARCHITECTURE = "architecture"
    DESIGN = "design"
    CONSTRUCTION = "construction"
    TESTING = "testing"
    DEPLOYMENT = "deployment"
    COMPLETE = "complete"


class QuestionCategory(Enum):
    """Categories of clarifying questions."""

    CLARIFICATION = "clarification"  # Ambiguous terms
    TECHNICAL = "technical"  # Technical constraints
    BUSINESS = "business"  # Business logic
    SECURITY = "security"  # Security requirements
    PERFORMANCE = "performance"  # Performance/scale
    UX = "ux"  # User experience
    INTEGRATION = "integration"  # External systems


@dataclass
class ClarifyingQuestion:
    """A question asked to refine the vision."""

    question: str
    category: QuestionCategory
    rationale: str
    examples: List[str] = field(default_factory=list)
    asked_by: Optional[str] = None  # Agent ID
    answer: Optional[str] = None
    impacts: List[str] = field(default_factory=list)  # What this affects
    timestamp: float = field(default_factory=time.time)


@dataclass
class Decision:
    """A decision made during the project lifecycle."""

    id: str
    phase: LifecyclePhase
    question: str
    decision: str
    made_by: str  # "human" or agent ID
    rationale: str
    alternatives_considered: List[str] = field(default_factory=list)
    confidence: float = 1.0  # 0.0 to 1.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HumanIntervention:
    """Records when a human overrides or corrects an agent decision."""

    phase: LifecyclePhase
    agent_proposed: str
    human_decided: str
    reason: str
    timestamp: float = field(default_factory=time.time)
    learned_from: bool = False  # Whether this was incorporated into learning


@dataclass
class Requirements:
    """Structured requirements extracted from vision."""

    functional: List[str] = field(default_factory=list)
    non_functional: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    user_stories: List[str] = field(default_factory=list)
    acceptance_criteria: List[str] = field(default_factory=list)
    out_of_scope: List[str] = field(default_factory=list)


@dataclass
class Architecture:
    """System architecture design."""

    overview: str
    components: List[Dict[str, Any]] = field(default_factory=list)
    data_model: Dict[str, Any] = field(default_factory=dict)
    integration_points: List[Dict[str, Any]] = field(default_factory=list)
    technology_choices: Dict[str, str] = field(default_factory=dict)
    design_patterns: List[str] = field(default_factory=list)
    non_functional_approach: Dict[str, str] = field(default_factory=dict)


@dataclass
class Design:
    """Detailed design specifications."""

    components: List[Dict[str, Any]] = field(default_factory=list)
    interfaces: List[Dict[str, Any]] = field(default_factory=list)
    data_structures: List[Dict[str, Any]] = field(default_factory=list)
    algorithms: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ProjectVision:
    """
    Single source of truth for the entire project.

    All agents reference this for context and alignment.
    Evolves through the lifecycle as understanding deepens.
    """

    id: str
    creator_intent: str  # Original user request
    refined_vision: str = ""  # After clarifying questions
    current_phase: LifecyclePhase = LifecyclePhase.SCOPING

    # Scoping outputs
    scope: Dict[str, Any] = field(default_factory=dict)
    success_criteria: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)

    # Requirements phase outputs
    requirements: Optional[Requirements] = None

    # Architecture phase outputs
    architecture: Optional[Architecture] = None

    # Design phase outputs
    design: Optional[Design] = None

    # Decision tracking
    decisions: List[Decision] = field(default_factory=list)
    questions: List[ClarifyingQuestion] = field(default_factory=list)
    open_questions: List[ClarifyingQuestion] = field(default_factory=list)

    # Human collaboration tracking
    human_interventions: List[HumanIntervention] = field(default_factory=list)

    # Metadata
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_decision(
        self,
        phase: LifecyclePhase,
        question: str,
        decision: str,
        made_by: str,
        rationale: str,
        confidence: float = 1.0,
        alternatives: Optional[List[str]] = None,
    ) -> Decision:
        """Record a decision made during the project."""
        decision_obj = Decision(
            id=f"decision_{len(self.decisions)}",
            phase=phase,
            question=question,
            decision=decision,
            made_by=made_by,
            rationale=rationale,
            confidence=confidence,
            alternatives_considered=alternatives or [],
        )
        self.decisions.append(decision_obj)
        self.updated_at = time.time()
        return decision_obj

    def record_intervention(
        self,
        phase: LifecyclePhase,
        agent_proposal: str,
        human_decision: str,
        reason: str,
    ) -> None:
        """Record when human overrides agent decision."""
        intervention = HumanIntervention(
            phase=phase,
            agent_proposed=agent_proposal,
            human_decided=human_decision,
            reason=reason,
        )
        self.human_interventions.append(intervention)
        self.updated_at = time.time()

    def add_question(
        self, question: ClarifyingQuestion
    ) -> None:
        """Add a clarifying question."""
        self.questions.append(question)
        if question.answer is None:
            self.open_questions.append(question)
        self.updated_at = time.time()

    def answer_question(self, question_idx: int, answer: str) -> None:
        """Answer a clarifying question."""
        if question_idx < len(self.questions):
            self.questions[question_idx].answer = answer
            # Remove from open questions
            self.open_questions = [
                q for q in self.open_questions
                if q != self.questions[question_idx]
            ]
            self.updated_at = time.time()

    def transition_phase(self, new_phase: LifecyclePhase) -> None:
        """Transition to a new lifecycle phase."""
        self.current_phase = new_phase
        self.updated_at = time.time()

    def get_context_for_phase(self, phase: LifecyclePhase) -> Dict[str, Any]:
        """
        Get relevant context for a specific phase.

        Agents can use this to understand what information is available.
        """
        context = {
            "vision": self.refined_vision or self.creator_intent,
            "scope": self.scope,
            "constraints": self.constraints,
            "success_criteria": self.success_criteria,
            "current_phase": self.current_phase.value,
            "decisions": [
                {
                    "question": d.question,
                    "decision": d.decision,
                    "rationale": d.rationale,
                }
                for d in self.decisions
            ],
        }

        if phase.value in [
            "architecture",
            "design",
            "construction",
            "testing",
            "deployment",
        ]:
            if self.requirements:
                context["requirements"] = {
                    "functional": self.requirements.functional,
                    "non_functional": self.requirements.non_functional,
                    "constraints": self.requirements.constraints,
                }

        if phase.value in ["design", "construction", "testing", "deployment"]:
            if self.architecture:
                context["architecture"] = {
                    "overview": self.architecture.overview,
                    "components": self.architecture.components,
                    "data_model": self.architecture.data_model,
                }

        if phase.value in ["construction", "testing", "deployment"]:
            if self.design:
                context["design"] = {
                    "components": self.design.components,
                    "interfaces": self.design.interfaces,
                }

        return context

    def format_readable(self) -> str:
        """Format vision for human reading."""
        output = []
        output.append(f"# Project Vision: {self.id}")
        output.append(f"\n## Original Intent")
        output.append(self.creator_intent)

        if self.refined_vision:
            output.append(f"\n## Refined Vision")
            output.append(self.refined_vision)

        if self.success_criteria:
            output.append(f"\n## Success Criteria")
            for i, criterion in enumerate(self.success_criteria, 1):
                output.append(f"{i}. {criterion}")

        if self.constraints:
            output.append(f"\n## Constraints")
            for constraint in self.constraints:
                output.append(f"- {constraint}")

        if self.requirements:
            output.append(f"\n## Requirements")
            output.append(f"\n### Functional")
            for req in self.requirements.functional:
                output.append(f"- {req}")
            output.append(f"\n### Non-Functional")
            for req in self.requirements.non_functional:
                output.append(f"- {req}")

        if self.architecture:
            output.append(f"\n## Architecture")
            output.append(self.architecture.overview)

        output.append(f"\n## Current Phase: {self.current_phase.value}")

        return "\n".join(output)
