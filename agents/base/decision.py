"""
Decision tracking and learning system for hybrid agent architecture.

This module provides infrastructure for logging agent decisions,
tracking human interventions, and learning from patterns.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class DecisionMode(Enum):
    """Mode of decision making."""

    AUTO_EXECUTE = "auto_execute"  # High confidence, execute automatically
    PROPOSE_APPROVE = "propose_approve"  # Medium confidence, need approval
    HUMAN_DECIDE = "human_decide"  # Low confidence, human decides
    COLLABORATIVE = "collaborative"  # Iterate with human


@dataclass
class ConfidenceScore:
    """Confidence assessment for a decision."""

    overall: float  # 0.0 to 1.0
    factors: Dict[str, float] = field(default_factory=dict)
    reasoning: str = ""

    @property
    def decision_mode(self) -> DecisionMode:
        """Determine decision mode based on confidence."""
        if self.overall >= 0.9:
            return DecisionMode.AUTO_EXECUTE
        elif self.overall >= 0.7:
            return DecisionMode.PROPOSE_APPROVE
        else:
            return DecisionMode.HUMAN_DECIDE


@dataclass
class AgentProposal:
    """A proposal from an agent."""

    agent_id: str
    task_id: str
    proposed_action: str
    alternatives: List[str] = field(default_factory=list)
    rationale: str = ""
    confidence: ConfidenceScore = field(
        default_factory=lambda: ConfidenceScore(overall=0.5)
    )
    timestamp: float = field(default_factory=time.time)

    def format_for_human(self) -> str:
        """Format proposal for human review."""
        output = [
            f"Proposal: {self.proposed_action}",
            f"Confidence: {self.confidence.overall:.0%}",
            f"\nRationale: {self.rationale}",
        ]

        if self.alternatives:
            output.append("\nAlternatives considered:")
            for i, alt in enumerate(self.alternatives, 1):
                output.append(f"  {i}. {alt}")

        if self.confidence.factors:
            output.append("\nConfidence factors:")
            for factor, score in self.confidence.factors.items():
                output.append(f"  - {factor}: {score:.0%}")

        return "\n".join(output)


@dataclass
class DecisionContext:
    """Context in which a decision is being made."""

    task_id: str
    phase: str  # LifecyclePhase value
    task_type: str
    vision_context: Dict[str, Any] = field(default_factory=dict)
    historical_context: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class DecisionRecord:
    """Record of a decision made (for learning)."""

    id: str
    timestamp: float
    context: DecisionContext
    agent_id: str
    agent_proposal: str
    final_decision: str
    made_by: str  # "human" or agent ID
    confidence: float
    was_override: bool
    outcome: Optional[str] = None  # "success", "failure", "partial"
    metadata: Dict[str, Any] = field(default_factory=dict)


class DecisionLog:
    """
    Logs all decisions for learning and analysis.

    This is the foundation for transitioning to full autonomy.
    """

    def __init__(self):
        self.records: List[DecisionRecord] = []
        self._next_id = 0

    def log(
        self,
        context: DecisionContext,
        agent_id: str,
        agent_proposal: str,
        final_decision: str,
        made_by: str,
        confidence: float,
    ) -> DecisionRecord:
        """Log a decision."""
        record = DecisionRecord(
            id=f"decision_{self._next_id}",
            timestamp=time.time(),
            context=context,
            agent_id=agent_id,
            agent_proposal=agent_proposal,
            final_decision=final_decision,
            made_by=made_by,
            confidence=confidence,
            was_override=agent_proposal != final_decision,
        )
        self.records.append(record)
        self._next_id += 1
        return record

    def update_outcome(self, decision_id: str, outcome: str) -> None:
        """Update the outcome of a decision."""
        for record in self.records:
            if record.id == decision_id:
                record.outcome = outcome
                break

    def query(self, **filters) -> List[DecisionRecord]:
        """Query decisions by filters."""
        results = self.records

        for key, value in filters.items():
            if key == "was_override":
                results = [r for r in results if r.was_override == value]
            elif key == "phase":
                results = [r for r in results if r.context.phase == value]
            elif key == "task_type":
                results = [r for r in results if r.context.task_type == value]
            elif key == "confidence_gte":
                results = [r for r in results if r.confidence >= value]
            elif key == "made_by":
                results = [r for r in results if r.made_by == value]
            elif key == "outcome":
                results = [r for r in results if r.outcome == value]

        return results

    def get_override_rate(self) -> float:
        """Calculate percentage of decisions that were overridden."""
        if not self.records:
            return 0.0
        overrides = sum(1 for r in self.records if r.was_override)
        return overrides / len(self.records)

    def get_confidence_calibration(self, threshold: float = 0.9) -> float:
        """
        Calculate confidence calibration.

        Returns: What % of high-confidence decisions were actually correct?
        """
        high_conf = [r for r in self.records if r.confidence >= threshold]
        if not high_conf:
            return 0.0

        correct = sum(1 for r in high_conf if not r.was_override)
        return correct / len(high_conf)

    def get_task_type_stats(self, task_type: str) -> Dict[str, Any]:
        """Get statistics for a specific task type."""
        tasks = self.query(task_type=task_type)

        if not tasks:
            return {
                "count": 0,
                "override_rate": 0.0,
                "avg_confidence": 0.0,
                "success_rate": 0.0,
            }

        overrides = sum(1 for t in tasks if t.was_override)
        successes = sum(1 for t in tasks if t.outcome == "success")
        total_confidence = sum(t.confidence for t in tasks)

        return {
            "count": len(tasks),
            "override_rate": overrides / len(tasks),
            "avg_confidence": total_confidence / len(tasks),
            "success_rate": successes / len(tasks) if tasks else 0.0,
        }


@dataclass
class LearningInsight:
    """An insight learned from decision patterns."""

    category: str
    description: str
    evidence: str
    confidence: float
    actionable: bool = False
    action: str = ""


class LearningAnalyzer:
    """Analyzes decision logs to extract learning insights."""

    def __init__(self, decision_log: DecisionLog):
        self.log = decision_log

    def analyze(self) -> List[LearningInsight]:
        """Run full analysis and return insights."""
        insights = []

        # Overall override analysis
        override_rate = self.log.get_override_rate()
        insights.append(
            LearningInsight(
                category="overall_performance",
                description=f"Humans override {override_rate:.1%} of agent decisions",
                evidence=f"Based on {len(self.log.records)} decisions",
                confidence=1.0 if len(self.log.records) >= 10 else 0.5,
            )
        )

        # Confidence calibration
        calibration = self.log.get_confidence_calibration(threshold=0.9)
        insights.append(
            LearningInsight(
                category="confidence_calibration",
                description=f"High-confidence decisions are correct {calibration:.1%} of time",
                evidence=f"Threshold: 90% confidence",
                confidence=1.0 if len(self.log.query(confidence_gte=0.9)) >= 5 else 0.5,
            )
        )

        # Task-specific insights
        task_types = set(r.context.task_type for r in self.log.records)
        for task_type in task_types:
            stats = self.log.get_task_type_stats(task_type)
            if stats["count"] >= 10 and stats["override_rate"] == 0:
                insights.append(
                    LearningInsight(
                        category="automation_candidate",
                        description=f"Task type '{task_type}' ready for full automation",
                        evidence=f"{stats['count']} decisions, 0 overrides, "
                        f"{stats['avg_confidence']:.0%} avg confidence",
                        confidence=stats["avg_confidence"],
                        actionable=True,
                        action=f"Upgrade '{task_type}' to AUTO_EXECUTE mode",
                    )
                )

        return insights

    def suggest_autonomy_level(self) -> Dict[str, Any]:
        """Suggest appropriate autonomy level based on performance."""
        override_rate = self.log.get_override_rate()
        calibration = self.log.get_confidence_calibration()
        total_decisions = len(self.log.records)

        # Calculate autonomy readiness score
        if total_decisions < 20:
            readiness = 0.0
            recommendation = "INSUFFICIENT_DATA"
        elif override_rate < 0.05 and calibration > 0.95:
            readiness = 0.95
            recommendation = "READY_FOR_HIGH_AUTONOMY"
        elif override_rate < 0.10 and calibration > 0.90:
            readiness = 0.80
            recommendation = "READY_FOR_MEDIUM_AUTONOMY"
        elif override_rate < 0.20 and calibration > 0.80:
            readiness = 0.60
            recommendation = "KEEP_HUMAN_IN_LOOP"
        else:
            readiness = 0.30
            recommendation = "INCREASE_HUMAN_OVERSIGHT"

        return {
            "readiness_score": readiness,
            "recommendation": recommendation,
            "override_rate": override_rate,
            "calibration": calibration,
            "total_decisions": total_decisions,
        }


class HumanInterface:
    """
    Interface for human interaction in hybrid mode.

    This is an abstract interface that can be implemented
    for different UIs (CLI, web, etc.)
    """

    async def ask_question(self, question: str) -> str:
        """Ask human a question and get answer."""
        raise NotImplementedError

    async def quick_approve(
        self, proposal: str, default: bool = True
    ) -> bool:
        """Quick yes/no approval."""
        raise NotImplementedError

    async def choose(
        self, prompt: str, options: List[str]
    ) -> str:
        """Choose from multiple options."""
        raise NotImplementedError

    async def review(
        self, content: str, approval_criteria: List[str]
    ) -> Dict[str, Any]:
        """
        Review content against criteria.

        Returns:
            {
                "approved": bool,
                "changes_requested": List[str],
                "comments": str
            }
        """
        raise NotImplementedError

    async def notify(self, message: str) -> None:
        """Send notification to human."""
        raise NotImplementedError


class CLIHumanInterface(HumanInterface):
    """Simple CLI implementation of human interface."""

    async def ask_question(self, question: str) -> str:
        """Ask question via CLI."""
        print(f"\n{question}")
        return input("Your answer: ")

    async def quick_approve(
        self, proposal: str, default: bool = True
    ) -> bool:
        """Quick approval via CLI."""
        print(f"\n{proposal}")
        default_str = "Y/n" if default else "y/N"
        response = input(f"Approve? ({default_str}): ").strip().lower()

        if not response:
            return default
        return response in ["y", "yes"]

    async def choose(
        self, prompt: str, options: List[str]
    ) -> str:
        """Choose from options via CLI."""
        print(f"\n{prompt}")
        for i, option in enumerate(options, 1):
            print(f"{i}. {option}")

        while True:
            try:
                choice = input(f"Choose (1-{len(options)}): ").strip()
                idx = int(choice) - 1
                if 0 <= idx < len(options):
                    return options[idx]
                print(f"Please choose between 1 and {len(options)}")
            except ValueError:
                print("Please enter a number")

    async def review(
        self, content: str, approval_criteria: List[str]
    ) -> Dict[str, Any]:
        """Review content via CLI."""
        print(f"\n=== REVIEW REQUIRED ===")
        print(content)
        print(f"\n=== APPROVAL CRITERIA ===")
        for i, criterion in enumerate(approval_criteria, 1):
            print(f"{i}. {criterion}")

        print(f"\n")
        response = input("Approve / Request Changes / Reject? (a/c/r): ").strip().lower()

        if response == "a":
            return {"approved": True, "changes_requested": [], "comments": ""}
        elif response == "c":
            changes = input("What changes are needed? ").strip()
            return {
                "approved": False,
                "changes_requested": [changes],
                "comments": changes,
            }
        else:
            reason = input("Reason for rejection: ").strip()
            return {"approved": False, "changes_requested": [], "comments": reason}

    async def notify(self, message: str) -> None:
        """Print notification."""
        print(f"\n[NOTIFICATION] {message}")
