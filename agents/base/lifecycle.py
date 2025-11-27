"""
Lifecycle management and phase gates for hybrid development workflow.

This module provides the phase gate system that enables human oversight
at critical transition points in the development lifecycle.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from agents.base.vision import LifecyclePhase, ProjectVision
from agents.base.decision import HumanInterface


class GateStatus(Enum):
    """Status of a phase gate review."""

    APPROVED = "approved"
    REWORK_NEEDED = "rework_needed"
    REJECTED = "rejected"
    INCOMPLETE = "incomplete"


@dataclass
class GateDecision:
    """Decision from a phase gate review."""

    status: GateStatus
    comments: str = ""
    changes_requested: List[str] = field(default_factory=list)
    approved_by: str = "human"

    @staticmethod
    def APPROVED(comments: str = "") -> "GateDecision":
        """Create approved decision."""
        return GateDecision(status=GateStatus.APPROVED, comments=comments)

    @staticmethod
    def REWORK(changes: List[str], comments: str = "") -> "GateDecision":
        """Create rework decision."""
        return GateDecision(
            status=GateStatus.REWORK_NEEDED,
            changes_requested=changes,
            comments=comments,
        )

    @staticmethod
    def REJECTED(reason: str) -> "GateDecision":
        """Create rejected decision."""
        return GateDecision(status=GateStatus.REJECTED, comments=reason)

    @staticmethod
    def INCOMPLETE(reason: str) -> "GateDecision":
        """Create incomplete decision."""
        return GateDecision(status=GateStatus.INCOMPLETE, comments=reason)


@dataclass
class PhaseArtifact:
    """An artifact produced during a lifecycle phase."""

    name: str
    description: str
    content: Any
    required: bool = True


class PhaseGate:
    """
    Checkpoint between lifecycle phases requiring human approval.

    Phase gates ensure quality and alignment before proceeding to next phase.
    """

    def __init__(
        self,
        from_phase: LifecyclePhase,
        to_phase: LifecyclePhase,
        required_artifacts: List[str],
        approval_criteria: List[str],
        name: Optional[str] = None,
    ):
        self.name = name or f"{from_phase.value}_to_{to_phase.value}"
        self.from_phase = from_phase
        self.to_phase = to_phase
        self.required_artifacts = required_artifacts
        self.approval_criteria = approval_criteria

    async def review(
        self,
        vision: ProjectVision,
        phase_outputs: Dict[str, Any],
        human: HumanInterface,
    ) -> GateDecision:
        """
        Present phase outputs for human review.

        Args:
            vision: Project vision with context
            phase_outputs: Outputs from the completed phase
            human: Interface to human reviewer

        Returns:
            GateDecision indicating approval, rework, or rejection
        """

        # Check all required artifacts are present
        missing = [
            artifact
            for artifact in self.required_artifacts
            if artifact not in phase_outputs
        ]

        if missing:
            return GateDecision.INCOMPLETE(f"Missing artifacts: {', '.join(missing)}")

        # Create review package
        review_content = self._format_review_package(vision, phase_outputs)

        # Present to human for review
        review_result = await human.review(review_content, self.approval_criteria)

        # Convert to gate decision
        if review_result["approved"]:
            return GateDecision.APPROVED(review_result.get("comments", ""))
        elif review_result["changes_requested"]:
            return GateDecision.REWORK(
                review_result["changes_requested"], review_result.get("comments", "")
            )
        else:
            return GateDecision.REJECTED(review_result.get("comments", "Rejected"))

    def _format_review_package(
        self, vision: ProjectVision, phase_outputs: Dict[str, Any]
    ) -> str:
        """Format review package for human consumption."""
        lines = []

        lines.append(f"{'='*60}")
        lines.append(f"PHASE GATE REVIEW: {self.name}")
        lines.append(f"{'='*60}")
        lines.append(f"\nTransition: {self.from_phase.value} → {self.to_phase.value}")

        # Vision context
        lines.append(f"\n--- PROJECT VISION ---")
        lines.append(vision.refined_vision or vision.creator_intent)

        # Phase outputs
        lines.append(f"\n--- PHASE OUTPUTS ---")
        for artifact_name in self.required_artifacts:
            if artifact_name in phase_outputs:
                lines.append(f"\n{artifact_name}:")
                lines.append(str(phase_outputs[artifact_name]))
            else:
                lines.append(f"\n{artifact_name}: MISSING")

        # Approval criteria
        lines.append(f"\n--- APPROVAL CRITERIA ---")
        for i, criterion in enumerate(self.approval_criteria, 1):
            lines.append(f"{i}. {criterion}")

        lines.append(f"\n{'='*60}")

        return "\n".join(lines)


class PhaseExecutionResult:
    """Result of executing a lifecycle phase."""

    def __init__(
        self,
        phase: LifecyclePhase,
        success: bool,
        outputs: Dict[str, Any],
        errors: List[str] = None,
    ):
        self.phase = phase
        self.success = success
        self.outputs = outputs or {}
        self.errors = errors or []


class WorkflowAborted(Exception):
    """Exception raised when workflow is aborted."""

    pass


class HybridLifecycleWorkflow:
    """
    Manages full project lifecycle with human oversight via phase gates.

    This is the hybrid approach: agents do the work, humans approve
    at critical transition points.
    """

    def __init__(self, project_name: str, creator_intent: str):
        self.project_name = project_name
        self.vision = ProjectVision(
            id=project_name, creator_intent=creator_intent
        )

        # Define standard phase gates
        self.gates = self._create_standard_gates()

        # Track execution
        self.phase_results: Dict[LifecyclePhase, PhaseExecutionResult] = {}
        self.gate_decisions: Dict[str, GateDecision] = {}

    def _create_standard_gates(self) -> List[PhaseGate]:
        """Create standard phase gates for software development."""
        return [
            PhaseGate(
                from_phase=LifecyclePhase.SCOPING,
                to_phase=LifecyclePhase.REQUIREMENTS,
                required_artifacts=["refined_vision", "success_criteria"],
                approval_criteria=[
                    "Vision is clear and unambiguous",
                    "Success criteria are measurable",
                    "Scope is realistic and well-defined",
                ],
                name="Vision Gate",
            ),
            PhaseGate(
                from_phase=LifecyclePhase.REQUIREMENTS,
                to_phase=LifecyclePhase.ARCHITECTURE,
                required_artifacts=["requirements", "constraints"],
                approval_criteria=[
                    "Requirements are complete and testable",
                    "Functional requirements are clear",
                    "Non-functional requirements are specified",
                    "Constraints are documented",
                ],
                name="Requirements Gate",
            ),
            PhaseGate(
                from_phase=LifecyclePhase.ARCHITECTURE,
                to_phase=LifecyclePhase.DESIGN,
                required_artifacts=["architecture", "data_model"],
                approval_criteria=[
                    "Architecture supports all requirements",
                    "Performance considerations are addressed",
                    "Security approach is defined",
                    "Integration points are identified",
                    "Technology choices are justified",
                ],
                name="Architecture Gate",
            ),
            PhaseGate(
                from_phase=LifecyclePhase.DESIGN,
                to_phase=LifecyclePhase.CONSTRUCTION,
                required_artifacts=["detailed_design", "interfaces"],
                approval_criteria=[
                    "Design is complete and detailed",
                    "Interfaces are well-defined",
                    "Data structures are specified",
                    "Design aligns with architecture",
                ],
                name="Design Gate",
            ),
            PhaseGate(
                from_phase=LifecyclePhase.CONSTRUCTION,
                to_phase=LifecyclePhase.TESTING,
                required_artifacts=["code", "unit_tests"],
                approval_criteria=[
                    "Code is complete",
                    "Code follows standards",
                    "Unit tests are written",
                    "Code review passed",
                ],
                name="Construction Gate",
            ),
            PhaseGate(
                from_phase=LifecyclePhase.TESTING,
                to_phase=LifecyclePhase.DEPLOYMENT,
                required_artifacts=["test_results", "test_coverage"],
                approval_criteria=[
                    "All tests pass",
                    "Test coverage meets requirements",
                    "No critical bugs",
                    "Acceptance criteria met",
                ],
                name="Testing Gate",
            ),
        ]

    def get_gate(self, from_phase: LifecyclePhase) -> Optional[PhaseGate]:
        """Get the gate for transitioning from a phase."""
        for gate in self.gates:
            if gate.from_phase == from_phase:
                return gate
        return None

    async def execute_phase(
        self,
        phase: LifecyclePhase,
        coordinator: Any,  # Coordinator type (avoid circular import)
    ) -> PhaseExecutionResult:
        """
        Execute a specific lifecycle phase.

        This is where agents do their work. Subclasses should override
        to implement phase-specific logic.
        """
        # Default implementation - subclasses override for specific phases
        outputs = {}
        errors = []

        try:
            # Phase-specific logic would go here
            # For now, return a placeholder result
            outputs = {"phase_completed": phase.value}
            success = True
        except Exception as e:
            errors.append(str(e))
            success = False

        result = PhaseExecutionResult(
            phase=phase, success=success, outputs=outputs, errors=errors
        )

        self.phase_results[phase] = result
        return result

    async def execute_with_gates(
        self,
        coordinator: Any,
        human: HumanInterface,
        start_phase: LifecyclePhase = LifecyclePhase.SCOPING,
        end_phase: LifecyclePhase = LifecyclePhase.DEPLOYMENT,
    ) -> ProjectVision:
        """
        Execute full lifecycle with human review at each phase gate.

        Args:
            coordinator: Agent coordinator
            human: Human interface for reviews
            start_phase: Phase to start from
            end_phase: Phase to end at

        Returns:
            Completed project vision

        Raises:
            WorkflowAborted: If a gate is rejected
        """

        current_phase = start_phase
        phase_order = list(LifecyclePhase)
        start_idx = phase_order.index(start_phase)
        end_idx = phase_order.index(end_phase)

        for phase_idx in range(start_idx, end_idx + 1):
            current_phase = phase_order[phase_idx]
            self.vision.transition_phase(current_phase)

            await human.notify(f"Starting phase: {current_phase.value}")

            # Execute the phase
            phase_result = await self.execute_phase(current_phase, coordinator)

            if not phase_result.success:
                await human.notify(
                    f"Phase {current_phase.value} failed: {phase_result.errors}"
                )
                raise WorkflowAborted(f"Phase {current_phase.value} failed")

            # Get the gate for this phase
            gate = self.get_gate(current_phase)

            if gate:
                # Present for gate review
                await human.notify(
                    f"Phase {current_phase.value} complete. Ready for gate review."
                )

                gate_decision = await gate.review(
                    self.vision, phase_result.outputs, human
                )

                self.gate_decisions[gate.name] = gate_decision

                if gate_decision.status == GateStatus.APPROVED:
                    await human.notify(
                        f"✓ {gate.name} approved. Proceeding to {gate.to_phase.value}"
                    )
                    # Update vision with phase outputs
                    self._update_vision_from_outputs(
                        current_phase, phase_result.outputs
                    )

                elif gate_decision.status == GateStatus.REWORK_NEEDED:
                    await human.notify(
                        f"⚠ {gate.name} requires rework: {gate_decision.comments}"
                    )
                    # TODO: Implement rework loop
                    # For now, we'll abort
                    raise WorkflowAborted(
                        f"Rework needed: {gate_decision.changes_requested}"
                    )

                else:  # REJECTED or INCOMPLETE
                    await human.notify(
                        f"✗ {gate.name} rejected: {gate_decision.comments}"
                    )
                    raise WorkflowAborted(f"{gate.name} rejected")

        await human.notify("🎉 Workflow completed successfully!")
        return self.vision

    def _update_vision_from_outputs(
        self, phase: LifecyclePhase, outputs: Dict[str, Any]
    ) -> None:
        """Update project vision with phase outputs."""

        if phase == LifecyclePhase.SCOPING:
            if "refined_vision" in outputs:
                self.vision.refined_vision = outputs["refined_vision"]
            if "success_criteria" in outputs:
                self.vision.success_criteria = outputs["success_criteria"]
            if "scope" in outputs:
                self.vision.scope = outputs["scope"]

        elif phase == LifecyclePhase.REQUIREMENTS:
            if "requirements" in outputs:
                self.vision.requirements = outputs["requirements"]

        elif phase == LifecyclePhase.ARCHITECTURE:
            if "architecture" in outputs:
                self.vision.architecture = outputs["architecture"]

        elif phase == LifecyclePhase.DESIGN:
            if "design" in outputs:
                self.vision.design = outputs["design"]

    def get_status(self) -> Dict[str, Any]:
        """Get current workflow status."""
        return {
            "project": self.project_name,
            "current_phase": self.vision.current_phase.value,
            "completed_phases": [
                phase.value for phase in self.phase_results.keys()
            ],
            "gate_decisions": {
                name: decision.status.value
                for name, decision in self.gate_decisions.items()
            },
            "vision_summary": {
                "intent": self.vision.creator_intent,
                "refined": self.vision.refined_vision,
                "success_criteria": self.vision.success_criteria,
            },
        }
