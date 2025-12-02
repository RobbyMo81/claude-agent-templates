#!/usr/bin/env python3
"""
Automated Full Workflow Test - For Stakeholder Validation

This script runs the complete 6-phase workflow with automated gate approvals
to validate end-to-end functionality without manual intervention.
"""

import asyncio
import sys
from datetime import datetime
from agents.base.coordinator import Coordinator
from agents.base.lifecycle import GateDecision, GateStatus
from agents.base.vision import LifecyclePhase
from agents.workflows.table_extension import TableExtensionWorkflow
from agents.specialized.team_assistant_agent import TeamAssistantAgent
from agents.specialized.code_agent import CodeAgent
from agents.specialized.schema_agent import SchemaAgent
from agents.specialized.test_agent import TestAgent


class AutomatedHumanInterface:
    """
    Automated human interface that auto-approves all gates for validation testing.
    """

    def __init__(self, log_file="workflow_execution.log"):
        self.log_file = log_file
        self.decisions = []
        self.log_lines = []

    def log(self, message):
        """Log a message to console and file."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] {message}"
        # Replace unicode characters that Windows console can't handle
        safe_line = line.encode('ascii', errors='replace').decode('ascii')
        print(safe_line)
        self.log_lines.append(line)  # Keep original in file

    async def review(self, content, approval_criteria):
        """Auto-approve all gates and log the decision."""
        self.log(f"\n{'='*70}")
        self.log(f"GATE REVIEW")
        self.log(f"{'='*70}")

        # Log content
        self.log(f"\nReview Content:")
        if isinstance(content, str):
            if len(content) > 200:
                self.log(f"{content[:200]}...")
            else:
                self.log(content)
        else:
            self.log(f"{type(content).__name__}")

        # Log approval criteria
        self.log(f"\nApproval Criteria:")
        for i, criterion in enumerate(approval_criteria, 1):
            self.log(f"  {i}. {criterion}")

        # Auto-approve
        self.log(f"\nDECISION: APPROVED [OK]")
        self.decisions.append({
            "decision": "APPROVED",
            "timestamp": datetime.now().isoformat()
        })

        return {
            "approved": True,
            "changes_requested": [],
            "comments": "Automated approval for validation test"
        }

    async def answer_question(self, question):
        """Auto-answer clarifying questions with defaults."""
        self.log(f"\nQUESTION: {question.question}")
        answer = "BC21"  # Default answer
        self.log(f"ANSWER (automated): {answer}")
        return answer

    async def notify(self, message):
        """Log notifications."""
        self.log(f"NOTIFY: {message}")

    def write_log(self):
        """Write execution log to file."""
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.log_lines))


async def main():
    """Run the full automated workflow test."""

    print("="*70)
    print("  AUTOMATED FULL WORKFLOW VALIDATION TEST")
    print("  Testing All 6 Phases with Auto-Approval")
    print("="*70)
    print()

    # Initialize automated human interface
    human = AutomatedHumanInterface("PHASE_1_FULL_COMPLETION_LOG.txt")

    # Test intent
    creator_intent = "Add loyalty points tracking to Customer table with points balance, tier level, and enrollment date"
    human.log(f"TEST INTENT: {creator_intent}")
    human.log(f"START TIME: {datetime.now().isoformat()}")
    human.log(f"")

    # Initialize coordinator and agents
    human.log("Initializing coordinator and agents...")
    coordinator = Coordinator(name="ValidationCoordinator")

    team_assistant = TeamAssistantAgent()
    code_agent = CodeAgent()
    schema_agent = SchemaAgent()
    test_agent = TestAgent()

    coordinator.register_agent(team_assistant)
    coordinator.register_agent(code_agent)
    coordinator.register_agent(schema_agent)
    coordinator.register_agent(test_agent)

    human.log(f"[OK] Registered {len(coordinator.agents)} agents")
    human.log("")

    # Create workflow
    human.log("Creating workflow...")
    workflow = TableExtensionWorkflow(creator_intent)
    human.log(f"[OK] Workflow created: {workflow.project_name}")
    human.log("")

    # Execute all phases
    human.log("="*70)
    human.log("STARTING FULL 6-PHASE EXECUTION")
    human.log("="*70)
    human.log("")

    try:
        final_vision = await workflow.execute_with_gates(
            coordinator=coordinator,
            human=human,
            start_phase=LifecyclePhase.SCOPING,
            end_phase=LifecyclePhase.TESTING,
        )

        human.log("")
        human.log("="*70)
        human.log("WORKFLOW EXECUTION COMPLETE")
        human.log("="*70)
        human.log("")

        # Get workflow status
        status = workflow.get_status()
        human.log(f"Current Phase: {status['current_phase']}")
        human.log(f"Completed Phases: {', '.join(status['completed_phases'])}")
        human.log("")

        # Validate all phases completed
        expected_phases = [
            "scoping", "requirements", "architecture",
            "design", "construction", "testing"
        ]
        actual_phases = status['completed_phases']

        human.log("="*70)
        human.log("PHASE COMPLETION VALIDATION")
        human.log("="*70)
        for phase in expected_phases:
            if phase in actual_phases:
                human.log(f"[OK] Phase {phase.upper()}: COMPLETED")
            else:
                human.log(f"[FAIL] Phase {phase.upper()}: FAILED")
        human.log("")

        # Check for artifacts
        human.log("="*70)
        human.log("ARTIFACT VALIDATION")
        human.log("="*70)

        if hasattr(workflow, 'phase_results') and workflow.phase_results:
            human.log(f"[OK] Artifacts generated: {len(workflow.phase_results)} phases")
            for phase, result in workflow.phase_results.items():
                if result.success and result.outputs:
                    human.log(f"  - {phase.value}: {len(result.outputs)} artifacts")
                    for artifact_name in result.outputs.keys():
                        human.log(f"    * {artifact_name}")
        else:
            human.log("[FAIL] No artifacts found in workflow.phase_results")

        human.log("")

        # Summary
        human.log("="*70)
        human.log("VALIDATION SUMMARY")
        human.log("="*70)

        all_phases_complete = all(p in actual_phases for p in expected_phases)

        if all_phases_complete:
            human.log("[OK] SUCCESS: All 6 phases completed successfully")
            human.log(f"[OK] Total gates approved: {len(human.decisions)}")
            human.log(f"[OK] End time: {datetime.now().isoformat()}")
            exit_code = 0
        else:
            human.log("[FAIL] FAILURE: Workflow incomplete")
            human.log(f"  Expected: {', '.join(expected_phases)}")
            human.log(f"  Completed: {', '.join(actual_phases)}")
            exit_code = 1

    except Exception as e:
        human.log("")
        human.log("="*70)
        human.log("VALIDATION FAILED WITH ERROR")
        human.log("="*70)
        human.log(f"Error: {str(e)}")
        import traceback
        human.log(f"\nTraceback:")
        human.log(traceback.format_exc())
        exit_code = 1

    # Write log file
    human.write_log()
    human.log("")
    human.log(f"Log written to: PHASE_1_FULL_COMPLETION_LOG.txt")

    return exit_code


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
