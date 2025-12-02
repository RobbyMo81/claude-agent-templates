#!/usr/bin/env python3
"""
Table Extension Full Demo - Complete 6-phase workflow demonstration.

This script runs through ALL 6 phases with human gates:
1. Scoping → Requirements Gate
2. Requirements → Architecture Gate
3. Architecture → Design Gate
4. Design → Construction Gate
5. Construction → Testing Gate
6. Testing → Complete

Usage:
    python examples/table_extension_full_demo.py
"""

import asyncio
from agents.base.coordinator import Coordinator
from agents.base.decision import CLIHumanInterface
from agents.base.vision import LifecyclePhase
from agents.workflows.table_extension import TableExtensionWorkflow
from agents.specialized.team_assistant_agent import TeamAssistantAgent
from agents.specialized.code_agent import CodeAgent
from agents.specialized.schema_agent import SchemaAgent
from agents.specialized.test_agent import TestAgent


async def main():
    """Run the full table extension workflow."""

    print("=" * 70)
    print("  FULL HYBRID AGENT WORKFLOW DEMONSTRATION")
    print("  Table Extension for Business Central")
    print("  (ALL 6 PHASES WITH HUMAN GATES)")
    print("=" * 70)
    print()

    # Step 1: User provides initial intent
    creator_intent = input(
        "What would you like to do? (or press Enter for demo): "
    ).strip()

    if not creator_intent:
        creator_intent = "Add loyalty points tracking to Customer table with points balance, tier level, and enrollment date"
        print(f"Using demo intent: {creator_intent}")

    print()

    # Step 2: Initialize the coordinator and agents
    print("Initializing agent team...")
    coordinator = Coordinator(name="TableExtensionCoordinator")

    # Register agents
    team_assistant = TeamAssistantAgent()
    code_agent = CodeAgent()
    schema_agent = SchemaAgent()
    test_agent = TestAgent()

    coordinator.register_agent(team_assistant)
    coordinator.register_agent(code_agent)
    coordinator.register_agent(schema_agent)
    coordinator.register_agent(test_agent)

    print(f"✓ Registered {len(coordinator.agents)} agents")
    print(f"  - {team_assistant.name} (Vision Facilitation)")
    print(f"  - {code_agent.name} (Code Generation)")
    print(f"  - {schema_agent.name} (Schema Management)")
    print(f"  - {test_agent.name} (Testing)")
    print()

    # Step 3: Create the workflow
    print("Creating workflow...")
    workflow = TableExtensionWorkflow(creator_intent)
    print(f"✓ Workflow created: {workflow.project_name}")
    print()

    # Step 4: Create human interface
    human = CLIHumanInterface()

    # Step 5: Execute workflow with ALL gates
    print("=" * 70)
    print("  STARTING FULL WORKFLOW EXECUTION")
    print("=" * 70)
    print()
    print("The workflow will proceed through ALL phases:")
    print("  1. Scoping (Vision Refinement)")
    print("  2. Requirements (Structured Requirements)")
    print("  3. Architecture (System Design)")
    print("  4. Design (Detailed Specifications)")
    print("  5. Construction (Code Generation)")
    print("  6. Testing (Test Generation)")
    print()
    print("You'll be asked to approve at each phase gate.")
    print("Type 'a' to approve, 'r' to request rework, or 'j' to reject.")
    print()

    input("Press Enter to begin...")
    print()

    try:
        # Execute ALL phases from SCOPING to DEPLOYMENT
        final_vision = await workflow.execute_with_gates(
            coordinator=coordinator,
            human=human,
            start_phase=LifecyclePhase.SCOPING,
            end_phase=LifecyclePhase.DEPLOYMENT,  # Run through all phases
        )

        print()
        print("=" * 70)
        print("  WORKFLOW COMPLETE!")
        print("=" * 70)
        print()
        print("Final Vision:")
        print(final_vision.format_readable())
        print()

        # Show workflow status
        status = workflow.get_status()
        print("Workflow Status:")
        print(f"  Current Phase: {status['current_phase']}")
        print(f"  Completed Phases: {', '.join(status['completed_phases'])}")
        print()

        # Show generated artifacts
        print("=" * 70)
        print("  GENERATED ARTIFACTS")
        print("=" * 70)
        print()

        if hasattr(workflow, 'outputs') and workflow.outputs:
            for phase, output in workflow.outputs.items():
                print(f"\n{phase.upper()}:")
                if isinstance(output, dict):
                    for key, value in output.items():
                        if isinstance(value, str) and len(value) > 200:
                            print(f"  {key}: {value[:200]}...")
                        else:
                            print(f"  {key}: {value}")
                else:
                    print(f"  {output}")

        print()

    except KeyboardInterrupt:
        print()
        print("❌ Workflow cancelled by user")
        print()
        return 1
    except Exception as e:
        print()
        print(f"❌ Workflow failed: {str(e)}")
        import traceback
        traceback.print_exc()
        print()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
