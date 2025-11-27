"""
Tests for Table Extension Workflow and Hybrid Architecture.

Tests the end-to-end hybrid workflow with TeamAssistantAgent and
phase gates for table extension development.
"""

import pytest
from agents.base.coordinator import Coordinator
from agents.base.vision import LifecyclePhase
from agents.specialized.team_assistant_agent import TeamAssistantAgent
from agents.specialized.code_agent import CodeAgent
from agents.specialized.schema_agent import SchemaAgent
from agents.specialized.test_agent import TestAgent
from agents.workflows.table_extension import TableExtensionWorkflow
from agents.base.agent import Task, AgentCapability


@pytest.fixture
def team_coordinator():
    """Create a coordinator with all necessary agents for table extension."""
    coordinator = Coordinator(name="TestCoordinator")

    # Register agents
    coordinator.register_agent(TeamAssistantAgent())
    coordinator.register_agent(CodeAgent())
    coordinator.register_agent(SchemaAgent())
    coordinator.register_agent(TestAgent())

    return coordinator


class TestTeamAssistantAgent:
    """Test the TeamAssistantAgent."""

    async def test_team_assistant_has_correct_capabilities(self):
        """Test that TeamAssistantAgent has required capabilities."""
        agent = TeamAssistantAgent()

        assert AgentCapability.VISION_FACILITATION in agent.capabilities
        assert AgentCapability.REQUIREMENTS_ANALYSIS in agent.capabilities
        assert AgentCapability.DECISION_SUPPORT in agent.capabilities

    async def test_generate_clarifying_questions(self):
        """Test question generation for table extensions."""
        agent = TeamAssistantAgent()

        task = Task(
            name="generate_questions",
            required_capabilities={AgentCapability.VISION_FACILITATION},
            data={"creator_intent": "Add Email field to Customer table"},
        )

        result = await agent.execute_task(task)

        assert result.completed
        assert result.result["status"] == "questions_generated"
        assert "questions" in result.result
        assert len(result.result["questions"]) > 0

    async def test_facilitate_vision_without_human(self):
        """Test vision facilitation in non-interactive mode."""
        agent = TeamAssistantAgent()

        task = Task(
            name="refine_vision",
            required_capabilities={AgentCapability.VISION_FACILITATION},
            data={
                "creator_intent": "Add Email and Phone fields to Customer table",
                "human_interface": None,  # No human interaction
            },
        )

        result = await agent.execute_task(task)

        assert result.completed
        assert result.result["status"] == "questions_generated"
        assert result.result["needs_human_input"] is True

    async def test_gather_requirements_from_vision(self):
        """Test requirements gathering."""
        from agents.base.vision import ProjectVision

        agent = TeamAssistantAgent()
        vision = ProjectVision(
            id="test_vision",
            creator_intent="Add Email and Phone fields to Customer table",
            refined_vision="Add Email (Text[80]) and Phone (Text[30]) fields to Customer table for contact information",
        )

        task = Task(
            name="gather_requirements",
            required_capabilities={AgentCapability.REQUIREMENTS_ANALYSIS},
            data={"vision": vision},
        )

        result = await agent.execute_task(task)

        assert result.completed
        assert result.result["status"] == "success"
        requirements = result.result["requirements"]
        assert requirements is not None
        assert len(requirements.functional) > 0
        assert len(requirements.non_functional) > 0


class TestTableExtensionWorkflow:
    """Test the TableExtensionWorkflow."""

    def test_workflow_initialization(self):
        """Test workflow is initialized correctly."""
        workflow = TableExtensionWorkflow("Add Email to Customer table")

        assert workflow.project_name.startswith("TableExtension_")
        assert workflow.vision.creator_intent == "Add Email to Customer table"
        assert workflow.vision.current_phase == LifecyclePhase.SCOPING

    def test_extract_table_name(self):
        """Test table name extraction from intent."""
        workflow = TableExtensionWorkflow("Extend Customer table")
        assert "Customer" in workflow.project_name

        workflow2 = TableExtensionWorkflow("Add fields to Vendor")
        assert "Vendor" in workflow2.project_name

    async def test_scoping_phase_execution(self, team_coordinator):
        """Test scoping phase executes successfully."""
        workflow = TableExtensionWorkflow("Add Email to Customer")

        result = await workflow.execute_phase(
            LifecyclePhase.SCOPING, team_coordinator
        )

        assert result.success
        assert result.phase == LifecyclePhase.SCOPING
        assert "refined_vision" in result.outputs
        assert "success_criteria" in result.outputs
        assert len(result.outputs["success_criteria"]) > 0

    async def test_requirements_phase_execution(self, team_coordinator):
        """Test requirements phase executes successfully."""
        from agents.base.vision import ProjectVision

        workflow = TableExtensionWorkflow("Add Email to Customer")
        workflow.vision = ProjectVision(
            id="test",
            creator_intent="Add Email to Customer",
            refined_vision="Add Email field to Customer table",
        )

        result = await workflow.execute_phase(
            LifecyclePhase.REQUIREMENTS, team_coordinator
        )

        assert result.success
        assert result.phase == LifecyclePhase.REQUIREMENTS
        assert "requirements" in result.outputs

    async def test_architecture_phase_execution(self, team_coordinator):
        """Test architecture phase executes successfully."""
        workflow = TableExtensionWorkflow("Add Email to Customer")

        result = await workflow.execute_phase(
            LifecyclePhase.ARCHITECTURE, team_coordinator
        )

        assert result.success
        assert result.phase == LifecyclePhase.ARCHITECTURE
        assert "architecture" in result.outputs
        assert "data_model" in result.outputs

    async def test_design_phase_execution(self, team_coordinator):
        """Test design phase executes successfully."""
        workflow = TableExtensionWorkflow("Add Email to Customer")

        result = await workflow.execute_phase(
            LifecyclePhase.DESIGN, team_coordinator
        )

        assert result.success
        assert result.phase == LifecyclePhase.DESIGN
        assert "design" in result.outputs

    async def test_construction_phase_execution(self, team_coordinator):
        """Test construction phase executes successfully."""
        workflow = TableExtensionWorkflow("Add Email to Customer")

        result = await workflow.execute_phase(
            LifecyclePhase.CONSTRUCTION, team_coordinator
        )

        assert result.success
        assert result.phase == LifecyclePhase.CONSTRUCTION
        assert "code" in result.outputs

    async def test_testing_phase_execution(self, team_coordinator):
        """Test testing phase executes successfully."""
        workflow = TableExtensionWorkflow("Add Email to Customer")

        result = await workflow.execute_phase(
            LifecyclePhase.TESTING, team_coordinator
        )

        assert result.success
        assert result.phase == LifecyclePhase.TESTING
        assert "test_results" in result.outputs

    async def test_full_workflow_phases(self, team_coordinator):
        """Test that all workflow phases can execute in sequence."""
        workflow = TableExtensionWorkflow("Add Email and Phone to Customer")

        phases = [
            LifecyclePhase.SCOPING,
            LifecyclePhase.REQUIREMENTS,
            LifecyclePhase.ARCHITECTURE,
            LifecyclePhase.DESIGN,
            LifecyclePhase.CONSTRUCTION,
            LifecyclePhase.TESTING,
        ]

        for phase in phases:
            result = await workflow.execute_phase(phase, team_coordinator)
            assert result.success, f"Phase {phase.value} failed: {result.errors}"

            # Update vision with outputs
            workflow._update_vision_from_outputs(phase, result.outputs)

    async def test_workflow_gates_defined(self):
        """Test that workflow has proper phase gates defined."""
        workflow = TableExtensionWorkflow("Add Email to Customer")

        assert len(workflow.gates) > 0

        # Check that we have gates for key transitions
        gate_names = [gate.name for gate in workflow.gates]
        assert "Vision Gate" in gate_names
        assert "Requirements Gate" in gate_names
        assert "Architecture Gate" in gate_names


class TestHybridWorkflowIntegration:
    """Integration tests for the complete hybrid workflow."""

    async def test_workflow_with_vision_context(self, team_coordinator):
        """Test that vision context flows through the workflow."""
        workflow = TableExtensionWorkflow("Add loyalty points to Customer")

        # Execute scoping
        scoping_result = await workflow.execute_phase(
            LifecyclePhase.SCOPING, team_coordinator
        )
        workflow._update_vision_from_outputs(
            LifecyclePhase.SCOPING, scoping_result.outputs
        )

        # Execute requirements with context from scoping
        req_result = await workflow.execute_phase(
            LifecyclePhase.REQUIREMENTS, team_coordinator
        )

        assert req_result.success
        # Vision context should be available to requirements phase
        assert workflow.vision.refined_vision is not None

    async def test_agent_coordination_in_workflow(self, team_coordinator):
        """Test that multiple agents coordinate within the workflow."""
        workflow = TableExtensionWorkflow("Add fields to Vendor table")

        # Execute multiple phases
        scoping = await workflow.execute_phase(
            LifecyclePhase.SCOPING, team_coordinator
        )
        workflow._update_vision_from_outputs(LifecyclePhase.SCOPING, scoping.outputs)

        construction = await workflow.execute_phase(
            LifecyclePhase.CONSTRUCTION, team_coordinator
        )

        # Both phases should succeed using different agents
        assert scoping.success
        assert construction.success

    async def test_workflow_status_tracking(self, team_coordinator):
        """Test that workflow tracks status correctly."""
        workflow = TableExtensionWorkflow("Add Email to Customer")

        # Execute a phase
        await workflow.execute_phase(LifecyclePhase.SCOPING, team_coordinator)

        # Check status
        status = workflow.get_status()

        assert status["project"] == workflow.project_name
        assert status["current_phase"] == LifecyclePhase.SCOPING.value
        assert len(status["completed_phases"]) > 0
