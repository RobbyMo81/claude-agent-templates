"""
Tests for workflow execution and orchestration.
"""

import pytest
from agents.base.workflow import (
    FeatureWorkflow,
    SchemaWorkflow,
    APIWorkflow,
    DeploymentWorkflow,
    WorkflowStatus,
)


@pytest.mark.workflow
class TestFeatureWorkflow:
    """Test feature development workflow."""

    async def test_feature_workflow_structure(self, sample_feature_spec):
        """Test that feature workflow has correct structure."""
        workflow = FeatureWorkflow("TestFeature", sample_feature_spec)

        assert len(workflow.tasks) == 3  # code, test, docs
        assert workflow.name == "feature_TestFeature"
        assert workflow.status == WorkflowStatus.PENDING

    async def test_feature_workflow_dependencies(self, sample_feature_spec):
        """Test that feature workflow tasks have correct dependencies."""
        workflow = FeatureWorkflow("TestFeature", sample_feature_spec)

        task_groups = workflow.get_task_dependency_order()

        # First group should have code task (no dependencies)
        assert len(task_groups[0]) == 1
        assert "generate_code" in task_groups[0][0].name

        # Second group should have test and docs (depend on code)
        assert len(task_groups[1]) == 2
        task_names = [t.name for t in task_groups[1]]
        assert any("test" in name for name in task_names)
        assert any("docs" in name for name in task_names)

    async def test_feature_workflow_execution(
        self, coordinator_with_agents, sample_feature_spec
    ):
        """Test complete feature workflow execution."""
        workflow = FeatureWorkflow("CustomerPortal", sample_feature_spec)

        result = await coordinator_with_agents.execute_workflow(workflow)

        assert result.status == WorkflowStatus.COMPLETED
        assert len(result.tasks_completed) == 3
        assert len(result.tasks_failed) == 0
        assert result.execution_time > 0

    async def test_feature_workflow_capability_check(
        self, coordinator_with_agents, sample_feature_spec
    ):
        """Test that workflow checks for required capabilities."""
        workflow = FeatureWorkflow("TestFeature", sample_feature_spec)

        # With all agents, should be able to execute
        can_execute = workflow.can_execute_with_agents(
            list(coordinator_with_agents.agents.values())
        )
        assert can_execute is True


@pytest.mark.workflow
class TestSchemaWorkflow:
    """Test schema migration workflow."""

    async def test_schema_workflow_structure(self, sample_schema_changes):
        """Test that schema workflow has correct structure."""
        workflow = SchemaWorkflow(sample_schema_changes)

        assert len(workflow.tasks) == 4  # analyze, generate, test, deploy
        assert workflow.name == "schema_migration"

    async def test_schema_workflow_execution(
        self, coordinator_with_agents, sample_schema_changes
    ):
        """Test complete schema workflow execution."""
        workflow = SchemaWorkflow(sample_schema_changes)

        result = await coordinator_with_agents.execute_workflow(workflow)

        assert result.status == WorkflowStatus.COMPLETED
        assert len(result.tasks_completed) == 4
        assert len(result.tasks_failed) == 0

    async def test_schema_workflow_sequential_execution(self, sample_schema_changes):
        """Test that schema tasks execute in correct order."""
        workflow = SchemaWorkflow(sample_schema_changes)

        task_groups = workflow.get_task_dependency_order()

        # Should have 4 groups (each task depends on previous)
        assert len(task_groups) == 4

        # Verify order
        assert "analyze" in task_groups[0][0].name
        assert "generate" in task_groups[1][0].name
        assert "test" in task_groups[2][0].name
        assert "deploy" in task_groups[3][0].name


@pytest.mark.workflow
class TestAPIWorkflow:
    """Test API integration workflow."""

    async def test_api_workflow_structure(self, sample_api_spec):
        """Test that API workflow has correct structure."""
        workflow = APIWorkflow(sample_api_spec)

        assert len(workflow.tasks) == 4  # design, code, test, docs
        assert workflow.name == "api_integration"

    async def test_api_workflow_execution(self, coordinator_with_agents, sample_api_spec):
        """Test complete API workflow execution."""
        workflow = APIWorkflow(sample_api_spec)

        result = await coordinator_with_agents.execute_workflow(workflow)

        assert result.status == WorkflowStatus.COMPLETED
        assert len(result.tasks_completed) == 4

    async def test_api_workflow_parallel_tasks(self, sample_api_spec):
        """Test that API workflow has parallel tasks where possible."""
        workflow = APIWorkflow(sample_api_spec)

        task_groups = workflow.get_task_dependency_order()

        # Should have 3 groups
        assert len(task_groups) == 3

        # Last group should have test and docs in parallel
        assert len(task_groups[2]) == 2


@pytest.mark.workflow
class TestDeploymentWorkflow:
    """Test full deployment workflow."""

    async def test_deployment_workflow_structure(self, sample_deployment_config):
        """Test that deployment workflow has correct structure."""
        workflow = DeploymentWorkflow(sample_deployment_config)

        assert len(workflow.tasks) == 5  # analyze, test, build, docs, deploy
        assert workflow.name == "full_deployment"

    async def test_deployment_workflow_execution(
        self, coordinator_with_agents, sample_deployment_config
    ):
        """Test complete deployment workflow execution."""
        workflow = DeploymentWorkflow(sample_deployment_config)

        result = await coordinator_with_agents.execute_workflow(workflow)

        assert result.status == WorkflowStatus.COMPLETED
        assert len(result.tasks_completed) == 5
        assert result.execution_time > 0

    async def test_deployment_workflow_final_task_dependencies(
        self, sample_deployment_config
    ):
        """Test that deploy task waits for build and docs."""
        workflow = DeploymentWorkflow(sample_deployment_config)

        # Find the deploy task
        deploy_task = next(t for t in workflow.tasks if t.name == "deploy")

        # Should depend on build and docs
        assert len(deploy_task.dependencies) == 2


@pytest.mark.workflow
class TestWorkflowCoordination:
    """Test workflow coordination features."""

    async def test_workflow_task_dependency_validation(self):
        """Test that circular dependencies are detected."""
        from agents.base.workflow import Workflow
        from agents.base.agent import Task

        workflow = Workflow("test")

        task1 = Task(name="task1")
        task2 = Task(name="task2", dependencies=[task1.id])
        task1.dependencies = [task2.id]  # Create circular dependency

        workflow.add_task(task1)
        workflow.add_task(task2)

        with pytest.raises(ValueError, match="Circular dependency"):
            workflow.get_task_dependency_order()

    async def test_workflow_required_capabilities(self, sample_feature_spec):
        """Test that workflows track required capabilities."""
        workflow = FeatureWorkflow("Test", sample_feature_spec)

        from agents.base.agent import AgentCapability

        # Feature workflow needs code gen, testing, and documentation
        assert AgentCapability.CODE_GENERATION in workflow.required_capabilities
        assert AgentCapability.TESTING in workflow.required_capabilities
        assert AgentCapability.DOCUMENTATION in workflow.required_capabilities

    async def test_workflow_status_transitions(
        self, coordinator_with_agents, sample_feature_spec
    ):
        """Test that workflow status transitions correctly."""
        workflow = FeatureWorkflow("Test", sample_feature_spec)

        assert workflow.status == WorkflowStatus.PENDING

        result = await coordinator_with_agents.execute_workflow(workflow)

        assert workflow.status in [WorkflowStatus.COMPLETED, WorkflowStatus.PARTIALLY_COMPLETED]
        assert result.workflow_id == workflow.id
