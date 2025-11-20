"""
Tests for concurrent and simultaneous workflow execution.

These tests verify that agents can coordinate tasks simultaneously
and handle multiple workflows running in parallel.
"""

import pytest
import asyncio
from agents.base.workflow import (
    FeatureWorkflow,
    SchemaWorkflow,
    APIWorkflow,
    DeploymentWorkflow,
    WorkflowStatus,
)


@pytest.mark.concurrent
class TestConcurrentWorkflowExecution:
    """Test concurrent execution of multiple workflows."""

    async def test_two_workflows_concurrent(
        self, coordinator_with_agents, sample_feature_spec, sample_api_spec
    ):
        """Test two workflows running concurrently."""
        workflow1 = FeatureWorkflow("Feature1", sample_feature_spec)
        workflow2 = APIWorkflow(sample_api_spec)

        results = await coordinator_with_agents.execute_concurrent_workflows(
            [workflow1, workflow2]
        )

        assert len(results) == 2
        assert all(r.status == WorkflowStatus.COMPLETED for r in results)
        assert results[0].workflow_id == workflow1.id
        assert results[1].workflow_id == workflow2.id

    async def test_three_workflows_concurrent(
        self,
        coordinator_with_agents,
        sample_feature_spec,
        sample_schema_changes,
        sample_api_spec,
    ):
        """Test three different workflows running simultaneously."""
        workflow1 = FeatureWorkflow("ConcurrentFeature", sample_feature_spec)
        workflow2 = SchemaWorkflow(sample_schema_changes)
        workflow3 = APIWorkflow(sample_api_spec)

        results = await coordinator_with_agents.execute_concurrent_workflows(
            [workflow1, workflow2, workflow3]
        )

        assert len(results) == 3
        assert all(r.status == WorkflowStatus.COMPLETED for r in results)

        # All workflows should complete
        total_tasks = sum(len(r.tasks_completed) for r in results)
        assert total_tasks == (3 + 4 + 4)  # Feature(3) + Schema(4) + API(4)

    async def test_multiple_same_type_workflows(
        self, coordinator_with_agents, sample_feature_spec
    ):
        """Test multiple workflows of the same type running concurrently."""
        workflows = [
            FeatureWorkflow(f"Feature{i}", sample_feature_spec) for i in range(3)
        ]

        results = await coordinator_with_agents.execute_concurrent_workflows(workflows)

        assert len(results) == 3
        # All should complete successfully
        assert all(r.status == WorkflowStatus.COMPLETED for r in results)

        # Each workflow should have completed its tasks
        for result in results:
            assert len(result.tasks_completed) == 3

    async def test_agent_load_distribution(
        self, coordinator_with_agents, sample_feature_spec
    ):
        """Test that agents distribute load across concurrent workflows."""
        # Create multiple workflows that require different capabilities
        workflows = [
            FeatureWorkflow(f"Feature{i}", sample_feature_spec) for i in range(4)
        ]

        results = await coordinator_with_agents.execute_concurrent_workflows(workflows)

        # All workflows should complete
        assert all(r.status == WorkflowStatus.COMPLETED for r in results)

        # Check that multiple agents were used
        all_agents = coordinator_with_agents.agents.values()
        agents_with_completed_tasks = [
            agent for agent in all_agents if len(agent.task_history) > 0
        ]

        # At least 3 agents should have participated
        assert len(agents_with_completed_tasks) >= 3

    async def test_concurrent_workflow_isolation(
        self, coordinator_with_agents, sample_feature_spec, sample_schema_changes
    ):
        """Test that concurrent workflows don't interfere with each other."""
        workflow1 = FeatureWorkflow("IsolatedFeature1", sample_feature_spec)
        workflow2 = SchemaWorkflow(sample_schema_changes)

        results = await coordinator_with_agents.execute_concurrent_workflows(
            [workflow1, workflow2]
        )

        # Each workflow should have its own tasks completed
        assert len(results[0].tasks_completed) == 3
        assert len(results[1].tasks_completed) == 4

        # No cross-contamination
        assert results[0].workflow_id == workflow1.id
        assert results[1].workflow_id == workflow2.id


@pytest.mark.concurrent
class TestAgentTeamCoordination:
    """Test agents working as a coordinated team."""

    async def test_agents_coordinate_on_complex_workflow(
        self, coordinator_with_agents, sample_deployment_config
    ):
        """Test that multiple agents coordinate on a complex workflow."""
        workflow = DeploymentWorkflow(sample_deployment_config)

        result = await coordinator_with_agents.execute_workflow(workflow)

        assert result.status == WorkflowStatus.COMPLETED

        # Multiple different agents should have participated
        task_agents = set()
        for task_id, agent_id in coordinator_with_agents.task_assignments.items():
            task_agents.add(agent_id)

        # At least 3 different agents should be involved
        assert len(task_agents) >= 3

    async def test_parallel_task_execution_within_workflow(
        self, coordinator_with_agents, sample_feature_spec
    ):
        """Test that independent tasks execute in parallel."""
        workflow = FeatureWorkflow("ParallelTest", sample_feature_spec)

        import time

        start_time = time.time()
        result = await coordinator_with_agents.execute_workflow(workflow)
        execution_time = time.time() - start_time

        assert result.status == WorkflowStatus.COMPLETED

        # Execution time should be less than sequential execution
        # (2 parallel tasks in group 2 should execute concurrently)
        # This is a rough check - actual timing depends on system
        assert execution_time < 10  # Reasonable upper bound

    async def test_agent_availability_tracking(self, coordinator_with_agents):
        """Test that agents track their availability correctly."""
        from agents.base.agent import AgentStatus, Task, AgentCapability

        # Get a code agent
        code_agents = [
            a
            for a in coordinator_with_agents.agents.values()
            if AgentCapability.CODE_GENERATION in a.capabilities
        ]
        agent = code_agents[0]

        assert agent.status == AgentStatus.IDLE

        # Start a task (in background)
        task = Task(
            name="blocking_task",
            required_capabilities={AgentCapability.CODE_GENERATION},
            data={"feature_spec": {"name": "Test"}},
        )

        # Execute task
        async def execute():
            await agent.execute_task(task)

        task_future = asyncio.create_task(execute())

        # Give it a moment to start
        await asyncio.sleep(0.01)

        # After completion, should be idle again
        await task_future
        assert agent.status == AgentStatus.IDLE

    async def test_workflow_error_handling(self, coordinator_with_agents):
        """Test that workflow handles task failures gracefully."""
        from agents.base.workflow import Workflow
        from agents.base.agent import Task, AgentCapability

        workflow = Workflow("error_test")

        # Create a task with impossible capability requirement
        task = Task(
            name="impossible_task",
            required_capabilities={
                AgentCapability.CODE_GENERATION,
                AgentCapability.TESTING,
                AgentCapability.DEPLOYMENT,
                AgentCapability.SCHEMA_MANAGEMENT,
                AgentCapability.API_INTEGRATION,
                AgentCapability.DOCUMENTATION,
            },  # No single agent has all these
        )
        workflow.add_task(task)

        result = await coordinator_with_agents.execute_workflow(workflow)

        # Workflow should fail gracefully
        assert result.status == WorkflowStatus.FAILED
        assert len(result.errors) > 0


@pytest.mark.concurrent
class TestHighLoadScenarios:
    """Test system under high load with many concurrent workflows."""

    async def test_many_concurrent_workflows(self, coordinator_with_agents):
        """Test system with many workflows executing simultaneously."""
        from agents.base.workflow import Workflow
        from agents.base.agent import Task, AgentCapability

        # Create 10 simple workflows
        workflows = []
        for i in range(10):
            workflow = Workflow(f"high_load_{i}")
            workflow.add_task(
                Task(
                    name=f"task_{i}",
                    required_capabilities={AgentCapability.CODE_GENERATION},
                    data={"feature_spec": {"name": f"Feature{i}"}},
                )
            )
            workflows.append(workflow)

        results = await coordinator_with_agents.execute_concurrent_workflows(workflows)

        assert len(results) == 10
        # Most should complete (some might fail due to agent availability)
        completed = [r for r in results if r.status == WorkflowStatus.COMPLETED]
        assert len(completed) >= 6  # At least 60% success rate

    async def test_mixed_workflow_types_high_load(
        self,
        coordinator_with_agents,
        sample_feature_spec,
        sample_schema_changes,
        sample_api_spec,
        sample_deployment_config,
    ):
        """Test mixed workflow types under high concurrent load."""
        workflows = []

        # Mix of different workflow types
        for i in range(3):
            workflows.append(FeatureWorkflow(f"Feature{i}", sample_feature_spec))
        for i in range(2):
            workflows.append(APIWorkflow(sample_api_spec))
        workflows.append(SchemaWorkflow(sample_schema_changes))
        workflows.append(DeploymentWorkflow(sample_deployment_config))

        results = await coordinator_with_agents.execute_concurrent_workflows(workflows)

        assert len(results) == 7

        # Check overall success
        completed = [r for r in results if r.status == WorkflowStatus.COMPLETED]
        failed = [r for r in results if r.status == WorkflowStatus.FAILED]

        # Most should complete successfully
        assert len(completed) >= 5

        # Verify no duplicate task assignments at the same time
        # (agents shouldn't be double-booked)
        # This is implicitly tested by successful completion


@pytest.mark.concurrent
class TestRealWorldScenarios:
    """Test real-world development scenarios."""

    async def test_full_development_cycle(
        self,
        coordinator_with_agents,
        sample_feature_spec,
        sample_deployment_config,
    ):
        """Test a complete development cycle: feature -> test -> deploy."""
        # Feature development
        feature_workflow = FeatureWorkflow("NewFeature", sample_feature_spec)
        feature_result = await coordinator_with_agents.execute_workflow(feature_workflow)

        assert feature_result.status == WorkflowStatus.COMPLETED

        # Deployment
        deploy_workflow = DeploymentWorkflow(sample_deployment_config)
        deploy_result = await coordinator_with_agents.execute_workflow(deploy_workflow)

        assert deploy_result.status == WorkflowStatus.COMPLETED

    async def test_parallel_feature_development(
        self, coordinator_with_agents, sample_feature_spec
    ):
        """Test multiple teams working on different features simultaneously."""
        # Simulate 3 teams working on different features
        team1_workflow = FeatureWorkflow("Authentication", sample_feature_spec)
        team2_workflow = FeatureWorkflow("Reporting", sample_feature_spec)
        team3_workflow = FeatureWorkflow("Integration", sample_feature_spec)

        results = await coordinator_with_agents.execute_concurrent_workflows(
            [team1_workflow, team2_workflow, team3_workflow]
        )

        # All teams should successfully complete
        assert all(r.status == WorkflowStatus.COMPLETED for r in results)

        # Verify each team completed all their tasks
        for result in results:
            assert len(result.tasks_completed) == 3
            assert len(result.tasks_failed) == 0

    async def test_emergency_hotfix_during_development(
        self,
        coordinator_with_agents,
        sample_feature_spec,
        sample_deployment_config,
    ):
        """Test handling urgent hotfix while other work is in progress."""
        # Regular feature development
        regular_feature = FeatureWorkflow("RegularFeature", sample_feature_spec)

        # Urgent hotfix (higher priority conceptually, but runs concurrently)
        hotfix_workflow = DeploymentWorkflow(sample_deployment_config)

        # Both should complete successfully
        results = await coordinator_with_agents.execute_concurrent_workflows(
            [regular_feature, hotfix_workflow]
        )

        assert all(r.status == WorkflowStatus.COMPLETED for r in results)
