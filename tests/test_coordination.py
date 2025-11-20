"""
Tests for agent coordination and self-organization.
"""

import pytest
from agents.base.agent import AgentCapability, AgentStatus, Task
from agents.base.workflow import WorkflowStatus


@pytest.mark.integration
class TestAgentCoordination:
    """Test agent coordination capabilities."""

    async def test_agent_registration(self, coordinator, code_agent, test_agent):
        """Test that agents can be registered with coordinator."""
        coordinator.register_agent(code_agent)
        coordinator.register_agent(test_agent)

        assert len(coordinator.agents) == 2
        assert code_agent.id in coordinator.agents
        assert test_agent.id in coordinator.agents

    async def test_peer_discovery(self, coordinator_with_agents, all_agents):
        """Test that agents are aware of each other."""
        # Each agent should know about all other agents
        for agent in all_agents:
            # Each agent should have n-1 peers (all others except itself)
            assert len(agent.peers) == len(all_agents) - 1

            # Verify no agent is its own peer
            assert agent.id not in agent.peers

    async def test_capability_broadcasting(self, code_agent, test_agent):
        """Test that agents can broadcast their capabilities."""
        code_agent.register_peer(test_agent)
        test_agent.register_peer(code_agent)

        await code_agent.broadcast_capability()

        # Test agent should have received the message
        assert not test_agent.message_queue.empty()

        message = await test_agent.message_queue.get()
        assert message.message_type == "capability_announcement"
        assert message.sender_id == code_agent.id
        assert "capabilities" in message.payload

    async def test_task_offer_system(self, code_agent):
        """Test that agents can offer to handle tasks."""
        task = Task(
            name="test_task",
            required_capabilities={AgentCapability.CODE_GENERATION},
        )

        # Agent should be able to handle this task
        assert code_agent.can_handle_task(task)

        # Agent should have high confidence
        confidence = code_agent._assess_task_confidence(task)
        assert confidence > 0.5

    async def test_self_organization_for_simple_task(
        self, coordinator_with_agents, sample_feature_spec
    ):
        """Test that agents self-organize to handle a simple task."""
        task = Task(
            name="generate_code",
            required_capabilities={AgentCapability.CODE_GENERATION},
            data={"feature_spec": sample_feature_spec},
        )

        # Find agents that can handle this task
        capable_agents = [
            agent for agent in coordinator_with_agents.agents.values() if agent.can_handle_task(task)
        ]

        assert len(capable_agents) > 0
        # CodeAgent and SchemaAgent both have CODE_GENERATION
        assert len(capable_agents) >= 2

    async def test_task_assignment_prevents_double_booking(self, coordinator_with_agents):
        """Test that agents don't get double-booked."""
        # Create multiple tasks
        tasks = [
            Task(
                name=f"task_{i}",
                required_capabilities={AgentCapability.CODE_GENERATION},
            )
            for i in range(3)
        ]

        # Self-organize task group
        assignments = await coordinator_with_agents._self_organize_task_group(tasks)

        # Each assigned agent should be unique
        assigned_agent_ids = list(assignments.values())
        assert len(assigned_agent_ids) == len(set(assigned_agent_ids))

    async def test_coordinator_tracks_workflow_results(
        self, coordinator_with_agents, sample_feature_spec
    ):
        """Test that coordinator tracks workflow execution."""
        from agents.base.workflow import FeatureWorkflow

        workflow = FeatureWorkflow("TestFeature", sample_feature_spec)

        result = await coordinator_with_agents.execute_workflow(workflow)

        assert workflow.id in coordinator_with_agents.workflow_results
        assert result.workflow_id == workflow.id

    async def test_system_status_reporting(self, coordinator_with_agents):
        """Test that coordinator can report system status."""
        status = coordinator_with_agents.get_system_status()

        assert "coordinator" in status
        assert "total_agents" in status
        assert status["total_agents"] == 6  # All specialized agents
        assert "agents" in status
        assert len(status["agents"]) == 6


@pytest.mark.integration
class TestAgentCommunication:
    """Test inter-agent communication."""

    async def test_direct_message(self, code_agent, test_agent):
        """Test direct messaging between agents."""
        code_agent.register_peer(test_agent)

        from agents.base.agent import AgentMessage

        message = AgentMessage(
            sender_id=code_agent.id,
            recipient_id=test_agent.id,
            message_type="custom_message",
            payload={"data": "test"},
        )

        await code_agent.send_message(message)

        # Test agent should receive the message
        assert not test_agent.message_queue.empty()
        received = await test_agent.message_queue.get()
        assert received.sender_id == code_agent.id
        assert received.payload["data"] == "test"

    async def test_broadcast_message(self, coordinator_with_agents, all_agents):
        """Test broadcasting messages to all peers."""
        sender = all_agents[0]

        from agents.base.agent import AgentMessage

        message = AgentMessage(
            sender_id=sender.id,
            recipient_id=None,  # Broadcast
            message_type="broadcast_test",
            payload={"announcement": "test"},
        )

        await sender.send_message(message)

        # All other agents should receive the message
        for agent in all_agents[1:]:
            assert not agent.message_queue.empty()

    async def test_task_completion_notification(self, code_agent, test_agent):
        """Test that agents notify peers when tasks complete."""
        code_agent.register_peer(test_agent)

        task = Task(
            name="test_notification",
            required_capabilities={AgentCapability.CODE_GENERATION},
            data={"feature_spec": {"name": "Test"}},
        )

        await code_agent.execute_task(task)

        # Test agent should receive task_started and task_completed messages
        messages = []
        while not test_agent.message_queue.empty():
            messages.append(await test_agent.message_queue.get())

        message_types = [msg.message_type for msg in messages]
        assert "task_started" in message_types
        assert "task_completed" in message_types
