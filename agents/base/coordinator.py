"""
Coordinator for managing agent self-organization and workflow execution.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Set
from agents.base.agent import Agent, AgentMessage, AgentStatus, Task
from agents.base.workflow import Workflow, WorkflowResult, WorkflowStatus


class Coordinator:
    """
    Coordinates agent self-organization and workflow execution.

    The coordinator facilitates agent discovery and task assignment,
    but agents organize themselves to complete workflows.
    """

    def __init__(self, name: str = "main_coordinator"):
        self.name = name
        self.agents: Dict[str, Agent] = {}
        self.active_workflows: Dict[str, Workflow] = {}
        self.workflow_results: Dict[str, WorkflowResult] = {}
        self.task_assignments: Dict[str, str] = {}  # task_id -> agent_id

    def register_agent(self, agent: Agent) -> None:
        """Register an agent with the coordinator."""
        self.agents[agent.id] = agent

        # Make all agents aware of each other for self-organization
        for existing_agent in self.agents.values():
            if existing_agent.id != agent.id:
                agent.register_peer(existing_agent)
                existing_agent.register_peer(agent)

    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from the coordinator."""
        if agent_id in self.agents:
            agent = self.agents.pop(agent_id)

            # Remove peer references
            for other_agent in self.agents.values():
                other_agent.unregister_peer(agent_id)

    async def execute_workflow(
        self, workflow: Workflow, timeout: Optional[float] = None
    ) -> WorkflowResult:
        """
        Execute a workflow by coordinating agent self-organization.

        The coordinator provides structure, but agents self-organize
        to complete tasks based on their capabilities.
        """
        start_time = time.time()
        self.active_workflows[workflow.id] = workflow
        workflow.status = WorkflowStatus.IN_PROGRESS

        result = WorkflowResult(
            workflow_id=workflow.id,
            status=WorkflowStatus.IN_PROGRESS,
        )

        try:
            # Check if we have enough agent capabilities
            if not workflow.can_execute_with_agents(list(self.agents.values())):
                missing_caps = self._get_missing_capabilities(workflow)
                result.status = WorkflowStatus.FAILED
                result.errors.append(f"Missing capabilities: {missing_caps}")
                return result

            # Get tasks in dependency order
            task_groups = workflow.get_task_dependency_order()

            # Execute task groups sequentially, tasks within groups concurrently
            for group_idx, task_group in enumerate(task_groups):
                # Let agents self-organize for this task group
                assignments = await self._self_organize_task_group(task_group)

                if not assignments:
                    result.status = WorkflowStatus.FAILED
                    result.errors.append(f"Failed to assign tasks in group {group_idx}")
                    break

                # Execute tasks in parallel
                task_results = await asyncio.gather(
                    *[
                        self._execute_assigned_task(task, agent_id)
                        for task, agent_id in assignments.items()
                    ],
                    return_exceptions=True,
                )

                # Process results
                for i, task_result in enumerate(task_results):
                    task = task_group[i]

                    if isinstance(task_result, Exception):
                        task.error = str(task_result)
                        result.tasks_failed.append(task)
                    elif task.completed:
                        result.tasks_completed.append(task)
                    else:
                        result.tasks_failed.append(task)

                # Check if any tasks failed
                if result.tasks_failed:
                    result.status = WorkflowStatus.PARTIALLY_COMPLETED
                    result.errors.append(f"Tasks failed in group {group_idx}")

            # Determine final status
            if not result.tasks_failed:
                result.status = WorkflowStatus.COMPLETED
            elif not result.tasks_completed:
                result.status = WorkflowStatus.FAILED
            else:
                result.status = WorkflowStatus.PARTIALLY_COMPLETED

        except asyncio.TimeoutError:
            result.status = WorkflowStatus.FAILED
            result.errors.append("Workflow execution timeout")
        except Exception as e:
            result.status = WorkflowStatus.FAILED
            result.errors.append(f"Workflow execution error: {str(e)}")
        finally:
            result.execution_time = time.time() - start_time
            workflow.status = result.status
            self.workflow_results[workflow.id] = result
            self.active_workflows.pop(workflow.id, None)

        return result

    async def _self_organize_task_group(
        self, task_group: List[Task]
    ) -> Dict[Task, str]:
        """
        Let agents self-organize to handle a group of tasks.

        Returns a mapping of tasks to agent IDs.
        """
        assignments: Dict[Task, str] = {}

        # Broadcast tasks and collect offers
        offers: Dict[str, List[tuple[str, float]]] = {
            task.id: [] for task in task_group
        }  # task_id -> [(agent_id, confidence)]

        # Request task assistance from agents
        for task in task_group:
            for agent in self.agents.values():
                if agent.can_handle_task(task) and agent.status == AgentStatus.IDLE:
                    confidence = agent._assess_task_confidence(task)
                    offers[task.id].append((agent.id, confidence))

        # Assign tasks based on best offers
        assigned_agents: Set[str] = set()

        for task in task_group:
            if not offers[task.id]:
                continue

            # Sort by confidence (highest first) and exclude already assigned agents
            available_offers = [
                (agent_id, conf)
                for agent_id, conf in offers[task.id]
                if agent_id not in assigned_agents
            ]

            if available_offers:
                available_offers.sort(key=lambda x: x[1], reverse=True)
                best_agent_id = available_offers[0][0]
                assignments[task] = best_agent_id
                assigned_agents.add(best_agent_id)

        return assignments

    async def _execute_assigned_task(self, task: Task, agent_id: str) -> Task:
        """Execute a task with the assigned agent."""
        agent = self.agents.get(agent_id)
        if not agent:
            task.error = f"Agent {agent_id} not found"
            return task

        self.task_assignments[task.id] = agent_id
        result = await agent.execute_task(task)
        return result

    def _get_missing_capabilities(self, workflow: Workflow) -> Set[str]:
        """Get capabilities required by workflow but not available in agents."""
        available_capabilities = set()
        for agent in self.agents.values():
            available_capabilities.update(agent.capabilities)

        missing = workflow.required_capabilities - available_capabilities
        return {cap.value for cap in missing}

    async def execute_concurrent_workflows(
        self, workflows: List[Workflow], timeout: Optional[float] = None
    ) -> List[WorkflowResult]:
        """
        Execute multiple workflows concurrently.

        This tests the agents' ability to coordinate on simultaneous tasks.
        """
        results = await asyncio.gather(
            *[self.execute_workflow(wf, timeout) for wf in workflows],
            return_exceptions=True,
        )

        workflow_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                workflow_results.append(
                    WorkflowResult(
                        workflow_id=workflows[i].id,
                        status=WorkflowStatus.FAILED,
                        errors=[str(result)],
                    )
                )
            else:
                workflow_results.append(result)

        return workflow_results

    def get_system_status(self) -> Dict[str, Any]:
        """Get status of the entire multi-agent system."""
        return {
            "coordinator": self.name,
            "total_agents": len(self.agents),
            "active_workflows": len(self.active_workflows),
            "completed_workflows": len(self.workflow_results),
            "agents": {agent_id: agent.get_status() for agent_id, agent in self.agents.items()},
            "workflows": {
                wf_id: {
                    "name": wf.name,
                    "status": wf.status.value,
                    "total_tasks": len(wf.tasks),
                }
                for wf_id, wf in self.active_workflows.items()
            },
        }
