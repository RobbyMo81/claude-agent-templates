"""
Workflow coordination system for multi-agent tasks.
"""

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from agents.base.agent import Agent, AgentCapability, Task


class WorkflowStatus(Enum):
    """Status of a workflow execution."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIALLY_COMPLETED = "partially_completed"


@dataclass
class WorkflowResult:
    """Result of a workflow execution."""

    workflow_id: str
    status: WorkflowStatus
    tasks_completed: List[Task] = field(default_factory=list)
    tasks_failed: List[Task] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    execution_time: float = 0.0


class Workflow:
    """
    Represents a coordinated workflow that can self-organize agents.

    Workflows define what needs to be done, and agents self-organize
    to complete the tasks based on their capabilities.
    """

    def __init__(
        self,
        name: str,
        workflow_id: Optional[str] = None,
    ):
        self.id = workflow_id or f"workflow_{name}_{asyncio.get_event_loop().time()}"
        self.name = name
        self.tasks: List[Task] = []
        self.status = WorkflowStatus.PENDING
        self.required_capabilities: Set[AgentCapability] = set()

    def add_task(self, task: Task) -> None:
        """Add a task to the workflow."""
        self.tasks.append(task)
        self.required_capabilities.update(task.required_capabilities)

    def can_execute_with_agents(self, agents: List[Agent]) -> bool:
        """Check if the given agents can execute this workflow."""
        available_capabilities = set()
        for agent in agents:
            available_capabilities.update(agent.capabilities)

        return self.required_capabilities.issubset(available_capabilities)

    def get_task_dependency_order(self) -> List[List[Task]]:
        """
        Get tasks organized in dependency order.
        Returns list of task groups that can be executed in parallel.
        """
        # Build dependency graph
        task_map = {task.id: task for task in self.tasks}
        completed_tasks: Set[str] = set()
        ordered_groups: List[List[Task]] = []

        while len(completed_tasks) < len(self.tasks):
            # Find tasks with no unmet dependencies
            ready_tasks = []
            for task in self.tasks:
                if task.id in completed_tasks:
                    continue

                dependencies_met = all(dep_id in completed_tasks for dep_id in task.dependencies)
                if dependencies_met:
                    ready_tasks.append(task)

            if not ready_tasks:
                # Circular dependency or error
                raise ValueError("Circular dependency detected in workflow tasks")

            ordered_groups.append(ready_tasks)
            completed_tasks.update(task.id for task in ready_tasks)

        return ordered_groups


class FeatureWorkflow(Workflow):
    """Workflow for developing a new feature in AL/Business Central."""

    def __init__(self, feature_name: str, feature_spec: Dict[str, Any]):
        super().__init__(f"feature_{feature_name}")

        # Task 1: Generate AL code
        code_task = Task(
            name=f"generate_code_{feature_name}",
            required_capabilities={AgentCapability.CODE_GENERATION},
            data={"feature_spec": feature_spec, "language": "AL"},
        )

        # Task 2: Generate tests (depends on code)
        test_task = Task(
            name=f"generate_tests_{feature_name}",
            required_capabilities={AgentCapability.TESTING},
            data={"feature_spec": feature_spec},
            dependencies=[code_task.id],
        )

        # Task 3: Generate documentation (depends on code)
        doc_task = Task(
            name=f"generate_docs_{feature_name}",
            required_capabilities={AgentCapability.DOCUMENTATION},
            data={"feature_spec": feature_spec},
            dependencies=[code_task.id],
        )

        self.add_task(code_task)
        self.add_task(test_task)
        self.add_task(doc_task)


class SchemaWorkflow(Workflow):
    """Workflow for schema changes in AL/Business Central."""

    def __init__(self, schema_changes: Dict[str, Any]):
        super().__init__("schema_migration")

        # Task 1: Analyze current schema
        analyze_task = Task(
            name="analyze_schema",
            required_capabilities={AgentCapability.SCHEMA_MANAGEMENT, AgentCapability.CODE_ANALYSIS},
            data={"changes": schema_changes},
        )

        # Task 2: Generate table extensions
        schema_task = Task(
            name="generate_table_extensions",
            required_capabilities={AgentCapability.SCHEMA_MANAGEMENT, AgentCapability.CODE_GENERATION},
            data={"changes": schema_changes},
            dependencies=[analyze_task.id],
        )

        # Task 3: Generate migration tests
        test_task = Task(
            name="generate_migration_tests",
            required_capabilities={AgentCapability.TESTING},
            data={"changes": schema_changes},
            dependencies=[schema_task.id],
        )

        # Task 4: Prepare deployment
        deploy_task = Task(
            name="prepare_deployment",
            required_capabilities={AgentCapability.DEPLOYMENT},
            data={"changes": schema_changes},
            dependencies=[test_task.id],
        )

        self.add_task(analyze_task)
        self.add_task(schema_task)
        self.add_task(test_task)
        self.add_task(deploy_task)


class APIWorkflow(Workflow):
    """Workflow for API integration in AL/Business Central."""

    def __init__(self, api_spec: Dict[str, Any]):
        super().__init__("api_integration")

        # Task 1: Design API integration
        design_task = Task(
            name="design_api_integration",
            required_capabilities={AgentCapability.API_INTEGRATION, AgentCapability.CODE_ANALYSIS},
            data={"api_spec": api_spec},
        )

        # Task 2: Generate API code
        api_code_task = Task(
            name="generate_api_code",
            required_capabilities={AgentCapability.API_INTEGRATION, AgentCapability.CODE_GENERATION},
            data={"api_spec": api_spec},
            dependencies=[design_task.id],
        )

        # Task 3: Generate integration tests
        test_task = Task(
            name="generate_api_tests",
            required_capabilities={AgentCapability.TESTING},
            data={"api_spec": api_spec},
            dependencies=[api_code_task.id],
        )

        # Task 4: Generate API documentation
        doc_task = Task(
            name="generate_api_docs",
            required_capabilities={AgentCapability.DOCUMENTATION},
            data={"api_spec": api_spec},
            dependencies=[api_code_task.id],
        )

        self.add_task(design_task)
        self.add_task(api_code_task)
        self.add_task(test_task)
        self.add_task(doc_task)


class DeploymentWorkflow(Workflow):
    """Full deployment workflow for AL/Business Central."""

    def __init__(self, deployment_config: Dict[str, Any]):
        super().__init__("full_deployment")

        # Task 1: Code analysis
        analyze_task = Task(
            name="analyze_code",
            required_capabilities={AgentCapability.CODE_ANALYSIS},
            data={"config": deployment_config},
        )

        # Task 2: Run tests
        test_task = Task(
            name="run_tests",
            required_capabilities={AgentCapability.TESTING},
            data={"config": deployment_config},
            dependencies=[analyze_task.id],
        )

        # Task 3: Build application
        build_task = Task(
            name="build_application",
            required_capabilities={AgentCapability.DEPLOYMENT},
            data={"config": deployment_config},
            dependencies=[test_task.id],
        )

        # Task 4: Generate deployment docs
        doc_task = Task(
            name="generate_deployment_docs",
            required_capabilities={AgentCapability.DOCUMENTATION},
            data={"config": deployment_config},
            dependencies=[build_task.id],
        )

        # Task 5: Deploy
        deploy_task = Task(
            name="deploy",
            required_capabilities={AgentCapability.DEPLOYMENT},
            data={"config": deployment_config},
            dependencies=[build_task.id, doc_task.id],
        )

        self.add_task(analyze_task)
        self.add_task(test_task)
        self.add_task(build_task)
        self.add_task(doc_task)
        self.add_task(deploy_task)
