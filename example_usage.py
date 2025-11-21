"""
Example usage of the multi-agent system.

This script demonstrates:
1. Creating a coordinator
2. Registering specialized agents
3. Executing individual workflows
4. Executing multiple concurrent workflows
"""

import asyncio
from agents.base.coordinator import Coordinator
from agents.specialized import (
    CodeAgent,
    TestAgent,
    SchemaAgent,
    APIAgent,
    DeploymentAgent,
    DocumentationAgent,
)
from agents.base.workflow import (
    FeatureWorkflow,
    SchemaWorkflow,
    APIWorkflow,
    DeploymentWorkflow,
    WorkflowStatus,
)


async def example_single_workflow():
    """Example: Execute a single feature workflow."""
    print("\n" + "=" * 60)
    print("Example 1: Single Feature Workflow")
    print("=" * 60)

    # Create coordinator
    coordinator = Coordinator(name="example_coordinator")

    # Register all specialized agents
    coordinator.register_agent(CodeAgent("CodeAgent-1"))
    coordinator.register_agent(TestAgent("TestAgent-1"))
    coordinator.register_agent(DocumentationAgent("DocAgent-1"))

    # Define feature specification
    feature_spec = {
        "name": "CustomerPortal",
        "object_type": "table",
        "fields": ["PortalID", "CustomerNo", "AccessLevel", "LastLogin"],
    }

    # Create and execute workflow
    workflow = FeatureWorkflow("CustomerPortal", feature_spec)
    print(f"\nExecuting workflow: {workflow.name}")
    print(f"Required capabilities: {[c.value for c in workflow.required_capabilities]}")

    result = await coordinator.execute_workflow(workflow)

    # Display results
    print(f"\nWorkflow Status: {result.status.value}")
    print(f"Tasks Completed: {len(result.tasks_completed)}/{len(workflow.tasks)}")
    print(f"Execution Time: {result.execution_time:.3f}s")

    if result.errors:
        print(f"Errors: {result.errors}")


async def example_concurrent_workflows():
    """Example: Execute multiple workflows concurrently."""
    print("\n" + "=" * 60)
    print("Example 2: Concurrent Workflows")
    print("=" * 60)

    # Create coordinator with all agents
    coordinator = Coordinator(name="concurrent_coordinator")

    # Register all agent types
    coordinator.register_agent(CodeAgent("CodeAgent-1"))
    coordinator.register_agent(TestAgent("TestAgent-1"))
    coordinator.register_agent(SchemaAgent("SchemaAgent-1"))
    coordinator.register_agent(APIAgent("APIAgent-1"))
    coordinator.register_agent(DeploymentAgent("DeployAgent-1"))
    coordinator.register_agent(DocumentationAgent("DocAgent-1"))

    print(f"\nRegistered {len(coordinator.agents)} agents")

    # Create multiple workflows
    workflows = [
        FeatureWorkflow(
            "Authentication",
            {
                "name": "UserAuth",
                "object_type": "codeunit",
                "fields": ["UserID", "Token"],
            },
        ),
        FeatureWorkflow(
            "Reporting",
            {
                "name": "SalesReport",
                "object_type": "page",
                "fields": ["ReportID", "Period"],
            },
        ),
        APIWorkflow(
            {
                "entity": "Customer",
                "type": "REST",
                "auth": "OAuth2",
                "endpoints": ["GET", "POST", "PATCH"],
            }
        ),
    ]

    print(f"\nExecuting {len(workflows)} workflows concurrently...")
    print("Workflows:")
    for wf in workflows:
        print(f"  - {wf.name} ({len(wf.tasks)} tasks)")

    # Execute concurrently
    results = await coordinator.execute_concurrent_workflows(workflows)

    # Display results
    print("\nResults:")
    for i, result in enumerate(results):
        workflow_name = workflows[i].name
        status_symbol = "✓" if result.status == WorkflowStatus.COMPLETED else "✗"
        print(f"  {status_symbol} {workflow_name}:")
        print(f"      Status: {result.status.value}")
        print(f"      Tasks: {len(result.tasks_completed)}/{len(workflows[i].tasks)}")
        print(f"      Time: {result.execution_time:.3f}s")

    # System status
    print("\nSystem Status:")
    status = coordinator.get_system_status()
    print(f"  Total Agents: {status['total_agents']}")
    print(f"  Completed Workflows: {status['completed_workflows']}")


async def example_complex_deployment():
    """Example: Complex deployment workflow."""
    print("\n" + "=" * 60)
    print("Example 3: Full Deployment Workflow")
    print("=" * 60)

    # Create coordinator with all agents
    coordinator = Coordinator(name="deployment_coordinator")

    # Register all necessary agents
    for AgentClass, name in [
        (CodeAgent, "CodeAgent-1"),
        (TestAgent, "TestAgent-1"),
        (DeploymentAgent, "DeployAgent-1"),
        (DocumentationAgent, "DocAgent-1"),
    ]:
        coordinator.register_agent(AgentClass(name))

    # Deployment configuration
    deployment_config = {
        "app_name": "CustomerManagement",
        "version": "2.0.0.0",
        "environment": "production",
    }

    # Create deployment workflow
    workflow = DeploymentWorkflow(deployment_config)

    print(f"\nExecuting deployment workflow: {workflow.name}")
    print(f"Tasks in workflow: {len(workflow.tasks)}")

    # Show task dependencies
    task_groups = workflow.get_task_dependency_order()
    print(f"\nTask execution groups (by dependency):")
    for i, group in enumerate(task_groups, 1):
        tasks_in_group = [t.name for t in group]
        print(f"  Group {i}: {tasks_in_group}")

    # Execute
    result = await coordinator.execute_workflow(workflow)

    # Results
    print(f"\nDeployment Status: {result.status.value}")
    print(f"Tasks Completed: {len(result.tasks_completed)}")
    print(f"Total Execution Time: {result.execution_time:.3f}s")

    if result.status == WorkflowStatus.COMPLETED:
        print("\n✓ Deployment successful!")
    else:
        print(f"\n✗ Deployment failed: {result.errors}")


async def example_high_load():
    """Example: High load scenario with many concurrent workflows."""
    print("\n" + "=" * 60)
    print("Example 4: High Load Test (10 Concurrent Workflows)")
    print("=" * 60)

    # Create coordinator
    coordinator = Coordinator(name="high_load_coordinator")

    # Register agents
    coordinator.register_agent(CodeAgent("CodeAgent-1"))
    coordinator.register_agent(TestAgent("TestAgent-1"))
    coordinator.register_agent(DocumentationAgent("DocAgent-1"))

    # Create 10 feature workflows
    workflows = [
        FeatureWorkflow(
            f"Feature{i}",
            {
                "name": f"Feature{i}",
                "object_type": "table",
                "fields": ["Field1", "Field2"],
            },
        )
        for i in range(10)
    ]

    print(f"\nExecuting {len(workflows)} workflows concurrently...")

    import time

    start_time = time.time()
    results = await coordinator.execute_concurrent_workflows(workflows)
    total_time = time.time() - start_time

    # Analyze results
    completed = sum(1 for r in results if r.status == WorkflowStatus.COMPLETED)
    failed = len(results) - completed

    print(f"\nResults:")
    print(f"  Total Workflows: {len(workflows)}")
    print(f"  Completed: {completed}")
    print(f"  Failed: {failed}")
    print(f"  Success Rate: {(completed/len(workflows)*100):.1f}%")
    print(f"  Total Time: {total_time:.3f}s")
    print(f"  Avg Time per Workflow: {(total_time/len(workflows)):.3f}s")


async def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Multi-Agent System Examples")
    print("=" * 60)

    # Run examples
    await example_single_workflow()
    await example_concurrent_workflows()
    await example_complex_deployment()
    await example_high_load()

    print("\n" + "=" * 60)
    print("Examples Complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
