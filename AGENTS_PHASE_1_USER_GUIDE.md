# User Guide: Agent System - Phase 1 Real World Test

This guide will walk you through conducting a "Phase 1" real-world test of the agent system. This test involves running a single, simple workflow to demonstrate the basic capabilities of the system.

## 1. Introduction

The agent system is a collection of specialized AI agents that work together to perform complex software development tasks. It consists of three main components:

*   **Coordinator**: The central hub that manages and directs the agents.
*   **Agents**: Specialized units that perform specific tasks, such as writing code (`CodeAgent`), running tests (`TestAgent`), or creating documentation (`DocumentationAgent`).
*   **Workflows**: A series of tasks that define a larger process, like creating a new feature or deploying an application.

This guide will show you how to run a pre-defined "Phase 1" test, which executes a single `FeatureWorkflow`.

## 2. Prerequisites

Before you begin, ensure you have the following installed:

*   **Python 3.10 or higher**
*   **Required Python packages**: You can install these using `pip`. Open your terminal or command prompt in the root directory of this project and run:

```bash
pip install -e .
```
This command will install the project in "editable" mode and all the dependencies listed in `pyproject.toml`.

## Setting Up a Virtual Environment

A virtual environment helps isolate dependencies for your project, ensuring that packages required for one project do not affect others. Here’s how to set up a virtual environment using Python’s built-in `venv` module:

### Steps to Create a Virtual Environment

1. **Open your terminal in the project directory.**
2. **Run the following command:**
   ```
   python -m venv venv
   ```
   This creates a folder named `venv` containing the virtual environment.

3. **Activate the virtual environment:**
   - On Windows:
     ```
     .\venv\Scripts\activate
     ```
       *(Do not use `source venv/bin/activate` on Windows; this will result in an error. Use the command above in PowerShell or Command Prompt.)*
   - On macOS/Linux:
     ```
     source venv/bin/activate
     ```

4. **Install project dependencies:**
   The project now includes a `requirements.txt` file listing all necessary Python packages for the agents application.

   ```powershell
   pip install -r requirements.txt
   ```

   This will install core dependencies such as numpy, pandas, matplotlib, and requests. If you need additional packages, you can add them to `requirements.txt` and rerun the install command.

5. **Deactivate the virtual environment when done testing the application. This is the final step:**


### Notes

* Always activate your virtual environment before running or developing your project.
* Add `venv/` to your `.gitignore` file to avoid committing the virtual environment to source control.

## 3. Running the Phase 1 Test

The file `example_usage.py` in the root directory contains several pre-built examples. The first example, `example_single_workflow`, is our "Phase 1" test.

To run all the examples, including the Phase 1 test, execute the following command in your terminal from the root directory of the project:

```bash
python example_usage.py
```

You will see the output for all the examples. The output for the Phase 1 test will look like this:

```
============================================================
Example 1: Single Feature Workflow
============================================================

Executing workflow: CustomerPortal
Required capabilities: ['code_generation', 'testing', 'documentation']

Workflow Status: Completed
Tasks Completed: 3/3
Execution Time: X.XXXs
```
*(Note: The execution time will vary)*

After you have finished running and testing the application, you can deactivate the virtual environment:

```powershell
deactivate
```

## 4. Understanding the Code

Let's break down the `example_single_workflow` function in `example_usage.py` to understand what's happening.

```python
async def example_single_workflow():
    """Example: Execute a single feature workflow."""
    print("\n" + "=" * 60)
    print("Example 1: Single Feature Workflow")
    print("=" * 60)

    # 1. Create coordinator
    coordinator = Coordinator(name="example_coordinator")

    # 2. Register specialized agents
    coordinator.register_agent(CodeAgent("CodeAgent-1"))
    coordinator.register_agent(TestAgent("TestAgent-1"))
    coordinator.register_agent(DocumentationAgent("DocAgent-1"))

    # 3. Define feature specification
    feature_spec = {
        "name": "CustomerPortal",
        "object_type": "table",
        "fields": ["PortalID", "CustomerNo", "AccessLevel", "LastLogin"],
    }

    # 4. Create and execute workflow
    workflow = FeatureWorkflow("CustomerPortal", feature_spec)
    print(f"\nExecuting workflow: {workflow.name}")
    print(f"Required capabilities: {[c.value for c in workflow.required_capabilities]}")

    result = await coordinator.execute_workflow(workflow)

    # 5. Display results
    print(f"\nWorkflow Status: {result.status.value}")
    print(f"Tasks Completed: {len(result.tasks_completed)}/{len(workflow.tasks)}")
    print(f"Execution Time: {result.execution_time:.3f}s")

    if result.errors:
        print(f"Errors: {result.errors}")
```

Here's what each part does:

1.  **Create coordinator**: An instance of the `Coordinator` is created to manage the workflow.
2.  **Register specialized agents**: The agents needed for this workflow (`CodeAgent`, `TestAgent`, and `DocumentationAgent`) are instantiated and registered with the coordinator.
3.  **Define feature specification**: A dictionary (`feature_spec`) defines the new feature to be built. In this case, it's a "CustomerPortal" table with several fields.
4.  **Create and execute workflow**: A `FeatureWorkflow` is created with the feature specification. The `coordinator.execute_workflow()` method is then called to run it. This is where the coordinator assigns tasks to the appropriate agents.
5.  **Display results**: The results of the workflow are printed, showing the status, number of completed tasks, and execution time.

## 5. Next Steps

This "Phase 1" test demonstrates the basic functionality of the agent system. The `example_usage.py` script also contains more advanced examples that you can explore:

*   **`example_concurrent_workflows`**: Shows how to run multiple workflows at the same time.
*   **`example_complex_deployment`**: Demonstrates a more complex workflow for deploying an application.
*   **`example_high_load`**: Simulates a high-load scenario with many concurrent workflows.

These can be considered "Phase 2" and beyond, and they build upon the concepts you've learned in this guide.
