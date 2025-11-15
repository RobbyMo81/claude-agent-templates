"""
Unit tests for individual agents.
"""

import pytest
from agents.base.agent import AgentCapability, AgentStatus, Task


@pytest.mark.unit
class TestCodeAgent:
    """Test CodeAgent functionality."""

    async def test_code_generation(self, code_agent, sample_feature_spec):
        """Test that CodeAgent can generate AL code."""
        task = Task(
            name="generate_code_test",
            required_capabilities={AgentCapability.CODE_GENERATION},
            data={"feature_spec": sample_feature_spec, "language": "AL"},
        )

        result = await code_agent.execute_task(task)

        assert result.completed is True
        assert result.error is None
        assert "generated_code" in result.result
        assert result.result["status"] == "success"
        assert code_agent.status == AgentStatus.IDLE

    async def test_code_analysis(self, code_agent):
        """Test that CodeAgent can analyze code."""
        task = Task(
            name="analyze_code_test",
            required_capabilities={AgentCapability.CODE_ANALYSIS},
            data={"code": "table 50100 MyTable { }"},
        )

        result = await code_agent.execute_task(task)

        assert result.completed is True
        assert "analysis" in result.result
        assert "quality_score" in result.result["analysis"]

    async def test_capability_check(self, code_agent):
        """Test that CodeAgent correctly identifies its capabilities."""
        assert AgentCapability.CODE_GENERATION in code_agent.capabilities
        assert AgentCapability.CODE_ANALYSIS in code_agent.capabilities
        assert AgentCapability.TESTING not in code_agent.capabilities


@pytest.mark.unit
class TestTestAgent:
    """Test TestAgent functionality."""

    async def test_test_generation(self, test_agent, sample_feature_spec):
        """Test that TestAgent can generate tests."""
        task = Task(
            name="generate_tests_feature",
            required_capabilities={AgentCapability.TESTING},
            data={"feature_spec": sample_feature_spec},
        )

        result = await test_agent.execute_task(task)

        assert result.completed is True
        assert "test_code" in result.result
        assert result.result["test_count"] >= 3

    async def test_test_execution(self, test_agent):
        """Test that TestAgent can run tests."""
        task = Task(
            name="run_tests",
            required_capabilities={AgentCapability.TESTING},
            data={"test_count": 10},
        )

        result = await test_agent.execute_task(task)

        assert result.completed is True
        assert result.result["tests_run"] == 10
        assert result.result["tests_passed"] >= 0


@pytest.mark.unit
class TestSchemaAgent:
    """Test SchemaAgent functionality."""

    async def test_schema_analysis(self, schema_agent, sample_schema_changes):
        """Test that SchemaAgent can analyze schema."""
        task = Task(
            name="analyze_schema",
            required_capabilities={AgentCapability.SCHEMA_MANAGEMENT},
            data={"changes": sample_schema_changes},
        )

        result = await schema_agent.execute_task(task)

        assert result.completed is True
        assert "impact_analysis" in result.result

    async def test_table_extension_generation(self, schema_agent, sample_schema_changes):
        """Test that SchemaAgent can generate table extensions."""
        task = Task(
            name="generate_table_extensions",
            required_capabilities={AgentCapability.SCHEMA_MANAGEMENT},
            data={"changes": sample_schema_changes},
        )

        result = await schema_agent.execute_task(task)

        assert result.completed is True
        assert "extensions_created" in result.result
        assert result.result["extensions_created"] == 2


@pytest.mark.unit
class TestAPIAgent:
    """Test APIAgent functionality."""

    async def test_api_design(self, api_agent, sample_api_spec):
        """Test that APIAgent can design APIs."""
        task = Task(
            name="design_api_integration",
            required_capabilities={AgentCapability.API_INTEGRATION},
            data={"api_spec": sample_api_spec},
        )

        result = await api_agent.execute_task(task)

        assert result.completed is True
        assert "design" in result.result
        assert "api_pages_needed" in result.result["design"]

    async def test_api_code_generation(self, api_agent, sample_api_spec):
        """Test that APIAgent can generate API code."""
        task = Task(
            name="generate_api_code",
            required_capabilities={AgentCapability.API_INTEGRATION},
            data={"api_spec": sample_api_spec},
        )

        result = await api_agent.execute_task(task)

        assert result.completed is True
        assert "api_page" in result.result


@pytest.mark.unit
class TestDeploymentAgent:
    """Test DeploymentAgent functionality."""

    async def test_app_build(self, deployment_agent, sample_deployment_config):
        """Test that DeploymentAgent can build apps."""
        task = Task(
            name="build_application",
            required_capabilities={AgentCapability.DEPLOYMENT},
            data={"config": sample_deployment_config},
        )

        result = await deployment_agent.execute_task(task)

        assert result.completed is True
        assert result.result["errors"] == 0
        assert "app_file" in result.result

    async def test_app_deployment(self, deployment_agent, sample_deployment_config):
        """Test that DeploymentAgent can deploy apps."""
        task = Task(
            name="deploy_app",
            required_capabilities={AgentCapability.DEPLOYMENT},
            data={"config": sample_deployment_config},
        )

        result = await deployment_agent.execute_task(task)

        assert result.completed is True
        assert result.result["app_published"] is True


@pytest.mark.unit
class TestDocumentationAgent:
    """Test DocumentationAgent functionality."""

    async def test_api_documentation(self, documentation_agent, sample_api_spec):
        """Test that DocumentationAgent can generate API docs."""
        task = Task(
            name="generate_api_docs",
            required_capabilities={AgentCapability.DOCUMENTATION},
            data={"api_spec": sample_api_spec},
        )

        result = await documentation_agent.execute_task(task)

        assert result.completed is True
        assert "documentation" in result.result
        assert result.result["format"] == "markdown"

    async def test_deployment_documentation(
        self, documentation_agent, sample_deployment_config
    ):
        """Test that DocumentationAgent can generate deployment docs."""
        task = Task(
            name="generate_deployment_docs",
            required_capabilities={AgentCapability.DOCUMENTATION},
            data={"config": sample_deployment_config},
        )

        result = await documentation_agent.execute_task(task)

        assert result.completed is True
        assert len(result.result["sections"]) > 0
