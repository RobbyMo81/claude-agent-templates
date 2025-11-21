"""
Pytest configuration and fixtures for agent testing.
"""

import pytest
from agents.base.coordinator import Coordinator
from agents.specialized.code_agent import CodeAgent
from agents.specialized.test_agent import TestAgent
from agents.specialized.schema_agent import SchemaAgent
from agents.specialized.api_agent import APIAgent
from agents.specialized.deployment_agent import DeploymentAgent
from agents.specialized.documentation_agent import DocumentationAgent


@pytest.fixture
def coordinator():
    """Create a coordinator instance for testing."""
    return Coordinator(name="test_coordinator")


@pytest.fixture
def code_agent():
    """Create a code agent instance for testing."""
    return CodeAgent(name="TestCodeAgent")


@pytest.fixture
def test_agent():
    """Create a test agent instance for testing."""
    return TestAgent(name="TestTestAgent")


@pytest.fixture
def schema_agent():
    """Create a schema agent instance for testing."""
    return SchemaAgent(name="TestSchemaAgent")


@pytest.fixture
def api_agent():
    """Create an API agent instance for testing."""
    return APIAgent(name="TestAPIAgent")


@pytest.fixture
def deployment_agent():
    """Create a deployment agent instance for testing."""
    return DeploymentAgent(name="TestDeploymentAgent")


@pytest.fixture
def documentation_agent():
    """Create a documentation agent instance for testing."""
    return DocumentationAgent(name="TestDocAgent")


@pytest.fixture
def all_agents(
    code_agent,
    test_agent,
    schema_agent,
    api_agent,
    deployment_agent,
    documentation_agent,
):
    """Create all specialized agents."""
    return [
        code_agent,
        test_agent,
        schema_agent,
        api_agent,
        deployment_agent,
        documentation_agent,
    ]


@pytest.fixture
def coordinator_with_agents(coordinator, all_agents):
    """Create a coordinator with all agents registered."""
    for agent in all_agents:
        coordinator.register_agent(agent)
    return coordinator


@pytest.fixture
def sample_feature_spec():
    """Sample feature specification for testing."""
    return {
        "name": "CustomerPortal",
        "object_type": "table",
        "fields": ["PortalID", "CustomerNo", "AccessLevel"],
        "test_count": 5,
    }


@pytest.fixture
def sample_schema_changes():
    """Sample schema changes for testing."""
    return {
        "tables": [
            {"name": "Customer", "fields": ["PreferredContact", "VIPStatus"]},
            {"name": "Vendor", "fields": ["RatingScore"]},
        ]
    }


@pytest.fixture
def sample_api_spec():
    """Sample API specification for testing."""
    return {
        "entity": "Customer",
        "type": "REST",
        "auth": "OAuth2",
        "endpoints": ["GET", "POST", "PATCH", "DELETE"],
    }


@pytest.fixture
def sample_deployment_config():
    """Sample deployment configuration for testing."""
    return {
        "app_name": "TestApp",
        "environment": "sandbox",
        "version": "1.0.0.0",
    }
