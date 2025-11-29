# Claude Agent Templates

**Structured agentic workflows for Microsoft Dynamics 365 Business Central / AL development**

[![License: MPL 2.0](https://img.shields.io/badge/License-MPL%202.0-brightgreen.svg)](https://opensource.org/licenses/MPL-2.0)
[![CI/CD](https://github.com/RobbyMo81/claude-agent-templates/workflows/CI%2FCD%20Pipeline/badge.svg)](https://github.com/RobbyMo81/claude-agent-templates/actions)

---

## Overview

Agent Army is a **multi-agent system** that automates AL (Application Language) development workflows for Business Central. It uses AI-powered agents to analyze code, generate new features, validate security and compliance, run tests, and deploy to BC environments.

### Key Features

- 🤖 **AI-Powered Code Generation**: Generate AL tables, pages, codeunits from specifications
- 🔍 **Intelligent Code Analysis**: Parse and understand AL codebases
- ✅ **Automated Testing**: Generate and execute AL tests with coverage tracking
- 🔒 **Security & Compliance**: Validate against OWASP top 10, AL coding standards
- 🚀 **Automated Deployment**: Deploy to BC dev/staging/production with approvals
- 💰 **Cost Optimized**: Aggressive caching reduces LLM costs by 97%
- 🌐 **Provider Agnostic**: Support for Claude, OpenAI, Azure OpenAI, local Ollama

---

## Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.12+ (for local development)
- Business Central environment (for deployment features)
- Anthropic API key (or OpenAI/Azure OpenAI)

### 1. Local Development (Docker Compose)
A Python-based multi-agent system for AL (Application Language) development with Microsoft Dynamics 365 Business Central. Agents self-organize to execute complex workflows simultaneously.

## Features

- **Self-Organizing Agents**: Agents coordinate autonomously based on capabilities
- **Concurrent Workflow Execution**: Multiple workflows run simultaneously with intelligent task distribution
- **AL/Business Central Focus**: Specialized agents for AL development tasks
- **Comprehensive Testing**: Full test suite including concurrent workflow tests
- **Flexible Architecture**: Easy to extend with new agents and workflows

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/RobbyMo81/claude-agent-templates.git
cd claude-agent-templates

# Start all services
make dev

# Or manually
docker-compose up
```

Access the services:
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Temporal UI**: http://localhost:8080
- **Database UI**: http://localhost:8081 (user: `agent_army`, password: `agent_army_dev`)
- **Redis UI**: http://localhost:8082

### 2. Create Your First Mission

```bash
curl -X POST http://localhost:8000/api/v1/missions \
  -H "Content-Type: application/json" \
  -d '{
    "specification": {
      "action": "generate_table",
      "target": {
        "name": "Customer",
        "fields": [
          {"name": "No.", "type": "Code[20]", "primary_key": true},
          {"name": "Name", "type": "Text[100]"},
          {"name": "CreditLimit", "type": "Decimal"}
        ]
      }
    },
    "environment": "dev"
  }'
```

Response:
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "running",
  "progress": 0.2,
  "current_step": "Analyzing existing code",
  "created_at": "2025-11-19T10:30:00Z"
}
```

### 3. Check Mission Status

```bash
curl http://localhost:8000/api/v1/missions/550e8400-e29b-41d4-a716-446655440000
```

---

## Architecture

Agent Army v2.0 uses a **modular monolith architecture** with clear separation of concerns:

```
┌─────────────────────────────────────────┐
│         API Gateway (FastAPI)           │
│     - REST API, GraphQL, WebSocket      │
└─────────────────┬───────────────────────┘
                  │
        ┌─────────┴─────────┐
        │  Temporal Workflows │
        │  (Orchestration)    │
        └─────────┬───────────┘
                  │
    ┌─────────────┼─────────────┐
    │             │             │
┌───▼────┐  ┌────▼────┐  ┌────▼────┐
│Analyzer│  │Generator│  │Validator│
│ Module │  │ Module  │  │ Module  │
└────────┘  └─────────┘  └─────────┘
    │             │             │
    └─────────────┼─────────────┘
                  │
        ┌─────────┴─────────┐
        │  PostgreSQL + Redis │
        └─────────────────────┘
```

**Core Modules**:
- **Analyzer**: Parse AL code, extract structures, dependency analysis
- **Generator**: AI-powered code generation using Claude/OpenAI
- **Validator**: Security scanning, compliance checks, performance analysis
- **Tester**: Generate and execute AL tests
- **Deployer**: Package and deploy to BC environments

**See**: [Enhanced Design Document](docs/architecture/02-enhanced-design.md) for full details.

---

## Documentation

### Architecture & Design
- 📋 [Pre-Design Specification v1.0](docs/architecture/00-pre-design-specification.md) - Initial design
- 🔴 [Red Team Review](docs/architecture/01-red-team-panel.md) - 58 critical findings
- ✅ [Enhanced Design v2.0](docs/architecture/02-enhanced-design.md) - Final production-ready design
- 📊 [Implementation Summary](docs/IMPLEMENTATION_SUMMARY.md) - Complete overview

### API & Development
- 🔌 [OpenAPI Specification](api/openapi.yaml) - REST API documentation
- 🗄️ [Database Schema](infrastructure/database/schema.sql) - PostgreSQL schema
- 🐳 [Docker Compose](docker-compose.yml) - Local development setup
- 🔧 [Makefile](Makefile) - Developer commands

---

## Development

### Available Commands

```bash
make dev            # Start local development environment
make test           # Run test suite
make test-cov       # Run tests with coverage report
make lint           # Run linters (black, isort, flake8, mypy)
make format         # Auto-format code
make db-shell       # Open PostgreSQL shell
make redis-shell    # Open Redis CLI
make logs           # Tail application logs
make clean          # Clean up all services and volumes
```

### Project Structure

```
claude-agent-templates/
├── src/                      # Application source code
│   ├── api/                  # FastAPI routes
│   ├── core/                 # Core business logic
│   │   ├── agents/           # Agent modules
│   │   ├── orchestrator/     # Temporal workflows
│   │   └── models/           # Data models
│   └── infrastructure/       # Database, cache, observability
├── tests/                    # Test suite
├── docs/                     # Documentation
├── api/                      # API specifications
├── infrastructure/           # IaC and database schemas
├── docker-compose.yml        # Local development
├── Dockerfile                # Production image
└── Makefile                  # Developer commands
```

---

## Deployment

### Staging

```bash
make deploy-staging
```

### Production

```bash
make deploy-prod  # Requires confirmation
```

**Deployment Strategy**: Canary deployment with 10% traffic → monitor for 5 minutes → promote to 100%

**See**: [CI/CD Pipeline](.github/workflows/ci-cd.yml) for full deployment automation.

---

## Configuration

### Environment Variables

```bash
# Application
APP_ENV=development|staging|production
DEBUG=true|false
LOG_LEVEL=DEBUG|INFO|WARNING|ERROR

# Database
DATABASE_URL=postgresql://user:password@host:5432/dbname

# Redis
REDIS_URL=redis://host:6379/0

# Temporal
TEMPORAL_HOST=temporal:7233
TEMPORAL_NAMESPACE=default

# LLM Provider
LLM_PROVIDER=claude|openai|azure|ollama
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

# Authentication
AUTH_ENABLED=true|false
AUTH0_DOMAIN=your-domain.auth0.com
AUTH0_CLIENT_ID=...

# Business Central
BC_ADMIN_API_URL=https://api.businesscentral.dynamics.com/...
BC_TENANT_ID=...
```

---

## Cost Optimization

Agent Army v2.0 is designed for cost efficiency:

- **LLM Costs**: $6,216/year (vs $189,000 initial estimate) = **97% reduction**
  - Aggressive caching (40% hit rate)
  - Intelligent model selection (Haiku for simple tasks, Sonnet for complex)
  - Budget controls per user

- **Infrastructure**: $512/month (vs $4,469 initial estimate) = **89% reduction**
  - Serverless compute (Cloud Run auto-scaling)
  - Managed services (no ops overhead)
  - Right-sized resources

**See**: [Enhanced Design - Cost Optimization](docs/architecture/02-enhanced-design.md#cost-optimization)

---

## Security

- 🔐 OAuth2/SAML authentication via Auth0
- 🛡️ Input validation with JSON Schema
- 🔒 PII redaction before LLM processing
- 🗝️ Secret management with Vault/Secret Manager
- 📝 Immutable audit logs (7-year retention)
- 🚨 Security scanning with Trivy + Semgrep
- ✅ GDPR-compliant data retention policies

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](docs/guides/CONTRIBUTING.md) for guidelines.

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Run linters and tests (`make lint && make test`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

---

## License

This project is licensed under the **Mozilla Public License 2.0** - see the [LICENSE](LICENSE) file for details.

---

## Support

- 📖 **Documentation**: [docs/](docs/)
- 🐛 **Issues**: [GitHub Issues](https://github.com/RobbyMo81/claude-agent-templates/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/RobbyMo81/claude-agent-templates/discussions)

---

## Acknowledgments

- **Microsoft Dynamics 365 Business Central** team for AL language
- **Anthropic** for Claude API
- **Temporal.io** for workflow orchestration
- **FastAPI** community

---

**Built with ❤️ by the Agent Army team**
# Install dependencies
pip install -e .

# Install development dependencies (optional)
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit              # Unit tests
pytest -m integration       # Integration tests
pytest -m workflow          # Workflow tests
pytest -m concurrent        # Concurrent workflow tests

# Run with coverage
pytest --cov=agents --cov-report=html
```

## Architecture

### Multi-Agent System

The system implements a self-organizing multi-agent architecture:

```
┌─────────────────────────────────────────────────┐
│              Coordinator                         │
│  - Facilitates agent discovery                  │
│  - Manages workflow execution                   │
│  - Tracks system status                         │
└─────────────────────────────────────────────────┘
                      │
        ┌─────────────┴─────────────┐
        ▼                           ▼
┌──────────────┐           ┌──────────────┐
│   Workflows  │           │    Agents    │
│              │           │              │
│ - Feature    │◄──────────┤ - Code       │
│ - Schema     │           │ - Test       │
│ - API        │           │ - Schema     │
│ - Deployment │           │ - API        │
└──────────────┘           │ - Deployment │
                           │ - Docs       │
                           └──────────────┘
```

### Specialized Agents

1. **CodeAgent**: Generates and analyzes AL code
2. **TestAgent**: Creates and runs AL tests
3. **SchemaAgent**: Manages database schema and table extensions
4. **APIAgent**: Handles API integration
5. **DeploymentAgent**: Compiles and deploys apps
6. **DocumentationAgent**: Generates documentation

### Pre-defined Workflows

- **FeatureWorkflow**: Complete feature development (code → tests + docs)
- **SchemaWorkflow**: Schema migration (analyze → generate → test → deploy)
- **APIWorkflow**: API integration (design → code → tests + docs)
- **DeploymentWorkflow**: Full deployment cycle

## Usage Example

```python
import asyncio
from agents.base.coordinator import Coordinator
from agents.specialized import (
    CodeAgent, TestAgent, DocumentationAgent
)
from agents.base.workflow import FeatureWorkflow

async def main():
    # Create coordinator
    coordinator = Coordinator()

    # Register agents
    coordinator.register_agent(CodeAgent())
    coordinator.register_agent(TestAgent())
    coordinator.register_agent(DocumentationAgent())

    # Define feature specification
    feature_spec = {
        "name": "CustomerPortal",
        "object_type": "table",
        "fields": ["PortalID", "CustomerNo", "AccessLevel"]
    }

    # Execute workflow
    workflow = FeatureWorkflow("CustomerPortal", feature_spec)
    result = await coordinator.execute_workflow(workflow)

    print(f"Workflow Status: {result.status}")
    print(f"Tasks Completed: {len(result.tasks_completed)}")
    print(f"Execution Time: {result.execution_time:.2f}s")

asyncio.run(main())
```

### Concurrent Workflows

```python
# Execute multiple workflows simultaneously
workflows = [
    FeatureWorkflow("Authentication", auth_spec),
    FeatureWorkflow("Reporting", report_spec),
    APIWorkflow(api_spec),
]

results = await coordinator.execute_concurrent_workflows(workflows)

for result in results:
    print(f"{result.workflow_id}: {result.status}")
```

## Testing

### Test Coverage

The test suite includes:

- **Unit Tests**: Individual agent functionality
- **Coordination Tests**: Agent communication and self-organization
- **Workflow Tests**: Task dependency resolution and execution
- **Concurrent Tests**: Simultaneous workflow execution and load distribution

### Running Specific Tests

```bash
# Test individual agents
pytest tests/test_agents.py -v

# Test coordination
pytest tests/test_coordination.py -v

# Test workflows
pytest tests/test_workflows.py -v

# Test concurrent execution (most comprehensive)
pytest tests/test_concurrent_workflows.py -v
```

### Test Scenarios

The concurrent workflow tests cover:

- Multiple workflows running simultaneously
- Agent load distribution
- Workflow isolation
- High-load scenarios (10+ concurrent workflows)
- Real-world development scenarios
- Error handling and recovery

## Development

### Code Quality

```bash
# Format code
black agents/ tests/

# Lint
ruff check agents/ tests/

# Type check
mypy agents/

# All checks
black agents/ tests/ && ruff check agents/ tests/ && mypy agents/
```

### Adding New Agents

1. Create a new agent class inheriting from `Agent`
2. Define capabilities in `__init__`
3. Implement `_execute_task_logic` method
4. Register in `agents/specialized/__init__.py`
5. Add tests in `tests/test_agents.py`

### Adding New Workflows

1. Create workflow class inheriting from `Workflow`
2. Define tasks with dependencies in `__init__`
3. Add to `agents/base/workflow.py`
4. Add tests in `tests/test_workflows.py`

## Documentation

See [CLAUDE.md](CLAUDE.md) for detailed architecture and development guidance.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please ensure:

1. All tests pass: `pytest`
2. Code is formatted: `black agents/ tests/`
3. No linting errors: `ruff check agents/ tests/`
4. Type checking passes: `mypy agents/`
