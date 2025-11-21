"""
Documentation Agent - Generates documentation for AL/Business Central projects.
"""

from typing import Any, Dict
from agents.base.agent import Agent, AgentCapability, Task


class DocumentationAgent(Agent):
    """
    Specialized agent for documentation generation.

    Capabilities:
    - Generate API documentation
    - Create user guides
    - Generate technical documentation
    - Create deployment guides
    """

    def __init__(self, name: str = "DocumentationAgent"):
        super().__init__(
            name=name,
            capabilities={AgentCapability.DOCUMENTATION},
        )

    async def _execute_task_logic(self, task: Task) -> Any:
        """Execute documentation tasks."""
        task_name = task.name.lower()

        if "api" in task_name:
            return await self._generate_api_docs(task)
        elif "deployment" in task_name:
            return await self._generate_deployment_docs(task)
        else:
            return await self._generate_technical_docs(task)

    async def _generate_api_docs(self, task: Task) -> Dict[str, Any]:
        """Generate API documentation."""
        api_spec = task.data.get("api_spec", {})

        docs = self._create_api_documentation(api_spec)

        return {
            "status": "success",
            "documentation": docs,
            "format": "markdown",
            "endpoints_documented": len(api_spec.get("endpoints", [])),
        }

    async def _generate_deployment_docs(self, task: Task) -> Dict[str, Any]:
        """Generate deployment documentation."""
        config = task.data.get("config", {})

        docs = self._create_deployment_documentation(config)

        return {
            "status": "success",
            "documentation": docs,
            "format": "markdown",
            "sections": ["Prerequisites", "Installation", "Configuration", "Troubleshooting"],
        }

    async def _generate_technical_docs(self, task: Task) -> Dict[str, Any]:
        """Generate technical documentation."""
        feature_spec = task.data.get("feature_spec", {})

        docs = self._create_technical_documentation(feature_spec)

        return {
            "status": "success",
            "documentation": docs,
            "format": "markdown",
            "sections": ["Overview", "Architecture", "Usage", "API Reference"],
        }

    def _create_api_documentation(self, spec: Dict[str, Any]) -> str:
        """Create API documentation template."""
        entity = spec.get("entity", "Entity")

        return f'''# {entity} API Documentation

## Overview
This API provides access to {entity} data in Business Central.

## Authentication
- **Type**: {spec.get("auth", "OAuth2")}
- **Scope**: API.ReadWrite.All

## Endpoints

### GET /{entity.lower()}s
Retrieve all {entity.lower()} records.

**Response**: 200 OK
```json
{{
  "value": [
    {{
      "id": "guid",
      "number": "string",
      "name": "string"
    }}
  ]
}}
```

### POST /{entity.lower()}s
Create a new {entity.lower()} record.

**Request Body**:
```json
{{
  "number": "string",
  "name": "string"
}}
```

**Response**: 201 Created

## Rate Limiting
- **Limit**: 1000 requests per hour
- **Headers**: X-RateLimit-Remaining

## Examples
See the examples directory for code samples in various languages.
'''

    def _create_deployment_documentation(self, config: Dict[str, Any]) -> str:
        """Create deployment documentation template."""
        app_name = config.get("app_name", "MyApp")

        return f'''# {app_name} Deployment Guide

## Prerequisites
- Business Central environment (version 20.0 or higher)
- AL Language extension for VS Code
- App dependencies installed

## Installation Steps

### 1. Download the App
Download the latest release from the releases page.

### 2. Install Dependencies
Ensure all required dependencies are installed in your environment.

### 3. Deploy the App
```powershell
Publish-NAVApp -ServerInstance BC200 -Path "./{app_name}_1.0.0.0.app"
Sync-NAVApp -ServerInstance BC200 -Name "{app_name}"
Install-NAVApp -ServerInstance BC200 -Name "{app_name}"
```

### 4. Verify Installation
Check that the app is installed and running:
```powershell
Get-NAVAppInfo -ServerInstance BC200 -Name "{app_name}"
```

## Configuration
After installation, configure the app settings in Business Central:
1. Navigate to Extension Management
2. Find {app_name}
3. Configure settings as needed

## Troubleshooting
- **Issue**: App fails to install
  - **Solution**: Check dependencies and version compatibility
- **Issue**: App not visible in UI
  - **Solution**: Verify user permissions and app configuration
'''

    def _create_technical_documentation(self, spec: Dict[str, Any]) -> str:
        """Create technical documentation template."""
        feature_name = spec.get("name", "Feature")

        return f'''# {feature_name} Technical Documentation

## Overview
{feature_name} provides functionality for...

## Architecture

### Components
- **Tables**: Data storage structures
- **Pages**: User interface components
- **Codeunits**: Business logic implementation
- **APIs**: External integration endpoints

### Data Flow
1. User input via pages
2. Validation in codeunits
3. Data storage in tables
4. External access via APIs

## Usage

### Basic Usage
```al
// Example code usage
local procedure Use{feature_name}()
var
    MyRecord: Record "{feature_name}";
begin
    MyRecord.Init();
    MyRecord.Insert(true);
end;
```

## API Reference
See API documentation for detailed endpoint information.

## Best Practices
- Follow AL coding conventions
- Implement proper error handling
- Use appropriate data types
- Document complex logic
'''
