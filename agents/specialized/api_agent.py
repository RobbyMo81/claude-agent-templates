"""
API Agent - Handles API integrations for Business Central.
"""

from typing import Any
from agents.base.agent import Agent, AgentCapability, Task


class APIAgent(Agent):
    """
    Specialized agent for API integration.

    Capabilities:
    - Design API integrations
    - Generate API pages and queries
    - Handle web service integration
    """

    def __init__(self, name: str = "APIAgent"):
        super().__init__(
            name=name,
            capabilities={
                AgentCapability.API_INTEGRATION,
                AgentCapability.CODE_GENERATION,
                AgentCapability.CODE_ANALYSIS,
            },
        )

    async def _execute_task_logic(self, task: Task) -> Any:
        """Execute API integration tasks."""
        task_name = task.name.lower()

        if "design" in task_name:
            return await self._design_api(task)
        elif "generate" in task_name or "create" in task_name:
            return await self._generate_api_code(task)
        else:
            return await self._design_api(task)

    async def _design_api(self, task: Task) -> Dict[str, Any]:
        """Design API integration architecture."""
        api_spec = task.data.get("api_spec", {})

        return {
            "status": "success",
            "api_type": api_spec.get("type", "REST"),
            "endpoints": api_spec.get("endpoints", []),
            "authentication": api_spec.get("auth", "OAuth2"),
            "design": {
                "api_pages_needed": 2,
                "queries_needed": 1,
                "codeunits_needed": 1,
            },
        }

    async def _generate_api_code(self, task: Task) -> Dict[str, Any]:
        """Generate API integration code."""
        api_spec = task.data.get("api_spec", {})

        api_page = self._create_api_page(api_spec)

        return {
            "status": "success",
            "api_page": api_page,
            "endpoints_created": len(api_spec.get("endpoints", [])),
        }

    def _create_api_page(self, spec: Dict[str, Any]) -> str:
        """Create API page template."""
        entity_name = spec.get("entity", "Customer")

        return f'''page 50300 "{entity_name}API"
{{
    PageType = API;
    APIPublisher = 'company';
    APIGroup = 'app';
    APIVersion = 'v1.0';
    EntityName = '{entity_name.lower()}';
    EntitySetName = '{entity_name.lower()}s';
    SourceTable = "{entity_name}";
    DelayedInsert = true;
    ODataKeyFields = SystemId;

    layout
    {{
        area(Content)
        {{
            repeater(Group)
            {{
                field(id; Rec.SystemId)
                {{
                    Editable = false;
                }}
                field(number; Rec."No.")
                {{
                }}
                field(name; Rec.Name)
                {{
                }}
            }}
        }}
    }}
}}
'''
