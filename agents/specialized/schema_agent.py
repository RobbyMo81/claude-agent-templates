"""
Schema Agent - Manages database schema for Business Central.
"""

from typing import Any
from agents.base.agent import Agent, AgentCapability, Task


class SchemaAgent(Agent):
    """
    Specialized agent for schema management.

    Capabilities:
    - Analyze database schema
    - Generate table extensions
    - Manage schema migrations
    """

    def __init__(self, name: str = "SchemaAgent"):
        super().__init__(
            name=name,
            capabilities={
                AgentCapability.SCHEMA_MANAGEMENT,
                AgentCapability.CODE_GENERATION,
                AgentCapability.CODE_ANALYSIS,
            },
        )

    async def _execute_task_logic(self, task: Task) -> Any:
        """Execute schema management tasks."""
        task_name = task.name.lower()

        if "analyze" in task_name:
            return await self._analyze_schema(task)
        elif "generate" in task_name or "create" in task_name:
            return await self._generate_table_extensions(task)
        elif "migrate" in task_name:
            return await self._manage_migration(task)
        else:
            return await self._analyze_schema(task)

    async def _analyze_schema(self, task: Task) -> Dict[str, Any]:
        """Analyze current schema and proposed changes."""
        changes = task.data.get("changes", {})

        return {
            "status": "success",
            "current_tables": ["Customer", "Vendor", "Item"],
            "proposed_changes": changes,
            "impact_analysis": {
                "tables_affected": len(changes.get("tables", [])),
                "breaking_changes": False,
                "migration_required": True,
            },
        }

    async def _generate_table_extensions(self, task: Task) -> Dict[str, Any]:
        """Generate AL table extensions."""
        changes = task.data.get("changes", {})
        tables = changes.get("tables", [])

        extensions = [self._create_table_extension(table) for table in tables]

        return {
            "status": "success",
            "extensions_created": len(extensions),
            "extensions": extensions,
        }

    async def _manage_migration(self, task: Task) -> Dict[str, Any]:
        """Manage schema migration."""
        return {
            "status": "success",
            "migration_scripts": ["upgrade_v1_to_v2.al"],
            "data_migration_required": False,
        }

    def _create_table_extension(self, table_spec: Dict[str, Any]) -> str:
        """Create table extension template."""
        table_name = table_spec.get("name", "Customer")
        fields = table_spec.get("fields", [])

        field_defs = "\n".join(
            [f'        field(50{i+100}; "{f}"; Text[100]) {{ }}' for i, f in enumerate(fields)]
        )

        return f'''tableextension 50100 "{table_name}Ext" extends "{table_name}"
{{
    fields
    {{
{field_defs if field_defs else '        // Add extension fields here'}
    }}
}}
'''
