"""
Code Agent - Generates and analyzes AL code for Business Central.
"""

from typing import Any
from agents.base.agent import Agent, AgentCapability, Task


class CodeAgent(Agent):
    """
    Specialized agent for AL code generation and analysis.

    Capabilities:
    - Generate AL code for Business Central objects (tables, pages, reports, etc.)
    - Analyze existing AL code for quality and best practices
    - Refactor AL code
    """

    def __init__(self, name: str = "CodeAgent"):
        super().__init__(
            name=name,
            capabilities={
                AgentCapability.CODE_GENERATION,
                AgentCapability.CODE_ANALYSIS,
            },
        )

    async def _execute_task_logic(self, task: Task) -> Any:
        """Execute code generation or analysis tasks."""
        task_name = task.name.lower()

        if "generate" in task_name or "create" in task_name:
            return await self._generate_code(task)
        elif "analyze" in task_name or "review" in task_name:
            return await self._analyze_code(task)
        elif "refactor" in task_name:
            return await self._refactor_code(task)
        else:
            return await self._generate_code(task)

    async def _generate_code(self, task: Task) -> Dict[str, Any]:
        """Generate AL code based on specifications."""
        feature_spec = task.data.get("feature_spec", {})
        language = task.data.get("language", "AL")

        # Simulate code generation
        generated_code = {
            "object_type": feature_spec.get("object_type", "table"),
            "object_name": feature_spec.get("name", "NewObject"),
            "code": self._create_al_template(feature_spec),
            "dependencies": feature_spec.get("dependencies", []),
            "language": language,
        }

        return {
            "status": "success",
            "generated_code": generated_code,
            "files_created": 1,
            "lines_of_code": len(generated_code["code"].split("\n")),
        }

    async def _analyze_code(self, task: Task) -> Dict[str, Any]:
        """Analyze AL code for quality and best practices."""
        code = task.data.get("code", "")

        # Simulate code analysis
        analysis = {
            "quality_score": 85,
            "issues": [
                {"severity": "warning", "message": "Consider using explicit naming conventions"},
                {"severity": "info", "message": "Code follows AL best practices"},
            ],
            "metrics": {
                "complexity": "medium",
                "maintainability": "high",
                "test_coverage": 0,
            },
        }

        return {
            "status": "success",
            "analysis": analysis,
        }

    async def _refactor_code(self, task: Task) -> Dict[str, Any]:
        """Refactor existing AL code."""
        code = task.data.get("code", "")

        return {
            "status": "success",
            "refactored": True,
            "improvements": ["Improved naming", "Reduced complexity"],
        }

    def _create_al_template(self, spec: Dict[str, Any]) -> str:
        """Create an AL code template based on specifications."""
        object_type = spec.get("object_type", "table")
        object_name = spec.get("name", "NewObject")
        fields = spec.get("fields", [])

        if object_type == "table":
            return self._create_table_template(object_name, fields)
        elif object_type == "page":
            return self._create_page_template(object_name, spec)
        elif object_type == "codeunit":
            return self._create_codeunit_template(object_name, spec)
        else:
            return f"// AL {object_type}: {object_name}\n// Auto-generated template"

    def _create_table_template(self, name: str, fields: list) -> str:
        """Create AL table template."""
        field_definitions = "\n".join(
            [f'        field({i+1}; "{f}"; Text[100]) {{ }}' for i, f in enumerate(fields)]
        )

        return f'''table 50100 "{name}"
{{
    DataClassification = CustomerContent;

    fields
    {{
{field_definitions if field_definitions else '        // Add fields here'}
    }}

    keys
    {{
        key(PK; "No.")
        {{
            Clustered = true;
        }}
    }}
}}
'''

    def _create_page_template(self, name: str, spec: Dict[str, Any]) -> str:
        """Create AL page template."""
        return f'''page 50100 "{name}"
{{
    PageType = Card;
    SourceTable = "{spec.get('source_table', 'TableName')}";
    ApplicationArea = All;
    UsageCategory = Administration;

    layout
    {{
        area(Content)
        {{
            group(General)
            {{
                // Add fields here
            }}
        }}
    }}
}}
'''

    def _create_codeunit_template(self, name: str, spec: Dict[str, Any]) -> str:
        """Create AL codeunit template."""
        return f'''codeunit 50100 "{name}"
{{
    // Add procedures here

    procedure ProcessData()
    begin
        // Implementation
    end;
}}
'''
