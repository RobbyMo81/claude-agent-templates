# 1. Navigate to project directory
  cd "C:\Users\RobMo\OneDrive\Documents\claude-agent-templates-new"

  # 2. Run the demo with the test project intent
  python examples/table_extension_demo.py

  When prompted: Enter this intent:
  Add loyalty points tracking to the Customer table with points balance, tier level, and enrollment date

  What to Observe

  The workflow will proceed through 6 phases with human gates:

  Phase 1: SCOPING

  - Agent: TeamAssistantAgent
  - Expected: Asks clarifying questions:
    - "What Business Central version?"
    - "Data classification for loyalty fields?"
    - "Should points field allow negatives?"
    - "What are the tier thresholds?"
    - "Which pages should display these fields?"

  ✅ SUCCESS CRITERIA:
  - Questions are relevant and AL-specific
  - You can answer or use defaults
  - Vision is refined with your answers

  Phase 2: REQUIREMENTS

  - Agent: TeamAssistantAgent
  - Expected: Extracts structured requirements:
    - Functional requirements (field specifications)
    - Non-functional requirements (performance, security)
    - Constraints (BC version compatibility)

  ✅ SUCCESS CRITERIA:
  - Requirements accurately reflect your answers
  - Human gate shows clear requirements list
  - You approve to continue

  Phase 3: ARCHITECTURE

  - Agent: SchemaAgent
  - Expected: Designs table extension structure:
    - Extension object definition
    - Field IDs (e.g., 50100-50102)
    - Primary key considerations
    - Page extension recommendations

  ✅ SUCCESS CRITERIA:
  - Design follows AL/BC best practices
  - Field IDs don't conflict with standard ranges
  - Architecture is clear and reviewable

  Phase 4: DESIGN

  - Agent: SchemaAgent + CodeAgent
  - Expected: Detailed specifications:
    - Exact field definitions with types
    - Data validation rules
    - Caption and description text
    - Page field placement

  ✅ SUCCESS CRITERIA:
  - Specifications are complete and precise
  - AL syntax is correct
  - Design matches your intent

  Phase 5: CONSTRUCTION

  - Agent: CodeAgent
  - Expected: Generates actual AL code:
  tableextension 50100 "Customer Loyalty Ext" extends Customer
  {
      fields
      {
          field(50100; "Loyalty Points"; Integer) { ... }
          field(50101; "Loyalty Tier"; Enum) { ... }
          field(50102; "Loyalty Enrollment Date"; Date) { ... }
      }
  }

  ✅ SUCCESS CRITERIA:
  - Code compiles (syntax-wise)
  - Follows BC naming conventions
  - Includes proper metadata

  Phase 6: TESTING

  - Agent: TestAgent
  - Expected: Generates test codeunit:
  codeunit 50100 "Customer Loyalty Tests"
  {
      [Test]
      procedure TestLoyaltyPointsField()
      begin
          // Test implementation
      end;
  }

  ✅ SUCCESS CRITERIA:
  - Test coverage for all fields
  - Proper test structure
  - Uses BC test framework

  Success Metrics for Phase 2 Readiness

  The system is ready for Phase 2 if:

  | Category            | Metric                                        | Pass Threshold          |
  |---------------------|-----------------------------------------------|-------------------------|
  | Agent Coordination  | All agents execute their tasks                | 100%                    |
  | Vision Context Flow | Each phase uses previous phase outputs        | Validated manually      |
  | Human Gates         | Gates present clear, reviewable artifacts     | 6 of 6 gates            |
  | Code Quality        | Generated AL code is syntactically valid      | 100%                    |
  | Error Handling      | System recovers gracefully from bad inputs    | Tested with 1 rejection |
  | Performance         | Full workflow completes in <5 minutes         | ✅                       |
  | Decision Learning   | System logs all decisions for future learning | Logs present            |

  Extended Validation Tests

  If the basic test succeeds, try these variations:

  Test 2: Rejection Workflow
  - Reject a gate (e.g., at architecture review)
  - Observe: Does system handle rework?
  - Expected: Workflow allows iteration

  Test 3: Concurrent Workflows
  # Run two extensions simultaneously
  python -c "
  import asyncio
  from agents.base.coordinator import Coordinator
  from agents.workflows.table_extension import TableExtensionWorkflow
  from agents.specialized import *

  async def test():
      coordinator = Coordinator()
      coordinator.register_agent(TeamAssistantAgent())
      coordinator.register_agent(CodeAgent())
      coordinator.register_agent(SchemaAgent())
      coordinator.register_agent(TestAgent())

      workflows = [
          TableExtensionWorkflow('Add Email field to Customer'),
          TableExtensionWorkflow('Add Phone field to Vendor')
      ]

      results = await coordinator.execute_concurrent_workflows(workflows)
      print(f'Completed {len([r for r in results if r.status == \"completed\"])} workflows')

  asyncio.run(test())
  "

  Expected: Both workflows complete without interference

  Test 4: Complex Extension
  - Intent: "Add a complete loyalty rewards system with points, tiers, transactions table, and redemption history"
  - Expected: TeamAssistant should ask more questions, potentially break into sub-projects

  What Would Disqualify Phase 2 Readiness?

  🚫 Critical Failures:
  - Agents fail to coordinate (tasks not assigned)
  - Generated code has syntax errors
  - Vision context lost between phases
  - Human gates don't present reviewable artifacts
  - System crashes on normal inputs

  ⚠️ Major Concerns:
  - Generated code is nonsensical (even if syntactically valid)
  - Questions from TeamAssistant are generic, not AL-specific
  - Workflow takes >10 minutes for simple extension
  - No decision logging occurs
  - Cannot handle rejection/rework

  ✅ Acceptable for Phase 2:
  - Generated code needs minor manual fixes
  - Some questions could be better
  - Performance could be faster
  - Decision logging is basic
  - Human must guide heavily (this IS Phase 1)

  Deliverables to Review

  After completing the test project, you should have:

  1. Generated AL Code Files:
    - Customer_Loyalty_Ext.al (table extension)
    - Customer_Loyalty_Tests.al (test codeunit)
  2. Documentation:
    - Requirements document
    - Architecture design
    - Implementation notes
  3. Decision Log:
    - All human decisions logged
    - Agent confidence scores recorded
    - Pattern data for Phase 2 learning
  4. Vision Artifact:
    - Complete ProjectVision object
    - Shows evolution from initial intent to final design