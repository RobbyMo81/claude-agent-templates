"""
Team Assistant Agent - Facilitates vision creation and requirements gathering.

This agent assists humans in refining their vision through clarifying questions
and structured dialogue, rather than making autonomous decisions.
"""

from typing import Any, Dict, List
from agents.base.agent import Agent, AgentCapability, Task
from agents.base.vision import (
    ProjectVision,
    ClarifyingQuestion,
    QuestionCategory,
    Requirements,
    LifecyclePhase,
)
from agents.base.decision import HumanInterface, ConfidenceScore


class TeamAssistantAgent(Agent):
    """
    Assists humans in making decisions through facilitation and clarification.

    Unlike autonomous agents, this agent:
    - Asks clarifying questions rather than making assumptions
    - Proposes options rather than deciding
    - Facilitates human decision-making
    """

    def __init__(self, name: str = "TeamAssistant"):
        super().__init__(
            name=name,
            capabilities={
                AgentCapability.VISION_FACILITATION,
                AgentCapability.REQUIREMENTS_ANALYSIS,
                AgentCapability.DECISION_SUPPORT,
                AgentCapability.HUMAN_COLLABORATION,
            },
        )
        self.question_templates = self._initialize_question_templates()

    def _initialize_question_templates(self) -> Dict[str, List[str]]:
        """Initialize AL/BC specific question templates."""
        return {
            "table_extension": [
                "What Business Central version are you targeting?",
                "Should these fields be available in all companies or specific ones?",
                "Do you need these fields to be visible on any specific pages?",
                "Are there any validation rules for these fields?",
                "Should these fields be included in any existing field groups?",
            ],
            "field_properties": [
                "What should be the maximum length for text fields?",
                "Should any fields be required (NotBlank)?",
                "Do any fields need to reference other tables (TableRelation)?",
                "Should any fields have specific default values?",
            ],
            "integration": [
                "Should these fields be exposed via API?",
                "Do you need these fields synced with external systems?",
                "Are there any web service requirements?",
            ],
            "data_classification": [
                "What data classification applies? (CustomerContent, EndUserIdentifiableInformation, etc.)",
                "Are there GDPR considerations for these fields?",
            ],
        }

    async def _execute_task_logic(self, task: Task) -> Any:
        """Execute vision facilitation or requirements analysis."""
        task_name = task.name.lower()

        if "vision" in task_name or "refine" in task_name:
            return await self._facilitate_vision(task)
        elif "requirements" in task_name or "gather" in task_name:
            return await self._gather_requirements(task)
        elif "question" in task_name:
            return await self._generate_questions(task)
        else:
            return await self._facilitate_vision(task)

    async def _facilitate_vision(self, task: Task) -> Dict[str, Any]:
        """Facilitate vision creation through structured questions."""
        creator_intent = task.data.get("creator_intent", "")
        human = task.data.get("human_interface")

        if not human:
            # Return questions without human interaction (for testing)
            questions = self._generate_clarifying_questions(creator_intent)
            return {
                "status": "questions_generated",
                "questions": [q.__dict__ for q in questions],
                "needs_human_input": True,
            }

        # Interactive mode with human
        questions = self._generate_clarifying_questions(creator_intent)
        answers = {}

        for question in questions:
            answer = await human.ask_question(
                f"{question.question}\n"
                f"(Category: {question.category.value})\n"
                f"Examples: {', '.join(question.examples) if question.examples else 'N/A'}"
            )
            question.answer = answer
            answers[question.question] = answer

        # Synthesize refined vision
        refined_vision = self._synthesize_vision(creator_intent, questions)
        confidence = self._assess_vision_confidence(questions)

        return {
            "status": "vision_refined",
            "refined_vision": refined_vision,
            "questions": questions,
            "answers": answers,
            "confidence": confidence.overall,
        }

    def _generate_clarifying_questions(
        self, intent: str
    ) -> List[ClarifyingQuestion]:
        """Generate smart clarifying questions based on intent."""
        questions = []
        intent_lower = intent.lower()

        # Detect what the user wants to do
        if "table" in intent_lower or "field" in intent_lower:
            # Table extension scenario
            questions.extend(self._questions_for_table_extension(intent))

        if "email" in intent_lower or "phone" in intent_lower:
            # Contact fields - add validation questions
            questions.append(
                ClarifyingQuestion(
                    question="Should email addresses be validated for format?",
                    category=QuestionCategory.TECHNICAL,
                    rationale="Email validation prevents data quality issues",
                    examples=["Yes, validate format", "No, allow any text"],
                )
            )

        # Always ask about data classification (BC requirement)
        questions.append(
            ClarifyingQuestion(
                question="What data classification applies to these fields?",
                category=QuestionCategory.SECURITY,
                rationale="Business Central requires data classification for GDPR compliance",
                examples=[
                    "CustomerContent",
                    "EndUserIdentifiableInformation",
                    "AccountData",
                ],
            )
        )

        return questions

    def _questions_for_table_extension(self, intent: str) -> List[ClarifyingQuestion]:
        """Generate questions specific to table extensions."""
        questions = []

        # Identify table name
        questions.append(
            ClarifyingQuestion(
                question="Which table are you extending?",
                category=QuestionCategory.CLARIFICATION,
                rationale="Need to know the base table for the extension",
                examples=["Customer", "Vendor", "Item", "Sales Header"],
            )
        )

        # Field specifications
        questions.append(
            ClarifyingQuestion(
                question="What field names and types do you need?",
                category=QuestionCategory.TECHNICAL,
                rationale="Need exact field specifications for code generation",
                examples=[
                    "Email (Text[80])",
                    "Phone (Text[30])",
                    "LoyaltyPoints (Integer)",
                ],
            )
        )

        # Page visibility
        questions.append(
            ClarifyingQuestion(
                question="Should these fields be visible on any pages?",
                category=QuestionCategory.UX,
                rationale="Determines if page extensions are needed",
                examples=[
                    "Customer Card page",
                    "Customer List page",
                    "Both",
                    "No page changes needed",
                ],
            )
        )

        return questions

    def _synthesize_vision(
        self, creator_intent: str, questions: List[ClarifyingQuestion]
    ) -> str:
        """Synthesize a refined vision from intent and answers."""
        vision_parts = [f"## Refined Vision\n", f"**Original Intent:** {creator_intent}\n"]

        # Group answers by category
        by_category: Dict[QuestionCategory, List[ClarifyingQuestion]] = {}
        for q in questions:
            if q.answer:
                if q.category not in by_category:
                    by_category[q.category] = []
                by_category[q.category].append(q)

        # Format by category
        for category, qs in by_category.items():
            vision_parts.append(f"\n**{category.value.title()}:**")
            for q in qs:
                vision_parts.append(f"- {q.question}: {q.answer}")

        return "\n".join(vision_parts)

    def _assess_vision_confidence(
        self, questions: List[ClarifyingQuestion]
    ) -> ConfidenceScore:
        """Assess confidence in the refined vision."""
        answered = sum(1 for q in questions if q.answer)
        total = len(questions)

        if total == 0:
            return ConfidenceScore(overall=0.0, reasoning="No questions asked")

        answer_rate = answered / total

        # High confidence if all questions answered with specific responses
        if answer_rate == 1.0:
            # Check for vague answers
            vague_answers = sum(
                1
                for q in questions
                if q.answer
                and any(
                    word in q.answer.lower()
                    for word in ["maybe", "don't know", "not sure", "unclear"]
                )
            )

            if vague_answers > 0:
                confidence = 0.7 - (vague_answers * 0.1)
            else:
                confidence = 0.95
        else:
            confidence = answer_rate * 0.8

        return ConfidenceScore(
            overall=confidence,
            factors={
                "answer_completeness": answer_rate,
                "answer_specificity": 1.0 if answer_rate == 1.0 else 0.7,
            },
            reasoning=f"{answered}/{total} questions answered with sufficient detail",
        )

    async def _gather_requirements(self, task: Task) -> Dict[str, Any]:
        """Extract structured requirements from vision."""
        vision = task.data.get("vision")
        if not vision:
            return {"status": "error", "error": "No vision provided"}

        # Extract requirements from vision
        requirements = Requirements()

        # Parse vision for functional requirements
        vision_text = vision.refined_vision or vision.creator_intent
        vision_lower = vision_text.lower()

        if "table" in vision_lower or "field" in vision_lower:
            requirements.functional.append("Extend Business Central table with new fields")
            requirements.functional.append(
                "Maintain data integrity with base table structure"
            )

        if "page" in vision_lower:
            requirements.functional.append("Add fields to page layout")

        # Non-functional requirements (AL specific)
        requirements.non_functional.append("Code must compile in AL Language")
        requirements.non_functional.append("Follow BC naming conventions")
        requirements.non_functional.append("Include proper data classification")
        requirements.non_functional.append("Support upgrade compatibility")

        # Constraints
        requirements.constraints.append("Must use AL Language")
        requirements.constraints.append("Must be compatible with target BC version")
        requirements.constraints.append("Field IDs must be in customization range (50000+)")

        # User stories (if we can infer them)
        if "email" in vision_lower or "phone" in vision_lower:
            requirements.user_stories.append(
                "As a user, I want to store contact information so I can communicate with customers"
            )

        # Acceptance criteria
        requirements.acceptance_criteria.append("Table extension compiles without errors")
        requirements.acceptance_criteria.append("Fields are accessible from base table")
        requirements.acceptance_criteria.append("Data classification is properly set")

        return {
            "status": "success",
            "requirements": requirements,
            "confidence": 0.85,
        }

    async def _generate_questions(self, task: Task) -> Dict[str, Any]:
        """Generate questions for a specific context."""
        context = task.data.get("context")

        # If no explicit context, infer from creator_intent
        if not context:
            creator_intent = task.data.get("creator_intent", "").lower()
            if "table" in creator_intent or "field" in creator_intent:
                context = "table_extension"
            elif "api" in creator_intent:
                context = "integration"
            else:
                context = "general"

        questions = []

        if context in self.question_templates:
            template_questions = self.question_templates[context]
            questions = [
                ClarifyingQuestion(
                    question=q,
                    category=QuestionCategory.TECHNICAL,
                    rationale=f"Important for {context} implementation",
                )
                for q in template_questions
            ]

        return {
            "status": "questions_generated",
            "questions": [q.__dict__ for q in questions],
            "count": len(questions),
        }
