#!/usr/bin/env python3
"""
Stress Test Suite for Hybrid Multi-Agent System

This script comprehensively tests the hybrid agent architecture under
various stress conditions including:
- High concurrency
- Complex workflows
- Decision tracking and learning
- Vision context propagation
- Phase gate validation
- Agent coordination
- Error recovery
- Performance benchmarking

Usage:
    python tests/stress_test.py
    python tests/stress_test.py --verbose
    python tests/stress_test.py --quick  # Skip long-running tests
"""

import asyncio
import time
import argparse
from typing import List, Dict, Any
from dataclasses import dataclass

from agents.base.coordinator import Coordinator
from agents.base.vision import ProjectVision, LifecyclePhase
from agents.base.decision import DecisionLog, LearningAnalyzer, CLIHumanInterface
from agents.base.lifecycle import HybridLifecycleWorkflow, PhaseGate, GateDecision
from agents.base.agent import Task, AgentCapability, AgentStatus

from agents.specialized import (
    TeamAssistantAgent,
    CodeAgent,
    SchemaAgent,
    TestAgent,
    APIAgent,
    DeploymentAgent,
    DocumentationAgent,
)

from agents.workflows.table_extension import TableExtensionWorkflow


@dataclass
class StressTestResult:
    """Result from a stress test."""
    test_name: str
    passed: bool
    duration: float
    details: Dict[str, Any]
    errors: List[str]


class MockHumanInterface:
    """Mock human interface for automated stress testing."""

    def __init__(self, auto_approve: bool = True):
        self.auto_approve = auto_approve
        self.interactions = []

    async def ask_question(self, question: str) -> str:
        self.interactions.append({"type": "question", "content": question})
        return "Test Answer"

    async def quick_approve(self, proposal: str, default: bool = True) -> bool:
        self.interactions.append({"type": "quick_approve", "content": proposal})
        return self.auto_approve

    async def choose(self, prompt: str, options: List[str]) -> str:
        self.interactions.append({"type": "choose", "prompt": prompt, "options": options})
        return options[0] if options else "default"

    async def review(self, content: str, approval_criteria: List[str]) -> Dict[str, Any]:
        self.interactions.append({
            "type": "review",
            "content": content,
            "criteria": approval_criteria
        })
        if self.auto_approve:
            return {"approved": True, "changes_requested": [], "comments": "Auto-approved"}
        else:
            return {
                "approved": False,
                "changes_requested": ["Test change request"],
                "comments": "Auto-rejected for testing"
            }

    async def notify(self, message: str) -> None:
        self.interactions.append({"type": "notification", "content": message})


class HybridAgentStressTest:
    """Comprehensive stress test suite for hybrid agent system."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results: List[StressTestResult] = []
        self.decision_log = DecisionLog()

    def log(self, message: str):
        """Log message if verbose mode enabled."""
        if self.verbose:
            print(f"[STRESS TEST] {message}")

    async def run_all_tests(self, quick_mode: bool = False) -> Dict[str, Any]:
        """Run all stress tests."""
        print("="*70)
        print("  HYBRID AGENT SYSTEM STRESS TEST SUITE")
        print("="*70)
        print()

        # Test categories
        tests = [
            # Basic functionality
            ("Team Assistant Agent Stress", self.test_team_assistant_stress),
            ("Vision System Stress", self.test_vision_system_stress),
            ("Decision Tracking Stress", self.test_decision_tracking_stress),
            ("Phase Gate Stress", self.test_phase_gate_stress),

            # Workflow tests
            ("Single Workflow Stress", self.test_single_workflow_stress),
            ("Sequential Workflows Stress", self.test_sequential_workflows_stress),

            # Concurrency tests
            ("Concurrent Workflows Stress", self.test_concurrent_workflows_stress),
            ("Agent Load Distribution", self.test_agent_load_distribution),

            # Edge cases
            ("Error Recovery Stress", self.test_error_recovery_stress),
            ("Vision Context Propagation", self.test_vision_context_propagation),
            ("Confidence Calibration", self.test_confidence_calibration),

            # Learning system
            ("Decision Learning Stress", self.test_decision_learning_stress),

            # Performance
            ("Performance Benchmark", self.test_performance_benchmark),
        ]

        if not quick_mode:
            tests.extend([
                ("High Concurrency Stress", self.test_high_concurrency_stress),
                ("Marathon Test (100 workflows)", self.test_marathon_workflows),
            ])

        # Run each test
        for test_name, test_func in tests:
            print(f"\n{'─'*70}")
            print(f"Running: {test_name}")
            print(f"{'─'*70}")

            try:
                result = await test_func()
                self.results.append(result)

                status = "✅ PASS" if result.passed else "❌ FAIL"
                print(f"{status} - {result.duration:.3f}s")

                if not result.passed:
                    print(f"Errors: {result.errors}")

            except Exception as e:
                print(f"❌ CRASH - {str(e)}")
                self.results.append(StressTestResult(
                    test_name=test_name,
                    passed=False,
                    duration=0.0,
                    details={},
                    errors=[str(e)]
                ))

        # Print summary
        return self._generate_summary()

    async def test_team_assistant_stress(self) -> StressTestResult:
        """Stress test TeamAssistantAgent with multiple scenarios."""
        start_time = time.time()
        errors = []

        agent = TeamAssistantAgent()

        # Test 1: Multiple vision refinements
        intents = [
            "Add Email to Customer",
            "Create loyalty points system",
            "Integrate with external API",
            "Build reporting dashboard",
            "Implement data migration"
        ]

        for intent in intents:
            task = Task(
                name="refine_vision",
                required_capabilities={AgentCapability.VISION_FACILITATION},
                data={"creator_intent": intent, "human_interface": None}
            )

            result = await agent.execute_task(task)
            if result.error:
                errors.append(f"Vision refinement failed for '{intent}': {result.error}")

        # Test 2: Requirements gathering
        vision = ProjectVision(
            id="test_vision",
            creator_intent="Complex multi-table schema change",
            refined_vision="Extend Customer, Vendor, and Item tables with custom fields"
        )

        req_task = Task(
            name="gather_requirements",
            required_capabilities={AgentCapability.REQUIREMENTS_ANALYSIS},
            data={"vision": vision}
        )

        result = await agent.execute_task(req_task)
        if result.error:
            errors.append(f"Requirements gathering failed: {result.error}")

        # Test 3: Question generation stress
        for i in range(20):
            q_task = Task(
                name="generate_questions",
                required_capabilities={AgentCapability.VISION_FACILITATION},
                data={"context": "table_extension"}
            )
            result = await agent.execute_task(q_task)
            if result.error:
                errors.append(f"Question generation {i} failed: {result.error}")

        duration = time.time() - start_time

        return StressTestResult(
            test_name="Team Assistant Agent Stress",
            passed=len(errors) == 0,
            duration=duration,
            details={"intents_processed": len(intents), "questions_generated": 20},
            errors=errors
        )

    async def test_vision_system_stress(self) -> StressTestResult:
        """Stress test ProjectVision system."""
        start_time = time.time()
        errors = []

        vision = ProjectVision(
            id="stress_test_vision",
            creator_intent="Build comprehensive BC extension"
        )

        # Stress test: Add many decisions
        for i in range(100):
            vision.add_decision(
                phase=LifecyclePhase.ARCHITECTURE,
                question=f"Decision {i}",
                decision=f"Choice {i}",
                made_by="human",
                rationale=f"Rationale {i}",
                confidence=0.8 + (i % 20) / 100
            )

        # Stress test: Phase transitions
        phases = list(LifecyclePhase)
        for phase in phases:
            vision.transition_phase(phase)
            context = vision.get_context_for_phase(phase)
            if not context:
                errors.append(f"Failed to get context for phase {phase.value}")

        # Stress test: Vision formatting
        formatted = vision.format_readable()
        if not formatted:
            errors.append("Vision formatting failed")

        duration = time.time() - start_time

        return StressTestResult(
            test_name="Vision System Stress",
            passed=len(errors) == 0,
            duration=duration,
            details={
                "decisions_added": 100,
                "phases_traversed": len(phases),
                "vision_size": len(formatted)
            },
            errors=errors
        )

    async def test_decision_tracking_stress(self) -> StressTestResult:
        """Stress test decision tracking and learning."""
        start_time = time.time()
        errors = []

        log = DecisionLog()

        # Log many decisions
        for i in range(500):
            from agents.base.decision import DecisionContext

            context = DecisionContext(
                task_id=f"task_{i}",
                phase="construction",
                task_type="code_generation" if i % 2 == 0 else "test_generation"
            )

            log.log(
                context=context,
                agent_id="agent_1",
                agent_proposal=f"Proposal {i}",
                final_decision=f"Proposal {i}" if i % 10 != 0 else f"Override {i}",
                made_by="human" if i % 10 == 0 else "agent_1",
                confidence=0.7 + (i % 30) / 100
            )

        # Test analytics
        override_rate = log.get_override_rate()
        if override_rate < 0 or override_rate > 1:
            errors.append(f"Invalid override rate: {override_rate}")

        calibration = log.get_confidence_calibration(threshold=0.9)
        if calibration < 0 or calibration > 1:
            errors.append(f"Invalid calibration: {calibration}")

        # Test learning analyzer
        analyzer = LearningAnalyzer(log)
        insights = analyzer.analyze()

        if len(insights) == 0:
            errors.append("No insights generated from 500 decisions")

        autonomy_level = analyzer.suggest_autonomy_level()
        if "readiness_score" not in autonomy_level:
            errors.append("Autonomy level assessment failed")

        duration = time.time() - start_time

        return StressTestResult(
            test_name="Decision Tracking Stress",
            passed=len(errors) == 0,
            duration=duration,
            details={
                "decisions_logged": len(log.records),
                "override_rate": override_rate,
                "calibration": calibration,
                "insights_generated": len(insights),
                "autonomy_score": autonomy_level.get("readiness_score", 0)
            },
            errors=errors
        )

    async def test_phase_gate_stress(self) -> StressTestResult:
        """Stress test phase gate system."""
        start_time = time.time()
        errors = []

        # Create multiple gates
        gates = [
            PhaseGate(
                from_phase=LifecyclePhase.SCOPING,
                to_phase=LifecyclePhase.REQUIREMENTS,
                required_artifacts=["vision", "success_criteria"],
                approval_criteria=["Vision is clear"]
            ),
            PhaseGate(
                from_phase=LifecyclePhase.REQUIREMENTS,
                to_phase=LifecyclePhase.ARCHITECTURE,
                required_artifacts=["requirements"],
                approval_criteria=["Requirements complete"]
            ),
        ]

        vision = ProjectVision(id="gate_test", creator_intent="Test")
        human = MockHumanInterface(auto_approve=True)

        # Test each gate with valid outputs
        for gate in gates:
            outputs = {artifact: f"Test {artifact}" for artifact in gate.required_artifacts}

            decision = await gate.review(vision, outputs, human)
            if decision.status.value != "approved":
                errors.append(f"Gate {gate.name} failed to approve valid outputs")

        # Test gate rejection
        human_reject = MockHumanInterface(auto_approve=False)
        decision = await gates[0].review(vision, {"vision": "test"}, human_reject)
        if decision.status.value == "approved":
            errors.append("Gate approved when it should have rejected")

        # Test missing artifacts
        decision = await gates[0].review(vision, {}, human)
        if decision.status.value != "incomplete":
            errors.append("Gate didn't detect missing artifacts")

        duration = time.time() - start_time

        return StressTestResult(
            test_name="Phase Gate Stress",
            passed=len(errors) == 0,
            duration=duration,
            details={
                "gates_tested": len(gates),
                "human_interactions": len(human.interactions)
            },
            errors=errors
        )

    async def test_single_workflow_stress(self) -> StressTestResult:
        """Stress test single workflow execution."""
        start_time = time.time()
        errors = []

        coordinator = self._create_full_coordinator()
        workflow = TableExtensionWorkflow("Add Email to Customer table")

        # Execute all phases
        phases = [
            LifecyclePhase.SCOPING,
            LifecyclePhase.REQUIREMENTS,
            LifecyclePhase.ARCHITECTURE,
            LifecyclePhase.DESIGN,
            LifecyclePhase.CONSTRUCTION,
            LifecyclePhase.TESTING,
        ]

        for phase in phases:
            result = await workflow.execute_phase(phase, coordinator)
            if not result.success:
                errors.append(f"Phase {phase.value} failed: {result.errors}")

        duration = time.time() - start_time

        return StressTestResult(
            test_name="Single Workflow Stress",
            passed=len(errors) == 0,
            duration=duration,
            details={"phases_completed": len(phases)},
            errors=errors
        )

    async def test_sequential_workflows_stress(self) -> StressTestResult:
        """Stress test multiple sequential workflows."""
        start_time = time.time()
        errors = []

        coordinator = self._create_full_coordinator()

        workflows = [
            TableExtensionWorkflow("Add Email to Customer"),
            TableExtensionWorkflow("Add Phone to Vendor"),
            TableExtensionWorkflow("Add Loyalty to Customer"),
            TableExtensionWorkflow("Add Rating to Item"),
            TableExtensionWorkflow("Add Notes to Sales Header"),
        ]

        for i, workflow in enumerate(workflows):
            result = await workflow.execute_phase(LifecyclePhase.SCOPING, coordinator)
            if not result.success:
                errors.append(f"Workflow {i} failed: {result.errors}")

        duration = time.time() - start_time

        return StressTestResult(
            test_name="Sequential Workflows Stress",
            passed=len(errors) == 0,
            duration=duration,
            details={"workflows_completed": len(workflows)},
            errors=errors
        )

    async def test_concurrent_workflows_stress(self) -> StressTestResult:
        """Stress test concurrent workflow execution."""
        start_time = time.time()
        errors = []

        coordinator = self._create_full_coordinator()

        # Create 10 concurrent workflows
        workflows = [
            TableExtensionWorkflow(f"Add Field{i} to Table{i}")
            for i in range(10)
        ]

        # Execute all concurrently
        tasks = [
            workflow.execute_phase(LifecyclePhase.SCOPING, coordinator)
            for workflow in workflows
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                errors.append(f"Workflow {i} crashed: {str(result)}")
            elif not result.success:
                errors.append(f"Workflow {i} failed: {result.errors}")

        duration = time.time() - start_time

        return StressTestResult(
            test_name="Concurrent Workflows Stress",
            passed=len(errors) == 0,
            duration=duration,
            details={
                "concurrent_workflows": len(workflows),
                "successful": len([r for r in results if not isinstance(r, Exception) and r.success])
            },
            errors=errors
        )

    async def test_agent_load_distribution(self) -> StressTestResult:
        """Test that work is distributed across agents."""
        start_time = time.time()
        errors = []

        coordinator = self._create_full_coordinator()

        # Execute multiple workflows
        workflows = [
            TableExtensionWorkflow(f"Workflow {i}")
            for i in range(5)
        ]

        for workflow in workflows:
            await workflow.execute_phase(LifecyclePhase.SCOPING, coordinator)
            await workflow.execute_phase(LifecyclePhase.CONSTRUCTION, coordinator)

        # Check that multiple agents were used
        agents_used = set()
        for agent in coordinator.agents.values():
            if len(agent.task_history) > 0:
                agents_used.add(agent.name)

        if len(agents_used) < 2:
            errors.append(f"Only {len(agents_used)} agents used, expected multiple")

        duration = time.time() - start_time

        return StressTestResult(
            test_name="Agent Load Distribution",
            passed=len(errors) == 0,
            duration=duration,
            details={
                "agents_used": len(agents_used),
                "agent_names": list(agents_used)
            },
            errors=errors
        )

    async def test_error_recovery_stress(self) -> StressTestResult:
        """Test system's ability to handle errors gracefully."""
        start_time = time.time()
        errors = []

        coordinator = Coordinator()
        # Intentionally don't register agents to cause failures

        workflow = TableExtensionWorkflow("Test workflow")

        # This should fail gracefully
        result = await workflow.execute_phase(LifecyclePhase.SCOPING, coordinator)

        if result.success:
            errors.append("Workflow succeeded when it should have failed (no agents)")

        if not result.errors:
            errors.append("No error message provided on failure")

        duration = time.time() - start_time

        return StressTestResult(
            test_name="Error Recovery Stress",
            passed=len(errors) == 0,
            duration=duration,
            details={"graceful_failure": not result.success},
            errors=errors
        )

    async def test_vision_context_propagation(self) -> StressTestResult:
        """Test that vision context flows through all phases."""
        start_time = time.time()
        errors = []

        coordinator = self._create_full_coordinator()
        workflow = TableExtensionWorkflow("Context propagation test")

        # Execute scoping
        scoping_result = await workflow.execute_phase(LifecyclePhase.SCOPING, coordinator)
        workflow._update_vision_from_outputs(LifecyclePhase.SCOPING, scoping_result.outputs)

        # Check vision was updated
        if not workflow.vision.refined_vision:
            errors.append("Vision not updated after scoping")

        # Execute requirements
        req_result = await workflow.execute_phase(LifecyclePhase.REQUIREMENTS, coordinator)

        # Vision context should be available
        context = workflow.vision.get_context_for_phase(LifecyclePhase.REQUIREMENTS)
        if "vision" not in context:
            errors.append("Vision context missing in requirements phase")

        duration = time.time() - start_time

        return StressTestResult(
            test_name="Vision Context Propagation",
            passed=len(errors) == 0,
            duration=duration,
            details={"context_keys": list(context.keys())},
            errors=errors
        )

    async def test_confidence_calibration(self) -> StressTestResult:
        """Test confidence scoring across agents."""
        start_time = time.time()
        errors = []

        agent = TeamAssistantAgent()

        # Test confidence assessment
        from agents.base.vision import ClarifyingQuestion, QuestionCategory

        questions = [
            ClarifyingQuestion("Q1", QuestionCategory.TECHNICAL, "R1", answer="A1"),
            ClarifyingQuestion("Q2", QuestionCategory.TECHNICAL, "R2", answer="A2"),
            ClarifyingQuestion("Q3", QuestionCategory.TECHNICAL, "R3", answer="A3"),
        ]

        confidence = agent._assess_vision_confidence(questions)

        if confidence.overall < 0 or confidence.overall > 1:
            errors.append(f"Invalid confidence score: {confidence.overall}")

        # Test with unanswered questions
        incomplete_questions = [
            ClarifyingQuestion("Q1", QuestionCategory.TECHNICAL, "R1"),  # No answer
            ClarifyingQuestion("Q2", QuestionCategory.TECHNICAL, "R2", answer="A2"),
        ]

        low_confidence = agent._assess_vision_confidence(incomplete_questions)
        if low_confidence.overall >= confidence.overall:
            errors.append("Incomplete questions should have lower confidence")

        duration = time.time() - start_time

        return StressTestResult(
            test_name="Confidence Calibration",
            passed=len(errors) == 0,
            duration=duration,
            details={
                "full_confidence": confidence.overall,
                "partial_confidence": low_confidence.overall
            },
            errors=errors
        )

    async def test_decision_learning_stress(self) -> StressTestResult:
        """Test decision learning system under load."""
        start_time = time.time()
        errors = []

        log = DecisionLog()

        # Simulate a learning scenario
        from agents.base.decision import DecisionContext

        # Pattern: email field sizing - agent learns over time
        for i in range(30):
            context = DecisionContext(
                task_id=f"email_{i}",
                phase="design",
                task_type="email_field_sizing"
            )

            # First 10: Agent proposes Text[80], human overrides to Text[100]
            # Next 20: Agent learns and proposes Text[100], human accepts
            if i < 10:
                log.log(context, "code_agent", "Text[80]", "Text[100]", "human", 0.8)
            else:
                log.log(context, "code_agent", "Text[100]", "Text[100]", "code_agent", 0.95)

        # Check learning
        task_stats = log.get_task_type_stats("email_field_sizing")

        if task_stats["count"] != 30:
            errors.append(f"Expected 30 decisions, got {task_stats['count']}")

        # Override rate should be around 33% (10 out of 30)
        expected_override = 10 / 30
        if abs(task_stats["override_rate"] - expected_override) > 0.05:
            errors.append(f"Override rate {task_stats['override_rate']} doesn't match expected {expected_override}")

        # Analyze for automation candidates
        analyzer = LearningAnalyzer(log)
        insights = analyzer.analyze()

        # Should identify this as automation candidate
        automation_found = any(
            i.category == "automation_candidate" and "email_field_sizing" in i.description
            for i in insights
        )

        if not automation_found:
            errors.append("Failed to identify automation candidate from clear pattern")

        duration = time.time() - start_time

        return StressTestResult(
            test_name="Decision Learning Stress",
            passed=len(errors) == 0,
            duration=duration,
            details={
                "decisions": 30,
                "override_rate": task_stats["override_rate"],
                "automation_candidates": sum(1 for i in insights if i.category == "automation_candidate")
            },
            errors=errors
        )

    async def test_performance_benchmark(self) -> StressTestResult:
        """Benchmark performance of key operations."""
        start_time = time.time()
        errors = []
        benchmarks = {}

        # Benchmark 1: Vision creation
        t1 = time.time()
        for i in range(100):
            vision = ProjectVision(id=f"bench_{i}", creator_intent=f"Intent {i}")
            vision.add_decision(LifecyclePhase.SCOPING, "Q", "D", "human", "R")
        benchmarks["vision_creation_per_100"] = time.time() - t1

        # Benchmark 2: Decision logging
        log = DecisionLog()
        from agents.base.decision import DecisionContext

        t2 = time.time()
        for i in range(1000):
            context = DecisionContext(f"task_{i}", "phase", "type")
            log.log(context, "agent", "prop", "dec", "human", 0.8)
        benchmarks["decision_logging_per_1000"] = time.time() - t2

        # Benchmark 3: Agent task execution
        agent = TeamAssistantAgent()
        t3 = time.time()
        for i in range(50):
            task = Task(
                name="test",
                required_capabilities={AgentCapability.VISION_FACILITATION},
                data={"creator_intent": f"Test {i}", "human_interface": None}
            )
            await agent.execute_task(task)
        benchmarks["agent_execution_per_50"] = time.time() - t3

        # Check for performance regressions
        if benchmarks["vision_creation_per_100"] > 1.0:
            errors.append(f"Vision creation too slow: {benchmarks['vision_creation_per_100']:.3f}s")

        if benchmarks["decision_logging_per_1000"] > 2.0:
            errors.append(f"Decision logging too slow: {benchmarks['decision_logging_per_1000']:.3f}s")

        if benchmarks["agent_execution_per_50"] > 5.0:
            errors.append(f"Agent execution too slow: {benchmarks['agent_execution_per_50']:.3f}s")

        duration = time.time() - start_time

        return StressTestResult(
            test_name="Performance Benchmark",
            passed=len(errors) == 0,
            duration=duration,
            details=benchmarks,
            errors=errors
        )

    async def test_high_concurrency_stress(self) -> StressTestResult:
        """Test system under very high concurrency."""
        start_time = time.time()
        errors = []

        coordinator = self._create_full_coordinator()

        # Create 50 concurrent workflows
        workflows = [
            TableExtensionWorkflow(f"Concurrent {i}")
            for i in range(50)
        ]

        # Execute all concurrently
        tasks = [
            workflow.execute_phase(LifecyclePhase.SCOPING, coordinator)
            for workflow in workflows
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        successful = sum(1 for r in results if not isinstance(r, Exception) and r.success)

        # Allow some failures under extreme load, but most should succeed
        if successful < 40:  # 80% success rate
            errors.append(f"Only {successful}/50 workflows succeeded under high concurrency")

        duration = time.time() - start_time

        return StressTestResult(
            test_name="High Concurrency Stress",
            passed=len(errors) == 0,
            duration=duration,
            details={
                "concurrent_workflows": 50,
                "successful": successful,
                "success_rate": successful / 50
            },
            errors=errors
        )

    async def test_marathon_workflows(self) -> StressTestResult:
        """Marathon test: 100 workflows sequentially."""
        start_time = time.time()
        errors = []

        coordinator = self._create_full_coordinator()

        successful = 0
        for i in range(100):
            workflow = TableExtensionWorkflow(f"Marathon {i}")
            result = await workflow.execute_phase(LifecyclePhase.SCOPING, coordinator)
            if result.success:
                successful += 1
            else:
                if len(errors) < 10:  # Only log first 10 errors
                    errors.append(f"Workflow {i} failed")

        # Should have very high success rate
        if successful < 95:
            errors.append(f"Only {successful}/100 workflows succeeded in marathon")

        duration = time.time() - start_time

        return StressTestResult(
            test_name="Marathon Test (100 workflows)",
            passed=len(errors) == 0,
            duration=duration,
            details={
                "total_workflows": 100,
                "successful": successful,
                "avg_time_per_workflow": duration / 100
            },
            errors=errors
        )

    def _create_full_coordinator(self) -> Coordinator:
        """Create a coordinator with all agents registered."""
        coordinator = Coordinator()
        coordinator.register_agent(TeamAssistantAgent())
        coordinator.register_agent(CodeAgent())
        coordinator.register_agent(SchemaAgent())
        coordinator.register_agent(TestAgent())
        coordinator.register_agent(APIAgent())
        coordinator.register_agent(DeploymentAgent())
        coordinator.register_agent(DocumentationAgent())
        return coordinator

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate test summary."""
        print("\n" + "="*70)
        print("  TEST SUMMARY")
        print("="*70)

        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed

        total_duration = sum(r.duration for r in self.results)

        print(f"\nTotal Tests: {total}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"Pass Rate: {passed/total*100:.1f}%")
        print(f"Total Duration: {total_duration:.2f}s")
        print()

        if failed > 0:
            print("Failed Tests:")
            for result in self.results:
                if not result.passed:
                    print(f"  ❌ {result.test_name}")
                    for error in result.errors[:3]:  # Show first 3 errors
                        print(f"     - {error}")
            print()

        # Performance highlights
        print("Performance Highlights:")
        for result in self.results:
            if "Performance" in result.test_name or "Benchmark" in result.test_name:
                print(f"  {result.test_name}:")
                for key, value in result.details.items():
                    if isinstance(value, float):
                        print(f"    - {key}: {value:.3f}s")

        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / total,
            "duration": total_duration,
            "results": self.results
        }


async def main():
    """Main entry point for stress tests."""
    parser = argparse.ArgumentParser(description="Hybrid Agent System Stress Test")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--quick", "-q", action="store_true", help="Skip long-running tests")
    args = parser.parse_args()

    stress_test = HybridAgentStressTest(verbose=args.verbose)
    summary = await stress_test.run_all_tests(quick_mode=args.quick)

    # Exit with appropriate code
    exit_code = 0 if summary["failed"] == 0 else 1
    return exit_code


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
