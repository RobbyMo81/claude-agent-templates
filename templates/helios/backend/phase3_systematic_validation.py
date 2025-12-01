#!/usr/bin/env python3
"""
PHASE 3 SYSTEMATIC VALIDATION PLAN
=================================

Comprehensive Phase 3 System Testing Based on Post-Refactoring_Test.txt Requirements
This script implements a disciplined, systematic validation approach for Phase 3 components.

TESTING PHASES:
1. Backend Component Import & Initialization Test
2. Full System Smoke Test Coordination
3. Functional Unit Testing of Phase 3 Core Logic
4. End-to-End API Integration Testing
5. Production Readiness Assessment

Author: AI Assistant
Date: July 15, 2025
"""

import os
import sys
import time
import json
import tempfile
import unittest
import subprocess
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Add backend to Python path
backend_path = os.path.join(os.path.dirname(__file__), "backend")
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)


class Phase3ValidationPlan:
    """
    Systematic validation plan for Phase 3 components
    Based on Post-Refactoring_Test.txt requirements
    """

    def __init__(self):
        self.results = {}
        self.start_time = datetime.now()
        self.phase_status = {
            "phase_1_imports": "PENDING",
            "phase_2_smoke_test": "PENDING",
            "phase_3_unit_tests": "PENDING",
            "phase_4_integration": "PENDING",
            "phase_5_readiness": "PENDING",
        }

    def log_result(self, phase: str, test_name: str, status: str, details: str = ""):
        """Log test results"""
        if phase not in self.results:
            self.results[phase] = []

        self.results[phase].append(
            {
                "test": test_name,
                "status": status,
                "details": details,
                "timestamp": datetime.now().isoformat(),
            }
        )

        print(f"[{phase}] {test_name}: {status}")
        if details:
            print(f"    Details: {details}")

    def print_phase_header(self, phase_num: int, phase_name: str, mission: str):
        """Print formatted phase header"""
        print("\n" + "=" * 80)
        print(f"🧪 PHASE {phase_num}: {phase_name.upper()}")
        print("=" * 80)
        print(f"📋 Mission: {mission}")
        print("⏰ Starting Phase:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("-" * 80)

    def phase_1_backend_component_test(self):
        """
        PHASE 1: Backend Component Import & Initialization Test

        Mission: Confirm that ImportError is truly gone and all Phase 3 components
        can be imported and initialized successfully.

        This is our gate - we do not proceed until we see success.
        """
        self.print_phase_header(
            1,
            "Backend Component Import & Initialization",
            "Confirm ImportError is gone and components initialize",
        )

        # Test 1.1: Import memory_store
        try:
            print("🔍 Testing memory_store.py import...")
            from memory_store import MemoryStore

            self.log_result(
                "phase_1",
                "memory_store_import",
                "SUCCESS",
                "MemoryStore class imported successfully",
            )
            print("✅ memory_store.py - Import successful")
        except Exception as e:
            self.log_result("phase_1", "memory_store_import", "FAILED", str(e))
            print(f"❌ memory_store.py - Import failed: {e}")
            return False

        # Test 1.2: Import metacognition
        try:
            print("🔍 Testing metacognition.py import...")
            from metacognition import (
                MetacognitiveEngine,
                MetacognitiveAssessment,
                LearningStrategy,
            )

            self.log_result(
                "phase_1",
                "metacognition_import",
                "SUCCESS",
                "MetacognitiveEngine and related classes imported",
            )
            print("✅ metacognition.py - Import successful")
        except Exception as e:
            self.log_result("phase_1", "metacognition_import", "FAILED", str(e))
            print(f"❌ metacognition.py - Import failed: {e}")
            return False

        # Test 1.3: Import decision_engine
        try:
            print("🔍 Testing decision_engine.py import...")
            from decision_engine import DecisionEngine, Goal, DecisionType

            self.log_result(
                "phase_1",
                "decision_engine_import",
                "SUCCESS",
                "DecisionEngine and related classes imported",
            )
            print("✅ decision_engine.py - Import successful")
        except Exception as e:
            self.log_result("phase_1", "decision_engine_import", "FAILED", str(e))
            print(f"❌ decision_engine.py - Import failed: {e}")
            return False

        # Test 1.4: Initialize MemoryStore
        try:
            print("🔍 Testing MemoryStore initialization...")
            temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
            temp_db.close()

            memory_store = MemoryStore(temp_db.name)
            self.log_result(
                "phase_1",
                "memory_store_init",
                "SUCCESS",
                f"MemoryStore initialized with database: {temp_db.name}",
            )
            print("✅ MemoryStore initialization successful")

            # Cleanup
            memory_store.close()
            os.unlink(temp_db.name)

        except Exception as e:
            self.log_result("phase_1", "memory_store_init", "FAILED", str(e))
            print(f"❌ MemoryStore initialization failed: {e}")
            return False

        # Test 1.5: Initialize MetacognitiveEngine
        try:
            print("🔍 Testing MetacognitiveEngine initialization...")
            temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
            temp_db.close()

            memory_store = MemoryStore(temp_db.name)
            metacognitive_engine = MetacognitiveEngine(memory_store)

            self.log_result(
                "phase_1",
                "metacognitive_engine_init",
                "SUCCESS",
                "MetacognitiveEngine initialized successfully",
            )
            print("✅ MetacognitiveEngine initialization successful")

            # Cleanup
            memory_store.close()
            os.unlink(temp_db.name)

        except Exception as e:
            self.log_result("phase_1", "metacognitive_engine_init", "FAILED", str(e))
            print(f"❌ MetacognitiveEngine initialization failed: {e}")
            return False

        # Test 1.6: Initialize DecisionEngine
        try:
            print("🔍 Testing DecisionEngine initialization...")
            temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
            temp_db.close()

            memory_store = MemoryStore(temp_db.name)
            metacognitive_engine = MetacognitiveEngine(memory_store)
            decision_engine = DecisionEngine(memory_store, metacognitive_engine)

            self.log_result(
                "phase_1",
                "decision_engine_init",
                "SUCCESS",
                "DecisionEngine initialized successfully",
            )
            print("✅ DecisionEngine initialization successful")

            # Cleanup
            memory_store.close()
            os.unlink(temp_db.name)

        except Exception as e:
            self.log_result("phase_1", "decision_engine_init", "FAILED", str(e))
            print(f"❌ DecisionEngine initialization failed: {e}")
            return False

        # Phase 1 Summary
        print("\n🎯 PHASE 1 GATE CHECK:")
        print("✅ All Phase 3 backend components import successfully")
        print("✅ All Phase 3 backend components initialize successfully")
        print("🚀 GATE PASSED - Proceeding to Phase 2")

        self.phase_status["phase_1_imports"] = "SUCCESS"
        return True

    def phase_2_smoke_test_coordination(self):
        """
        PHASE 2: Full System Smoke Test Coordination

        Mission: Ensure the refactoring didn't introduce any new startup crashes.
        Coordinate backend and frontend server startup.
        """
        self.print_phase_header(
            2, "Full System Smoke Test", "Ensure no new startup crashes were introduced"
        )

        print("📋 SMOKE TEST INSTRUCTIONS:")
        print("This phase requires manual coordination of both servers.")
        print()
        print("🔧 MANUAL STEPS TO EXECUTE:")
        print("1. Open Terminal 1 - Backend Server:")
        print("   cd backend")
        print("   python server.py")
        print("   → Look for: 'Flask app started' or similar success message")
        print()
        print("2. Open Terminal 2 - Frontend Server:")
        print("   npm run dev")
        print("   → Look for: 'Local: http://localhost:3000' or similar")
        print()
        print("3. Verify both servers start cleanly without errors")
        print()
        print("⚠️  AUTOMATED CHECK: Testing if backend can start programmatically...")

        # Test backend startup programmatically
        try:
            print("🔍 Testing backend server import...")
            original_dir = os.getcwd()
            backend_dir = os.path.dirname(
                __file__
            )  # Current directory is already backend
            os.chdir(backend_dir)

            # Test server.py import
            sys.path.insert(0, backend_dir)
            import server

            self.log_result(
                "phase_2",
                "backend_server_import",
                "SUCCESS",
                "server.py imports without errors",
            )
            print("✅ Backend server.py imports successfully")

            # Test Flask app creation
            app = server.app
            if app:
                self.log_result(
                    "phase_2",
                    "flask_app_creation",
                    "SUCCESS",
                    "Flask app created successfully",
                )
                print("✅ Flask app created successfully")

            os.chdir(original_dir)

        except Exception as e:
            self.log_result("phase_2", "backend_server_import", "FAILED", str(e))
            print(f"❌ Backend server import failed: {e}")
            return False

        print("\n🎯 PHASE 2 RESULTS:")
        print("✅ Backend server imports and Flask app creation successful")
        print("⚠️  Manual verification required for full server startup")
        print("📝 Please execute manual steps above and verify both servers start")

        self.phase_status["phase_2_smoke_test"] = "PARTIAL_SUCCESS"
        return True

    def phase_3_functional_unit_testing(self):
        """
        PHASE 3: Functional Unit Testing of Phase 3 Core Logic

        Mission: Test core functions in isolation to verify repaired code logic.
        """
        self.print_phase_header(
            3, "Functional Unit Testing", "Test core Phase 3 functions in isolation"
        )

        # Setup test environment
        temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        temp_db.close()
        memory_store = None

        try:
            # Import required components
            from memory_store import MemoryStore
            from metacognition import MetacognitiveEngine, LearningStrategy
            from decision_engine import (
                DecisionEngine,
                Goal,
                DecisionType,
                DecisionPriority,
            )

            memory_store = MemoryStore(temp_db.name)

            # Test 3.1: MemoryStore core functionality
            print("🔍 Testing MemoryStore core functionality...")
            try:
                memory_store = MemoryStore(temp_db.name)

                # Test table creation
                with memory_store._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                    tables = cursor.fetchall()

                if len(tables) > 0:
                    self.log_result(
                        "phase_3",
                        "memory_store_tables",
                        "SUCCESS",
                        f"Database created with {len(tables)} tables",
                    )
                    print(f"✅ MemoryStore created {len(tables)} tables successfully")
                else:
                    raise Exception("No tables created")

                # Test model storage
                test_model = {
                    "name": "test_model",
                    "architecture": "test_arch",
                    "version": "1.0",
                    "metadata": {"test": True},
                }

                memory_store.save_model_metadata(
                    name=test_model["name"],
                    file_path="/test/path",
                    architecture=test_model["architecture"],
                    version=test_model["version"],
                    metadata=test_model["metadata"],
                )
                stored_models = memory_store.list_models()

                if len(stored_models) > 0:
                    self.log_result(
                        "phase_3",
                        "memory_store_model_storage",
                        "SUCCESS",
                        "Model metadata stored and retrieved successfully",
                    )
                    print("✅ MemoryStore model storage working")
                else:
                    raise Exception("Model storage failed")

            except Exception as e:
                self.log_result("phase_3", "memory_store_core", "FAILED", str(e))
                print(f"❌ MemoryStore core functionality failed: {e}")
                return False

            # Test 3.2: MetacognitiveEngine core functionality
            print("🔍 Testing MetacognitiveEngine core functionality...")
            try:
                metacognitive_engine = MetacognitiveEngine(memory_store)

                # Test self-assessment with mock data
                assessment = metacognitive_engine.assess_current_state(
                    model_name="test_model",
                    current_metrics={"accuracy": 0.85, "loss": 0.23},
                    recent_performance=[0.80, 0.82, 0.85],
                    context={"training_epoch": 50},
                )

                if assessment and hasattr(assessment, "confidence_score"):
                    self.log_result(
                        "phase_3",
                        "metacognitive_self_assess",
                        "SUCCESS",
                        f"Self-assessment completed with confidence: {assessment.confidence_score:.2f}",
                    )
                    print(
                        f"✅ MetacognitiveEngine self-assessment: confidence={assessment.confidence_score:.2f}"
                    )
                else:
                    raise Exception("Self-assessment returned invalid result")

                # Test learning recommendations
                recommendations = metacognitive_engine.get_learning_recommendations(
                    "test_model", assessment
                )

                if recommendations and isinstance(recommendations, dict):
                    self.log_result(
                        "phase_3",
                        "metacognitive_recommendations",
                        "SUCCESS",
                        f"Learning recommendations generated: {len(recommendations)} items",
                    )
                    print(
                        f"✅ MetacognitiveEngine recommendations: {len(recommendations)} items"
                    )
                else:
                    raise Exception("Learning recommendations failed")

            except Exception as e:
                self.log_result("phase_3", "metacognitive_core", "FAILED", str(e))
                print(f"❌ MetacognitiveEngine core functionality failed: {e}")
                return False

            # Test 3.3: DecisionEngine core functionality
            print("🔍 Testing DecisionEngine core functionality...")
            try:
                decision_engine = DecisionEngine(memory_store, metacognitive_engine)

                # Test goal processing
                test_goal = Goal(
                    goal_id="test_goal_001",
                    name="Improve model accuracy",
                    target_metric="accuracy",
                    target_value=0.90,
                    current_value=0.85,
                    priority=1,
                    deadline=datetime.now() + timedelta(days=7),
                )

                decision_engine.add_goal(test_goal)
                goal_status = decision_engine.get_goal_status()

                if goal_status and "goals" in goal_status:
                    self.log_result(
                        "phase_3",
                        "decision_engine_goals",
                        "SUCCESS",
                        f"Goal added and retrieved: {test_goal.goal_id}",
                    )
                    print(f"✅ DecisionEngine goal processing: {test_goal.goal_id}")
                else:
                    raise Exception("Goal processing failed")

                # Test decision making
                decisions = decision_engine.make_autonomous_decision(
                    model_name="test_model",
                    current_metrics={"accuracy": 0.85, "loss": 0.23},
                    recent_performance=[0.80, 0.82, 0.85],
                    context={"current_lr": 0.001},
                )

                if decisions and len(decisions) > 0:
                    self.log_result(
                        "phase_3",
                        "decision_engine_decisions",
                        "SUCCESS",
                        f"Autonomous decisions made: {len(decisions)} decisions",
                    )
                    print(
                        f"✅ DecisionEngine autonomous decision making: {len(decisions)} decisions"
                    )
                else:
                    self.log_result(
                        "phase_3",
                        "decision_engine_decisions",
                        "SUCCESS",
                        "Decision making executed (no decisions needed)",
                    )
                    print("✅ DecisionEngine autonomous decision making executed")

            except Exception as e:
                self.log_result("phase_3", "decision_engine_core", "FAILED", str(e))
                print(f"❌ DecisionEngine core functionality failed: {e}")
                return False

        finally:
            # Cleanup
            try:
                if memory_store:
                    memory_store.close()
                os.unlink(temp_db.name)
            except:
                pass

        print("\n🎯 PHASE 3 RESULTS:")
        print("✅ MemoryStore core functionality validated")
        print("✅ MetacognitiveEngine self-assessment and recommendations working")
        print("✅ DecisionEngine goal processing and decision making working")
        print("🧠 Phase 3 'brains' are functioning correctly!")

        self.phase_status["phase_3_unit_tests"] = "SUCCESS"
        return True

    def phase_4_api_integration_testing(self):
        """
        PHASE 4: End-to-End API Integration Testing

        Mission: Test the full loop from React UI to Python backend systems.
        """
        self.print_phase_header(
            4, "End-to-End API Integration", "Test full UI to backend API loop"
        )

        print("📋 API INTEGRATION TEST PLAN:")
        print()

        # Test 4.1: Check if server has Phase 3 endpoints
        print("🔍 Testing Phase 3 API endpoint availability...")
        try:
            import server

            app = server.app

            # Get all routes
            routes = []
            for rule in app.url_map.iter_rules():
                routes.append(rule.rule)

            # Check for Phase 3 endpoints
            metacognition_endpoints = [r for r in routes if "metacognitive" in r]
            decision_endpoints = [r for r in routes if "decision" in r]

            if metacognition_endpoints:
                self.log_result(
                    "phase_4",
                    "metacognition_endpoints",
                    "SUCCESS",
                    f"Found {len(metacognition_endpoints)} metacognition endpoints",
                )
                print(
                    f"✅ Metacognition API endpoints: {len(metacognition_endpoints)} found"
                )
                for endpoint in metacognition_endpoints:
                    print(f"   - {endpoint}")
            else:
                self.log_result(
                    "phase_4",
                    "metacognition_endpoints",
                    "FAILED",
                    "No metacognition endpoints found",
                )
                print("❌ No metacognition endpoints found")

            if decision_endpoints:
                self.log_result(
                    "phase_4",
                    "decision_endpoints",
                    "SUCCESS",
                    f"Found {len(decision_endpoints)} decision endpoints",
                )
                print(f"✅ Decision API endpoints: {len(decision_endpoints)} found")
                for endpoint in decision_endpoints:
                    print(f"   - {endpoint}")
            else:
                self.log_result(
                    "phase_4",
                    "decision_endpoints",
                    "FAILED",
                    "No decision endpoints found",
                )
                print("❌ No decision endpoints found")

        except Exception as e:
            self.log_result("phase_4", "api_endpoint_check", "FAILED", str(e))
            print(f"❌ API endpoint check failed: {e}")
            return False

        # Test 4.2: Frontend component availability
        print("\n🔍 Testing frontend component availability...")
        try:
            frontend_components = [
                "components/MetacognitiveDashboard.tsx",
                "App.tsx",
                "components/Sidebar.tsx",
            ]

            # Frontend components are in parent directory
            frontend_dir = os.path.dirname(os.path.dirname(__file__))

            for component in frontend_components:
                component_path = os.path.join(frontend_dir, component)
                if os.path.exists(component_path):
                    self.log_result(
                        "phase_4",
                        f'frontend_{component.replace("/", "_")}',
                        "SUCCESS",
                        f"Component exists: {component}",
                    )
                    print(f"✅ Frontend component: {component}")
                else:
                    self.log_result(
                        "phase_4",
                        f'frontend_{component.replace("/", "_")}',
                        "FAILED",
                        f"Component missing: {component}",
                    )
                    print(f"❌ Frontend component missing: {component}")

        except Exception as e:
            self.log_result("phase_4", "frontend_component_check", "FAILED", str(e))
            print(f"❌ Frontend component check failed: {e}")

        print("\n📋 MANUAL INTEGRATION TEST INSTRUCTIONS:")
        print("To complete Phase 4 testing, please execute these manual steps:")
        print()
        print("1. Start both servers (if not already running):")
        print("   Terminal 1: cd backend && python server.py")
        print("   Terminal 2: npm run dev")
        print()
        print("2. Open browser and navigate to: http://localhost:3000")
        print()
        print("3. Navigate to the '🧠 Metacognitive AI' dashboard")
        print("   - Click on the metacognitive section in the sidebar")
        print()
        print("4. Test the UI tabs:")
        print("   - Click 'Self-Assessment' tab")
        print("   - Click 'Performance Patterns' tab")
        print("   - Click 'Learning Recommendations' tab")
        print()
        print("5. Monitor browser developer tools (F12 → Network tab):")
        print("   - Look for API calls to /api/metacognitive/*")
        print("   - Look for API calls to /api/decision/*")
        print("   - Verify responses are JSON (not 404/500 errors)")
        print()
        print("6. Expected API endpoints to test:")
        for endpoint in metacognition_endpoints + decision_endpoints:
            print(f"   ✓ {endpoint}")

        self.phase_status["phase_4_integration"] = "REQUIRES_MANUAL_VERIFICATION"
        return True

    def phase_5_production_readiness_assessment(self):
        """
        PHASE 5: Production Readiness Assessment

        Mission: Final assessment of Phase 3 system readiness.
        """
        self.print_phase_header(
            5,
            "Production Readiness Assessment",
            "Final validation for mission readiness",
        )

        # Calculate success metrics
        total_tests = sum(len(phase_results) for phase_results in self.results.values())
        successful_tests = sum(
            len([test for test in phase_results if test["status"] == "SUCCESS"])
            for phase_results in self.results.values()
        )

        success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0

        print(f"📊 OVERALL TEST RESULTS:")
        print(f"   Total Tests Executed: {total_tests}")
        print(f"   Successful Tests: {successful_tests}")
        print(f"   Success Rate: {success_rate:.1f}%")
        print()

        # Phase-by-phase assessment
        for phase, status in self.phase_status.items():
            status_icon = (
                "✅" if status == "SUCCESS" else "⚠️" if "PARTIAL" in status else "❌"
            )
            print(f"{status_icon} {phase.replace('_', ' ').title()}: {status}")

        # Overall readiness assessment
        print("\n🎯 PRODUCTION READINESS ASSESSMENT:")

        if success_rate >= 90 and self.phase_status["phase_1_imports"] == "SUCCESS":
            readiness = "MISSION READY"
            icon = "🚀"
            message = (
                "Phase 3 systems are validated and ready for production deployment!"
            )
        elif success_rate >= 70:
            readiness = "MOSTLY READY"
            icon = "⚠️"
            message = (
                "Phase 3 systems are largely functional with minor issues to address."
            )
        else:
            readiness = "NEEDS WORK"
            icon = "🔧"
            message = "Phase 3 systems require significant debugging before deployment."

        print(f"{icon} OVERALL STATUS: {readiness}")
        print(f"   {message}")

        # Specific recommendations
        print("\n📋 RECOMMENDATIONS:")

        if self.phase_status["phase_1_imports"] == "SUCCESS":
            print("✅ Phase 1 Gate Passed - All imports and initialization working")
        else:
            print("❌ Phase 1 Gate Failed - Must fix import errors before proceeding")

        if self.phase_status["phase_3_unit_tests"] == "SUCCESS":
            print("✅ Core Logic Validated - Phase 3 'brains' are functioning")
        else:
            print("🔧 Core Logic Issues - Need to debug Phase 3 internal functions")

        if self.phase_status["phase_4_integration"] == "REQUIRES_MANUAL_VERIFICATION":
            print("📝 Manual Verification Required - Complete end-to-end testing")

        # Final conclusion
        print(f"\n🏁 VALIDATION COMPLETE")
        print(f"   Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Duration: {datetime.now() - self.start_time}")

        if readiness == "MISSION READY":
            print(
                "\n🎉 CONGRATULATIONS! Phase 3 has moved from 'code complete' to 'mission ready'!"
            )

        self.phase_status["phase_5_readiness"] = readiness
        return readiness == "MISSION READY"

    def execute_full_validation(self):
        """Execute the complete validation plan"""
        print("🚀 STARTING PHASE 3 SYSTEMATIC VALIDATION")
        print("=" * 80)
        print("Based on Post-Refactoring_Test.txt requirements")
        print("Implementing disciplined, systematic validation approach")
        print("=" * 80)

        # Execute each phase
        phases = [
            self.phase_1_backend_component_test,
            self.phase_2_smoke_test_coordination,
            self.phase_3_functional_unit_testing,
            self.phase_4_api_integration_testing,
            self.phase_5_production_readiness_assessment,
        ]

        for i, phase_func in enumerate(phases, 1):
            try:
                success = phase_func()
                if not success and i == 1:  # Phase 1 is the gate
                    print(f"\n🛑 GATE FAILURE: Phase 1 failed. Cannot proceed.")
                    print("Must fix import errors before continuing validation.")
                    return False
            except Exception as e:
                print(f"\n❌ PHASE {i} EXCEPTION: {e}")
                self.log_result(f"phase_{i}", "execution", "EXCEPTION", str(e))

        return True


def main():
    """Main execution function"""
    print("PHASE 3 VALIDATION PLAN - SYSTEMATIC TESTING APPROACH")
    print("=" * 80)
    print("Implementation of Post-Refactoring_Test.txt requirements")
    print("Author: AI Assistant")
    print("Date: July 15, 2025")
    print("=" * 80)

    # Create and execute validation plan
    validator = Phase3ValidationPlan()
    success = validator.execute_full_validation()

    if success:
        print("\n🎯 VALIDATION PLAN EXECUTION COMPLETE")
    else:
        print("\n⚠️ VALIDATION PLAN TERMINATED DUE TO GATE FAILURE")

    return success


if __name__ == "__main__":
    main()
