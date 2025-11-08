#!/usr/bin/env python3
"""
Unit tests for server.py
Tests the Flask application initialization and basic functionality
"""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock
import tempfile

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))


class TestServerInitialization(unittest.TestCase):
    """Test server initialization and basic functionality"""

    def setUp(self):
        """Set up test environment"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        self.temp_db.close()

    def tearDown(self):
        """Clean up test environment"""
        try:
            os.unlink(self.temp_db.name)
        except:
            pass

    def test_server_import(self):
        """Test that server.py can be imported without errors"""
        try:
            import server

            self.assertTrue(hasattr(server, "app"))
            self.assertTrue(hasattr(server, "DEPENDENCIES_AVAILABLE"))
            print("✅ Server import successful")
        except Exception as e:
            self.fail(f"Failed to import server: {e}")

    def test_flask_app_creation(self):
        """Test that Flask app is created properly"""
        import server

        self.assertIsNotNone(server.app)
        self.assertEqual(server.app.name, "server")
        print("✅ Flask app creation successful")

    def test_dependencies_check(self):
        """Test dependencies availability check"""
        import server

        self.assertIsInstance(server.DEPENDENCIES_AVAILABLE, bool)
        print(f"✅ Dependencies available: {server.DEPENDENCIES_AVAILABLE}")

    def test_basic_routes_exist(self):
        """Test that basic routes are registered"""
        import server

        app = server.app

        # Get all registered routes
        routes = [rule.rule for rule in app.url_map.iter_rules()]

        # Check for essential routes
        essential_routes = [
            "/health",
            "/",
            "/api/models",
            "/api/metacognitive/assessment",
            "/api/decisions/make",
        ]

        for route in essential_routes:
            self.assertIn(route, routes, f"Essential route {route} not found")

        print(f"✅ All {len(essential_routes)} essential routes registered")
        print(f"   Total routes: {len(routes)}")

    def test_health_endpoint(self):
        """Test the health check endpoint"""
        import server

        app = server.app

        with app.test_client() as client:
            response = client.get("/health")
            self.assertEqual(response.status_code, 200)

            data = response.get_json()
            self.assertEqual(data["status"], "healthy")
            self.assertEqual(data["service"], "helios-backend")

        print("✅ Health endpoint working correctly")

    def test_root_endpoint(self):
        """Test the root endpoint"""
        import server

        app = server.app

        with app.test_client() as client:
            response = client.get("/")
            self.assertEqual(response.status_code, 200)

            data = response.get_json()
            self.assertEqual(data["service"], "Helios Backend API")
            self.assertIn("endpoints", data)
            self.assertIsInstance(data["endpoints"], list)

        print("✅ Root endpoint working correctly")

    def test_models_endpoint_basic(self):
        """Test the models endpoint basic functionality"""
        import server

        app = server.app

        with app.test_client() as client:
            response = client.get("/api/models")
            self.assertEqual(response.status_code, 200)

            data = response.get_json()
            self.assertIsInstance(data, list)

        print("✅ Models endpoint basic functionality working")

    def test_request_json_handling(self):
        """Test proper handling of request.json in POST endpoints"""
        import server

        app = server.app

        # Test metacognitive assessment with null data
        with app.test_client() as client:
            response = client.post(
                "/api/metacognitive/assessment", content_type="application/json"
            )
            # Print response for debugging
            print(f"Response status: {response.status_code}")
            if response.status_code == 500:
                print(f"Response data: {response.get_json()}")
            # Should not crash, should return error or handle gracefully
            self.assertIn(
                response.status_code, [400, 503]
            )  # Bad request or service unavailable

        print("✅ Request JSON handling test passed")

    def test_error_handlers(self):
        """Test error handlers"""
        import server

        app = server.app

        with app.test_client() as client:
            # Test 404 handler
            response = client.get("/nonexistent-endpoint")
            self.assertEqual(response.status_code, 404)

            data = response.get_json()
            self.assertEqual(data["error"], "Endpoint not found")

        print("✅ Error handlers working correctly")


class TestServerWithDependencies(unittest.TestCase):
    """Test server functionality when dependencies are available"""

    def test_server_with_dependencies(self):
        """Test server behavior when ML dependencies are available"""
        import server

        if server.DEPENDENCIES_AVAILABLE:
            # Test that components are initialized
            self.assertIsNotNone(server.memory_store)
            self.assertIsNotNone(server.metacognitive_engine)
            self.assertIsNotNone(server.decision_engine)
            self.assertIsNotNone(server.cross_model_analytics)
            print("✅ All Phase 3 & 4 components initialized successfully")
        else:
            # Test that components are None when dependencies not available
            self.assertIsNone(server.memory_store)
            self.assertIsNone(server.metacognitive_engine)
            self.assertIsNone(server.decision_engine)
            self.assertIsNone(server.cross_model_analytics)
            print("⚠️ ML dependencies not available - running in mock mode")


def main():
    """Run all server tests"""
    print("🧪 RUNNING SERVER.PY UNIT TESTS")
    print("=" * 50)

    # Create test suite
    suite = unittest.TestSuite()

    # Add test cases
    suite.addTests(
        unittest.TestLoader().loadTestsFromTestCase(TestServerInitialization)
    )
    suite.addTests(
        unittest.TestLoader().loadTestsFromTestCase(TestServerWithDependencies)
    )

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("🎉 ALL SERVER TESTS PASSED!")
        print(f"✅ Ran {result.testsRun} tests successfully")
    else:
        print("❌ SOME TESTS FAILED!")
        print(f"Failed: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")

        # Print details of failures
        for test, trace in result.failures + result.errors:
            print(f"\n❌ {test}: {trace}")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
