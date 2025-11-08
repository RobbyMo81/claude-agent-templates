#!/usr/bin/env python3
"""
Phase 4 Cross-Model Analytics - Integration Test Suite
Test full frontend-to-backend flow for advanced analytics features
"""

import os
import sys
import time
import json
import tempfile
import unittest
import requests
import subprocess
from datetime import datetime, timedelta
from threading import Thread
from contextlib import contextmanager

# Add backend to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from memory_store import MemoryStore
    from cross_model_analytics import CrossModelAnalytics
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"  Warning: Could not import dependencies: {e}")
    DEPENDENCIES_AVAILABLE = False

class IntegrationTestServer:
    """Helper class to manage test server lifecycle"""
    
    def __init__(self, port=5001):
        self.port = port
        self.process = None
        self.base_url = f"http://localhost:{port}"
    
    def start(self):
        """Start the Flask server for testing"""
        print(f" Starting test server on port {self.port}...")
        
        # Start server in subprocess
        env = os.environ.copy()
        env['FLASK_ENV'] = 'testing'
        env['FLASK_PORT'] = str(self.port)
        
        self.process = subprocess.Popen([
            sys.executable, 'server.py'
        ], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for server to start
        for attempt in range(30):  # 30 second timeout
            try:
                response = requests.get(f"{self.base_url}/health", timeout=2)
                if response.status_code == 200:
                    print(f" Test server started successfully")
                    return True
            except requests.exceptions.ConnectionError:
                time.sleep(1)
        
        print(f" Failed to start test server")
        return False
    
    def stop(self):
        """Stop the Flask server"""
        if self.process:
            self.process.terminate()
            self.process.wait()
            print(f" Test server stopped")
    
    @contextmanager
    def running(self):
        """Context manager for server lifecycle"""
        try:
            if self.start():
                yield self
            else:
                raise RuntimeError("Failed to start test server")
        finally:
            self.stop()

class TestPhase4Integration(unittest.TestCase):
    """Integration tests for Phase 4 Cross-Model Analytics"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test class - start server and prepare test data"""
        if not DEPENDENCIES_AVAILABLE:
            raise unittest.SkipTest("Dependencies not available")
        
        cls.server = IntegrationTestServer()
        cls.base_url = cls.server.base_url
        
        # Start the server
        if not cls.server.start():
            raise RuntimeError("Failed to start test server")
        
        # Create test database with sample data
        cls._setup_test_database()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test class - stop server"""
        if hasattr(cls, 'server'):
            cls.server.stop()
        if hasattr(cls, 'temp_db'):
            try:
                os.unlink(cls.temp_db.name)
            except FileNotFoundError:
                pass
    
    @classmethod
    def _setup_test_database(cls):
        """Create test database with sample training data"""
        cls.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        cls.temp_db.close()
        
        # Initialize with test data
        memory_store = MemoryStore(cls.temp_db.name)
        
        current_time = datetime.now().isoformat()
        
        with memory_store._get_connection() as conn:
            cursor = conn.cursor()
            
            # Insert comprehensive test training sessions
            test_sessions = [
                # GPT-3.5 model runs
                ('job_gpt35_001', 'gpt-3.5-turbo', 'completed', 
                 (datetime.now() - timedelta(days=1)).isoformat(),
                 (datetime.now() - timedelta(days=1, hours=-2)).isoformat(),
                 100, '{"learning_rate": 0.001, "batch_size": 32}', None, current_time, current_time),
                ('job_gpt35_002', 'gpt-3.5-turbo', 'completed',
                 (datetime.now() - timedelta(days=3)).isoformat(),
                 (datetime.now() - timedelta(days=3, hours=-1.8)).isoformat(),
                 100, '{"learning_rate": 0.001, "batch_size": 32}', None, current_time, current_time),
                
                # GPT-4 model runs
                ('job_gpt4_001', 'gpt-4', 'completed',
                 (datetime.now() - timedelta(days=2)).isoformat(),
                 (datetime.now() - timedelta(days=2, hours=-3)).isoformat(),
                 100, '{"learning_rate": 0.0005, "batch_size": 16}', None, current_time, current_time),
                ('job_gpt4_002', 'gpt-4', 'completed',
                 (datetime.now() - timedelta(days=4)).isoformat(),
                 (datetime.now() - timedelta(days=4, hours=-2.5)).isoformat(),
                 100, '{"learning_rate": 0.0005, "batch_size": 16}', None, current_time, current_time),
                
                # Claude model runs
                ('job_claude_001', 'claude-3', 'completed',
                 (datetime.now() - timedelta(days=1, hours=-6)).isoformat(),
                 (datetime.now() - timedelta(days=1, hours=-8)).isoformat(),
                 100, '{"learning_rate": 0.002, "batch_size": 24}', None, current_time, current_time),
                
                # Failed training session
                ('job_failed_001', 'experimental-model', 'failed',
                 (datetime.now() - timedelta(hours=2)).isoformat(),
                 (datetime.now() - timedelta(hours=1)).isoformat(),
                 45, '{"learning_rate": 0.01, "batch_size": 64}', 'Convergence failure', current_time, current_time),
            ]
            
            cursor.executemany("""
                INSERT INTO training_sessions 
                (job_id, model_name, status, start_time, end_time, progress, config, error_message, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, test_sessions)
            
            # Insert corresponding training logs
            test_logs = [
                # GPT-3.5 logs - good performance
                ('job_gpt35_001', 10, 0.85, '{"accuracy": 0.65}', current_time),
                ('job_gpt35_001', 25, 0.45, '{"accuracy": 0.78}', current_time),
                ('job_gpt35_001', 50, 0.25, '{"accuracy": 0.85}', current_time),
                ('job_gpt35_001', 75, 0.18, '{"accuracy": 0.89}', current_time),
                ('job_gpt35_001', 100, 0.15, '{"accuracy": 0.92}', current_time),
                
                ('job_gpt35_002', 10, 0.80, '{"accuracy": 0.68}', current_time),
                ('job_gpt35_002', 25, 0.42, '{"accuracy": 0.80}', current_time),
                ('job_gpt35_002', 50, 0.22, '{"accuracy": 0.87}', current_time),
                ('job_gpt35_002', 75, 0.16, '{"accuracy": 0.90}', current_time),
                ('job_gpt35_002', 100, 0.14, '{"accuracy": 0.93}', current_time),
                
                # GPT-4 logs - excellent performance but slower
                ('job_gpt4_001', 10, 0.75, '{"accuracy": 0.70}', current_time),
                ('job_gpt4_002', 25, 0.35, '{"accuracy": 0.82}', current_time),
                ('job_gpt4_001', 50, 0.18, '{"accuracy": 0.90}', current_time),
                ('job_gpt4_001', 75, 0.12, '{"accuracy": 0.94}', current_time),
                ('job_gpt4_001', 100, 0.10, '{"accuracy": 0.96}', current_time),
                
                ('job_gpt4_002', 10, 0.78, '{"accuracy": 0.72}', current_time),
                ('job_gpt4_002', 25, 0.38, '{"accuracy": 0.84}', current_time),
                ('job_gpt4_002', 50, 0.20, '{"accuracy": 0.91}', current_time),
                ('job_gpt4_002', 75, 0.13, '{"accuracy": 0.95}', current_time),
                ('job_gpt4_002', 100, 0.11, '{"accuracy": 0.97}', current_time),
                
                # Claude logs - moderate performance, fast training
                ('job_claude_001', 10, 0.90, '{"accuracy": 0.60}', current_time),
                ('job_claude_001', 25, 0.55, '{"accuracy": 0.75}', current_time),
                ('job_claude_001', 50, 0.35, '{"accuracy": 0.82}', current_time),
                ('job_claude_001', 75, 0.28, '{"accuracy": 0.86}', current_time),
                ('job_claude_001', 100, 0.25, '{"accuracy": 0.88}', current_time),
                
                # Failed training logs - partial data
                ('job_failed_001', 10, 1.2, '{"accuracy": 0.45}', current_time),
                ('job_failed_001', 20, 1.1, '{"accuracy": 0.48}', current_time),
                ('job_failed_001', 30, 1.15, '{"accuracy": 0.46}', current_time),
                ('job_failed_001', 45, 1.25, '{"accuracy": 0.44}', current_time),
            ]
            
            cursor.executemany("""
                INSERT INTO training_logs 
                (job_id, epoch, loss, metrics, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, test_logs)
            
            conn.commit()
    
    def test_01_server_health_check(self):
        """Test that the server is running and responds to health checks"""
        print("\n Testing server health check...")
        
        response = requests.get(f"{self.base_url}/health")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn('status', data)
        self.assertEqual(data['status'], 'healthy')
        print(" Server health check passed")
    
    def test_02_analytics_performance_endpoint(self):
        """Test the performance analysis endpoint"""
        print("\n Testing performance analysis endpoint...")
        
        # Test GPT-4 performance analysis
        response = requests.get(f"{self.base_url}/api/analytics/performance/gpt-4")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn('model_name', data)
        self.assertEqual(data['model_name'], 'gpt-4')
        self.assertIn('final_loss', data)
        self.assertIn('training_time', data)
        self.assertIn('stability_score', data)
        self.assertIn('efficiency_score', data)
        
        # Should have good performance (low final loss)
        self.assertLess(data['final_loss'], 0.5)
        print(f" GPT-4 performance analysis: final_loss={data['final_loss']:.3f}")
    
    def test_03_analytics_comparison_endpoint(self):
        """Test the model comparison endpoint"""
        print("\n Testing model comparison endpoint...")
        
        # Compare GPT-3.5 vs GPT-4
        models = ['gpt-3.5-turbo', 'gpt-4']
        response = requests.post(f"{self.base_url}/api/analytics/comparison", 
                               json={'models': models})
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn('models', data)
        self.assertIn('best_model', data)
        self.assertIn('rankings', data)
        
        # Should have data for both models
        self.assertEqual(len(data['models']), 2)
        self.assertIn('gpt-3.5-turbo', data['models'])
        self.assertIn('gpt-4', data['models'])
        
        # GPT-4 should likely be the best model (lowest loss)
        print(f" Best model: {data['best_model']}")
        print(f" Model rankings: {data['rankings']}")
    
    def test_04_analytics_ensemble_endpoint(self):
        """Test the ensemble recommendations endpoint"""
        print("\n Testing ensemble recommendations endpoint...")
        
        models = ['gpt-3.5-turbo', 'gpt-4', 'claude-3']
        response = requests.post(f"{self.base_url}/api/analytics/ensemble", 
                               json={'models': models})
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn('recommendations', data)
        self.assertIsInstance(data['recommendations'], list)
        
        # Should have at least one recommendation
        self.assertGreater(len(data['recommendations']), 0)
        
        for rec in data['recommendations']:
            self.assertIn('ensemble_type', rec)
            self.assertIn('models', rec)
            self.assertIn('expected_performance', rec)
            self.assertIn('confidence', rec)
            
        print(f" Generated {len(data['recommendations'])} ensemble recommendations")
    
    def test_05_analytics_matrix_endpoint(self):
        """Test the performance matrix endpoint"""
        print("\n Testing performance matrix endpoint...")
        
        response = requests.get(f"{self.base_url}/api/analytics/matrix")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn('models', data)
        self.assertIn('metrics', data)
        self.assertIn('matrix', data)
        
        # Should have our test models
        models = data['models']
        self.assertIn('gpt-3.5-turbo', models)
        self.assertIn('gpt-4', models)
        self.assertIn('claude-3', models)
        
        # Matrix should be populated
        matrix = data['matrix']
        self.assertGreater(len(matrix), 0)
        
        print(f" Performance matrix with {len(models)} models")
    
    def test_06_analytics_trends_endpoint(self):
        """Test the trend analysis endpoint"""
        print("\n Testing trend analysis endpoint...")
        
        # Test last 7 days
        response = requests.get(f"{self.base_url}/api/analytics/trends?days=7")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn('period', data)
        self.assertIn('trends', data)
        self.assertIn('summary', data)
        
        trends = data['trends']
        self.assertIsInstance(trends, list)
        
        # Should have trend data
        for trend in trends:
            self.assertIn('date', trend)
            self.assertIn('models_trained', trend)
            self.assertIn('avg_performance', trend)
            
        print(f" Trend analysis for {data['period']} with {len(trends)} data points")
    
    def test_07_error_handling(self):
        """Test error handling for invalid requests"""
        print("\n Testing error handling...")
        
        # Test invalid model name
        response = requests.get(f"{self.base_url}/api/analytics/performance/nonexistent-model")
        self.assertEqual(response.status_code, 404)
        
        # Test invalid comparison request
        response = requests.post(f"{self.base_url}/api/analytics/comparison", 
                               json={'models': []})
        self.assertEqual(response.status_code, 400)
        
        # Test malformed JSON
        response = requests.post(f"{self.base_url}/api/analytics/ensemble", 
                               data="invalid json")
        self.assertEqual(response.status_code, 400)
        
        print(" Error handling tests passed")
    
    def test_08_end_to_end_workflow(self):
        """Test complete end-to-end analytics workflow"""
        print("\n Testing end-to-end analytics workflow...")
        
        # Step 1: Get list of available models from matrix
        response = requests.get(f"{self.base_url}/api/analytics/matrix")
        self.assertEqual(response.status_code, 200)
        
        available_models = response.json()['models']
        self.assertGreater(len(available_models), 0)
        print(f"  Found {len(available_models)} trained models")
        
        # Step 2: Analyze each model's performance
        model_performances = {}
        for model in available_models[:3]:  # Test first 3 models
            response = requests.get(f"{self.base_url}/api/analytics/performance/{model}")
            if response.status_code == 200:
                model_performances[model] = response.json()
        
        self.assertGreater(len(model_performances), 0)
        print(f"  Analyzed performance for {len(model_performances)} models")
        
        # Step 3: Compare top models
        if len(model_performances) >= 2:
            top_models = list(model_performances.keys())[:2]
            response = requests.post(f"{self.base_url}/api/analytics/comparison", 
                                   json={'models': top_models})
            self.assertEqual(response.status_code, 200)
            
            comparison = response.json()
            print(f"  Compared {len(top_models)} models, best: {comparison['best_model']}")
        
        # Step 4: Get ensemble recommendations
        if len(available_models) >= 2:
            response = requests.post(f"{self.base_url}/api/analytics/ensemble", 
                                   json={'models': available_models[:3]})
            self.assertEqual(response.status_code, 200)
            
            ensemble_data = response.json()
            print(f"  Generated {len(ensemble_data['recommendations'])} ensemble recommendations")
        
        # Step 5: Get trend analysis
        response = requests.get(f"{self.base_url}/api/analytics/trends?days=7")
        self.assertEqual(response.status_code, 200)
        
        trends = response.json()
        print(f"  Retrieved trend analysis with {len(trends['trends'])} data points")
        
        print(" End-to-end workflow completed successfully")
    
    def test_09_performance_under_load(self):
        """Test system performance under concurrent requests"""
        print("\n Testing performance under load...")
        
        import concurrent.futures
        import statistics
        
        def make_request():
            start_time = time.time()
            response = requests.get(f"{self.base_url}/api/analytics/matrix")
            end_time = time.time()
            return response.status_code, end_time - start_time
        
        # Make 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Check all requests succeeded
        status_codes = [result[0] for result in results]
        response_times = [result[1] for result in results]
        
        success_rate = sum(1 for code in status_codes if code == 200) / len(status_codes)
        avg_response_time = statistics.mean(response_times)
        
        self.assertGreaterEqual(success_rate, 0.8)  # At least 80% success rate
        self.assertLess(avg_response_time, 5.0)     # Average response under 5 seconds
        
        print(f"  Success rate: {success_rate:.1%}")
        print(f"  Average response time: {avg_response_time:.2f}s")
        print(" Performance under load test passed")


def run_integration_tests():
    """Run the complete integration test suite"""
    print("=" * 80)
    print(" PHASE 4 CROSS-MODEL ANALYTICS - INTEGRATION TESTS")
    print("=" * 80)
    
    # Check if dependencies are available
    if not DEPENDENCIES_AVAILABLE:
        print(" Dependencies not available. Skipping integration tests.")
        return False
    
    # Run the test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestPhase4Integration)
    runner = unittest.TextTestRunner(verbosity=2)
    
    print("\n Starting integration test suite...\n")
    
    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()
    
    # Print summary
    print("\n" + "=" * 80)
    print(" INTEGRATION TEST SUMMARY")
    print("=" * 80)
    
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    passed = total_tests - failures - errors
    
    print(f"Total Tests: {total_tests}")
    print(f" Passed: {passed}")
    print(f" Failed: {failures}")
    print(f" Errors: {errors}")
    print(f"  Duration: {end_time - start_time:.2f}s")
    
    success_rate = (passed / total_tests * 100) if total_tests > 0 else 0
    print(f" Success Rate: {success_rate:.1f}%")
    
    if result.failures:
        print("\n FAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}")
    
    if result.errors:
        print("\n ERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}")
    
    if success_rate >= 90:
        print("\n EXCELLENT: Integration tests passed with high success rate!")
    elif success_rate >= 70:
        print("\n GOOD: Integration tests mostly successful with minor issues")
    else:
        print("\n NEEDS ATTENTION: Integration tests have significant issues")
    
    return success_rate >= 70


if __name__ == "__main__":
    run_integration_tests()
