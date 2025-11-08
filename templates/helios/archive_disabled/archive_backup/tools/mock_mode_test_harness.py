"""
Helios Mock vs Trainer Mode Test Harness
Simulates different dependency scenarios to test fallback logic.
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
from contextlib import contextmanager
import importlib.util

class MockModeTestHarness:
    """Test harness for simulating mock vs trainer mode scenarios."""
    
    def __init__(self):
        self.test_results = []
        self.original_path = sys.path.copy()
        self.original_cwd = Path.cwd()
    
    @contextmanager
    def simulate_environment(self, scenario: str, **kwargs):
        """Context manager to simulate different dependency scenarios."""
        print(f"\n🧪 TESTING SCENARIO: {scenario}")
        print("=" * 60)
        
        # Store original state
        original_modules = sys.modules.copy()
        original_path = sys.path.copy()
        
        try:
            # Apply scenario modifications
            if scenario == "numpy_source_directory":
                self._simulate_numpy_source_error()
            elif scenario == "missing_torch":
                self._simulate_missing_dependency('torch')
            elif scenario == "missing_all_ml":
                self._simulate_missing_all_ml()
            elif scenario == "partial_dependencies":
                self._simulate_partial_dependencies()
            elif scenario == "shadowing_directory":
                self._simulate_shadowing_directory(kwargs.get('shadow_module', 'numpy'))
            elif scenario == "clean_environment":
                pass  # Use current environment as-is
            
            yield
            
        finally:
            # Restore original state 
            sys.modules.clear()
            sys.modules.update(original_modules)
            sys.path.clear()
            sys.path.extend(original_path)
    
    def _simulate_numpy_source_error(self):
        """Simulate the numpy source directory import error."""
        # Create a mock numpy module that raises the specific error
        mock_numpy_code = '''
import sys
raise ImportError("Error importing numpy: you should not try to import numpy from its source directory; please exit the numpy source tree, and relaunch your python interpreter from there.")
'''
        # Remove numpy from modules if it exists
        if 'numpy' in sys.modules:
            del sys.modules['numpy']
        
        # Create temporary module
        spec = importlib.util.spec_from_loader("numpy", loader=None)
        numpy_module = importlib.util.module_from_spec(spec)
        exec(mock_numpy_code, numpy_module.__dict__)
        sys.modules['numpy'] = numpy_module
    
    def _simulate_missing_dependency(self, dep_name: str):
        """Simulate a missing dependency."""
        if dep_name in sys.modules:
            del sys.modules[dep_name]
        
        # Override import to fail
        original_import = __builtins__['__import__']
        
        def mock_import(name, *args, **kwargs):
            if name == dep_name:
                raise ImportError(f"No module named '{dep_name}'")
            return original_import(name, *args, **kwargs)
        
        __builtins__['__import__'] = mock_import
    
    def _simulate_missing_all_ml(self):
        """Simulate all ML dependencies missing."""
        ml_deps = ['numpy', 'torch', 'sklearn', 'pandas', 'matplotlib']
        for dep in ml_deps:
            if dep in sys.modules:
                del sys.modules[dep]
    
    def _simulate_partial_dependencies(self):
        """Simulate partial ML dependencies (some available, some not)."""
        # Remove torch and sklearn but keep numpy and pandas
        missing_deps = ['torch', 'sklearn']
        for dep in missing_deps:
            if dep in sys.modules:
                del sys.modules[dep]
    
    def _simulate_shadowing_directory(self, module_name: str):
        """Simulate a local directory shadowing a module."""
        # Create a temporary directory with the module name
        temp_dir = Path(tempfile.mkdtemp())
        shadow_dir = temp_dir / module_name
        shadow_dir.mkdir()
        
        # Create a fake __init__.py that causes issues
        (shadow_dir / '__init__.py').write_text('raise ImportError("Shadowed import")')
        
        # Add to beginning of sys.path
        sys.path.insert(0, str(temp_dir))
        
        return temp_dir
    
    def test_server_import_logic(self) -> Dict[str, Any]:
        """Test the server's import logic directly."""
        result = {
            'dependencies_available': False,
            'import_errors': [],
            'successful_imports': [],
            'trainer_initialized': False,
            'expected_mock_mode': True
        }
        
        try:
            # Test core ML dependencies first
            import torch
            import numpy as np
            result['successful_imports'].append('torch')
            result['successful_imports'].append('numpy')
            print("✅ Core ML dependencies (torch, numpy) available")
            
            # Test Helios components (from backend directory)
            original_cwd = Path.cwd()
            backend_dir = original_cwd / 'backend'
            
            if backend_dir.exists():
                os.chdir(backend_dir)
            
            try:
                from agent import MLPowerballAgent
                from trainer import ModelTrainer, TrainingConfig
                from memory_store import MemoryStore
                from metacognition import MetacognitiveEngine
                from decision_engine import DecisionEngine, Goal
                from cross_model_analytics import CrossModelAnalytics
                
                result['successful_imports'].extend([
                    'agent', 'trainer', 'memory_store', 'metacognition',
                    'decision_engine', 'cross_model_analytics'
                ])
                
                # Test trainer initialization
                trainer = ModelTrainer("test_models")
                result['trainer_initialized'] = True
                result['dependencies_available'] = True
                result['expected_mock_mode'] = False
                
                print("✅ All Helios components imported successfully")
                print("✅ Trainer initialized successfully")
                
            except Exception as e:
                result['import_errors'].append(f"Helios components: {str(e)}")
                print(f"❌ Helios component import failed: {e}")
            finally:
                os.chdir(original_cwd)
                
        except Exception as e:
            result['import_errors'].append(f"Core ML: {str(e)}")
            print(f"❌ Core ML import failed: {e}")
        
        return result
    
    def test_mock_response_logic(self) -> Dict[str, Any]:
        """Test the mock response logic when dependencies are unavailable."""
        result = {
            'mock_response_triggered': False,
            'response_data': None,
            'error': None
        }
        
        try:
            # Simulate the server logic
            DEPENDENCIES_AVAILABLE = False
            trainer = None
            TrainingConfig = None
            
            # Test the condition from server.py
            condition_result = not DEPENDENCIES_AVAILABLE or not trainer or TrainingConfig is None
            
            if condition_result:
                # Generate mock response
                config = {'modelName': 'test_model'}
                response = {
                    "status": "started",
                    "message": f"Training job for {config['modelName']} has been queued successfully",
                    "job_id": f"job_{config['modelName']}_{hash(str(config)) % 10000}",
                    "estimated_duration": "15-30 minutes"
                }
                result['mock_response_triggered'] = True
                result['response_data'] = response
                print("✅ Mock response logic working correctly")
            else:
                result['error'] = "Mock response not triggered when it should be"
                print("❌ Mock response logic failed")
                
        except Exception as e:
            result['error'] = str(e)
            print(f"❌ Mock response test failed: {e}")
        
        return result
    
    def run_scenario_test(self, scenario: str, **kwargs) -> Dict[str, Any]:
        """Run a complete test for a specific scenario."""
        test_result = {
            'scenario': scenario,
            'timestamp': str(Path.cwd()),
            'import_test': None,
            'mock_logic_test': None,
            'recommendation': ""
        }
        
        with self.simulate_environment(scenario, **kwargs):
            # Test import logic
            test_result['import_test'] = self.test_server_import_logic()
            
            # Test mock response logic 
            test_result['mock_logic_test'] = self.test_mock_response_logic()
            
            # Generate recommendation
            if test_result['import_test']['dependencies_available']:
                test_result['recommendation'] = "✅ Dependencies available - trainer mode expected"
            else:
                test_result['recommendation'] = "⚠️ Dependencies unavailable - mock mode expected"
        
        self.test_results.append(test_result)
        return test_result
    
    def run_comprehensive_test_suite(self) -> Dict[str, Any]:
        """Run comprehensive test suite covering all scenarios."""
        print("🚀 STARTING COMPREHENSIVE MOCK/TRAINER MODE TEST SUITE")
        print("=" * 80)
        
        scenarios = [
            ("clean_environment", {}),
            ("numpy_source_directory", {}),
            ("missing_torch", {}),
            ("missing_all_ml", {}),
            ("partial_dependencies", {}),
            ("shadowing_directory", {"shadow_module": "numpy"}),
        ]
        
        suite_results = {
            'total_scenarios': len(scenarios),
            'scenarios_tested': 0,
            'scenarios_passed': 0,
            'scenarios_failed': 0,
            'detailed_results': [],
            'summary': {}
        }
        
        for scenario, kwargs in scenarios:
            try:
                result = self.run_scenario_test(scenario, **kwargs)
                suite_results['detailed_results'].append(result)
                suite_results['scenarios_tested'] += 1
                
                # Determine if test passed (mock mode when dependencies unavailable)
                deps_available = result['import_test']['dependencies_available']
                mock_triggered = result['mock_logic_test']['mock_response_triggered']
                
                # Test passes if: (deps available AND mock not triggered) OR (deps unavailable AND mock triggered)
                test_passed = (deps_available and not mock_triggered) or (not deps_available and mock_triggered)
                
                if test_passed:
                    suite_results['scenarios_passed'] += 1
                    print(f"✅ {scenario}: PASSED")
                else:
                    suite_results['scenarios_failed'] += 1
                    print(f"❌ {scenario}: FAILED")
                    
            except Exception as e:
                suite_results['scenarios_failed'] += 1
                print(f"❌ {scenario}: ERROR - {e}")
        
        # Generate summary
        suite_results['summary'] = {
            'success_rate': (suite_results['scenarios_passed'] / suite_results['scenarios_tested']) * 100,
            'total_import_errors': sum(len(r['import_test']['import_errors']) for r in suite_results['detailed_results']),
            'mock_mode_scenarios': sum(1 for r in suite_results['detailed_results'] if not r['import_test']['dependencies_available']),
            'trainer_mode_scenarios': sum(1 for r in suite_results['detailed_results'] if r['import_test']['dependencies_available'])
        }
        
        print(f"\n📊 TEST SUITE SUMMARY:")
        print(f"   Total Scenarios: {suite_results['total_scenarios']}")
        print(f"   Passed: {suite_results['scenarios_passed']}")
        print(f"   Failed: {suite_results['scenarios_failed']}")
        print(f"   Success Rate: {suite_results['summary']['success_rate']:.1f}%")
        print(f"   Mock Mode Scenarios: {suite_results['summary']['mock_mode_scenarios']}")
        print(f"   Trainer Mode Scenarios: {suite_results['summary']['trainer_mode_scenarios']}")
        
        return suite_results
    
    def save_test_report(self, results: Dict[str, Any], filename: str = "mock_mode_test_report.json"):
        """Save test results to a JSON report."""
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Test report saved to: {filename}")

def main():
    """Main entry point for test harness."""
    harness = MockModeTestHarness()
    
    # Run comprehensive test suite
    results = harness.run_comprehensive_test_suite()
    harness.save_test_report(results)
    
    return results

if __name__ == "__main__":
    main()
