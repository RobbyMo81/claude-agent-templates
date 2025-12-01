"""
Eastern Time Migration Test Runner
Comprehensive test suite for validating Eastern Time implementation.
"""

import pytest
import sys
import os
from pathlib import Path
import json
from datetime import datetime
import pytz

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_eastern_time_tests():
    """Run comprehensive Eastern Time migration tests"""
    
    print("=" * 80)
    print("EASTERN TIME MIGRATION TEST SUITE")
    print("=" * 80)
    
    # Test configuration
    test_files = [
        "tests/test_eastern_time.py",
        "tests/test_eastern_time_integration.py", 
        "tests/test_system_eastern_time.py",
        "tests/test_e2e_eastern_time.py"
    ]
    
    # Prepare test environment
    et_tz = pytz.timezone('US/Eastern')
    current_et = et_tz.localize(datetime(2025, 8, 26, 14, 30, 0))
    
    print(f"Test Environment:")
    print(f"  Current ET Time: {current_et}")
    print(f"  Timezone: {et_tz}")
    print(f"  Project Root: {project_root}")
    print()
    
    # Run each test suite
    results = {}
    
    for test_file in test_files:
        test_path = project_root / test_file
        
        if test_path.exists():
            print(f"Running {test_file}...")
            print("-" * 40)
            
            # Run pytest for this file
            pytest_args = [
                str(test_path),
                "-v",  # Verbose output
                "--tb=short",  # Short traceback format
                "--no-header",  # No header
                "--maxfail=5"  # Stop after 5 failures
            ]
            
            try:
                exit_code = pytest.main(pytest_args)
                results[test_file] = {
                    'status': 'PASSED' if exit_code == 0 else 'FAILED',
                    'exit_code': exit_code
                }
                
                if exit_code == 0:
                    print(f"✓ {test_file} - PASSED")
                else:
                    print(f"✗ {test_file} - FAILED (exit code: {exit_code})")
                    
            except Exception as e:
                results[test_file] = {
                    'status': 'ERROR',
                    'error': str(e)
                }
                print(f"✗ {test_file} - ERROR: {e}")
            
            print()
            
        else:
            results[test_file] = {
                'status': 'NOT_FOUND',
                'error': f"Test file not found: {test_path}"
            }
            print(f"? {test_file} - NOT FOUND")
            print()
    
    # Summary
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for result in results.values() if result['status'] == 'PASSED')
    failed = sum(1 for result in results.values() if result['status'] == 'FAILED')
    errors = sum(1 for result in results.values() if result['status'] == 'ERROR')
    not_found = sum(1 for result in results.values() if result['status'] == 'NOT_FOUND')
    total = len(results)
    
    print(f"Total Test Files: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Errors: {errors}")
    print(f"Not Found: {not_found}")
    print()
    
    # Detailed results
    for test_file, result in results.items():
        status_symbol = {
            'PASSED': '✓',
            'FAILED': '✗',
            'ERROR': '!',
            'NOT_FOUND': '?'
        }.get(result['status'], '?')
        
        print(f"{status_symbol} {test_file}: {result['status']}")
        if 'error' in result:
            print(f"    {result['error']}")
    
    print()
    
    # Migration readiness assessment
    print("=" * 80)
    print("MIGRATION READINESS ASSESSMENT")
    print("=" * 80)
    
    if passed == total and total > 0:
        print("🟢 READY FOR MIGRATION")
        print("All tests passed. Eastern Time implementation is ready for deployment.")
        migration_status = "READY"
        
    elif passed > failed and passed > 0:
        print("🟡 PARTIALLY READY")
        print("Some tests passed. Review failures and fix before full migration.")
        migration_status = "PARTIAL"
        
    else:
        print("🔴 NOT READY")
        print("Significant test failures. Fix issues before attempting migration.")
        migration_status = "NOT_READY"
    
    # Save results
    test_results = {
        'timestamp': datetime.now().isoformat(),
        'migration_status': migration_status,
        'summary': {
            'total': total,
            'passed': passed,
            'failed': failed,
            'errors': errors,
            'not_found': not_found
        },
        'detailed_results': results,
        'recommendations': generate_recommendations(results)
    }
    
    results_file = project_root / 'eastern_time_test_results.json'
    with open(results_file, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"Detailed results saved to: {results_file}")
    print()
    
    return migration_status


def generate_recommendations(results):
    """Generate recommendations based on test results"""
    recommendations = []
    
    failed_tests = [test for test, result in results.items() if result['status'] in ['FAILED', 'ERROR']]
    
    if not failed_tests:
        recommendations.append({
            'priority': 'HIGH',
            'category': 'MIGRATION',
            'action': 'Proceed with Eastern Time migration',
            'description': 'All tests passed. Ready for production deployment.'
        })
    else:
        for test in failed_tests:
            if 'test_eastern_time.py' in test:
                recommendations.append({
                    'priority': 'CRITICAL',
                    'category': 'CORE_FUNCTIONS',
                    'action': 'Fix core Eastern Time functions',
                    'description': 'Core timeutils functions need implementation before proceeding.'
                })
            elif 'integration' in test:
                recommendations.append({
                    'priority': 'HIGH', 
                    'category': 'INTEGRATION',
                    'action': 'Fix module integration issues',
                    'description': 'Integration between modules needs attention.'
                })
            elif 'system' in test:
                recommendations.append({
                    'priority': 'MEDIUM',
                    'category': 'SYSTEM_WIDE',
                    'action': 'Address system-wide consistency',
                    'description': 'System-wide Eastern Time consistency needs work.'
                })
            elif 'e2e' in test:
                recommendations.append({
                    'priority': 'LOW',
                    'category': 'END_TO_END',
                    'action': 'Fix end-to-end workflow issues',
                    'description': 'End-to-end workflows may need adjustment after core fixes.'
                })
    
    # Add general recommendations
    recommendations.append({
        'priority': 'MEDIUM',
        'category': 'BACKUP',
        'action': 'Create system backup before migration',
        'description': 'Backup current UTC-based system before applying changes.'
    })
    
    recommendations.append({
        'priority': 'LOW',
        'category': 'MONITORING',
        'action': 'Set up post-migration monitoring',
        'description': 'Monitor system behavior after Eastern Time migration.'
    })
    
    return recommendations


def check_prerequisites():
    """Check if prerequisites for testing are met"""
    print("Checking prerequisites...")
    
    prerequisites = []
    
    # Check if required modules exist
    required_modules = [
        'app/utils/timeutils.py',
        'app/production_provider.py',
        'app/auth.py',
        'config.toml'
    ]
    
    for module in required_modules:
        module_path = project_root / module
        if module_path.exists():
            prerequisites.append(f"✓ {module} - EXISTS")
        else:
            prerequisites.append(f"? {module} - NOT FOUND")
    
    # Check if test environment is set up
    if (project_root / '.venv').exists():
        prerequisites.append("✓ Virtual environment - EXISTS")
    else:
        prerequisites.append("? Virtual environment - NOT FOUND")
    
    # Check if pytest is available
    try:
        import pytest
        prerequisites.append("✓ pytest - AVAILABLE")
    except ImportError:
        prerequisites.append("✗ pytest - NOT AVAILABLE")
    
    # Check if required packages are available
    required_packages = ['pytz', 'datetime']
    for package in required_packages:
        try:
            __import__(package)
            prerequisites.append(f"✓ {package} - AVAILABLE")
        except ImportError:
            prerequisites.append(f"✗ {package} - NOT AVAILABLE")
    
    print("\nPrerequisites:")
    for prereq in prerequisites:
        print(f"  {prereq}")
    
    missing_critical = sum(1 for prereq in prerequisites if prereq.startswith('✗'))
    
    if missing_critical > 0:
        print(f"\n⚠️  {missing_critical} critical prerequisites missing!")
        print("Install missing packages and ensure environment is set up correctly.")
        return False
    else:
        print("\n✓ All critical prerequisites met.")
        return True


if __name__ == '__main__':
    print("Eastern Time Migration Test Runner")
    print(f"Project: {project_root}")
    print(f"Python: {sys.version}")
    print()
    
    # Check prerequisites first
    if not check_prerequisites():
        print("Prerequisites not met. Please fix issues and try again.")
        sys.exit(1)
    
    print()
    
    # Run the test suite
    try:
        migration_status = run_eastern_time_tests()
        
        if migration_status == "READY":
            sys.exit(0)  # Success
        elif migration_status == "PARTIAL":
            sys.exit(1)  # Partial success - review needed
        else:
            sys.exit(2)  # Not ready - fixes needed
            
    except KeyboardInterrupt:
        print("\n\nTest run interrupted by user.")
        sys.exit(3)
    except Exception as e:
        print(f"\n\nUnexpected error during test run: {e}")
        sys.exit(4)
