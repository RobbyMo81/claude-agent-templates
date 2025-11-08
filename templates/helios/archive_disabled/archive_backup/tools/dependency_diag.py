"""
Helios Dependency Diagnostics
Comprehensive diagnostics for ML dependency issues and mock-mode fallback logic.
"""

import os
import sys
import platform
import importlib
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dependency_diag.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

ML_DEPENDENCIES = [
    'numpy', 'torch', 'sklearn', 'pandas', 
    'matplotlib', 'scipy', 'torchvision'
]

HELIOS_MODULES = [
    'agent', 'trainer', 'memory_store', 'metacognition', 
    'decision_engine', 'cross_model_analytics'
]

class DependencyDiagnostics:
    """Comprehensive dependency diagnostics for Helios."""
    
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'python_env': {},
            'path_issues': {},
            'import_tests': {},
            'env_vars': {},
            'helios_modules': {},
            'recommendations': []
        }
    
    def banner(self, title: str) -> None:
        """Print a formatted banner."""
        border = "=" * 80
        print(f"\n{border}")
        print(f"{title.upper().center(80)}")
        print(f"{border}")
        logger.info(f"Starting: {title}")
    
    def check_python_env(self) -> None:
        """Check Python environment details."""
        self.banner("Python Environment Analysis")
        
        env_info = {
            'python_version': platform.python_version(),
            'python_implementation': platform.python_implementation(),
            'executable_path': sys.executable,
            'virtual_env_active': 'venv' in sys.executable.lower() or 'VIRTUAL_ENV' in os.environ,
            'virtual_env_path': os.environ.get('VIRTUAL_ENV', 'Not Set'),
            'platform': platform.platform(),
            'architecture': platform.architecture()[0],
            'sys_path_entries': len(sys.path),
            'working_directory': str(Path.cwd())
        }
        
        self.results['python_env'] = env_info
        
        print(f"Python Version      : {env_info['python_version']}")
        print(f"Implementation      : {env_info['python_implementation']}")
        print(f"Executable Path     : {env_info['executable_path']}")
        print(f"Virtual Environment : {env_info['virtual_env_active']}")
        if env_info['virtual_env_active']:
            print(f"Virtual Env Path    : {env_info['virtual_env_path']}")
        print(f"Platform            : {env_info['platform']}")
        print(f"Architecture        : {env_info['architecture']}")
        print(f"Working Directory   : {env_info['working_directory']}")
        print(f"sys.path entries    : {env_info['sys_path_entries']}")
        
        logger.info(f"Python environment: {env_info['python_version']} on {env_info['platform']}")
    
    def check_cwd_shadowing(self) -> None:
        """Check for local folders that might shadow imports."""
        self.banner("Path Collision & Shadowing Analysis")
        
        cwd = Path.cwd()
        suspicious_dirs = []
        sys_path_issues = []
        
        # Check for shadowing directories
        for dep in ML_DEPENDENCIES + HELIOS_MODULES:
            local_path = cwd / dep
            if local_path.is_dir():
                suspicious_dirs.append({
                    'name': dep,
                    'path': str(local_path),
                    'has_init': (local_path / '__init__.py').exists(),
                    'file_count': len(list(local_path.rglob('*.py')))
                })
        
        # Check sys.path for potential issues
        for i, path in enumerate(sys.path):
            path_obj = Path(path)
            if path_obj.exists():
                if path_obj == cwd:
                    sys_path_issues.append(f"Current directory is in sys.path at position {i}")
                elif 'numpy' in str(path_obj).lower() and path_obj.is_dir():
                    # Check if it's a source directory
                    if (path_obj / 'numpy' / 'setup.py').exists():
                        sys_path_issues.append(f"Potential numpy source directory in sys.path: {path}")
        
        self.results['path_issues'] = {
            'suspicious_dirs': suspicious_dirs,
            'sys_path_issues': sys_path_issues,
            'cwd': str(cwd)
        }
        
        if suspicious_dirs:
            print("⚠️  DETECTED LOCAL FOLDERS THAT MAY SHADOW IMPORTS:")
            for item in suspicious_dirs:
                print(f"   📁 {item['name']}")
                print(f"      Path: {item['path']}")
                print(f"      Has __init__.py: {item['has_init']}")
                print(f"      Python files: {item['file_count']}")
                print()
        else:
            print("✅ No suspicious shadowing directories detected")
        
        if sys_path_issues:
            print("⚠️  SYS.PATH ISSUES DETECTED:")
            for issue in sys_path_issues:
                print(f"   • {issue}")
        else:
            print("✅ No sys.path issues detected")
            
        logger.info(f"Path analysis: {len(suspicious_dirs)} suspicious dirs, {len(sys_path_issues)} sys.path issues")
    
    def check_imports(self) -> None:
        """Test imports for all ML dependencies."""
        self.banner("ML Dependency Import Analysis")
        
        import_results = {}
        
        for lib in ML_DEPENDENCIES:
            try:
                module = importlib.import_module(lib)
                version = getattr(module, '__version__', 'Unknown')
                location = getattr(module, '__file__', 'Unknown')
                
                import_results[lib] = {
                    'status': 'success',
                    'version': version,
                    'location': location,
                    'error': None
                }
                print(f"{lib:12} : ✅ v{version}")
                print(f"{'':12}   📍 {location}")
                
            except Exception as e:
                import_results[lib] = {
                    'status': 'failed',
                    'version': None,
                    'location': None,
                    'error': f"{e.__class__.__name__}: {str(e)}"
                }
                print(f"{lib:12} : ❌ {e.__class__.__name__}: {str(e)}")
                
                # Try to get more specific information
                if 'numpy' in lib.lower() and 'source directory' in str(e):
                    self.results['recommendations'].append(
                        "CRITICAL: numpy source directory detected. Move out of numpy source or use different directory."
                    )
        
        self.results['import_tests'] = import_results
        
        # Summary
        successful = sum(1 for r in import_results.values() if r['status'] == 'success')
        total = len(import_results)
        print(f"\n📊 IMPORT SUMMARY: {successful}/{total} successful")
        
        logger.info(f"Import tests: {successful}/{total} successful")
    
    def check_helios_modules(self) -> None:
        """Test imports for Helios-specific modules."""
        self.banner("Helios Module Import Analysis")
        
        # Change to backend directory for Helios module imports
        original_cwd = Path.cwd()
        backend_dir = original_cwd / 'backend'
        
        if backend_dir.exists():
            os.chdir(backend_dir)
            print(f"📍 Changed to backend directory: {backend_dir}")
        
        helios_results = {}
        
        for module in HELIOS_MODULES:
            try:
                imported_module = importlib.import_module(module)
                location = getattr(imported_module, '__file__', 'Unknown')
                
                helios_results[module] = {
                    'status': 'success',
                    'location': location,
                    'error': None
                }
                print(f"{module:20} : ✅ Imported")
                print(f"{'':20}   📍 {location}")
                
            except Exception as e:
                helios_results[module] = {
                    'status': 'failed',
                    'location': None,
                    'error': f"{e.__class__.__name__}: {str(e)}"
                }
                print(f"{module:20} : ❌ {e.__class__.__name__}: {str(e)}")
        
        # Restore original directory
        os.chdir(original_cwd)
        
        self.results['helios_modules'] = helios_results
        
        # Summary
        successful = sum(1 for r in helios_results.values() if r['status'] == 'success')
        total = len(helios_results)
        print(f"\n📊 HELIOS MODULE SUMMARY: {successful}/{total} successful")
        
        logger.info(f"Helios module tests: {successful}/{total} successful")
    
    def check_env_vars(self) -> None:
        """Check relevant environment variables."""
        self.banner("Environment Variables Analysis")
        
        relevant_vars = [
            'PYTHONPATH', 'VIRTUAL_ENV', 'PATH', 'PYTHONIOENCODING',
            'CONDA_DEFAULT_ENV', 'CONDA_PREFIX', 'PIP_PREFIX'
        ]
        
        env_data = {}
        
        for var in relevant_vars:
            value = os.environ.get(var, 'Not Set')
            env_data[var] = value
            
            if value != 'Not Set':
                print(f"{var:18} : {value}")
                # Check for potential issues
                if var == 'PYTHONPATH' and 'numpy' in value.lower():
                    self.results['recommendations'].append(
                        f"PYTHONPATH contains numpy-related paths: {value}"
                    )
            else:
                print(f"{var:18} : {value}")
        
        self.results['env_vars'] = env_data
        logger.info("Environment variables checked")
    
    def generate_recommendations(self) -> None:
        """Generate recommendations based on findings."""
        self.banner("Recommendations & Solutions")
        
        # Analyze results and generate recommendations
        if self.results['path_issues']['suspicious_dirs']:
            for suspicious in self.results['path_issues']['suspicious_dirs']:
                if suspicious['name'] in ML_DEPENDENCIES:
                    self.results['recommendations'].append(
                        f"Remove or rename local '{suspicious['name']}' directory: {suspicious['path']}"
                    )
        
        failed_imports = [
            lib for lib, result in self.results['import_tests'].items()
            if result['status'] == 'failed'
        ]
        
        if failed_imports:
            self.results['recommendations'].append(
                f"Install missing ML dependencies: {', '.join(failed_imports)}"
            )
        
        # Check if this looks like a numpy source directory issue
        numpy_result = self.results['import_tests'].get('numpy', {})
        if (numpy_result.get('status') == 'failed' and 
            'source directory' in str(numpy_result.get('error', ''))):
            self.results['recommendations'].append(
                "CRITICAL: You are running from numpy source directory. Change to a different directory."
            )
        
        # Print recommendations
        if self.results['recommendations']:
            print("🔧 RECOMMENDED ACTIONS:")
            for i, rec in enumerate(self.results['recommendations'], 1):
                print(f"   {i}. {rec}")
        else:
            print("✅ No critical issues detected")
        
        # Determine expected server behavior
        ml_available = all(
            result['status'] == 'success' 
            for result in self.results['import_tests'].values()
        )
        
        print(f"\n🎯 EXPECTED SERVER BEHAVIOR:")
        print(f"   DEPENDENCIES_AVAILABLE = {ml_available}")
        print(f"   trainer = {'ModelTrainer instance' if ml_available else 'None'}")
        print(f"   Mock mode = {'Disabled' if ml_available else 'Enabled'}")
        
        logger.info(f"Analysis complete. ML dependencies available: {ml_available}")
    
    def save_report(self, filename: str = "dependency_report.json") -> None:
        """Save detailed report to JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\n💾 Detailed report saved to: {filename}")
        logger.info(f"Report saved to {filename}")
    
    def run_all(self) -> Dict[str, Any]:
        """Run all diagnostic checks."""
        print("🔍 HELIOS DEPENDENCY DIAGNOSTICS")
        print(f"Started at: {self.results['timestamp']}")
        
        try:
            self.check_python_env()
            self.check_cwd_shadowing() 
            self.check_imports()
            self.check_helios_modules()
            self.check_env_vars()
            self.generate_recommendations()
            self.save_report()
            
            print("\n✔️  Dependency diagnostics completed successfully")
            logger.info("Diagnostics completed successfully")
            
        except Exception as e:
            print(f"\n❌ Diagnostics failed: {e}")
            logger.error(f"Diagnostics failed: {e}")
            raise
        
        return self.results

def main():
    """Main entry point."""
    diag = DependencyDiagnostics()
    return diag.run_all()

if __name__ == "__main__":
    main()
