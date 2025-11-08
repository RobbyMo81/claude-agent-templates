#!/usr/bin/env python3
"""
Helios Path Auto-Fix Script
Automatically fixes critical path issues identified by the path analyzer
"""

import os
import re
from pathlib import Path
from typing import Dict, List

class HeliosPathFixer:
    """Automated fixer for Helios path issues"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.fixes_applied = 0
        self.files_modified = []
        
        # Define fix patterns
        self.fix_patterns = {
            # Fix backend/venv to venv
            'backend_venv_fix': {
                'pattern': re.compile(r'([/\\])backend([/\\])venv([/\\]?)'),
                'replacement': r'\1venv\3',
                'description': 'Fix backend/venv to root-level venv'
            },
            
            # Fix Python executable paths
            'python_exe_fix': {
                'pattern': re.compile(r'backend[/\\]venv[/\\]Scripts[/\\]python\.exe'),
                'replacement': 'venv/Scripts/python.exe',
                'description': 'Fix Python executable path'
            },
            
            # Fix PowerShell paths with proper escaping
            'powershell_path_fix': {
                'pattern': re.compile(r'C:/Users/RobMo/OneDrive/Documents/helios/backend/venv/Scripts/python\.exe'),
                'replacement': 'C:/Users/RobMo/OneDrive/Documents/helios/venv/Scripts/python.exe',
                'description': 'Fix hardcoded PowerShell Python path'
            }
        }
    
    def fix_file(self, file_path: Path) -> bool:
        """Fix path issues in a single file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            original_content = content
            fixes_in_file = 0
            
            # Apply each fix pattern
            for fix_name, fix_info in self.fix_patterns.items():
                old_content = content
                content = fix_info['pattern'].sub(fix_info['replacement'], content)
                
                if content != old_content:
                    fixes_in_file += 1
                    print(f"  ✓ Applied {fix_info['description']}")
            
            # Write back if changes were made
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                self.fixes_applied += fixes_in_file
                self.files_modified.append(str(file_path.relative_to(self.project_root)))
                return True
            
            return False
            
        except Exception as e:
            print(f"  ❌ Error fixing {file_path}: {e}")
            return False
    
    def fix_critical_files(self) -> None:
        """Fix critical files identified by the analyzer"""
        
        critical_files = [
            "DEVELOPMENT_ISSUES_RESOLVED.md",
            "NPM Enviroment Startup.txt"
        ]
        
        print("🔧 APPLYING CRITICAL PATH FIXES")
        print("=" * 50)
        
        for filename in critical_files:
            file_path = self.project_root / filename
            if file_path.exists():
                print(f"\n📁 Fixing: {filename}")
                if self.fix_file(file_path):
                    print(f"  ✅ Fixed: {filename}")
                else:
                    print(f"  ℹ️  No changes needed: {filename}")
            else:
                print(f"  ⚠️  File not found: {filename}")
    
    def print_summary(self) -> None:
        """Print summary of fixes applied"""
        print("\n" + "=" * 60)
        print("📊 FIX SUMMARY")
        print("=" * 60)
        print(f"Files Modified: {len(self.files_modified)}")
        print(f"Total Fixes Applied: {self.fixes_applied}")
        
        if self.files_modified:
            print("\n📋 Modified Files:")
            for file_path in self.files_modified:
                print(f"  • {file_path}")
        
        print("\n✅ Critical path fixes completed!")
        print("💡 Next steps:")
        print("  1. Run system validation to verify fixes")
        print("  2. Test VS Code tasks and Python environment")
        print("  3. Check that server startup works correctly")


def main():
    """Main execution function"""
    project_root = r"C:\Users\RobMo\OneDrive\Documents\helios"
    
    print("🚀 Helios Path Auto-Fix")
    print("This script will fix critical path issues in your Helios project")
    print(f"Project root: {project_root}")
    print()
    
    # Create fixer and apply fixes
    fixer = HeliosPathFixer(project_root)
    fixer.fix_critical_files()
    fixer.print_summary()


if __name__ == "__main__":
    main()
