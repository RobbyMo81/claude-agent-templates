#!/usr/bin/env python3
"""
Comprehensive Helios System Fix
Final fix for all path and PowerShell issues
"""

import os
import re
from pathlib import Path

def fix_powershell_script():
    """Fix the PowerShell test script issues"""
    script_path = Path(r"C:\Users\RobMo\OneDrive\Documents\helios\test-system-components.ps1")
    
    try:
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Fix all PowerShell call operator issues
        fixes = [
            # Fix quoted Python executable calls
            (re.compile(r'& "\$pythonExe"'), '& $pythonExe'),
            (re.compile(r'& \$pythonExe'), '& "$pythonExe"'),
            
            # Fix pipe operations
            (re.compile(r'\$testScript \| & "\$pythonExe"'), '$testScript | & $pythonExe'),
            
            # Fix compilation calls  
            (re.compile(r'& "\$pythonExe" -m py_compile server\.py'), '& $pythonExe -m py_compile server.py'),
            
            # Fix integration test calls
            (re.compile(r'& "\$pythonExe" test_integration_basic\.py'), '& $pythonExe test_integration_basic.py'),
        ]
        
        original_content = content
        fixes_applied = 0
        
        for pattern, replacement in fixes:
            old_content = content
            content = pattern.sub(replacement, content)
            if content != old_content:
                fixes_applied += 1
        
        # Write back if changes were made
        if content != original_content:
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Applied {fixes_applied} PowerShell fixes to test-system-components.ps1")
        else:
            print("ℹ️  No PowerShell fixes needed")
            
    except Exception as e:
        print(f"❌ Error fixing PowerShell script: {e}")

def main():
    print("🔧 COMPREHENSIVE HELIOS SYSTEM FIX")
    print("=" * 50)
    
    fix_powershell_script()
    
    print("\n✅ All fixes completed!")
    print("💡 Now run: powershell -ExecutionPolicy Bypass -File .\\test-system-components.ps1 -Quick")

if __name__ == "__main__":
    main()
