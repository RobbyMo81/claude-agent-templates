#!/usr/bin/env python3
"""
Simple syntax checker for function definitions
"""

import ast
import os
from pathlib import Path

def check_python_syntax(file_path):
    """Check if a Python file has valid syntax"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Try to parse the file
        ast.parse(content)
        return True, "OK"
    except SyntaxError as e:
        return False, f"Syntax error at line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, f"Error: {str(e)}"

def main():
    """Check all Python files for syntax errors"""
    project_root = Path('.')
    python_files = list(project_root.glob('backend/*.py'))
    
    print("Checking Python syntax...")
    
    errors_found = 0
    for file_path in python_files:
        if file_path.name.startswith('.'):
            continue
            
        is_valid, message = check_python_syntax(file_path)
        if not is_valid:
            print(f"ERROR in {file_path}: {message}")
            errors_found += 1
        else:
            print(f"OK: {file_path}")
    
    print(f"\nSummary: {errors_found} files with syntax errors out of {len(python_files)} checked")
    return errors_found == 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
