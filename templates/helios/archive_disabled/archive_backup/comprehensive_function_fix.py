#!/usr/bin/env python3
"""
Search for and fix ALL remaining function definition syntax errors
"""

import re
from pathlib import Path

def find_and_fix_all_function_errors():
    """Find and fix all remaining function definition syntax errors"""
    
    files_to_check = [
        'backend/memory_store.py',
        'backend/metacognition.py'
    ]
    
    for file_path in files_to_check:
        full_path = Path(file_path)
        if not full_path.exists():
            continue
            
        print(f"\n=== Checking {file_path} ===")
        
        with open(full_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        fixed_any = False
        
        for i, line in enumerate(lines):
            line_num = i + 1
            stripped = line.strip()
            
            # Check for function definitions that start with 'def ' and contain comma but no colon
            if (stripped.startswith('def ') and 
                ',' in stripped and 
                ':' not in stripped):
                
                print(f"Line {line_num}: Found problematic function definition")
                print(f"  Current: {stripped}")
                
                # Check if this is a multiline function definition
                if '(' in stripped and ')' not in stripped:
                    # Find the end of the function parameters
                    j = i + 1
                    while j < len(lines) and ')' not in lines[j]:
                        j += 1
                    
                    if j < len(lines) and ')' in lines[j]:
                        closing_line = lines[j].rstrip()
                        if not closing_line.endswith(':'):
                            lines[j] = closing_line + ':\n'
                            fixed_any = True
                            print(f"  Fixed: Added colon to line {j+1}")
                
                elif '(' in stripped and ')' in stripped:
                    # Single line function definition
                    if not stripped.endswith(':'):
                        lines[i] = line.rstrip() + ':\n'
                        fixed_any = True
                        print(f"  Fixed: Added colon to line {line_num}")
        
        # Also check for functions ending with comma instead of colon
        for i, line in enumerate(lines):
            line_num = i + 1
            stripped = line.strip()
            
            if (stripped.startswith('def ') and 
                stripped.endswith(',') and 
                '(' in stripped and 
                ')' not in stripped):
                
                # This is likely a multiline function starting with comma error
                print(f"Line {line_num}: Found function with comma error")
                print(f"  Current: {stripped}")
                
                # Replace comma with proper continuation
                if stripped.endswith(',:'):
                    # Already has colon after comma, just remove comma
                    lines[i] = line.replace(',:', ':\n')
                    fixed_any = True
                    print(f"  Fixed: Removed comma before colon")
                elif stripped.endswith(','):
                    # Remove comma, but need to find where to add colon
                    lines[i] = line.rstrip()[:-1] + '\n'  # Remove comma
                    
                    # Find closing parenthesis and add colon there
                    j = i + 1
                    while j < len(lines) and ')' not in lines[j]:
                        j += 1
                    
                    if j < len(lines) and ')' in lines[j]:
                        closing_line = lines[j].rstrip()
                        if not closing_line.endswith(':'):
                            lines[j] = closing_line + ':\n'
                            fixed_any = True
                            print(f"  Fixed: Removed comma from line {line_num}, added colon to line {j+1}")
        
        if fixed_any:
            with open(full_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            print(f" Updated {file_path}")
        else:
            print(f"  No changes needed for {file_path}")

if __name__ == "__main__":
    print(" Searching for ALL remaining function syntax errors...")
    find_and_fix_all_function_errors()
    print("\n Complete!")
