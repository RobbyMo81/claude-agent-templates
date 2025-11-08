#!/usr/bin/env python3
"""
Complete fix for all remaining truncated function definitions
"""

import re
from pathlib import Path


def fix_all_remaining_functions():
    """Fix all remaining truncated function definitions"""

    # Read each file and fix all function definitions missing proper syntax
    files_to_fix = [
        "backend/decision_engine.py",
        "backend/memory_store.py",
        "backend/metacognition.py",
    ]

    for file_path in files_to_fix:
        full_path = Path(file_path)
        if not full_path.exists():
            print(f"File not found: {file_path}")
            continue

        print(f"Fixing {file_path}...")

        with open(full_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        modified = False
        for i, line in enumerate(lines):
            # Look for function definitions that need fixing
            stripped = line.strip()

            # Check for function definitions ending with comma
            if (
                stripped.startswith("def ")
                and "(" in stripped
                and ")" not in stripped
                and stripped.endswith(",")
            ):
                # This is a multiline function definition starting with comma
                # Look for the end of the parameter list
                j = i + 1
                while j < len(lines) and ")" not in lines[j]:
                    j += 1

                # Check if the line with ')' exists and doesn't end with ':'
                if j < len(lines):
                    closing_line = lines[j].rstrip()
                    if ")" in closing_line and not closing_line.endswith(":"):
                        # Add colon
                        lines[j] = closing_line + ":\n"
                        modified = True
                        print(f"  Fixed multiline function at line {i+1}")

            # Check for single line function definitions ending with comma but no colon
            elif (
                stripped.startswith("def ")
                and "(" in stripped
                and ")" in stripped
                and stripped.endswith(",")
                and not stripped.endswith(":,")
            ):
                # Replace trailing comma with colon
                lines[i] = line.rstrip()[:-1] + ":\n"
                modified = True
                print(f"  Fixed single line function at line {i+1}")

        if modified:
            with open(full_path, "w", encoding="utf-8") as f:
                f.writelines(lines)
            print(f"  Updated {file_path}")
        else:
            print(f"  No changes needed in {file_path}")


def fix_specific_remaining_errors():
    """Fix the specific remaining syntax errors detected"""

    specific_fixes = [
        # decision_engine.py line 638
        ("backend/decision_engine.py", 638, "def get_decision_history(self,"),
        # memory_store.py line 459
        ("backend/memory_store.py", 459, "def update_prediction_outcome(self,"),
        # metacognition.py line 305
        ("backend/metacognition.py", 305, "def analyze_performance_patterns(self,"),
    ]

    for file_path, line_num, expected_start in specific_fixes:
        full_path = Path(file_path)
        if not full_path.exists():
            continue

        with open(full_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if line_num <= len(lines):
            line_index = line_num - 1
            current_line = lines[line_index].strip()

            if current_line.startswith("def ") and not current_line.endswith(":"):
                # Check if this is a multiline function
                if current_line.endswith(","):
                    # Find the end of the function signature
                    j = line_index + 1
                    while j < len(lines) and ")" not in lines[j]:
                        j += 1

                    if j < len(lines) and ")" in lines[j]:
                        # Fix the closing line
                        closing_line = lines[j].rstrip()
                        if not closing_line.endswith(":"):
                            lines[j] = closing_line + ":\n"
                            print(f"Fixed {file_path} line {line_num}")

                            with open(full_path, "w", encoding="utf-8") as f:
                                f.writelines(lines)


if __name__ == "__main__":
    print("🔧 Fixing all remaining truncated functions...")
    fix_all_remaining_functions()
    print()
    fix_specific_remaining_errors()
    print("✅ Complete!")
