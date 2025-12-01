#!/usr/bin/env python3
"""
Focused Helios Code Quality Test
===============================

This test focuses only on our project files, excluding node_modules and other dependencies.
Specifically targets potential truncation issues from our refactoring process.
"""

import os
import re
import ast
from pathlib import Path
from typing import List, Dict, Any


class FocusedCodeQualityTester:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.issues = []

        # Define patterns to exclude
        self.exclude_patterns = [
            "node_modules",
            ".git",
            "__pycache__",
            ".pytest_cache",
            "dist",
            "build",
            ".vscode",
            "venv",
            "env",
        ]

    def should_analyze_file(self, file_path: Path) -> bool:
        """Determine if we should analyze this file"""
        path_str = str(file_path)

        # Exclude patterns
        for pattern in self.exclude_patterns:
            if pattern in path_str:
                return False

        # Only analyze our project files
        extensions = [".py", ".ts", ".tsx", ".js", ".jsx"]
        return any(file_path.suffix == ext for ext in extensions)

    def add_issue(
        self,
        file_path: str,
        line_num: int,
        issue_type: str,
        description: str,
        severity: str = "warning",
    ):
        """Add an issue to the report"""
        self.issues.append(
            {
                "file": file_path,
                "line": line_num,
                "type": issue_type,
                "description": description,
                "severity": severity,
                "line_content": self.get_line_content(file_path, line_num),
            }
        )

    def get_line_content(self, file_path: str, line_num: int) -> str:
        """Get the actual content of a specific line"""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
                if 0 < line_num <= len(lines):
                    return lines[line_num - 1].strip()
        except:
            pass
        return ""

    def analyze_project_files(self):
        """Analyze all project files"""
        project_files = [
            f
            for f in self.project_root.rglob("*")
            if f.is_file() and self.should_analyze_file(f)
        ]

        print(f"Analyzing {len(project_files)} project files...")

        for file_path in project_files:
            try:
                if file_path.suffix == ".py":
                    self.analyze_python_file(file_path)
                else:
                    self.analyze_frontend_file(file_path)
            except Exception as e:
                self.add_issue(
                    str(file_path), 0, "file_error", f"Error analyzing: {e}", "error"
                )

    def analyze_python_file(self, file_path: Path):
        """Analyze Python files for truncation and other issues"""
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            lines = content.splitlines()

        # Check syntax first
        try:
            ast.parse(content)
        except SyntaxError as e:
            self.add_issue(
                str(file_path),
                e.lineno or 0,
                "syntax_error",
                f"Syntax error: {e.msg}",
                "error",
            )
            return  # Skip further analysis if syntax is broken

        # Line-by-line analysis
        for line_num, line in enumerate(lines, 1):
            self.check_python_line_issues(file_path, line_num, line)

        # Function completeness check
        self.check_python_function_completeness(file_path, lines)

    def check_python_line_issues(self, file_path: Path, line_num: int, line: str):
        """Check a Python line for specific issues"""
        stripped = line.strip()

        # 1. Check for truncated function definitions
        if stripped.startswith("def "):
            if not stripped.endswith(":"):
                self.add_issue(
                    str(file_path),
                    line_num,
                    "truncated_function_def",
                    "Function definition missing colon",
                    "error",
                )

            # Check for unmatched parentheses in function signature
            if "(" in stripped:
                open_count = stripped.count("(")
                close_count = stripped.count(")")
                if open_count != close_count:
                    self.add_issue(
                        str(file_path),
                        line_num,
                        "unmatched_function_parens",
                        "Unmatched parentheses in function definition",
                        "error",
                    )

        # 2. Check for truncated class definitions
        if stripped.startswith("class "):
            if not stripped.endswith(":"):
                self.add_issue(
                    str(file_path),
                    line_num,
                    "truncated_class_def",
                    "Class definition missing colon",
                    "error",
                )

        # 3. Check for incomplete import statements
        if (
            stripped.startswith("from ")
            and " import " not in stripped
            and not stripped.endswith("\\")
        ):
            self.add_issue(
                str(file_path),
                line_num,
                "incomplete_import",
                "Import statement appears incomplete",
            )

        # 4. Check for lines that look like they were cut off
        if len(line) > 200 and not line.endswith(("\\", ",", ")", "}", "]")):
            # Very long lines that don't end with continuation characters might be truncated
            self.add_issue(
                str(file_path),
                line_num,
                "possible_truncation",
                "Very long line without proper continuation",
            )

        # 5. Check for incomplete string literals
        if not stripped.startswith("#"):  # Not a comment
            single_quotes = stripped.count("'")
            double_quotes = stripped.count('"')
            if (single_quotes % 2 != 0) or (double_quotes % 2 != 0):
                # Skip triple-quoted strings
                if not ('"""' in stripped or "'''" in stripped):
                    self.add_issue(
                        str(file_path),
                        line_num,
                        "unmatched_quotes",
                        "Unmatched quotes in line",
                    )

        # 6. Check for suspicious line endings that might indicate truncation
        suspicious_endings = [
            " and",
            " or",
            " +",
            " -",
            " *",
            " /",
            " =",
            " ==",
            " !=",
            " <",
            " >",
        ]
        if any(stripped.endswith(ending) for ending in suspicious_endings):
            self.add_issue(
                str(file_path),
                line_num,
                "suspicious_line_ending",
                f"Line ends with operator, possible truncation",
            )

    def check_python_function_completeness(self, file_path: Path, lines: List[str]):
        """Check if Python functions are complete"""
        in_function = False
        function_start = 0
        function_indent = 0

        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            current_indent = len(line) - len(line.lstrip()) if line.strip() else 0

            # Function start
            if stripped.startswith("def "):
                in_function = True
                function_start = line_num
                function_indent = current_indent

            # Check if we've exited the function
            elif in_function and line.strip():
                if current_indent <= function_indent:
                    in_function = False

            # Check if function ends without proper body
            elif in_function and line_num == len(lines):
                # Look back to see if function has any body
                has_body = False
                for check_line in lines[function_start:]:
                    if check_line.strip() and not check_line.strip().startswith("def "):
                        check_indent = len(check_line) - len(check_line.lstrip())
                        if check_indent > function_indent:
                            has_body = True
                            break

                if not has_body:
                    self.add_issue(
                        str(file_path),
                        function_start,
                        "empty_function",
                        "Function appears to have no body",
                    )

    def analyze_frontend_file(self, file_path: Path):
        """Analyze TypeScript/JavaScript files"""
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            lines = content.splitlines()

        for line_num, line in enumerate(lines, 1):
            self.check_frontend_line_issues(file_path, line_num, line)

        # Check component completeness
        self.check_component_completeness(file_path, lines)

    def check_frontend_line_issues(self, file_path: Path, line_num: int, line: str):
        """Check frontend file lines for issues"""
        stripped = line.strip()

        # 1. Check for truncated function definitions
        if "function" in stripped and "(" in stripped:
            open_parens = stripped.count("(")
            close_parens = stripped.count(")")
            if open_parens != close_parens:
                self.add_issue(
                    str(file_path),
                    line_num,
                    "unmatched_function_parens",
                    "Unmatched parentheses in function",
                )

        # 2. Check for incomplete interface definitions
        if (
            stripped.startswith("interface ")
            and "{" not in stripped
            and not stripped.endswith("{")
        ):
            self.add_issue(
                str(file_path),
                line_num,
                "incomplete_interface",
                "Interface definition may be incomplete",
            )

        # 3. Check for incomplete import statements
        if stripped.startswith("import ") and "from" in stripped:
            if not (
                stripped.endswith(";")
                or stripped.endswith("'")
                or stripped.endswith('"')
            ):
                self.add_issue(
                    str(file_path),
                    line_num,
                    "incomplete_import",
                    "Import statement may be incomplete",
                )

        # 4. Check for JSX truncation
        if "<" in stripped and ">" in stripped:
            # Count JSX tags
            open_tags = len(re.findall(r"<[^/!][^>]*[^/]>", stripped))  # Opening tags
            close_tags = len(re.findall(r"</[^>]*>", stripped))  # Closing tags
            self_closing = len(re.findall(r"<[^>]*/>", stripped))  # Self-closing tags

            # If we have more opening tags than closing + self-closing, might be truncated
            if open_tags > (close_tags + self_closing):
                self.add_issue(
                    str(file_path),
                    line_num,
                    "unclosed_jsx",
                    "JSX tags may be unclosed or truncated",
                )

        # 5. Check for very long lines that might be truncated
        if len(line) > 300 and not any(
            line.strip().endswith(x) for x in [",", ";", "{", "}", "(", ")"]
        ):
            self.add_issue(
                str(file_path),
                line_num,
                "possible_truncation",
                "Very long line without proper ending",
            )

    def check_component_completeness(self, file_path: Path, lines: List[str]):
        """Check React component completeness"""
        for line_num, line in enumerate(lines, 1):
            # Look for React component definitions
            if re.search(r"const\s+\w+:\s*React\.FC", line):
                component_start = line_num

                # Check if component has return statement within reasonable distance
                has_return = False
                for check_line in lines[
                    line_num : line_num + 50
                ]:  # Check next 50 lines
                    if "return" in check_line:
                        has_return = True
                        break

                if not has_return:
                    self.add_issue(
                        str(file_path),
                        component_start,
                        "missing_component_return",
                        "React component may be missing return statement",
                    )

    def generate_focused_report(self):
        """Generate a focused report on our project files"""
        print("\n" + "=" * 80)
        print("FOCUSED HELIOS CODE QUALITY REPORT")
        print("=" * 80)

        # Group issues by severity
        errors = [i for i in self.issues if i["severity"] == "error"]
        warnings = [i for i in self.issues if i["severity"] == "warning"]

        print(f"\n📊 SUMMARY:")
        print(f"   Total Issues: {len(self.issues)}")
        print(f"   Errors: {len(errors)}")
        print(f"   Warnings: {len(warnings)}")

        # Show critical errors first
        if errors:
            print(f"\n🚨 CRITICAL ERRORS ({len(errors)}):")
            for error in errors:
                print(f"   📍 {error['file']}:{error['line']}")
                print(f"      Type: {error['type']}")
                print(f"      Issue: {error['description']}")
                if error["line_content"]:
                    print(f"      Code: {error['line_content']}")
                print()

        # Group warnings by type
        warning_types = {}
        for warning in warnings:
            w_type = warning["type"]
            if w_type not in warning_types:
                warning_types[w_type] = []
            warning_types[w_type].append(warning)

        if warning_types:
            print(f"\n⚠️  WARNINGS BY TYPE:")
            for w_type, w_list in warning_types.items():
                print(f"   {w_type.replace('_', ' ').title()}: {len(w_list)}")

                # Show first few examples
                for warning in w_list[:3]:
                    print(
                        f"     • {warning['file']}:{warning['line']} - {warning['description']}"
                    )
                if len(w_list) > 3:
                    print(f"     ... and {len(w_list) - 3} more")
                print()

        # Recommendations
        print("💡 RECOMMENDATIONS:")
        if errors:
            print("   1. 🔴 Fix critical errors immediately")
        if any("truncat" in i["type"] for i in self.issues):
            print("   2. 📝 Review truncated code sections")
        if any("unmatched" in i["type"] for i in self.issues):
            print("   3. 🔗 Check bracket/quote matching")
        if any("incomplete" in i["type"] for i in self.issues):
            print("   4. ✅ Complete incomplete statements")

        print("\n" + "=" * 80)

        return len(errors) == 0


def main():
    """Run focused code quality test"""
    project_root = os.path.dirname(os.path.abspath(__file__))

    print("Starting Focused Helios Code Quality Analysis...")
    print(f"Project Root: {project_root}")

    tester = FocusedCodeQualityTester(project_root)
    tester.analyze_project_files()

    success = tester.generate_focused_report()

    return 0 if success else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
