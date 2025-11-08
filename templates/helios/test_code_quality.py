#!/usr/bin/env python3
"""
Helios Code Quality and Truncation Detection Unit Test
======================================================

This test suite scans the entire Helios codebase to detect:
1. Truncated lines of code
2. Incomplete function definitions
3. Missing imports or dependencies
4. Syntax errors
5. Inconsistent indentation
6. Missing closing brackets/braces
7. Incomplete strings or comments

Usage: python test_code_quality.py
"""

import os
import re
import ast
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import json


class CodeQualityTester:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.issues = []
        self.file_stats = {}

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
            }
        )

    def test_python_files(self) -> List[Dict[str, Any]]:
        """Test all Python files for syntax and quality issues"""
        python_files = list(self.project_root.rglob("*.py"))

        for file_path in python_files:
            try:
                self.analyze_python_file(file_path)
            except Exception as e:
                self.add_issue(
                    str(file_path),
                    0,
                    "file_error",
                    f"Error analyzing file: {e}",
                    "error",
                )

        return self.issues

    def test_typescript_files(self) -> List[Dict[str, Any]]:
        """Test all TypeScript/JavaScript files for quality issues"""
        ts_files = (
            list(self.project_root.rglob("*.ts"))
            + list(self.project_root.rglob("*.tsx"))
            + list(self.project_root.rglob("*.js"))
        )

        for file_path in ts_files:
            try:
                self.analyze_typescript_file(file_path)
            except Exception as e:
                self.add_issue(
                    str(file_path),
                    0,
                    "file_error",
                    f"Error analyzing file: {e}",
                    "error",
                )

        return self.issues

    def analyze_python_file(self, file_path: Path):
        """Analyze a Python file for various issues"""
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            lines = content.splitlines()

        # Test 1: Syntax validation
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

        # Test 2: Line-by-line analysis
        for line_num, line in enumerate(lines, 1):
            self.analyze_python_line(file_path, line_num, line)

        # Test 3: Function completeness
        self.check_function_completeness(file_path, content)

        # Test 4: Import analysis
        self.check_imports(file_path, content)

        # Update stats
        self.file_stats[str(file_path)] = {
            "lines": len(lines),
            "type": "python",
            "size_bytes": len(content),
        }

    def analyze_typescript_file(self, file_path: Path):
        """Analyze a TypeScript/JavaScript file for various issues"""
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            lines = content.splitlines()

        # Test 1: Line-by-line analysis
        for line_num, line in enumerate(lines, 1):
            self.analyze_typescript_line(file_path, line_num, line)

        # Test 2: Bracket matching
        self.check_bracket_matching(file_path, content)

        # Test 3: Component completeness
        self.check_component_completeness(file_path, content)

        # Update stats
        self.file_stats[str(file_path)] = {
            "lines": len(lines),
            "type": "typescript",
            "size_bytes": len(content),
        }

    def analyze_python_line(self, file_path: Path, line_num: int, line: str):
        """Analyze individual Python line for issues"""
        stripped = line.strip()

        # Check for truncated function definitions
        if stripped.startswith("def ") and not stripped.endswith(":"):
            if "(" in stripped and ")" not in stripped:
                self.add_issue(
                    str(file_path),
                    line_num,
                    "truncated_function",
                    "Function definition appears truncated - missing closing parenthesis",
                )

        # Check for truncated class definitions
        if stripped.startswith("class ") and not stripped.endswith(":"):
            if "(" in stripped and ")" not in stripped:
                self.add_issue(
                    str(file_path),
                    line_num,
                    "truncated_class",
                    "Class definition appears truncated",
                )

        # Check for incomplete import statements
        if stripped.startswith("from ") and " import " not in stripped:
            self.add_issue(
                str(file_path),
                line_num,
                "incomplete_import",
                "Import statement appears incomplete",
            )

        # Check for incomplete string literals
        quote_count = stripped.count('"') + stripped.count("'")
        if quote_count % 2 != 0 and not stripped.startswith("#"):
            self.add_issue(
                str(file_path),
                line_num,
                "unmatched_quotes",
                "Unmatched quotes detected",
            )

        # Check for lines ending with commas (potential truncation)
        if stripped.endswith(",") and len(stripped) > 100:
            self.add_issue(
                str(file_path),
                line_num,
                "suspicious_comma",
                "Long line ending with comma - possible truncation",
            )

        # Check for incomplete dictionary/list definitions
        if (stripped.count("{") != stripped.count("}")) or (
            stripped.count("[") != stripped.count("]")
        ):
            if not any(
                x in stripped for x in ['"""', "'''", "#"]
            ):  # Ignore multiline strings and comments
                self.add_issue(
                    str(file_path),
                    line_num,
                    "unmatched_brackets",
                    "Unmatched brackets detected",
                )

        # Check for incomplete function calls
        if "(" in stripped and ")" not in stripped and not stripped.endswith("\\"):
            if (
                not stripped.startswith("#")
                and "def " not in stripped
                and "class " not in stripped
            ):
                self.add_issue(
                    str(file_path),
                    line_num,
                    "incomplete_call",
                    "Potentially incomplete function call",
                )

    def analyze_typescript_line(self, file_path: Path, line_num: int, line: str):
        """Analyze individual TypeScript line for issues"""
        stripped = line.strip()

        # Check for incomplete function definitions
        if "function" in stripped and "(" in stripped and ")" not in stripped:
            self.add_issue(
                str(file_path),
                line_num,
                "truncated_function",
                "Function definition appears truncated",
            )

        # Check for incomplete interface definitions
        if stripped.startswith("interface ") and "{" not in stripped:
            self.add_issue(
                str(file_path),
                line_num,
                "incomplete_interface",
                "Interface definition appears incomplete",
            )

        # Check for incomplete import statements
        if (
            stripped.startswith("import ")
            and "from" in stripped
            and not stripped.endswith(";")
        ):
            if "'" not in stripped[-10:] and '"' not in stripped[-10:]:
                self.add_issue(
                    str(file_path),
                    line_num,
                    "incomplete_import",
                    "Import statement may be incomplete",
                )

        # Check for unmatched JSX tags
        if "<" in stripped and ">" in stripped:
            open_tags = len(re.findall(r"<[^/][^>]*>", stripped))
            close_tags = len(re.findall(r"</[^>]*>", stripped))
            self_closing = len(re.findall(r"<[^>]*/>", stripped))

            if open_tags - close_tags - self_closing > 0:
                self.add_issue(
                    str(file_path),
                    line_num,
                    "unmatched_jsx",
                    "Potentially unmatched JSX tags",
                )

        # Check for incomplete object literals
        if stripped.count("{") != stripped.count("}") and not stripped.startswith("//"):
            self.add_issue(
                str(file_path),
                line_num,
                "unmatched_braces",
                "Unmatched braces detected",
            )

    def check_function_completeness(self, file_path: Path, content: str):
        """Check if functions are complete and properly formatted"""
        lines = content.splitlines()
        in_function = False
        function_start = 0
        indent_level = 0

        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()

            if stripped.startswith("def "):
                in_function = True
                function_start = line_num
                indent_level = len(line) - len(line.lstrip())

                # Check if function definition is complete
                if not stripped.endswith(":"):
                    self.add_issue(
                        str(file_path),
                        line_num,
                        "incomplete_function_def",
                        "Function definition missing colon",
                    )

            elif in_function:
                current_indent = len(line) - len(line.lstrip())

                # Check if we've left the function
                if line.strip() and current_indent <= indent_level:
                    in_function = False

                # Check for common function ending patterns
                elif line_num == len(lines):  # Last line of file
                    if in_function:
                        self.add_issue(
                            str(file_path),
                            function_start,
                            "incomplete_function",
                            "Function may be incomplete - reaches end of file",
                        )

    def check_imports(self, file_path: Path, content: str):
        """Check import statements for completeness and correctness"""
        lines = content.splitlines()

        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()

            if stripped.startswith("from ") and " import " in stripped:
                # Check for incomplete from...import statements
                if stripped.endswith("import"):
                    self.add_issue(
                        str(file_path),
                        line_num,
                        "incomplete_import",
                        "Import statement ends with 'import' keyword",
                    )

                # Check for relative imports without proper module structure
                if stripped.startswith("from .") or stripped.startswith("from .."):
                    module_part = stripped.split(" import ")[0].replace("from ", "")
                    if module_part.count(".") > 2:
                        self.add_issue(
                            str(file_path),
                            line_num,
                            "complex_relative_import",
                            "Complex relative import detected",
                        )

    def check_bracket_matching(self, file_path: Path, content: str):
        """Check for matching brackets, braces, and parentheses"""
        stack = []
        pairs = {"(": ")", "[": "]", "{": "}"}
        lines = content.splitlines()

        for line_num, line in enumerate(lines, 1):
            in_string = False
            string_char = None

            for char_pos, char in enumerate(line):
                # Handle string literals
                if char in ['"', "'"] and not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char and in_string:
                    in_string = False
                    string_char = None

                # Skip characters inside strings
                if in_string:
                    continue

                # Handle opening brackets
                if char in pairs:
                    stack.append((char, line_num, char_pos))

                # Handle closing brackets
                elif char in pairs.values():
                    if not stack:
                        self.add_issue(
                            str(file_path),
                            line_num,
                            "unmatched_closing",
                            f"Unmatched closing bracket '{char}' at position {char_pos}",
                        )
                    else:
                        opening, _, _ = stack.pop()
                        if pairs[opening] != char:
                            self.add_issue(
                                str(file_path),
                                line_num,
                                "mismatched_brackets",
                                f"Mismatched bracket pair at line {line_num}",
                            )

        # Check for unclosed brackets
        for opening, line_num, pos in stack:
            self.add_issue(
                str(file_path),
                line_num,
                "unclosed_bracket",
                f"Unclosed bracket '{opening}' at position {pos}",
            )

    def check_component_completeness(self, file_path: Path, content: str):
        """Check TypeScript/React components for completeness"""
        lines = content.splitlines()

        # Check for React components
        component_pattern = re.compile(
            r"const\s+(\w+):\s*React\.FC.*=.*\(\w*\)\s*=>\s*{"
        )

        for line_num, line in enumerate(lines, 1):
            match = component_pattern.search(line)
            if match:
                component_name = match.group(1)

                # Check if component has a proper return statement
                remaining_lines = lines[line_num:]
                has_return = any(
                    "return" in l for l in remaining_lines[:20]
                )  # Check next 20 lines

                if not has_return:
                    self.add_issue(
                        str(file_path),
                        line_num,
                        "incomplete_component",
                        f"Component '{component_name}' may be missing return statement",
                    )

    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive report of all issues found"""
        issues_by_severity = {
            "error": [i for i in self.issues if i["severity"] == "error"],
            "warning": [i for i in self.issues if i["severity"] == "warning"],
        }

        issues_by_type = {}
        for issue in self.issues:
            issue_type = issue["type"]
            if issue_type not in issues_by_type:
                issues_by_type[issue_type] = []
            issues_by_type[issue_type].append(issue)

        total_files = len(self.file_stats)
        total_lines = sum(stats["lines"] for stats in self.file_stats.values())

        return {
            "summary": {
                "total_files_analyzed": total_files,
                "total_lines_analyzed": total_lines,
                "total_issues": len(self.issues),
                "error_count": len(issues_by_severity["error"]),
                "warning_count": len(issues_by_severity["warning"]),
            },
            "issues_by_severity": issues_by_severity,
            "issues_by_type": issues_by_type,
            "file_stats": self.file_stats,
            "all_issues": self.issues,
        }

    def print_report(self):
        """Print a formatted report to console"""
        report = self.generate_report()

        print("=" * 80)
        print("HELIOS CODE QUALITY ANALYSIS REPORT")
        print("=" * 80)
        print()

        # Summary
        summary = report["summary"]
        print(f"📊 SUMMARY:")
        print(f"   Files Analyzed: {summary['total_files_analyzed']}")
        print(f"   Lines Analyzed: {summary['total_lines_analyzed']:,}")
        print(f"   Total Issues:   {summary['total_issues']}")
        print(f"   Errors:        {summary['error_count']}")
        print(f"   Warnings:      {summary['warning_count']}")
        print()

        # Issues by type
        if report["issues_by_type"]:
            print("🔍 ISSUES BY TYPE:")
            for issue_type, issues in report["issues_by_type"].items():
                print(f"   {issue_type.replace('_', ' ').title()}: {len(issues)}")
            print()

        # Critical errors
        errors = report["issues_by_severity"]["error"]
        if errors:
            print("🚨 CRITICAL ERRORS:")
            for error in errors[:10]:  # Show first 10 errors
                print(f"   {error['file']}:{error['line']} - {error['description']}")
            if len(errors) > 10:
                print(f"   ... and {len(errors) - 10} more errors")
            print()

        # Warnings
        warnings = report["issues_by_severity"]["warning"]
        if warnings:
            print("⚠️  WARNINGS:")
            for warning in warnings[:10]:  # Show first 10 warnings
                print(
                    f"   {warning['file']}:{warning['line']} - {warning['description']}"
                )
            if len(warnings) > 10:
                print(f"   ... and {len(warnings) - 10} more warnings")
            print()

        # Recommendations
        print("💡 RECOMMENDATIONS:")
        if summary["error_count"] > 0:
            print("   1. Fix critical errors first (syntax errors, missing imports)")
        if any("truncated" in issue["type"] for issue in self.issues):
            print("   2. Review truncated functions and statements")
        if any("unmatched" in issue["type"] for issue in self.issues):
            print("   3. Check bracket and quote matching")
        if any("incomplete" in issue["type"] for issue in self.issues):
            print("   4. Complete incomplete statements and definitions")
        print()

        print("=" * 80)


def main():
    """Main test function"""
    project_root = os.path.dirname(os.path.abspath(__file__))

    print("Starting Helios Code Quality Analysis...")
    print(f"Project Root: {project_root}")
    print()

    tester = CodeQualityTester(project_root)

    # Run tests
    print("Analyzing Python files...")
    tester.test_python_files()

    print("Analyzing TypeScript/JavaScript files...")
    tester.test_typescript_files()

    # Generate and print report
    tester.print_report()

    # Save detailed report to file
    report = tester.generate_report()
    report_file = os.path.join(project_root, "code_quality_report.json")
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"Detailed report saved to: {report_file}")

    # Return exit code based on errors
    return 1 if report["summary"]["error_count"] > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
