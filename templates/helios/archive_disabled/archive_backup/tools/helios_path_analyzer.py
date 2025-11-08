#!/usr/bin/env python3
"""
Helios Path Analyzer - Enhanced Directory and Path Review Tool
Comprehensive analysis tool for detecting path inconsistencies and configuration issues
"""

import os
import re
import json
import csv
from pathlib import Path
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass
from datetime import datetime
import argparse

@dataclass
class PathMatch:
    """Data class for storing path match information"""
    file_path: str
    line_number: int
    line_content: str
    match_type: str
    severity: str
    suggested_fix: str = ""

class HeliosPathAnalyzer:
    """Enhanced analyzer for Helios project path consistency"""
    
    def __init__(self, search_root: str):
        self.search_root = Path(search_root)
        self.matches: List[PathMatch] = []
        self.stats = {
            'files_scanned': 0,
            'files_skipped': 0,
            'total_matches': 0,
            'critical_issues': 0,
            'warnings': 0
        }
        
        # Define patterns to search for
        self.patterns = {
            'backend_venv': {
                'regex': re.compile(r'[/\\]backend[/\\]venv[/\\]?'),
                'severity': 'CRITICAL',
                'description': 'Old backend/venv path (should be root-level venv)',
                'suggested_fix': 'Replace with root-level venv path'
            },
            'hardcoded_paths': {
                'regex': re.compile(r'C:[/\\]Users[/\\][^/\\]+[/\\].*[/\\]helios', re.IGNORECASE),
                'severity': 'WARNING',
                'description': 'Hardcoded absolute path',
                'suggested_fix': 'Use relative paths or environment variables'
            },
            'python_exe_backend': {
                'regex': re.compile(r'backend[/\\]venv[/\\]Scripts[/\\]python\.exe'),
                'severity': 'CRITICAL',
                'description': 'Python executable pointing to old backend venv',
                'suggested_fix': 'Update to venv/Scripts/python.exe'
            },
            'npm_paths': {
                'regex': re.compile(r'node_modules[/\\]\.bin', re.IGNORECASE),
                'severity': 'INFO',
                'description': 'Node.js binary path reference',
                'suggested_fix': 'Verify npm installation consistency'
            },
            'docker_references': {
                'regex': re.compile(r'["\']?/app/[^"\']*["\']?'),
                'severity': 'INFO',
                'description': 'Docker container path reference',
                'suggested_fix': 'Ensure Docker paths align with local development'
            },
            'vscode_settings': {
                'regex': re.compile(r'\.vscode[/\\]'),
                'severity': 'INFO',
                'description': 'VS Code configuration reference',
                'suggested_fix': 'Verify VS Code settings consistency'
            }
        }
        
        # File extensions to analyze
        self.text_extensions = {
            '.py', '.js', '.ts', '.tsx', '.json', '.jsonc', '.ps1', '.sh', '.bat',
            '.yml', '.yaml', '.md', '.txt', '.cfg', '.ini', '.env', '.dockerfile',
            '.html', '.css', '.scss', '.xml', '.toml'
        }
        
        # Directories to skip
        self.skip_dirs = {
            '__pycache__', '.git', 'node_modules', '.pytest_cache', '.mypy_cache',
            'dist', 'build', '.vscode', 'venv', '.env', 'logs'
        }

    def should_analyze_file(self, file_path: Path) -> bool:
        """Determine if a file should be analyzed"""
        # Skip if in excluded directory
        for part in file_path.parts:
            if part in self.skip_dirs:
                return False
        
        # Only analyze text files
        return file_path.suffix.lower() in self.text_extensions

    def analyze_file(self, file_path: Path) -> List[PathMatch]:
        """Analyze a single file for path issues"""
        matches = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f, start=1):
                    for pattern_name, pattern_info in self.patterns.items():
                        if pattern_info['regex'].search(line):
                            match = PathMatch(
                                file_path=str(file_path.relative_to(self.search_root)),
                                line_number=line_num,
                                line_content=line.strip(),
                                match_type=pattern_name,
                                severity=pattern_info['severity'],
                                suggested_fix=pattern_info['suggested_fix']
                            )
                            matches.append(match)
                            
                            # Update stats
                            if pattern_info['severity'] == 'CRITICAL':
                                self.stats['critical_issues'] += 1
                            elif pattern_info['severity'] == 'WARNING':
                                self.stats['warnings'] += 1
            
            self.stats['files_scanned'] += 1
            
        except (UnicodeDecodeError, PermissionError, OSError) as e:
            self.stats['files_skipped'] += 1
            # Optionally log skipped files
            # print(f"Skipped {file_path}: {e}")
        
        return matches

    def scan_directory(self) -> None:
        """Scan the entire directory structure"""
        print(f"🔍 Scanning Helios project at: {self.search_root}")
        print("─" * 60)
        
        for file_path in self.search_root.rglob('*'):
            if file_path.is_file() and self.should_analyze_file(file_path):
                file_matches = self.analyze_file(file_path)
                self.matches.extend(file_matches)
        
        self.stats['total_matches'] = len(self.matches)

    def generate_report(self, output_format: str = 'console') -> None:
        """Generate analysis report in specified format"""
        
        if output_format == 'console':
            self._print_console_report()
        elif output_format == 'json':
            self._save_json_report()
        elif output_format == 'csv':
            self._save_csv_report()
        elif output_format == 'all':
            self._print_console_report()
            self._save_json_report()
            self._save_csv_report()

    def _print_console_report(self) -> None:
        """Print detailed console report"""
        print("\n" + "=" * 80)
        print("📊 HELIOS PATH ANALYSIS REPORT")
        print("=" * 80)
        
        # Statistics
        print(f"Files Scanned: {self.stats['files_scanned']}")
        print(f"Files Skipped: {self.stats['files_skipped']}")
        print(f"Total Issues Found: {self.stats['total_matches']}")
        print(f"Critical Issues: {self.stats['critical_issues']}")
        print(f"Warnings: {self.stats['warnings']}")
        print()
        
        # Group matches by severity
        critical_matches = [m for m in self.matches if m.severity == 'CRITICAL']
        warning_matches = [m for m in self.matches if m.severity == 'WARNING']
        info_matches = [m for m in self.matches if m.severity == 'INFO']
        
        # Critical Issues
        if critical_matches:
            print("🚨 CRITICAL ISSUES (Need Immediate Fix)")
            print("-" * 50)
            for match in critical_matches:
                print(f"📁 {match.file_path} (Line {match.line_number})")
                print(f"   Type: {match.match_type}")
                print(f"   Content: {match.line_content[:100]}...")
                print(f"   💡 Fix: {match.suggested_fix}")
                print()
        
        # Warnings
        if warning_matches:
            print("⚠️  WARNINGS (Should Be Reviewed)")
            print("-" * 50)
            for match in warning_matches[:10]:  # Limit to first 10
                print(f"📁 {match.file_path} (Line {match.line_number})")
                print(f"   Type: {match.match_type}")
                print(f"   💡 Fix: {match.suggested_fix}")
                print()
            if len(warning_matches) > 10:
                print(f"   ... and {len(warning_matches) - 10} more warnings")
                print()
        
        # Summary by file type
        file_types = {}
        for match in self.matches:
            ext = Path(match.file_path).suffix or 'no_extension'
            file_types[ext] = file_types.get(ext, 0) + 1
        
        if file_types:
            print("📈 ISSUES BY FILE TYPE")
            print("-" * 50)
            for ext, count in sorted(file_types.items(), key=lambda x: x[1], reverse=True):
                print(f"   {ext}: {count} issues")
            print()

    def _save_json_report(self) -> None:
        """Save report as JSON file"""
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'search_root': str(self.search_root),
            'statistics': self.stats,
            'matches': [
                {
                    'file_path': match.file_path,
                    'line_number': match.line_number,
                    'line_content': match.line_content,
                    'match_type': match.match_type,
                    'severity': match.severity,
                    'suggested_fix': match.suggested_fix
                }
                for match in self.matches
            ]
        }
        
        output_file = self.search_root / 'helios_path_analysis.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        print(f"📄 JSON report saved to: {output_file}")

    def _save_csv_report(self) -> None:
        """Save report as CSV file"""
        output_file = self.search_root / 'helios_path_analysis.csv'
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'File Path', 'Line Number', 'Line Content', 
                'Match Type', 'Severity', 'Suggested Fix'
            ])
            
            for match in self.matches:
                writer.writerow([
                    match.file_path,
                    match.line_number,
                    match.line_content,
                    match.match_type,
                    match.severity,
                    match.suggested_fix
                ])
        
        print(f"📊 CSV report saved to: {output_file}")

    def get_fix_suggestions(self) -> Dict[str, List[str]]:
        """Generate automated fix suggestions"""
        suggestions = {
            'critical_fixes': [],
            'recommended_actions': [],
            'verification_steps': []
        }
        
        # Critical path fixes
        critical_matches = [m for m in self.matches if m.severity == 'CRITICAL']
        if critical_matches:
            suggestions['critical_fixes'].extend([
                "Update all backend/venv references to root-level venv",
                "Fix Python executable paths in configuration files",
                "Update VS Code tasks.json with correct paths"
            ])
        
        # Recommended actions
        if self.stats['warnings'] > 0:
            suggestions['recommended_actions'].extend([
                "Replace hardcoded paths with relative paths",
                "Use environment variables for user-specific paths",
                "Standardize path separators for cross-platform compatibility"
            ])
        
        # Verification steps
        suggestions['verification_steps'].extend([
            "Run system validation tests after path fixes",
            "Test virtual environment activation",
            "Verify all VS Code tasks work correctly"
        ])
        
        return suggestions


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Helios Path Analyzer')
    parser.add_argument('--root', default=r"C:\Users\RobMo\OneDrive\Documents\helios",
                       help='Root directory to analyze')
    parser.add_argument('--format', choices=['console', 'json', 'csv', 'all'],
                       default='console', help='Output format')
    parser.add_argument('--fix-suggestions', action='store_true',
                       help='Show automated fix suggestions')
    
    args = parser.parse_args()
    
    # Create analyzer and run scan
    analyzer = HeliosPathAnalyzer(args.root)
    analyzer.scan_directory()
    
    # Generate report
    analyzer.generate_report(args.format)
    
    # Show fix suggestions if requested
    if args.fix_suggestions:
        suggestions = analyzer.get_fix_suggestions()
        print("\n" + "=" * 80)
        print("🔧 AUTOMATED FIX SUGGESTIONS")
        print("=" * 80)
        
        if suggestions['critical_fixes']:
            print("🚨 Critical Fixes:")
            for fix in suggestions['critical_fixes']:
                print(f"   • {fix}")
            print()
        
        if suggestions['recommended_actions']:
            print("💡 Recommended Actions:")
            for action in suggestions['recommended_actions']:
                print(f"   • {action}")
            print()
        
        if suggestions['verification_steps']:
            print("✅ Verification Steps:")
            for step in suggestions['verification_steps']:
                print(f"   • {step}")


if __name__ == "__main__":
    main()
