#!/usr/bin/env python3
"""
Problem Analysis Script for Helios Backend
Parses pylint output and provides structured analysis
"""

import json
import os
import re
from collections import defaultdict, Counter
from typing import Dict, List, Any

def parse_pylint_text_output(filename: str) -> Dict[str, Any]:
    """Parse text format pylint output and extract structured data"""
    
    if not os.path.exists(filename):
        print(f" File {filename} not found")
        return {}
    
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract the final rating
    rating_match = re.search(r'Your code has been rated at ([\d\.]+)/10', content)
    overall_rating = float(rating_match.group(1)) if rating_match else 0.0
    
    # Parse individual issues
    issues = []
    issue_pattern = r'([^:]+):(\d+):(\d+): ([A-Z]\d+): (.+) \(([^)]+)\)'
    
    for match in re.finditer(issue_pattern, content):
        file_path, line, column, code, message, rule_name = match.groups()
        issues.append({
            'file': file_path.strip(),
            'line': int(line),
            'column': int(column),
            'code': code,
            'message': message.strip(),
            'rule': rule_name,
            'severity': get_severity(code[0])
        })
    
    # Group by severity and type
    by_severity = defaultdict(list)
    by_type = defaultdict(list)
    by_file = defaultdict(list)
    
    for issue in issues:
        by_severity[issue['severity']].append(issue)
        by_type[issue['rule']].append(issue)
        by_file[issue['file']].append(issue)
    
    # Count statistics
    severity_counts = {k: len(v) for k, v in by_severity.items()}
    type_counts = Counter(issue['rule'] for issue in issues)
    file_counts = {k: len(v) for k, v in by_file.items()}
    
    return {
        'overall_rating': overall_rating,
        'total_issues': len(issues),
        'issues': issues,
        'by_severity': dict(by_severity),
        'by_type': dict(by_type),
        'by_file': dict(by_file),
        'severity_counts': severity_counts,
        'type_counts': dict(type_counts.most_common()),
        'file_counts': file_counts
    }

def get_severity(code_letter: str) -> str:
    """Map pylint code letters to severity levels"""
    severity_map = {
        'C': 'convention',      # Convention violations
        'R': 'refactor',        # Refactoring suggestions
        'W': 'warning',         # Warnings
        'E': 'error',           # Errors
        'F': 'fatal'            # Fatal errors
    }
    return severity_map.get(code_letter, 'unknown')

def print_problem_summary(analysis: Dict[str, Any]):
    """Print a formatted summary of problems"""
    
    print("=" * 80)
    print(" PYLINT ANALYSIS SUMMARY")
    print("=" * 80)
    
    print(f" Overall Rating: {analysis['overall_rating']:.2f}/10")
    print(f" Total Issues: {analysis['total_issues']}")
    
    print(f"\n Issues by Severity:")
    for severity, count in analysis['severity_counts'].items():
        emoji = {
            'fatal': '',
            'error': '', 
            'warning': '',
            'refactor': '',
            'convention': ''
        }.get(severity, '')
        print(f"   {emoji} {severity.title()}: {count}")
    
    print(f"\n  Top Issue Types:")
    for issue_type, count in list(analysis['type_counts'].items())[:10]:
        print(f"   - {issue_type}: {count}")
    
    print(f"\n Issues by File:")
    sorted_files = sorted(analysis['file_counts'].items(), key=lambda x: x[1], reverse=True)
    for file_path, count in sorted_files[:10]:
        print(f"   - {os.path.basename(file_path)}: {count}")
    
    # Recommendations based on analysis
    print(f"\n Recommendations:")
    
    if analysis['overall_rating'] < 5.0:
        print("   CRITICAL: Code quality needs immediate attention")
    elif analysis['overall_rating'] < 7.0:
        print("   MODERATE: Several improvements needed")
    elif analysis['overall_rating'] < 8.5:
        print("   GOOD: Minor cleanup recommended")
    else:
        print("   EXCELLENT: Code quality is very high")
    
    # Specific recommendations
    if analysis['severity_counts'].get('fatal', 0) > 0:
        print("   Fix fatal errors immediately")
    
    if analysis['severity_counts'].get('error', 0) > 0:
        print("   Address all errors before deployment")
    
    if analysis['type_counts'].get('trailing-whitespace', 0) > 10:
        print("   Consider auto-formatting to fix whitespace issues")
    
    if analysis['type_counts'].get('line-too-long', 0) > 5:
        print("   Review line length settings or refactor long lines")
    
    print("=" * 80)

def main():
    """Main function to analyze problems.txt"""
    
    problems_file = 'problems.txt'
    
    if not os.path.exists(problems_file):
        print(f" {problems_file} not found. Run 'Export Problems' task first.")
        return
    
    print(" Analyzing pylint output...")
    analysis = parse_pylint_text_output(problems_file)
    
    if analysis:
        print_problem_summary(analysis)
        
        # Save structured analysis
        with open('problems-analysis.json', 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        print(f"\n Detailed analysis saved to: problems-analysis.json")
    else:
        print(" Failed to analyze problems file")

if __name__ == "__main__":
    main()
