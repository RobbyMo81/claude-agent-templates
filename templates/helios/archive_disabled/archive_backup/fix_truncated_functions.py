#!/usr/bin/env python3
"""
Fix Truncated Function Definitions
=================================

This script automatically fixes the truncated function definitions identified by our test.
"""

import re
from pathlib import Path

def fix_truncated_functions():
    """Fix all truncated function definitions in our codebase"""
    
    # Define files with truncated functions and their line patterns
    fixes = [
        # metacognition.py
        ('backend/metacognition.py', [
            (142, r'def _calculate_confidence\(self,', 'def _calculate_confidence(self,'),
            (173, r'def _predict_performance\(self,', 'def _predict_performance(self,'),  
            (197, r'def _estimate_uncertainty\(self,', 'def _estimate_uncertainty(self,'),
            (223, r'def _identify_knowledge_gaps\(self,', 'def _identify_knowledge_gaps(self,'),
            (256, r'def _recommend_learning_strategy\(self,', 'def _recommend_learning_strategy(self,'),
            (305, r'def analyze_performance_patterns\(self,', 'def analyze_performance_patterns(self,'),
            (360, r'def _detect_trend_pattern\(self,', 'def _detect_trend_pattern(self,'),
            (393, r'def _detect_cyclical_pattern\(self,', 'def _detect_cyclical_pattern(self,'),
            (425, r'def _detect_anomaly_patterns\(self,', 'def _detect_anomaly_patterns(self,'),
            (479, r'def get_learning_recommendations\(self,', 'def get_learning_recommendations(self,'),
        ]),
        
        # decision_engine.py  
        ('backend/decision_engine.py', [
            (236, r'def make_autonomous_decision\(self,', 'def make_autonomous_decision(self,'),
            (291, r'def _make_parameter_decisions\(self,', 'def _make_parameter_decisions(self,'),
            (336, r'def _make_strategy_decisions\(self,', 'def _make_strategy_decisions(self,'),
            (377, r'def _make_resource_decisions\(self,', 'def _make_resource_decisions(self,'),
            (419, r'def _make_goal_decisions\(self,', 'def _make_goal_decisions(self,'),
            (638, r'def get_decision_history\(self,', 'def get_decision_history(self,'),
        ]),
        
        # memory_store.py
        ('backend/memory_store.py', [
            (224, r'def save_model_metadata\(self,', 'def save_model_metadata(self,'),
            (313, r'def create_training_session\(self,', 'def create_training_session(self,'),
            (337, r'def update_training_session\(self,', 'def update_training_session(self,'),
            (393, r'def add_training_log\(self,', 'def add_training_log(self,'),
            (435, r'def save_prediction\(self,', 'def save_prediction(self,'),
            (459, r'def update_prediction_outcome\(self,', 'def update_prediction_outcome(self,'),
            (478, r'def get_model_predictions\(self,', 'def get_model_predictions(self,'),
            (504, r'def store_context\(self,', 'def store_context(self,'),
            (526, r'def get_context\(self,', 'def get_context(self,'),
            (567, r'def log_event\(self,', 'def log_event(self,'),
            (587, r'def get_recent_events\(self,', 'def get_recent_events(self,'),
            (624, r'def store_enhanced_journal_entry\(self,', 'def store_enhanced_journal_entry(self,'),
            (649, r'def get_enhanced_journal_entries\(self,', 'def get_enhanced_journal_entries(self,'),
            (700, r'def store_knowledge_fragment\(self,', 'def store_knowledge_fragment(self,'),
            (721, r'def get_knowledge_fragments\(self,', 'def get_knowledge_fragments(self,'),
            (776, r'def store_performance_metric\(self,', 'def store_performance_metric(self,'),
            (797, r'def get_performance_metrics\(self,', 'def get_performance_metrics(self,'),
            (838, r'def log_memory_operation\(self,', 'def log_memory_operation(self,'),
            (859, r'def compact_memory\(self,', 'def compact_memory(self,'),
        ]),
        
        # agent.py
        ('backend/agent.py', [
            (29, r'def __init__\(self,', 'def __init__(self,'),
            (182, r'def __init__\(self,', 'def __init__(self,'),
        ]),
        
        # trainer.py
        ('backend/trainer.py', [
            (225, r'def __init__\(self,', 'def __init__(self,'),
            (246, r'def start_training_job\(self,', 'def start_training_job(self,'),
        ]),
    ]
    
    project_root = Path('.')
    
    for file_path, line_fixes in fixes:
        full_path = project_root / file_path
        if not full_path.exists():
            print(f"  File not found: {full_path}")
            continue
            
        print(f" Fixing {file_path}...")
        
        # Read the file
        with open(full_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Apply fixes
        fixes_applied = 0
        for line_num, pattern, replacement in line_fixes:
            if line_num <= len(lines):
                line_index = line_num - 1
                original_line = lines[line_index]
                
                # Check if this line needs fixing (missing colon)
                if original_line.strip().startswith('def ') and not original_line.rstrip().endswith(':'):
                    # Add colon to the end
                    lines[line_index] = original_line.rstrip() + ':\n'
                    fixes_applied += 1
                    print(f"   Fixed line {line_num}")
        
        # Write the file back
        if fixes_applied > 0:
            with open(full_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            print(f"   Applied {fixes_applied} fixes to {file_path}")
        else:
            print(f"   No fixes needed for {file_path}")

def fix_multiline_function_definitions():
    """Fix multiline function definitions that span multiple lines"""
    
    files_to_check = [
        'backend/metacognition.py',
        'backend/decision_engine.py', 
        'backend/memory_store.py',
        'backend/agent.py',
        'backend/trainer.py'
    ]
    
    project_root = Path('.')
    
    for file_path in files_to_check:
        full_path = project_root / file_path
        if not full_path.exists():
            continue
            
        print(f" Checking multiline functions in {file_path}...")
        
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.splitlines()
        
        modified = False
        new_lines = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            # Check if this is a function definition line
            if line.strip().startswith('def ') and '(' in line:
                # Check if function signature spans multiple lines
                if '(' in line and ')' not in line:
                    # Collect the full function signature
                    func_lines = [line]
                    j = i + 1
                    
                    while j < len(lines) and ')' not in lines[j]:
                        func_lines.append(lines[j])
                        j += 1
                    
                    # Add the line with closing parenthesis
                    if j < len(lines):
                        func_lines.append(lines[j])
                        
                        # Check if the last line ends with colon
                        last_line = func_lines[-1].rstrip()
                        if not last_line.endswith(':'):
                            func_lines[-1] = last_line + ':\n'
                            modified = True
                            print(f"   Fixed multiline function at line {i+1}")
                    
                    new_lines.extend(func_lines)
                    i = j + 1
                else:
                    # Single line function - check for colon
                    if not line.rstrip().endswith(':'):
                        new_lines.append(line.rstrip() + ':')
                        modified = True
                        print(f"   Fixed single line function at line {i+1}")
                    else:
                        new_lines.append(line)
                    i += 1
            else:
                new_lines.append(line)
                i += 1
        
        # Write back if modified
        if modified:
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(new_lines) + '\n')
            print(f"   Updated {file_path}")

if __name__ == "__main__":
    print(" Starting truncated function fix...")
    
    fix_truncated_functions()
    print()
    fix_multiline_function_definitions()
    
    print("\n Function definition fixes complete!")
    print(" Run the focused test again to verify fixes...")
