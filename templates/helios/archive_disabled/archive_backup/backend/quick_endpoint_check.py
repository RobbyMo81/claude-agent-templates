#!/usr/bin/env python3
"""Quick check of API endpoints"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

try:
    import server
    app = server.app
    
    # Get all routes
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append(rule.rule)
    
    # Check for Phase 3 endpoints
    metacognitive_endpoints = [r for r in routes if 'metacognitive' in r]
    decision_endpoints = [r for r in routes if 'decision' in r]
    
    print(f"Metacognitive endpoints found: {len(metacognitive_endpoints)}")
    for endpoint in metacognitive_endpoints:
        print(f"  - {endpoint}")
    
    print(f"Decision endpoints found: {len(decision_endpoints)}")
    for endpoint in decision_endpoints:
        print(f"  - {endpoint}")
    
    total_endpoints = len(metacognitive_endpoints) + len(decision_endpoints)
    print(f"Total Phase 3 endpoints: {total_endpoints}")
    
    if total_endpoints >= 7:  # 3 metacognitive + 4 decision
        print("SUCCESS: All expected Phase 3 endpoints found!")
    else:
        print("WARNING: Some Phase 3 endpoints may be missing")
        
except Exception as e:
    print(f"ERROR: {e}")
