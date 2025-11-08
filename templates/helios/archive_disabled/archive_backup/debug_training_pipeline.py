#!/usr/bin/env python3
"""
Debug script to test the mock data path end-to-end
"""

import sys
import os
import requests
import json
from datetime import datetime

def test_mock_data_locally():
    """Test mock data generation directly without dependencies."""
    print("🧪 Testing Mock Data Generation Locally...")
    print("=" * 60)
    
    try:
        # Test the logic directly without imports
        import random
        import pandas as pd
        from datetime import datetime, timedelta
        
        # Set seed for consistent mock data
        random.seed(42)
        
        mock_data = []
        num_records = 10
        
        # Generate dates going back from today
        end_date = datetime.now()
        start_date = end_date - timedelta(days=num_records * 3)
        
        for i in range(num_records):
            # Generate draw date
            draw_date = start_date + timedelta(days=i * 3)
            
            # Generate 5 unique white balls (1-69)
            white_balls = sorted(random.sample(range(1, 70), 5))
            
            # Generate 1 powerball (1-26)
            powerball = random.randint(1, 26)
            
            mock_data.append({
                'draw_date': draw_date.strftime('%Y-%m-%d'),
                'white_ball_1': white_balls[0],
                'white_ball_2': white_balls[1],
                'white_ball_3': white_balls[2],
                'white_ball_4': white_balls[3],
                'white_ball_5': white_balls[4],
                'powerball': powerball
            })
        
        df = pd.DataFrame(mock_data)
        
        print(f"✅ Local mock generation successful!")
        print(f"📊 Shape: {df.shape}")
        print(f"📋 Columns: {list(df.columns)}")
        print(f"🎱 Sample:")
        print(df.head(3).to_string())
        return True
        
    except Exception as e:
        print(f"❌ Local test failed: {e}")
        return False

def test_trainer_import():
    """Test if we can import the trainer module."""
    print("\n🧪 Testing Trainer Import...")
    print("=" * 60)
    
    try:
        # Try to import without torch dependencies
        import importlib.util
        
        spec = importlib.util.spec_from_file_location("trainer", "backend/trainer.py")
        if spec is None:
            print("❌ Could not find trainer.py")
            return False
            
        print("✅ trainer.py file found")
        
        # Check if we can at least parse the module
        with open("backend/trainer.py", 'r') as f:
            content = f.read()
            
        if "elif data_source == 'mock':" in content:
            print("✅ Mock data handler found in code")
        else:
            print("❌ Mock data handler NOT found in code")
            return False
            
        if "_generate_mock_data" in content:
            print("✅ Mock data generator function found")
        else:
            print("❌ Mock data generator function NOT found")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def test_api_endpoint():
    """Test the API endpoint if server is running."""
    print("\n🧪 Testing API Endpoint...")
    print("=" * 60)
    
    try:
        # Try different ports that might be running
        ports = [8080, 5001, 3000, 5173]
        
        for port in ports:
            try:
                url = f"http://localhost:{port}/api/train"
                print(f"🔍 Trying {url}...")
                
                payload = {
                    "modelName": "debug_test",
                    "epochs": 1,
                    "learningRate": 0.001,
                    "dataSource": "mock"
                }
                
                response = requests.post(url, json=payload, timeout=5)
                
                if response.status_code == 200:
                    print(f"✅ Server found on port {port}")
                    print(f"📄 Response: {response.json()}")
                    return True
                elif response.status_code in [400, 500]:
                    print(f"⚠️ Server found on port {port} but returned error:")
                    print(f"📄 Response: {response.text}")
                    return True
                    
            except requests.exceptions.RequestException:
                continue
                
        print("❌ No active server found on common ports")
        return False
        
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

def check_environment():
    """Check Python environment and dependencies."""
    print("\n🧪 Checking Environment...")
    print("=" * 60)
    
    print(f"🐍 Python version: {sys.version}")
    print(f"📁 Working directory: {os.getcwd()}")
    print(f"📦 Python path: {sys.executable}")
    
    # Check for common packages
    packages = ['pandas', 'numpy', 'torch', 'flask', 'requests']
    for package in packages:
        try:
            __import__(package)
            print(f"✅ {package} available")
        except ImportError:
            print(f"❌ {package} NOT available")

def main():
    """Run all diagnostic tests."""
    print("🔬 Helios Training Pipeline Debug Analysis")
    print("=" * 60)
    print(f"⏰ Started at: {datetime.now()}")
    print()
    
    results = {
        'environment': check_environment(),
        'local_mock': test_mock_data_locally(),
        'trainer_import': test_trainer_import(),
        'api_endpoint': test_api_endpoint()
    }
    
    print("\n" + "=" * 60)
    print("📊 SUMMARY REPORT")
    print("=" * 60)
    
    for test, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test:15}: {status}")
    
    overall = all(results.values())
    print(f"\n🎯 Overall Status: {'✅ ALL TESTS PASSED' if overall else '❌ ISSUES DETECTED'}")
    
    if not overall:
        print("\n🚨 LIKELY ISSUES:")
        if not results['trainer_import']:
            print("   - trainer.py module has import/syntax issues")
        if not results['api_endpoint']:
            print("   - Server not running or not accessible")
        if not results['local_mock']:
            print("   - Core mock logic has dependency issues")

if __name__ == '__main__':
    main()
