#!/usr/bin/env python3
"""
Recent Tasks API Test Script
Test the new /api/recent-tasks endpoint
"""

import requests
import json
from datetime import datetime

def test_recent_tasks_api():
    """Test the recent tasks API endpoint"""
    
    print("🧪 Testing Recent Tasks API...")
    print(f"🕐 Test time: {datetime.now()}")
    print("-" * 50)
    
    # Test the new endpoint
    try:
        print("📡 Testing /api/recent-tasks endpoint...")
        response = requests.get('http://localhost:5000/api/recent-tasks', timeout=10)
        
        print(f"📊 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ API Response successful!")
            print(f"📋 Status: {data.get('status')}")
            print(f"📊 Task count: {data.get('count', 0)}")
            
            if 'tasks' in data and data['tasks']:
                print("\n📝 Recent Tasks:")
                for i, task in enumerate(data['tasks'], 1):
                    print(f"  {i}. {task.get('icon', '📋')} {task.get('task')} - {task.get('status')}")
                    print(f"     📄 {task.get('description')}")
                    print(f"     🕐 {task.get('timestamp')}")
                    print()
            
            if 'system_stats' in data:
                stats = data['system_stats']
                print("📊 System Stats:")
                print(f"  📈 Stocks: {stats.get('stocks', 0)}")
                print(f"  💹 Prices: {stats.get('prices', 0)}")
                print(f"  📅 Latest date: {stats.get('latest_date', 'N/A')}")
                
        else:
            print(f"❌ API Error: {response.status_code}")
            try:
                error_data = response.json()
                print(f"📄 Error details: {error_data}")
            except:
                print(f"📄 Raw response: {response.text}")
                
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Flask app not running on localhost:5000")
        print("💡 Start the Flask app first: python app.py")
    except requests.exceptions.Timeout:
        print("❌ Timeout Error: API response took too long")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    print("-" * 50)
    print("🧪 Test completed!")

def test_dashboard_endpoints():
    """Test related dashboard endpoints"""
    
    endpoints = [
        '/health',
        '/api/dashboard-stats', 
        '/api/system-info',
        '/api/automation/status'
    ]
    
    print("\n🔍 Testing related endpoints...")
    
    for endpoint in endpoints:
        try:
            print(f"📡 Testing {endpoint}...")
            response = requests.get(f'http://localhost:5000{endpoint}', timeout=5)
            
            if response.status_code == 200:
                print(f"  ✅ {endpoint} - OK")
            else:
                print(f"  ❌ {endpoint} - Error {response.status_code}")
                
        except Exception as e:
            print(f"  ❌ {endpoint} - Exception: {e}")

if __name__ == "__main__":
    print("🚀 BIST Dashboard Recent Tasks Test")
    print("=" * 50)
    
    test_recent_tasks_api()
    test_dashboard_endpoints()
    
    print("\n📋 How to use:")
    print("1. Start Flask app: python app.py")
    print("2. Open dashboard: http://localhost:5000/dashboard")
    print("3. Check Recent Tasks section in the dashboard")
    print("4. Verify tasks are loading properly")
