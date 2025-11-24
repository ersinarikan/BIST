#!/usr/bin/env python3
"""
Start Automation Pipeline
Simple script to start the automation pipeline
"""
import sys
import os
import time

# Setup environment
os.environ.setdefault('PYTHONWARNINGS', 'ignore')
sys.path.insert(0, '/opt/bist-pattern')

from working_automation import get_working_automation_pipeline  # type: ignore  # noqa: E402

pipeline = get_working_automation_pipeline()

print(f"Pipeline exists: {pipeline is not None}")
print(f"Currently running: {pipeline.is_running if pipeline else False}")

if pipeline and not pipeline.is_running:
    print("Starting automation pipeline...")
    success = pipeline.start_scheduler()
    print(f"Start result: {success}")
    
    if success:
        print("✅ Automation started successfully!")
        print("   Waiting for first cycle...")
        time.sleep(10)
        print(f"   Cycle count: {pipeline.cycle_count}")
        print(f"   Last run stats: {pipeline.last_run_stats}")
    else:
        print("❌ Failed to start automation")
        sys.exit(1)
elif pipeline and pipeline.is_running:
    print("ℹ️  Automation is already running")
    print(f"   Cycle count: {pipeline.cycle_count}")
else:
    print("❌ Pipeline not available")
    sys.exit(1)
