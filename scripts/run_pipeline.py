import subprocess
from datetime import datetime
import os
from pathlib import Path

def run_pipeline():
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Create timestamped log file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"pipeline_{timestamp}.log"
    
    # Command to run
    cmd = ["python", "scripts/test_model_pipeline.py"]
    
    # Run the command with nohup-like behavior
    with open(log_file, 'w') as f:
        process = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setpgrp  # This makes it continue running if VSCode is closed
        )
    
    print(f"Started pipeline process (PID: {process.pid})")
    print(f"Logging output to: {log_file}")
    print("Process is running in background. You can close VSCode safely.")

if __name__ == "__main__":
    run_pipeline() 