import sys
import os
from scripts.run_pipeline import stop_pipeline

def main():
    if len(sys.argv) != 2:
        print("Usage: python -m scripts.stop_pipeline <PID>")
        return
        
    try:
        pid = int(sys.argv[1])
        stop_pipeline(pid)
    except ValueError:
        print(f"Invalid PID: {sys.argv[1]}")

if __name__ == "__main__":
    main() 