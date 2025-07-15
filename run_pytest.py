#!/usr/bin/env python3
"""
Pytest wrapper for the HEP Foundation pipeline test.

This script runs the pipeline test with improved output management:
- Logs everything to a single, easy-to-find log file
- Only shows warnings/errors in the terminal
- Provides clear feedback about log locations
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_pytest_with_filtered_output():
    """Run pytest and filter output to show only warnings/errors in terminal."""

    # Setup paths
    results_dir = Path("_test_results")
    results_dir.mkdir(exist_ok=True)
    log_file = results_dir / "test_run.log"

    print("=" * 60)
    print("HEP Foundation Pipeline Test")
    print("=" * 60)
    print(f"Starting time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Full logs will be saved to: _test_results/test_run.log")
    print("Only warnings and errors will be shown below.")
    print("=" * 60)
    print()

    # Run pytest with unbuffered output
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "pytest",
        "tests/test_pipeline.py",
        "-v",
        "--tb=short",
        "--no-header",
    ]

    try:
        # Start the process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )

        # Process output line by line
        important_lines = []
        total_lines = 0

        with open(log_file, "w") as f:
            f.write("HEP Foundation Pipeline Test Log\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write("=" * 80 + "\n\n")

            for line in process.stdout:
                total_lines += 1

                # Write everything to log file
                f.write(line)
                f.flush()  # Ensure immediate writing

                # Check if line contains important information
                line_lower = line.lower().strip()
                is_important = (
                    "error" in line_lower
                    or "warning" in line_lower
                    or "failed" in line_lower
                    or "exception" in line_lower
                    or "traceback" in line_lower
                    or line.startswith("FAILED ")
                    or line.startswith("ERROR ")
                    or "assertion" in line_lower
                    or "::" in line
                    and ("PASSED" in line or "FAILED" in line)
                    or line.strip().startswith("=")
                    and ("FAILURES" in line or "ERRORS" in line)
                )

                if is_important:
                    important_lines.append(line.rstrip())
                    print(line.rstrip())

        # Wait for process to complete
        return_code = process.wait()

        # Write completion info to log
        with open(log_file, "a") as f:
            f.write(
                f"\n\nTest completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            f.write(f"Exit code: {return_code}\n")
            f.write(f"Total lines processed: {total_lines}\n")

        print()
        print("=" * 60)
        print(f"Test completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Exit code: {return_code}")
        print()
        print("📄 Full logs available at: _test_results/test_run.log")

        if return_code == 0:
            print("✅ Test completed successfully!")
        else:
            print("❌ Test failed - check logs for details")
            print()
            print("Recent important output:")
            for line in important_lines[-10:]:  # Show last 10 important lines
                print(f"  {line}")

        print("=" * 60)

        return return_code

    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        with open(log_file, "a") as f:
            f.write(
                f"\n\nTest interrupted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
        return 130

    except Exception as e:
        print(f"\n💥 Error running test: {e}")
        with open(log_file, "a") as f:
            f.write(f"\n\nError running test: {e}\n")
        return 1


if __name__ == "__main__":
    exit_code = run_pytest_with_filtered_output()
    sys.exit(exit_code)
