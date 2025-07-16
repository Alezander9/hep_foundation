#!/usr/bin/env python3
"""
Pytest wrapper for the HEP Foundation pipeline test.

This script runs the pipeline test with simple output management:
- All logs go to pytest.log (handled by test setup)
- Only warnings, errors, and progress are shown in stdout
- Clean start and end messages
"""

import subprocess
import sys
from datetime import datetime


def run_pytest_with_filtered_output():
    """Run pytest and show only warnings/errors/progress in stdout."""
    start_time = datetime.now()

    # Print starting message
    print("=" * 60)
    print("HEP Foundation Pipeline Test")
    print("=" * 60)
    print(f"Starting time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("Full logs saved to: _test_results/pytest.log")
    print("Showing only warnings, errors, and progress below:")
    print("=" * 60)
    print()

    # Run pytest
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
        # Start the process and capture output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )

        # Filter output - only show warnings, errors, and progress
        for line in process.stdout:
            line_lower = line.lower().strip()

            # Check if line contains warnings, errors, or progress
            if any(
                keyword in line_lower
                for keyword in [
                    "warning",
                    "error",
                    "failed",
                    "exception",
                    "traceback",
                    "assertion",
                    "critical",
                    "progress",
                ]
            ):
                print(line.rstrip())

        # Wait for process to complete
        return_code = process.wait()

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Print ending message
        print()
        print("=" * 60)
        print(f"Test completed: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Duration: {duration:.1f}s")

        if return_code == 0:
            print("✅ Test completed successfully!")
        else:
            print(
                "❌ Test failed with exit code: {return_code} - check pytest.log for details"
            )

        print("=" * 60)

        return return_code

    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        return 130

    except Exception as e:
        print(f"\n💥 Error running test: {e}")
        return 1


if __name__ == "__main__":
    exit_code = run_pytest_with_filtered_output()
    sys.exit(exit_code)
