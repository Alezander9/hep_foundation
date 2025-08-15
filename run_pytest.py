#!/usr/bin/env .venv/bin/python
"""
Pytest wrapper for the HEP Foundation pipeline test.

This script runs the pipeline test with simple output management:
- All logs go to pytest.log (handled by test setup)
- Only warnings, errors, and progress are shown in stdout
- Clean start and end messages
"""

import os
import subprocess
import sys
from datetime import datetime


def is_on_login_node():
    """Check if we're running on a login node by examining hostname."""
    try:
        hostname = os.uname().nodename
        # Perlmutter login nodes have 'login' in hostname
        return "login" in hostname.lower()
    except Exception:
        return False


def resubmit_to_compute_node():
    """Resubmit this script to run on a compute node."""
    print("=" * 70)
    print("HEP Foundation Pipeline Test - Auto Node Allocation")
    print("=" * 70)
    print("Detected login node - requesting interactive compute node...")
    print("This will run faster and avoid login node restrictions.")
    print("=" * 70)

    # Get the current script path and working directory
    script_path = os.path.abspath(__file__)
    workspace_dir = os.path.dirname(script_path)

    # Build salloc command to run this same script on compute node
    cmd = [
        "salloc",
        "--nodes=1",
        "--constraint=cpu",
        "--qos=interactive",
        "--time=5",
        "--account=m2616",
        "bash",
        "-c",
        f"cd {workspace_dir} && source .venv/bin/activate && python {script_path} --force-local",
    ]

    try:
        print("Running:", " ".join(cmd[:7]) + " [bash command]")
        print()

        # Execute and let salloc handle all the allocation and execution
        result = subprocess.run(cmd, check=False)
        return result.returncode

    except KeyboardInterrupt:
        print("\n🛑 Node allocation interrupted")
        return 130
    except Exception as e:
        print(f"\n💥 Error requesting compute node: {e}")
        print("Falling back to login node execution...")
        return None  # Signal to fall back to local execution


def run_pytest_with_filtered_output():
    """Run pytest and show only warnings/errors/progress in stdout."""
    start_time = datetime.now()

    # Print starting message
    print("=" * 60)
    print("HEP Foundation Pipeline Test")
    print("=" * 60)
    print(f"Starting time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("Full logs saved to: _test_results/pytest.log")
    print("Showing only warnings, errors, progress, and templog below:")
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

        # Filter output - only show warnings, errors, progress, or templog
        for line in process.stdout:
            line_lower = line.lower().strip()

            # Check if line contains warnings, errors, progress, or templog
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
                    "templog",
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
            print("✅ Test completed successfully! - check _test_results/")
        else:
            print(
                f"❌ Test failed with exit code: {return_code} - check _test_results/pytest.log for details"
            )
            if duration < 1:
                print(
                    "The test failed instantly, this is often an import error, did you remember to activate the venv?"
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
    # Check for force-local flag (used when resubmitted to compute node)
    force_local = "--force-local" in sys.argv

    # If on login node and not forced local, try to resubmit to compute node
    if not force_local and is_on_login_node():
        exit_code = resubmit_to_compute_node()
        if exit_code is not None:
            sys.exit(exit_code)
        # If resubmission failed, fall through to local execution

    # Run the actual test (either forced local or on compute node)
    exit_code = run_pytest_with_filtered_output()
    sys.exit(exit_code)
