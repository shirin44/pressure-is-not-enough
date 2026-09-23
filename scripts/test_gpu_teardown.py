"""Real-failure-path tests for scripts/gpu_teardown.py.

Run: python3 scripts/test_gpu_teardown.py

These invoke the actual script as a subprocess (black-box, matching how it's
really called) against REAL AWS calls with deliberately bad inputs -- an
invalid profile and a nonexistent instance id -- not mocks. Both are safe:
neither touches any real, billable resource. The whole point of this suite
(per the incident that motivated gpu_teardown.py) is that a script which
"looks like" it handles failure can still silently report success; only an
actual failing call, checked end to end including the process exit code and
the absence of any false "TEARDOWN FULLY CONFIRMED" banner, verifies that.

A fourth test runs the real script, read-only in effect, against this
project's actual GPU instance (expected already stopped) to confirm the
idempotent "already stopped" happy path also exits 0 with a genuine fresh
verification -- not just the failure paths.
"""
from __future__ import annotations

import subprocess
import sys

SCRIPT = "scripts/gpu_teardown.py"
REAL_INSTANCE_ID = "i-REDACTED"
REAL_REGION = "us-east-1"
REAL_PROFILE = "research"

failures: list[str] = []


def run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, SCRIPT, *args], capture_output=True, text=True)


def check(name: str, condition: bool, detail: str) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    if not condition:
        print(f"       {detail}")
        failures.append(name)


print("=== Test 1: invalid/nonexistent profile must fail loudly, exit 1, no false success ===")
proc = run(["--instance-id", REAL_INSTANCE_ID, "--region", REAL_REGION,
            "--profile", "totally-bogus-profile-that-does-not-exist"])
check("exit code is 1 (credentials failure)", proc.returncode == 1,
      f"got exit code {proc.returncode}; stdout={proc.stdout!r} stderr={proc.stderr!r}")
check("stderr states credentials expired/invalid and manual verification required",
      "CREDENTIALS EXPIRED OR INVALID" in proc.stderr and "MANUAL VERIFICATION REQUIRED" in proc.stderr,
      f"stderr={proc.stderr!r}")
check("stdout/stderr never claims teardown confirmed",
      "TEARDOWN FULLY CONFIRMED" not in proc.stdout and "TEARDOWN FULLY CONFIRMED" not in proc.stderr,
      f"stdout={proc.stdout!r} stderr={proc.stderr!r}")

print()
print("=== Test 2: nonexistent instance id (valid credentials) must fail loudly, exit 2, no false success ===")
proc = run(["--instance-id", "i-0000000000bogus99", "--region", REAL_REGION, "--profile", REAL_PROFILE])
check("exit code is 2 (AWS call rejected before teardown could be confirmed)", proc.returncode == 2,
      f"got exit code {proc.returncode}; stdout={proc.stdout!r} stderr={proc.stderr!r}")
check("stderr reports the failure explicitly",
      "FAILURE" in proc.stderr,
      f"stderr={proc.stderr!r}")
check("stdout/stderr never claims teardown confirmed",
      "TEARDOWN FULLY CONFIRMED" not in proc.stdout and "TEARDOWN FULLY CONFIRMED" not in proc.stderr,
      f"stdout={proc.stdout!r} stderr={proc.stderr!r}")

print()
print("=== Test 3: malformed security-group revoke request (valid credentials, valid real instance, "
      "already stopped) -- exercises the exit-4 cleanup-failure path without touching the instance's own state ===")
proc = run(["--instance-id", REAL_INSTANCE_ID, "--region", REAL_REGION, "--profile", REAL_PROFILE,
            "--revoke-ssh-cidr", "203.0.113.99/32",  # TEST-NET-3, guaranteed not an existing rule
            "--security-group-id", "sg-REDACTED"])
check("exit code is 4 (stopped, but cleanup step failed)", proc.returncode == 4,
      f"got exit code {proc.returncode}; stdout={proc.stdout!r} stderr={proc.stderr!r}")
check("stdout still confirms the instance itself is stopped before the cleanup failure",
      "already stopped" in proc.stdout or "'stopped'" in proc.stdout,
      f"stdout={proc.stdout!r}")
check("stderr reports the cleanup failure explicitly, distinct from a 'not stopped' failure",
      "FAILURE during cleanup" in proc.stderr,
      f"stderr={proc.stderr!r}")

print()
print("=== Test 4 (happy path, real, read-only in effect): already-stopped real instance -> exit 0 ===")
proc = run(["--instance-id", REAL_INSTANCE_ID, "--region", REAL_REGION, "--profile", REAL_PROFILE])
check("exit code is 0", proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}")
check("detects already-stopped and skips issuing a redundant stop call",
      "already stopped" in proc.stdout, f"stdout={proc.stdout!r}")
check("performs the mandatory fresh post-stop verification and confirms it",
      "fresh describe-instances reports" in proc.stdout and "TEARDOWN FULLY CONFIRMED" in proc.stdout,
      f"stdout={proc.stdout!r}")

print()
if failures:
    print(f"RESULT: {len(failures)} FAILED: {failures}")
    sys.exit(1)
print("RESULT: all tests passed.")
