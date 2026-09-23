"""Verified GPU-instance teardown: stop, confirm, and (optionally) restore/revoke.

Added 2026-08-27 after an incident where the Step 14 full-run teardown used a
compound shell command ending in an unconditional trailing `echo`:

    aws ec2 stop-instances ...; aws ec2 wait instance-stopped ...; echo "STOPPED"

When the AWS SSO session token expired mid-sequence (a multi-hour run), the
`stop-instances` and `wait` calls both failed, but the trailing `echo`'s own
exit code (always 0) was what the background task reported -- so the task
was marked "completed successfully" while the instance kept running,
undetected until a later, unrelated state check happened to show a
`LaunchTime` inconsistent with a stop having occurred.

This script exists so that never happens silently again. It replaces every
ad hoc chained stop/wait/echo invocation used earlier in this project with a
single, tested, reusable procedure. The load-bearing design choices:

1. Every AWS call's exit code is checked explicitly, in Python control flow
   -- never chained with `;`/`&&` into a shell one-liner whose overall exit
   code could come from an unrelated trailing command.
2. Credential validity is checked BEFORE issuing the stop, and re-checked at
   every subsequent step (a token can expire mid-run, not just at the start).
   Any credential failure aborts immediately and loudly: "CREDENTIALS
   EXPIRED OR INVALID -- TEARDOWN NOT CONFIRMED -- MANUAL VERIFICATION
   REQUIRED", non-zero exit. Nothing downstream can paper over this.
3. Teardown is only ever declared complete after a FRESH describe-instances
   call, issued as the last action, whose State.Name is read and checked
   to literally equal "stopped" -- this is mandatory, not optional, and not
   satisfied by the exit code of the stop or wait calls alone (both can
   "succeed" -- e.g. stop-instances accepted, wait exits 0 on a transient
   state -- without the instance actually being stopped yet).
4. If the instance is not stopped within the timeout, the stop call is
   retried explicitly once, then re-verified; if it still isn't stopped,
   the script fails loudly (non-zero exit) rather than finishing silently.

Distinct exit codes so a caller (or a human reading the output) knows
exactly what failed:
    0 = teardown fully confirmed (fresh state check == "stopped"; any
        requested cleanup -- instance-type restore, SG rule revoke --
        also succeeded)
    1 = credentials invalid/expired at some point -- teardown NOT confirmed
    2 = the stop-instances call itself failed (bad instance id, permissions,
        capacity/state conflict, etc.)
    3 = instance did not reach "stopped" within timeout, even after one
        explicit retry -- manual intervention required
    4 = instance IS confirmed stopped, but a requested cleanup step
        (instance-type restore or security-group rule revoke) failed

Usage:
    python3 scripts/gpu_teardown.py --instance-id i-xxxxxxxx --region us-east-1 \\
        --profile research [--restore-instance-type g5.xlarge] \\
        [--revoke-ssh-cidr 1.2.3.4/32 --security-group-id sg-xxxxxxxx] \\
        [--timeout-seconds 300] [--poll-interval-seconds 15]

Run before every commit that changes this file, and before relying on it for
a real run: `python3 scripts/test_gpu_teardown.py` (exercises real failure
paths -- an invalid profile and a nonexistent instance id -- not just the
happy path).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass


class CredentialsInvalid(Exception):
    """Raised the instant any AWS call fails in a way consistent with an
    expired/invalid credential -- caller must abort immediately, not retry
    or fall through to a later step."""


class StopCallFailed(Exception):
    """The stop-instances call itself was rejected by AWS."""


@dataclass
class InstanceState:
    state_name: str
    instance_type: str
    public_ip: str | None


def _run_aws(args: list[str], profile: str, region: str) -> subprocess.CompletedProcess:
    cmd = ["aws", *args, "--profile", profile, "--region", region, "--output", "json"]
    return subprocess.run(cmd, capture_output=True, text=True)


def _check_not_credentials_failure(proc: subprocess.CompletedProcess, step: str) -> None:
    if proc.returncode != 0 and (
        "Token has expired" in proc.stderr
        or "ExpiredToken" in proc.stderr
        or "sso" in proc.stderr.lower()
        or "Unable to locate credentials" in proc.stderr
    ):
        raise CredentialsInvalid(f"during '{step}': {proc.stderr.strip()}")


def check_credentials(profile: str, region: str) -> None:
    proc = _run_aws(["sts", "get-caller-identity"], profile, region)
    if proc.returncode != 0:
        raise CredentialsInvalid(f"during initial credential check: {proc.stderr.strip()}")


def get_state(instance_id: str, profile: str, region: str) -> InstanceState:
    proc = _run_aws(
        ["ec2", "describe-instances", "--instance-ids", instance_id,
         "--query", "Reservations[].Instances[].{S:State.Name,T:InstanceType,P:PublicIpAddress}"],
        profile, region,
    )
    _check_not_credentials_failure(proc, "describe-instances")
    if proc.returncode != 0:
        raise RuntimeError(f"describe-instances failed: {proc.stderr.strip()}")
    rows = json.loads(proc.stdout)
    if not rows:
        raise RuntimeError(f"describe-instances returned no data for {instance_id}: {proc.stdout!r}")
    row = rows[0]
    return InstanceState(state_name=row["S"], instance_type=row["T"], public_ip=row.get("P"))


def issue_stop(instance_id: str, profile: str, region: str) -> None:
    proc = _run_aws(["ec2", "stop-instances", "--instance-ids", instance_id], profile, region)
    _check_not_credentials_failure(proc, "stop-instances")
    if proc.returncode != 0:
        raise StopCallFailed(proc.stderr.strip())


def poll_until_stopped(instance_id: str, profile: str, region: str,
                        timeout_seconds: int, poll_interval_seconds: int) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        state = get_state(instance_id, profile, region)
        print(f"  ... state={state.state_name} (polling)")
        if state.state_name == "stopped":
            return True
        time.sleep(poll_interval_seconds)
    return False


def restore_instance_type(instance_id: str, target_type: str, profile: str, region: str) -> None:
    proc = _run_aws(
        ["ec2", "modify-instance-attribute", "--instance-id", instance_id,
         "--instance-type", json.dumps({"Value": target_type})],
        profile, region,
    )
    _check_not_credentials_failure(proc, "modify-instance-attribute")
    if proc.returncode != 0:
        raise RuntimeError(f"instance-type restore to {target_type} failed: {proc.stderr.strip()}")


def revoke_ssh_rule(security_group_id: str, cidr: str, profile: str, region: str) -> None:
    perms = json.dumps([{
        "IpProtocol": "tcp", "FromPort": 22, "ToPort": 22,
        "IpRanges": [{"CidrIp": cidr}],
    }])
    proc = _run_aws(
        ["ec2", "revoke-security-group-ingress", "--group-id", security_group_id,
         "--ip-permissions", perms],
        profile, region,
    )
    _check_not_credentials_failure(proc, "revoke-security-group-ingress")
    if proc.returncode != 0:
        raise RuntimeError(f"revoking SSH rule for {cidr} on {security_group_id} failed: {proc.stderr.strip()}")


def teardown(*, instance_id: str, profile: str, region: str,
             restore_instance_type_to: str | None,
             revoke_ssh_cidr: str | None, security_group_id: str | None,
             timeout_seconds: int, poll_interval_seconds: int) -> int:
    print(f"===== GPU TEARDOWN: instance={instance_id} region={region} profile={profile} =====")

    try:
        print("Step 1/4: checking credentials...")
        check_credentials(profile, region)
        print("  credentials OK")

        print("Step 2/4: checking current state...")
        state = get_state(instance_id, profile, region)
        print(f"  current state: {state.state_name} (type={state.instance_type}, public_ip={state.public_ip})")

        if state.state_name != "stopped":
            print("Step 3/4: issuing stop-instances...")
            issue_stop(instance_id, profile, region)
            print("  stop-instances accepted; polling for 'stopped'...")
            reached = poll_until_stopped(instance_id, profile, region, timeout_seconds, poll_interval_seconds)
            if not reached:
                print(f"  NOT stopped within {timeout_seconds}s -- retrying stop-instances once...")
                issue_stop(instance_id, profile, region)
                reached = poll_until_stopped(instance_id, profile, region, timeout_seconds, poll_interval_seconds)
            if not reached:
                print(f"FAILURE: instance {instance_id} did not reach 'stopped' after retry. "
                      f"MANUAL INTERVENTION REQUIRED.", file=sys.stderr)
                return 3
        else:
            print("Step 3/4: already stopped, skipping stop call.")

        print("Step 4/4 (mandatory): fresh post-stop verification...")
        final_state = get_state(instance_id, profile, region)
        print(f"  fresh describe-instances reports: State.Name = '{final_state.state_name}'")
        if final_state.state_name != "stopped":
            print(f"FAILURE: fresh verification shows '{final_state.state_name}', not 'stopped'. "
                  f"TEARDOWN NOT CONFIRMED. MANUAL INTERVENTION REQUIRED.", file=sys.stderr)
            return 3

    except CredentialsInvalid as e:
        print("=" * 70, file=sys.stderr)
        print("CREDENTIALS EXPIRED OR INVALID -- TEARDOWN NOT CONFIRMED", file=sys.stderr)
        print("MANUAL VERIFICATION REQUIRED.", file=sys.stderr)
        print(f"Detail: {e}", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        return 1
    except StopCallFailed as e:
        print(f"FAILURE: stop-instances call itself was rejected: {e}", file=sys.stderr)
        return 2
    except RuntimeError as e:
        # Any other AWS-call rejection before we can confirm a stopped state (e.g. a
        # bad/nonexistent instance id at the initial describe-instances check) -- must
        # never fall through uncaught or be mistaken for a confirmed teardown.
        print(f"FAILURE: an AWS call failed before teardown could be confirmed: {e}", file=sys.stderr)
        return 2

    # Instance is confirmed stopped from here on. Cleanup failures are real but
    # distinct from "not stopped" -- reported loudly, own exit code, never silent.
    cleanup_failed = False
    try:
        if restore_instance_type_to:
            print(f"Optional: restoring instance type to {restore_instance_type_to}...")
            restore_instance_type(instance_id, restore_instance_type_to, profile, region)
            print("  restored.")
        if revoke_ssh_cidr and security_group_id:
            print(f"Optional: revoking SSH rule for {revoke_ssh_cidr} on {security_group_id}...")
            revoke_ssh_rule(security_group_id, revoke_ssh_cidr, profile, region)
            print("  revoked.")
    except CredentialsInvalid as e:
        print("=" * 70, file=sys.stderr)
        print("CREDENTIALS EXPIRED DURING CLEANUP -- instance IS confirmed stopped, "
              "but cleanup (type restore / SG revoke) is UNCONFIRMED.", file=sys.stderr)
        print(f"Detail: {e}", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        return 1
    except RuntimeError as e:
        print(f"FAILURE during cleanup (instance IS stopped): {e}", file=sys.stderr)
        cleanup_failed = True

    if cleanup_failed:
        return 4

    print("=" * 70)
    print(f"TEARDOWN FULLY CONFIRMED: {instance_id} is 'stopped' (freshly verified).")
    print("=" * 70)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--restore-instance-type", default=None)
    parser.add_argument("--revoke-ssh-cidr", default=None)
    parser.add_argument("--security-group-id", default=None)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--poll-interval-seconds", type=int, default=15)
    args = parser.parse_args()

    if args.revoke_ssh_cidr and not args.security_group_id:
        parser.error("--revoke-ssh-cidr requires --security-group-id")

    exit_code = teardown(
        instance_id=args.instance_id, profile=args.profile, region=args.region,
        restore_instance_type_to=args.restore_instance_type,
        revoke_ssh_cidr=args.revoke_ssh_cidr, security_group_id=args.security_group_id,
        timeout_seconds=args.timeout_seconds, poll_interval_seconds=args.poll_interval_seconds,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
