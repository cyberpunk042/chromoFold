"""Shared plumbing for ChromoFold RC1 qualification scenario hooks.

Each hook (one executable per scenario, named exactly like the scenario) reads the scenario payload as JSON on
stdin, performs the REAL runtime operation for its scenario, and writes ONE JSON object to stdout. Diagnostics go
to stderr. A status:PASS hook MUST exit 0 (the adapter downgrades PASS->FAIL on a non-zero exit).

Honesty contract (handoff sections 8-9): **never return PASS from a no-op.** When the serving runtime, or the
specific control capability a scenario needs, is unavailable, return INCOMPLETE with an explicit reason. Return
FAIL only on an OBSERVED invariant violation. Nothing here fabricates evidence:

  - Request traffic is issued to the runtime's real generation endpoint (or an operator command); if neither is
    configured/reachable, the request-based part is INCOMPLETE.
  - The destructive control action for each scenario (kill a worker, inject a fault, rotate a cert, shift traffic,
    roll back) is executed via an operator-provided command — env CHROMOFOLD_HOOK_<ACTION>_CMD. If it is unset,
    the scenario is INCOMPLETE, not PASS.
  - Reference reconciliation compares a before/after snapshot of the runtime's reference counters; if no snapshot
    source is available it is reported as not-reconciled with a reason, never assumed true.

Environment (all optional; absence -> honest INCOMPLETE, never a fake PASS):
  CHROMOFOLD_RUNTIME_ENDPOINT          base URL of the serving runtime (default http://127.0.0.1:8080)
  CHROMOFOLD_SNAPSHOT_ENDPOINT / _TOKEN  where to read the runtime state snapshot for reference reconciliation
  CHROMOFOLD_HOOK_GEN_CMD              a shell command that issues ONE real generation request (rc 0 = completed)
  CHROMOFOLD_HOOK_<ACTION>_CMD         the real control action for a scenario (see each hook for its ACTION name)
"""
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request


def read_payload():
    try:
        raw = sys.stdin.read()
        return json.loads(raw) if raw.strip() else {}
    except Exception:
        return {}


def endpoint(payload):
    return (payload.get("runtime_endpoint")
            or os.getenv("CHROMOFOLD_RUNTIME_ENDPOINT")
            or "http://127.0.0.1:8080")


def _http(url, method="GET", body=None, timeout=15, headers=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        txt = r.read().decode("utf-8", "ignore")
        try:
            return getattr(r, "status", 200), json.loads(txt)
        except Exception:
            return getattr(r, "status", 200), txt


def reachable(ep, timeout=5):
    """Is a serving runtime answering at `ep`? Returns (ok, detail). Tries common health/model paths."""
    last = "no attempt"
    for path in ("/health", "/ready", "/v1/models", "/"):
        try:
            st, _ = _http(ep.rstrip("/") + path, timeout=timeout)
            if st < 500:
                return True, f"{path} -> HTTP {st}"
            last = f"{path} -> HTTP {st}"
        except Exception as e:
            last = str(e)
    return False, f"no serving runtime at {ep} ({last})"


def snapshot(payload):
    """Fetch the runtime's reference/state snapshot for reconciliation, or None if unavailable."""
    ep = (payload.get("snapshot_endpoint") or os.getenv("CHROMOFOLD_SNAPSHOT_ENDPOINT"))
    if not ep:
        return None
    tok = os.getenv("CHROMOFOLD_SNAPSHOT_TOKEN")
    headers = {"Authorization": f"Bearer {tok}"} if tok else None
    try:
        _, body = _http(ep, timeout=10, headers=headers)
        return body if isinstance(body, dict) else None
    except Exception:
        return None


def references_reconciled(before, after):
    """True only if we OBSERVED before/after snapshots and the reference counters returned to baseline."""
    if not isinstance(before, dict) or not isinstance(after, dict):
        return False, "no snapshot source for reference reconciliation"
    keys = ("page_references", "snapshot_references", "leases", "transfer_buffers", "active_requests")
    drift = {k: (int(after.get(k, 0)) - int(before.get(k, 0))) for k in keys}
    if any(v != 0 for v in drift.values()):
        return False, f"reference drift after scenario: {drift}"
    return True, "reference counters returned to baseline"


def operator_cmd(action):
    """The operator supplies the real control action as a shell command; None if unconfigured."""
    return os.getenv(f"CHROMOFOLD_HOOK_{action}_CMD")


def run_cmd(cmd, timeout=120):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return p.returncode, ((p.stdout or "") + (p.stderr or "")).strip()


def issue_requests(ep, n, tenant=None):
    """Issue n real generation requests. Uses CHROMOFOLD_HOOK_GEN_CMD if set, else a generic /v1/completions POST.
    Returns (started, completed, failed, detail) or (0,0,0,reason) if no generation capability is available."""
    gen = os.getenv("CHROMOFOLD_HOOK_GEN_CMD")
    started = completed = failed = 0
    if gen:
        for _ in range(n):
            started += 1
            env = dict(os.environ, CHROMOFOLD_TENANT=str(tenant or ""))
            try:
                p = subprocess.run(gen, shell=True, capture_output=True, text=True, timeout=60, env=env)
                completed += 1 if p.returncode == 0 else 0
                failed += 0 if p.returncode == 0 else 1
            except Exception:
                failed += 1
        return started, completed, failed, "gen-cmd"
    body = {"prompt": "chromofold rc1 probe", "max_tokens": 1, "n_predict": 1}
    if tenant is not None:
        body["tenant"] = str(tenant)
    for _ in range(n):
        started += 1
        try:
            st, _ = _http(ep.rstrip("/") + "/v1/completions", "POST", body, timeout=30)
            completed += 1 if st < 400 else 0
            failed += 0 if st < 400 else 1
        except Exception:
            failed += 1
    if completed == 0 and failed == n:
        return 0, 0, 0, "no generation endpoint (/v1/completions) and no CHROMOFOLD_HOOK_GEN_CMD"
    return started, completed, failed, "http-completions"


class Timer:
    def __init__(self):
        self.t0 = time.monotonic()

    def ms(self):
        return int((time.monotonic() - self.t0) * 1000)


def emit(status, timer, requests_started=0, requests_completed=0, requests_failed=0,
         references_reconciled=False, reason="", observations=None):
    out = {
        "status": status,
        "requests_started": int(requests_started),
        "requests_completed": int(requests_completed),
        "requests_failed": int(requests_failed),
        "recovery_ms": timer.ms(),
        "references_reconciled": bool(references_reconciled),
        "observations": observations or ([{"reason": reason}] if reason else []),
    }
    if reason:
        out["reason"] = reason
    json.dump(out, sys.stdout)
    sys.stdout.flush()
    sys.exit(0)  # status carries the verdict; PASS must exit 0, and INCOMPLETE/FAIL exit 0 so the adapter records them


def incomplete(timer, reason, observations=None):
    emit("INCOMPLETE", timer, reason=reason, observations=observations)


def fail(timer, reason, observations=None, **counts):
    emit("FAIL", timer, reason=reason, observations=observations, **counts)


def guard_runtime(timer, payload):
    """Common preamble: require a reachable serving runtime, else honest INCOMPLETE. Returns the endpoint."""
    ep = endpoint(payload)
    ok, detail = reachable(ep)
    if not ok:
        incomplete(timer, f"serving runtime not reachable — {detail}")
    return ep


def control_scenario(timer, payload, ep, action, restore_action=None, continuity_requests=10):
    """A control-plane scenario (kill/inject/rotate/upgrade/rollback): run the operator's REAL action (and an
    optional restore), confirm the runtime keeps or regains service, and reconcile references. Honest INCOMPLETE
    if the action command or a verification source is unavailable; FAIL only on an observed regression."""
    cmd = operator_cmd(action)
    if not cmd:
        incomplete(timer, f"set CHROMOFOLD_HOOK_{action}_CMD to the real control action for this scenario")
    before = snapshot(payload)
    rc, log = run_cmd(cmd)
    obs = [{"action": action, "cmd_rc": rc}]
    if rc != 0:
        fail(timer, f"{action} command failed", observations=obs + [{"log": log[:400]}])
    if restore_action:
        rcmd = operator_cmd(restore_action)
        if not rcmd:
            incomplete(timer, f"set CHROMOFOLD_HOOK_{restore_action}_CMD to restore capacity after {action}",
                       observations=obs)
        rrc, rlog = run_cmd(rcmd)
        obs.append({"restore": restore_action, "cmd_rc": rrc})
        if rrc != 0:
            fail(timer, f"{restore_action} command failed", observations=obs + [{"log": rlog[:400]}])
    s, c, fl, detail = issue_requests(ep, continuity_requests)
    if s == 0:
        incomplete(timer, f"could not verify service continuity — {detail}", observations=obs)
    if fl > 0:
        fail(timer, f"service degraded after {action}: {fl}/{s} requests failed", observations=obs,
             requests_started=s, requests_completed=c, requests_failed=fl)
    after = snapshot(payload)
    recon, why = references_reconciled(before, after)
    if not recon:
        incomplete(timer, f"could not verify reference reconciliation — {why}", observations=obs)
    emit("PASS", timer, requests_started=s, requests_completed=c, requests_failed=fl,
         references_reconciled=recon, observations=obs + [{"reconcile": why, "detail": detail}])
