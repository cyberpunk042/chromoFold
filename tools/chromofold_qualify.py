#!/usr/bin/env python3
import argparse, hashlib, json, pathlib, sys

SENSITIVE = {"prompt", "completion", "authorization", "private_key", "token", "secret"}

def load(path): return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
def redacted(value):
    if isinstance(value, dict):
        return {k: ("[REDACTED]" if k.lower() in SENSITIVE else redacted(v)) for k, v in value.items()}
    if isinstance(value, list): return [redacted(v) for v in value]
    return value

def evaluate(e):
    required = ["runtime", "security", "operations", "release"]
    if any(k not in e for k in required): return "INCOMPLETE", "missing qualification sections"
    r, s, o, rel = e["runtime"], e["security"], e["operations"], e["release"]
    checks = [r.get("requests",0)>0, r.get("prompt_leaks",1)==0, r.get("correctness_failures",1)==0,
              r.get("cuda_errors",1)==0, r.get("cross_tenant_contamination",1)==0,
              r.get("unreconciled_references",1)==0, s.get("mtls_observed") is True,
              s.get("audit_chain_verified") is True, o.get("worker_failure_recovered") is True,
              o.get("rolling_upgrade_observed") is True, o.get("rollback_observed") is True,
              rel.get("qualified_digest") == rel.get("promoted_digest")]
    return ("PASS", "eligible for v1.0.0-rc1") if all(checks) else ("FAIL", "one or more release gates failed")

def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="cmd",required=True)
    e=sub.add_parser("evaluate"); e.add_argument("evidence")
    b=sub.add_parser("bundle"); b.add_argument("source"); b.add_argument("output")
    a=p.parse_args()
    if a.cmd=="evaluate":
        evidence=load(a.evidence); decision, reason=evaluate(evidence)
        out={"decision":decision,"reason":reason,"evidence_sha256":hashlib.sha256(pathlib.Path(a.evidence).read_bytes()).hexdigest()}
        print(json.dumps(out,sort_keys=True)); return 0 if decision=="PASS" else 1
    source=redacted(load(a.source)); pathlib.Path(a.output).write_text(json.dumps(source,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(a.output); return 0
if __name__=="__main__": raise SystemExit(main())
