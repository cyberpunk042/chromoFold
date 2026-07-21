#!/usr/bin/env python3
import argparse
import hashlib
import json
import pathlib
import subprocess
import sys

MARKER_BEGIN = "// CHROMOFOLD_M11_BEGIN"
MARKER_END = "// CHROMOFOLD_M11_END"


def fail(message: str) -> None:
    raise SystemExit(message)


def git_head(root: pathlib.Path) -> str:
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


def load_manifest(path: pathlib.Path) -> dict:
    return json.loads(path.read_text())


def verify(root: pathlib.Path, manifest: dict) -> None:
    expected = manifest["upstream"]["commit"]
    actual = git_head(root)
    if actual != expected:
        fail(f"llama.cpp pin mismatch: expected {expected}, got {actual}")
    for rel in manifest["required_files"]:
        if not (root / rel).is_file():
            fail(f"required upstream file missing: {rel}")
    for anchor in manifest["anchors"]:
        text = (root / anchor["file"]).read_text(errors="strict")
        if anchor["symbol"] not in text:
            fail(f"upstream anchor missing: {anchor['file']}::{anchor['symbol']}")


def insert_once(path: pathlib.Path, anchor: str, block: str) -> bool:
    text = path.read_text()
    if MARKER_BEGIN in text:
        return False
    index = text.find(anchor)
    if index < 0:
        fail(f"cannot patch {path}: anchor {anchor!r} not found")
    replacement = f"{MARKER_BEGIN}\n{block.rstrip()}\n{MARKER_END}\n"
    path.write_text(text[:index] + replacement + text[index:])
    return True


def apply(root: pathlib.Path) -> list[str]:
    changed = []
    patches = [
        ("common/arg.cpp", "common_params_context", "// ChromoFold CLI registration is generated here.\nstatic const char * chromofold_server_cli_anchor = \"--kv-cache-backend\";"),
        ("examples/server/server.cpp", "server_context", "// Own one ChromoFold runtime per llama_context and bind slot lifecycle here.\n#include \"chromofold/llama_server_runtime.h\""),
        ("src/llama-context.cpp", "llama_decode", "// Route interleaved sequence K/V appends through ChromoFold before graph execution.\nextern int cf_llama_server_decode_hook(void *, const void *);"),
        ("src/llama-kv-cache.cpp", "llama_kv_cache_seq_cp", "// Mirror sequence copy/remove/shift into the ChromoFold authoritative cache.\nextern int cf_llama_server_seq_cp_hook(void *, int64_t, int64_t, int32_t, int32_t);"),
        ("src/llama-graph.cpp", "build_attn", "// Replace supported dense KV attention nodes with the ChromoFold compressed dispatcher.\nextern void * cf_llama_server_attention_hook(void *, const void *);"),
    ]
    for rel, anchor, block in patches:
        if insert_once(root / rel, anchor, block):
            changed.append(rel)
    return changed


def fingerprint(root: pathlib.Path, files: list[str]) -> dict:
    result = {}
    for rel in files:
        result[rel] = hashlib.sha256((root / rel).read_bytes()).hexdigest()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkout", type=pathlib.Path)
    parser.add_argument("--manifest", type=pathlib.Path, default=pathlib.Path(__file__).with_name("pinned_server_manifest.json"))
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--report", type=pathlib.Path)
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    verify(args.checkout, manifest)
    changed = [] if args.verify_only else apply(args.checkout)
    report = {
        "pin": git_head(args.checkout),
        "changed": changed,
        "idempotent": not changed and any(MARKER_BEGIN in (args.checkout / rel).read_text() for rel in manifest["required_files"]),
        "fingerprints": fingerprint(args.checkout, manifest["required_files"]),
    }
    if args.report:
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
