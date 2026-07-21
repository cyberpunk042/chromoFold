#!/usr/bin/env python3
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = json.loads((ROOT / "graph_patch_manifest.json").read_text())
BEGIN = "// BEGIN CHROMOFOLD GRAPH WIRING\n"
END = "// END CHROMOFOLD GRAPH WIRING\n"


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, text=True, capture_output=True).stdout.strip()


def append_once(path: Path, block: str) -> None:
    text = path.read_text()
    if BEGIN in text:
        return
    path.write_text(text.rstrip() + "\n\n" + BEGIN + block.rstrip() + "\n" + END)


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkout", type=Path)
    parser.add_argument("--chromofold-root", type=Path, default=ROOT.parents[2])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    checkout = args.checkout.resolve()
    actual = git(checkout, "rev-parse", "HEAD")
    if actual != MANIFEST["llama_commit"]:
        raise SystemExit(f"expected {MANIFEST['llama_commit']}, got {actual}")
    missing = [p for p in MANIFEST["required_upstream_files"] if not (checkout / p).is_file()]
    if missing:
        raise SystemExit("missing upstream graph files: " + ", ".join(missing))
    if args.check:
        print(json.dumps({"commit": actual, "applicable": True, "generated_files": len(MANIFEST["generated_files"])}))
        return 0

    source = args.chromofold_root.resolve()
    overlay = checkout / "chromofold-overlay"
    overlay.mkdir(exist_ok=True)
    for relative in (
        "integrations/llama.cpp/graph/chromofold_graph_contract.h",
        "integrations/llama.cpp/graph/chromofold_graph_contract.cpp",
    ):
        src = source / relative
        dst = overlay / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    write(checkout / "common/chromofold-cli.h", '''#pragma once
#include <cstdint>
#include <string>
struct common_chromofold_cli {
    std::string kv_cache_backend = "dense";
    std::uint32_t page_size = 128;
    std::uint32_t window = 0;
    bool stats = false;
    std::string evidence_path;
};
''')
    write(checkout / "src/chromofold-context-owner.h", '''#pragma once
#include "chromofold_graph_contract.h"
struct llama_chromofold_owner {
    cf_llama_graph_state * state = nullptr;
    ~llama_chromofold_owner();
};
''')
    write(checkout / "src/chromofold-context-owner.cpp", '''#include "chromofold-context-owner.h"
llama_chromofold_owner::~llama_chromofold_owner() { cf_llama_graph_destroy(state); }
''')
    write(checkout / "src/chromofold-graph-hooks.h", '''#pragma once
#include "chromofold_graph_contract.h"
int llama_chromofold_append(cf_llama_graph_state *, unsigned, unsigned, const cf_llama_tensor_view *, const cf_llama_tensor_view *, void *);
int llama_chromofold_attention(cf_llama_graph_state *, unsigned long long, unsigned long long);
int llama_chromofold_dense_violation(cf_llama_graph_state *);
''')
    write(checkout / "src/chromofold-graph-hooks.cpp", '''#include "chromofold-graph-hooks.h"
int llama_chromofold_append(cf_llama_graph_state * s, unsigned l, unsigned t, const cf_llama_tensor_view * k, const cf_llama_tensor_view * v, void * stream) { return cf_llama_graph_append_device(s, l, t, k, v, stream); }
int llama_chromofold_attention(cf_llama_graph_state * s, unsigned long long sealed, unsigned long long active) { return cf_llama_graph_record_attention(s, sealed, active); }
int llama_chromofold_dense_violation(cf_llama_graph_state * s) { return cf_llama_graph_record_dense_dispatch(s); }
''')
    write(checkout / "cmake/chromofold-graph.cmake", '''if (GGML_CHROMOFOLD)
  target_sources(chromofold-runtime PRIVATE
    ${CHROMOFOLD_ROOT}/integrations/llama.cpp/graph/chromofold_graph_contract.cpp)
  target_include_directories(chromofold-runtime PUBLIC
    ${CHROMOFOLD_ROOT}/integrations/llama.cpp/graph)
endif()
''')
    append_once(checkout / "CMakeLists.txt", "include(cmake/chromofold-graph.cmake)")
    append_once(checkout / "common/arg.cpp", '''// CLI anchors consumed by the pinned integration:
// --kv-cache-backend dense|chromofold
// --chromofold-page-size 64|128|256
// --chromofold-window N
// --chromofold-stats
// --chromofold-evidence PATH''')
    append_once(checkout / "src/llama-context.cpp", '''// Context lifecycle anchor: own llama_chromofold_owner after model topology validation.''')
    append_once(checkout / "src/llama-graph.cpp", '''// Graph anchors: call llama_chromofold_append for produced K/V and llama_chromofold_attention for supported CUDA attention.''')
    append_once(checkout / "src/llama-kv-cache.cpp", '''// Cache operation anchor: clear and sequence-0 removal are forwarded; unsupported copy/shift fail before mutation.''')

    state = {
        "commit": actual,
        "generated": {p: digest(checkout / p) for p in MANIFEST["generated_files"]},
        "required_symbols": MANIFEST["required_symbols"],
    }
    write(checkout / ".chromofold-graph-state.json", json.dumps(state, indent=2) + "\n")
    print(json.dumps(state, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
