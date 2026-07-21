#!/usr/bin/env python3
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = json.loads((ROOT / "patch_manifest.json").read_text())
BEGIN = "# BEGIN CHROMOFOLD RUNTIME\n"
END = "# END CHROMOFOLD RUNTIME\n"


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
        raise SystemExit("missing upstream files: " + ", ".join(missing))
    if args.check:
        print(json.dumps({"commit": actual, "applicable": True, "files": len(MANIFEST["generated_files"])}))
        return 0

    source = args.chromofold_root.resolve()
    overlay = checkout / "chromofold-overlay"
    overlay.mkdir(exist_ok=True)
    for relative in (
        "include/chromofold",
        "src/runtime",
        "src/cuda",
        "integrations/llama.cpp/chromofold_kv_adapter.h",
        "integrations/llama.cpp/chromofold_kv_adapter.cpp",
        "integrations/llama.cpp/chromofold_runtime_bridge.h",
        "integrations/llama.cpp/chromofold_runtime_bridge.cpp",
    ):
        src = source / relative
        dst = overlay / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            shutil.rmtree(dst) if dst.is_dir() else dst.unlink()
        shutil.copytree(src, dst) if src.is_dir() else shutil.copy2(src, dst)

    write(checkout / "cmake/chromofold-runtime.cmake", """option(GGML_CHROMOFOLD \"Enable ChromoFold compressed KV backend\" OFF)\nif (GGML_CHROMOFOLD)\n  if (NOT GGML_CUDA)\n    message(FATAL_ERROR \"GGML_CHROMOFOLD requires GGML_CUDA\")\n  endif()\n  set(CHROMOFOLD_ROOT \"${CMAKE_CURRENT_LIST_DIR}/../chromofold-overlay\")\n  add_library(chromofold-runtime STATIC\n    ${CHROMOFOLD_ROOT}/integrations/llama.cpp/chromofold_kv_adapter.cpp\n    ${CHROMOFOLD_ROOT}/integrations/llama.cpp/chromofold_runtime_bridge.cpp\n    ${CHROMOFOLD_ROOT}/src/runtime/kv_gpu_fixture.cpp\n    ${CHROMOFOLD_ROOT}/src/runtime/compressed_kv_cache.cu\n    ${CHROMOFOLD_ROOT}/src/cuda/kv_cuda_owner.cu\n    ${CHROMOFOLD_ROOT}/src/cuda/paged_kv_attention.cu)\n  target_include_directories(chromofold-runtime PUBLIC\n    ${CHROMOFOLD_ROOT}/include\n    ${CHROMOFOLD_ROOT}/integrations/llama.cpp)\n  target_compile_features(chromofold-runtime PUBLIC cxx_std_17)\n  target_link_libraries(chromofold-runtime PUBLIC CUDA::cudart)\nendif()\n""")
    write(checkout / "common/chromofold-options.h", """#pragma once\n#include <cstdint>\n#include <string>\nstruct common_chromofold_params {\n    std::string backend = \"dense\";\n    std::uint32_t page_size = 128;\n    std::uint32_t window = 0;\n    bool stats = false;\n    std::string evidence_path;\n};\n""")
    write(checkout / "src/chromofold-runtime-hook.h", """#pragma once\n#include \"chromofold_runtime_bridge.h\"\nstruct llama_chromofold_runtime {\n    cf_llama_runtime * handle = nullptr;\n    ~llama_chromofold_runtime();\n};\n""")
    write(checkout / "src/chromofold-runtime-hook.cpp", """#include \"chromofold-runtime-hook.h\"\nllama_chromofold_runtime::~llama_chromofold_runtime() { cf_llama_runtime_destroy(handle); }\n""")
    append_once(checkout / "CMakeLists.txt", "include(cmake/chromofold-runtime.cmake)")

    state = {
        "commit": actual,
        "generated": {p: digest(checkout / p) for p in MANIFEST["generated_files"]},
        "overlay": "chromofold-overlay",
    }
    write(checkout / ".chromofold-runtime-state.json", json.dumps(state, indent=2) + "\n")
    print(json.dumps(state, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
