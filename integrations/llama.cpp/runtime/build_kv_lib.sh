#!/usr/bin/env bash
# Precompile the ChromoFold compressed-KV engine into a position-independent static lib that
# llama-common links (see common/CMakeLists.txt block guarded by GGML_CHROMOFOLD). Building with
# nvcc here keeps the CUDA language out of llama's CMake scope. -fPIC is required because
# llama-common is a shared library.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
OUT="${1:-$REPO/build/libchromofold_kv}"
ARCH="${ARCH:-sm_75}"
mkdir -p "$OUT"
rm -f "$OUT"/*.o "$OUT"/*.a
for src in \
    integrations/llama.cpp/chromofold_kv_adapter.cpp \
    src/runtime/compressed_kv_cache.cu \
    src/cuda/kv_cuda_owner.cu \
    src/cuda/paged_kv_attention.cu \
    src/runtime/kv_gpu_fixture.cpp; do
    nvcc -O2 -std=c++17 -arch="$ARCH" -Xcompiler -fPIC \
        -I"$REPO/include" -I"$REPO/integrations/llama.cpp" \
        -c "$REPO/$src" -o "$OUT/$(basename "${src%.*}").o"
done
ar rcs "$OUT/libchromofold_kv.a" "$OUT"/*.o
echo "built $OUT/libchromofold_kv.a ($(stat -c%s "$OUT/libchromofold_kv.a") bytes, $(nm "$OUT/libchromofold_kv.a" | grep -c cf_llama_kv) cf_llama_kv symbols)"
