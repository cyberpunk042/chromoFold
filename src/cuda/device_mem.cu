// device_mem.cu — thin, stable-ABI device-memory + stream helpers (chromofold_search.h, device-native section).
// These let a caller that cannot (or does not want to) link cudart — e.g. a Rust `-sys` FFI — allocate resident
// device buffers, move data in/out, and manage a stream, then drive the device-native cf_fm_*_async kernels with
// zero host round-trip in the hot loop (P5). The engine keeps owning all CUDA; the caller holds opaque void*
// device pointers. Every entry point honors the null_arg_contract before any CUDA call.
#include "chromofold/chromofold.h"

#include <cuda_runtime.h>
#include <cstddef>

extern "C" {

cf_status cf_device_alloc(size_t nbytes, void** out) {
    if (out == nullptr) return CF_ERR_INVALID_ARGUMENT;
    *out = nullptr;
    if (nbytes == 0) nbytes = 1;  // a 0-byte request still yields a unique, freeable pointer
    return cudaMalloc(out, nbytes) == cudaSuccess ? CF_OK : CF_ERR_CUDA;
}

void cf_device_free(void* dptr) {
    if (dptr != nullptr) cudaFree(dptr);
}

cf_status cf_device_h2d(void* ddst, const void* hsrc, size_t nbytes) {
    if (ddst == nullptr || hsrc == nullptr) return CF_ERR_INVALID_ARGUMENT;
    if (nbytes == 0) return CF_OK;
    return cudaMemcpy(ddst, hsrc, nbytes, cudaMemcpyHostToDevice) == cudaSuccess ? CF_OK : CF_ERR_CUDA;
}

cf_status cf_device_d2h(void* hdst, const void* dsrc, size_t nbytes) {
    if (hdst == nullptr || dsrc == nullptr) return CF_ERR_INVALID_ARGUMENT;
    if (nbytes == 0) return CF_OK;
    return cudaMemcpy(hdst, dsrc, nbytes, cudaMemcpyDeviceToHost) == cudaSuccess ? CF_OK : CF_ERR_CUDA;
}

cf_status cf_stream_create(void** out) {
    if (out == nullptr) return CF_ERR_INVALID_ARGUMENT;
    *out = nullptr;
    cudaStream_t s = nullptr;
    if (cudaStreamCreate(&s) != cudaSuccess) return CF_ERR_CUDA;
    *out = static_cast<void*>(s);
    return CF_OK;
}

void cf_stream_destroy(void* stream) {
    if (stream != nullptr) cudaStreamDestroy(static_cast<cudaStream_t>(stream));
}

cf_status cf_stream_sync(void* stream) {
    // NULL == the default stream, which is a valid sync target.
    return cudaStreamSynchronize(static_cast<cudaStream_t>(stream)) == cudaSuccess ? CF_OK : CF_ERR_CUDA;
}

}  // extern "C"
