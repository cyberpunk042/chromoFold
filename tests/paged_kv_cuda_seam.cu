#include "chromofold/kv_cuda.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>

int main() {
    cf_kv_paged_attention_desc desc{};
    desc.struct_size = sizeof(desc);
    desc.abi_version = CF_KV_CUDA_ABI_VERSION;
    desc.queries = reinterpret_cast<const float *>(uintptr_t{0x1000});
    desc.output = reinterpret_cast<float *>(uintptr_t{0x2000});
    desc.query_count = 1;
    desc.query_head_count = 8;
    desc.kv_head_count = 2;
    desc.gqa_group_size = 4;
    desc.head_dim = 64;
    desc.softmax_scale = 1.0f / std::sqrt(64.0f);

    assert(cf_kv_validate_paged_attention_desc(&desc) == CF_OK);

    cf_kv_paged_attention_desc invalid = desc;
    invalid.gqa_group_size = 3;
    assert(cf_kv_validate_paged_attention_desc(&invalid) == CF_ERR_INVALID_ARGUMENT);

    invalid = desc;
    invalid.page_count = 1;
    invalid.pages = nullptr;
    assert(cf_kv_validate_paged_attention_desc(&invalid) == CF_ERR_INVALID_ARGUMENT);

    invalid = desc;
    invalid.active_token_count = 1;
    assert(cf_kv_validate_paged_attention_desc(&invalid) == CF_ERR_INVALID_ARGUMENT);

    invalid = desc;
    invalid.head_dim = CF_KV_MAX_HEAD_DIM + 1;
    assert(cf_kv_validate_paged_attention_desc(&invalid) == CF_ERR_INVALID_ARGUMENT);

    invalid = desc;
    invalid.softmax_scale = NAN;
    assert(cf_kv_validate_paged_attention_desc(&invalid) == CF_ERR_INVALID_ARGUMENT);

    std::cout << "paged CUDA KV ABI seam passed\n";
    return 0;
}
