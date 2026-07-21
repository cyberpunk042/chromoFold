#include "chromofold_runtime_bridge.h"

#include <cstdio>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

struct cf_llama_runtime {
    cf_llama_runtime_options options{};
    std::unique_ptr<cf_llama_kv_adapter, void (*)(cf_llama_kv_adapter *)> adapter{nullptr, cf_llama_kv_destroy};
    cf_llama_runtime_counters counters{};
    std::string last_error;
};

namespace {

template <typename Operation>
int guarded(cf_llama_runtime * runtime, Operation && operation) {
    if (runtime == nullptr) return -1;
    try {
        operation();
        runtime->last_error.clear();
        return 0;
    } catch (const std::exception & error) {
        runtime->last_error = error.what();
    } catch (...) {
        runtime->last_error = "unknown ChromoFold runtime error";
    }
    runtime->counters.rejected_operations++;
    return -1;
}

bool contiguous_f32(const cf_llama_tensor_view & view) {
    return view.data != nullptr && view.scalar_type == CF_LLAMA_SCALAR_F32 && view.token_count != 0 &&
           view.head_count != 0 && view.head_dim != 0 && view.element_stride_bytes == sizeof(float) &&
           view.head_stride_bytes == static_cast<uint64_t>(view.head_dim) * sizeof(float) &&
           view.token_stride_bytes == static_cast<uint64_t>(view.head_count) * view.head_stride_bytes;
}

}  // namespace

extern "C" int cf_llama_runtime_options_validate(const cf_llama_runtime_options * options) {
    if (options == nullptr || options->struct_size != sizeof(cf_llama_runtime_options)) return -1;
    if (cf_llama_kv_options_validate(&options->kv) != 0) return -1;
    if (options->kv.backend == CF_LLAMA_KV_BACKEND_CHROMOFOLD && options->allow_dense_fallback != 0) return -1;
    if (options->kv.backend == CF_LLAMA_KV_BACKEND_CHROMOFOLD && options->require_cuda == 0) return -1;
    return 0;
}

extern "C" cf_llama_runtime * cf_llama_runtime_create(const cf_llama_runtime_options * options) {
    if (cf_llama_runtime_options_validate(options) != 0) return nullptr;
    try {
        auto runtime = std::make_unique<cf_llama_runtime>();
        runtime->options = *options;
        if (options->kv.backend == CF_LLAMA_KV_BACKEND_CHROMOFOLD) {
            runtime->adapter.reset(cf_llama_kv_create(&options->kv));
            if (!runtime->adapter) return nullptr;
        }
        return runtime.release();
    } catch (...) {
        return nullptr;
    }
}

extern "C" void cf_llama_runtime_destroy(cf_llama_runtime * runtime) { delete runtime; }

extern "C" int cf_llama_runtime_append(cf_llama_runtime * runtime,
                                        uint32_t layer,
                                        uint32_t token_begin,
                                        const cf_llama_tensor_view * keys,
                                        const cf_llama_tensor_view * values,
                                        void * stream) {
    return guarded(runtime, [&] {
        if (runtime->options.kv.backend != CF_LLAMA_KV_BACKEND_CHROMOFOLD || !runtime->adapter) {
            throw std::runtime_error("ChromoFold backend is not selected");
        }
        if (keys == nullptr || values == nullptr || !contiguous_f32(*keys) || !contiguous_f32(*values)) {
            throw std::runtime_error("initial runtime bridge requires contiguous fp32 K/V tensors");
        }
        if (keys->token_count != values->token_count || keys->head_count != values->head_count ||
            keys->head_dim != values->head_dim || keys->head_count != runtime->options.kv.kv_head_count ||
            keys->head_dim != runtime->options.kv.head_dim) {
            throw std::runtime_error("K/V tensor topology does not match configured cache");
        }
        const auto * k = static_cast<const float *>(keys->data);
        const auto * v = static_cast<const float *>(values->data);
        for (uint32_t head = 0; head < keys->head_count; ++head) {
            std::vector<float> head_k(static_cast<size_t>(keys->token_count) * keys->head_dim);
            std::vector<float> head_v(head_k.size());
            for (uint32_t token = 0; token < keys->token_count; ++token) {
                const size_t source = (static_cast<size_t>(token) * keys->head_count + head) * keys->head_dim;
                const size_t destination = static_cast<size_t>(token) * keys->head_dim;
                for (uint32_t dim = 0; dim < keys->head_dim; ++dim) {
                    head_k[destination + dim] = k[source + dim];
                    head_v[destination + dim] = v[source + dim];
                }
            }
            if (cf_llama_kv_append(runtime->adapter.get(), layer, head, token_begin, head_k.data(), head_v.data(),
                                   keys->token_count, stream) != 0) {
                throw std::runtime_error(cf_llama_kv_last_error(runtime->adapter.get()));
            }
        }
        runtime->counters.append_calls++;
        runtime->counters.appended_values += static_cast<uint64_t>(keys->token_count) * keys->head_count * keys->head_dim * 2u;
    });
}

extern "C" int cf_llama_runtime_record_compressed_launch(cf_llama_runtime * runtime,
                                                           uint64_t sealed_values,
                                                           uint64_t active_values) {
    return guarded(runtime, [&] {
        if (runtime->options.kv.backend != CF_LLAMA_KV_BACKEND_CHROMOFOLD) {
            throw std::runtime_error("compressed launch recorded for dense backend");
        }
        runtime->counters.compressed_attention_launches++;
        runtime->counters.sealed_values_consumed += sealed_values;
        runtime->counters.active_values_consumed += active_values;
    });
}

extern "C" int cf_llama_runtime_record_dense_fallback(cf_llama_runtime * runtime) {
    return guarded(runtime, [&] {
        runtime->counters.dense_fallback_launches++;
        if (runtime->options.allow_dense_fallback == 0) throw std::runtime_error("dense fallback is disabled");
    });
}

extern "C" int cf_llama_runtime_record_cuda_error(cf_llama_runtime * runtime) {
    return guarded(runtime, [&] {
        runtime->counters.cuda_errors++;
        throw std::runtime_error("CUDA execution failed");
    });
}

extern "C" int cf_llama_runtime_snapshot_get(const cf_llama_runtime * runtime,
                                               cf_llama_runtime_snapshot * snapshot) {
    if (runtime == nullptr || snapshot == nullptr) return -1;
    *snapshot = {};
    snapshot->backend_selected = runtime->options.kv.backend;
    snapshot->initialized = 1;
    snapshot->dense_fallback_allowed = runtime->options.allow_dense_fallback;
    snapshot->counters = runtime->counters;
    if (runtime->adapter && cf_llama_kv_get_stats(runtime->adapter.get(), &snapshot->kv) != 0) return -1;
    snapshot->evidence_complete = runtime->options.kv.backend == CF_LLAMA_KV_BACKEND_CHROMOFOLD &&
                                  runtime->counters.compressed_attention_launches > 0 &&
                                  runtime->counters.sealed_values_consumed > 0 &&
                                  runtime->counters.dense_fallback_launches == 0 &&
                                  runtime->counters.cuda_errors == 0;
    return 0;
}

extern "C" int cf_llama_runtime_write_json(const cf_llama_runtime * runtime, const char * path) {
    if (runtime == nullptr || path == nullptr) return -1;
    cf_llama_runtime_snapshot snapshot{};
    if (cf_llama_runtime_snapshot_get(runtime, &snapshot) != 0) return -1;
    FILE * file = std::fopen(path, "wb");
    if (file == nullptr) return -1;
    const int result = std::fprintf(file,
        "{\n  \"backend_selected\":%u,\n  \"evidence_complete\":%u,\n  \"kv\":{\"appended_tokens\":%llu,\"sealed_tokens\":%llu,\"sealed_pages\":%llu,\"active_tokens\":%llu,\"dense_active_bytes\":%llu,\"compressed_bytes\":%llu,\"descriptor_bytes\":%llu},\n  \"counters\":{\"append_calls\":%llu,\"appended_values\":%llu,\"compressed_attention_launches\":%llu,\"sealed_values_consumed\":%llu,\"active_values_consumed\":%llu,\"dense_fallback_launches\":%llu,\"rejected_operations\":%llu,\"cuda_errors\":%llu}\n}\n",
        snapshot.backend_selected, snapshot.evidence_complete,
        static_cast<unsigned long long>(snapshot.kv.appended_tokens),
        static_cast<unsigned long long>(snapshot.kv.sealed_tokens),
        static_cast<unsigned long long>(snapshot.kv.sealed_pages),
        static_cast<unsigned long long>(snapshot.kv.active_tokens),
        static_cast<unsigned long long>(snapshot.kv.dense_active_bytes),
        static_cast<unsigned long long>(snapshot.kv.compressed_bytes),
        static_cast<unsigned long long>(snapshot.kv.descriptor_bytes),
        static_cast<unsigned long long>(snapshot.counters.append_calls),
        static_cast<unsigned long long>(snapshot.counters.appended_values),
        static_cast<unsigned long long>(snapshot.counters.compressed_attention_launches),
        static_cast<unsigned long long>(snapshot.counters.sealed_values_consumed),
        static_cast<unsigned long long>(snapshot.counters.active_values_consumed),
        static_cast<unsigned long long>(snapshot.counters.dense_fallback_launches),
        static_cast<unsigned long long>(snapshot.counters.rejected_operations),
        static_cast<unsigned long long>(snapshot.counters.cuda_errors));
    std::fclose(file);
    return result < 0 ? -1 : 0;
}

extern "C" const char * cf_llama_runtime_last_error(const cf_llama_runtime * runtime) {
    return runtime == nullptr ? "null runtime" : runtime->last_error.c_str();
}
