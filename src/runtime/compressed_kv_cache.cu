#include "chromofold/compressed_kv_cache.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace chromofold {
namespace {

struct ActivePage {
    std::uint32_t token_begin = 0;
    std::vector<float> keys;
    std::vector<float> values;
};

struct HeadState {
    ActivePage active;
    std::vector<cuda_runtime::DeviceKvPage> sealed;
};

std::size_t slot_index(const KvCacheConfig& config, std::uint32_t layer, std::uint32_t head) {
    if (layer >= config.layer_count || head >= config.kv_head_count) throw std::out_of_range("KV layer/head out of range");
    return static_cast<std::size_t>(layer) * config.kv_head_count + head;
}

void validate_config(const KvCacheConfig& config) {
    if (config.layer_count == 0 || config.kv_head_count == 0 || config.query_head_count == 0 ||
        config.head_dim == 0 || config.head_dim > CF_KV_MAX_HEAD_DIM || config.page_size == 0 ||
        config.gqa_group_size == 0 ||
        config.query_head_count != config.kv_head_count * config.gqa_group_size) {
        throw std::invalid_argument("invalid compressed KV cache configuration");
    }
}

}  // namespace

struct CompressedKvCache::Impl {
    explicit Impl(const KvCacheConfig& config)
        : heads(static_cast<std::size_t>(config.layer_count) * config.kv_head_count) {}

    ~Impl() {
        if (active_k_dev != nullptr) cudaFree(active_k_dev);
        if (active_v_dev != nullptr) cudaFree(active_v_dev);
    }

    std::vector<HeadState> heads;
    cuda_runtime::DeviceKvPageArray descriptor_array;
    std::uint64_t appended_tokens = 0;
    std::uint64_t sealed_tokens = 0;
    float* active_k_dev = nullptr;   // interleaved [active_token_count, kv_head_count, head_dim] fp32 tail
    float* active_v_dev = nullptr;
};

CompressedKvCache::CompressedKvCache(KvCacheConfig config) : impl_(nullptr), config_(config) {
    validate_config(config_);
    impl_ = std::make_unique<Impl>(config_);
}

CompressedKvCache::CompressedKvCache(CompressedKvCache&&) noexcept = default;
CompressedKvCache& CompressedKvCache::operator=(CompressedKvCache&&) noexcept = default;
CompressedKvCache::~CompressedKvCache() = default;

void CompressedKvCache::append(std::uint32_t layer,
                               std::uint32_t kv_head,
                               std::uint32_t token_begin,
                               const float* keys,
                               const float* values,
                               std::uint32_t token_count,
                               void* stream) {
    if (keys == nullptr || values == nullptr || token_count == 0) throw std::invalid_argument("invalid KV append");
    HeadState& state = impl_->heads[slot_index(config_, layer, kv_head)];
    const std::uint32_t active_tokens = static_cast<std::uint32_t>(state.active.keys.size() / config_.head_dim);
    if (active_tokens == 0) {
        state.active.token_begin = token_begin;
    } else if (token_begin != state.active.token_begin + active_tokens) {
        throw std::invalid_argument("KV append must be logically contiguous");
    }

    const std::size_t values_count = static_cast<std::size_t>(token_count) * config_.head_dim;
    state.active.keys.insert(state.active.keys.end(), keys, keys + values_count);
    state.active.values.insert(state.active.values.end(), values, values + values_count);
    impl_->appended_tokens += token_count;
    seal_full_pages(stream);
}

void CompressedKvCache::seal_full_pages(void* stream) {
    for (std::uint32_t layer = 0; layer < config_.layer_count; ++layer) {
        for (std::uint32_t head = 0; head < config_.kv_head_count; ++head) {
            HeadState& state = impl_->heads[slot_index(config_, layer, head)];
            while (state.active.keys.size() / config_.head_dim >= config_.page_size) {
                const std::size_t count = static_cast<std::size_t>(config_.page_size) * config_.head_dim;
                std::vector<float> keys(state.active.keys.begin(), state.active.keys.begin() + count);
                std::vector<float> values(state.active.values.begin(), state.active.values.begin() + count);
                auto encoded = (config_.codec == KvCodecMode::fixed_int8_huffman)
                    ? gpu_fixture::encode_int8_page(keys, values, state.active.token_begin, config_.page_size, config_.head_dim, head, 64)
                    : gpu_fixture::encode_int4_page(keys, values, state.active.token_begin, config_.page_size, config_.head_dim, head, 64);
                auto device_page = cuda_runtime::DeviceKvPage::upload(encoded, stream);
                state.sealed.push_back(std::move(device_page));
                state.active.keys.erase(state.active.keys.begin(), state.active.keys.begin() + count);
                state.active.values.erase(state.active.values.begin(), state.active.values.begin() + count);
                state.active.token_begin += config_.page_size;
                impl_->sealed_tokens += config_.page_size;
            }
        }
    }
}

void CompressedKvCache::flush(void* stream) {
    for (std::uint32_t layer = 0; layer < config_.layer_count; ++layer) {
        for (std::uint32_t head = 0; head < config_.kv_head_count; ++head) {
            HeadState& state = impl_->heads[slot_index(config_, layer, head)];
            const std::uint32_t tokens = static_cast<std::uint32_t>(state.active.keys.size() / config_.head_dim);
            if (tokens == 0) continue;
            auto encoded = (config_.codec == KvCodecMode::fixed_int8_huffman)
                ? gpu_fixture::encode_int8_page(state.active.keys, state.active.values, state.active.token_begin, tokens, config_.head_dim, head, 64)
                : gpu_fixture::encode_int4_page(state.active.keys, state.active.values, state.active.token_begin, tokens, config_.head_dim, head, 64);
            state.sealed.push_back(cuda_runtime::DeviceKvPage::upload(encoded, stream));
            impl_->sealed_tokens += tokens;
            state.active = {};
        }
    }
}

KvAttentionView CompressedKvCache::attention_view(std::uint32_t layer, void* stream) {
    if (layer >= config_.layer_count) throw std::out_of_range("KV layer out of range");
    KvAttentionView view;
    for (std::uint32_t head = 0; head < config_.kv_head_count; ++head) {
        const HeadState& state = impl_->heads[slot_index(config_, layer, head)];
        for (const auto& page : state.sealed) view.host_descriptors.push_back(page.descriptor());
    }
    if (!view.host_descriptors.empty()) {
        impl_->descriptor_array = cuda_runtime::DeviceKvPageArray::upload(view.host_descriptors, stream);
        view.device_descriptors = impl_->descriptor_array.device_data();
        view.page_count = impl_->descriptor_array.size();
    }

    // Expose the unsealed active tail as a dense fp32 device buffer with the kernel's expected layout
    // [active_token_count, kv_head_count, head_dim] (all kv heads share token_begin/token_count by construction).
    const std::uint32_t hd = config_.head_dim, kvh = config_.kv_head_count;
    const HeadState& h0 = impl_->heads[slot_index(config_, layer, 0)];
    const std::uint32_t active_tokens = static_cast<std::uint32_t>(h0.active.keys.size() / hd);
    if (active_tokens > 0) {
        std::vector<float> kbuf(static_cast<std::size_t>(active_tokens) * kvh * hd);
        std::vector<float> vbuf(kbuf.size());
        for (std::uint32_t head = 0; head < kvh; ++head) {
            const HeadState& st = impl_->heads[slot_index(config_, layer, head)];
            const std::uint32_t st_tokens = static_cast<std::uint32_t>(st.active.keys.size() / hd);
            if (st_tokens != active_tokens || st.active.token_begin != h0.active.token_begin) {
                throw std::runtime_error("KV active tails are not aligned across heads");
            }
            for (std::uint32_t tok = 0; tok < active_tokens; ++tok) {
                const std::size_t dst = (static_cast<std::size_t>(tok) * kvh + head) * hd;
                const std::size_t src = static_cast<std::size_t>(tok) * hd;
                std::copy(st.active.keys.begin() + src, st.active.keys.begin() + src + hd, kbuf.begin() + dst);
                std::copy(st.active.values.begin() + src, st.active.values.begin() + src + hd, vbuf.begin() + dst);
            }
        }
        cudaStream_t s = static_cast<cudaStream_t>(stream);
        if (impl_->active_k_dev != nullptr) { cudaFree(impl_->active_k_dev); impl_->active_k_dev = nullptr; }
        if (impl_->active_v_dev != nullptr) { cudaFree(impl_->active_v_dev); impl_->active_v_dev = nullptr; }
        const std::size_t bytes = kbuf.size() * sizeof(float);
        if (cudaMalloc(reinterpret_cast<void**>(&impl_->active_k_dev), bytes) != cudaSuccess ||
            cudaMalloc(reinterpret_cast<void**>(&impl_->active_v_dev), bytes) != cudaSuccess) {
            throw std::runtime_error("failed to allocate active KV tail on device");
        }
        cudaMemcpyAsync(impl_->active_k_dev, kbuf.data(), bytes, cudaMemcpyHostToDevice, s);
        cudaMemcpyAsync(impl_->active_v_dev, vbuf.data(), bytes, cudaMemcpyHostToDevice, s);
        if (s == nullptr) cudaStreamSynchronize(nullptr);
        view.active_k = impl_->active_k_dev;
        view.active_v = impl_->active_v_dev;
        view.active_token_begin = h0.active.token_begin;
        view.active_token_count = active_tokens;
    }
    return view;
}

void CompressedKvCache::remove_sequence(std::uint64_t sequence_id) {
    if (sequence_id != 0) throw std::invalid_argument("current ChromoFold cache supports sequence 0 only");
    clear();
}

void CompressedKvCache::copy_sequence(std::uint64_t source_sequence_id, std::uint64_t destination_sequence_id) {
    if (source_sequence_id == destination_sequence_id) return;
    throw std::runtime_error("sequence copy requires the forthcoming shared-page table");
}

void CompressedKvCache::shift_sequence(std::uint64_t sequence_id, std::int32_t delta) {
    if (sequence_id != 0 || delta != 0) throw std::runtime_error("non-zero context shift is not implemented");
}

void CompressedKvCache::clear() {
    impl_ = std::make_unique<Impl>(config_);
}

KvCacheStats CompressedKvCache::stats() const noexcept {
    KvCacheStats result;
    result.appended_tokens = impl_->appended_tokens;
    result.sealed_tokens = impl_->sealed_tokens;
    for (const HeadState& state : impl_->heads) {
        result.sealed_pages += state.sealed.size();
        result.active_tokens += state.active.keys.size() / config_.head_dim;
        result.dense_active_bytes += (state.active.keys.size() + state.active.values.size()) * sizeof(float);
        for (const auto& page : state.sealed) result.compressed_bytes += page.resident_bytes();
    }
    result.descriptor_bytes = result.sealed_pages * sizeof(cf_kv_device_page);
    return result;
}

}  // namespace chromofold
