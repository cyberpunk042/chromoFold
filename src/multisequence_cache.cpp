#include "chromofold/multisequence_cache.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

struct SharedPage {
    uint64_t page_id = 0;
    uint64_t generation = 0;
    uint32_t token_begin = 0;
    uint32_t token_count = 0;
    uint64_t resident_bytes = 0;
    uint32_t refcount = 0;
    bool retired = false;
};

struct PageRef {
    uint64_t page_id = 0;
    uint32_t logical_token_begin = 0;
    uint32_t token_count = 0;
};

struct LayerSequenceState {
    std::vector<PageRef> sealed_pages;
    uint32_t active_token_begin = 0;
    uint32_t active_token_count = 0;
    uint64_t active_generation = 0;
    uint64_t descriptor_generation = 0;
};

struct SpeculationState {
    cf_speculation_id id = 0;
    uint32_t baseline_token_count = 0;
    uint64_t baseline_generation = 0;
    bool active = false;
};

struct SequenceState {
    int64_t id = 0;
    std::vector<LayerSequenceState> layers;
    uint64_t topology_generation = 0;
    int64_t logical_shift = 0;
    SpeculationState speculation;
};

struct RetiredPage {
    uint64_t page_id = 0;
    uint64_t resident_bytes = 0;
    uint64_t retirement_generation = 0;
};

bool valid_config(const cf_multisequence_config & config) {
    return config.layer_count > 0 && config.kv_head_count > 0 &&
           config.query_head_count > 0 && config.head_dim > 0 &&
           config.page_size > 0 && config.gqa_group_size > 0 &&
           config.query_head_count == config.kv_head_count * config.gqa_group_size;
}

} // namespace

struct cf_multisequence_cache {
    cf_multisequence_config config{};
    mutable std::mutex mutex;
    std::map<int64_t, SequenceState> sequences;
    std::unordered_map<uint64_t, SharedPage> page_store;
    std::vector<RetiredPage> retired;
    cf_multisequence_counters counters{};
    uint64_t next_page_id = 1;
    uint64_t global_generation = 1;
    cf_speculation_id next_speculation_id = 1;
    std::string last_error;
};

namespace {

void set_error(cf_multisequence_cache * cache, const std::string & message) {
    cache->last_error = message;
}

void update_active_peak(cf_multisequence_cache * cache) {
    cache->counters.active_sequences = cache->sequences.size();
    cache->counters.active_sequences_peak = std::max(
        cache->counters.active_sequences_peak,
        cache->counters.active_sequences);
}

SharedPage & resolve_page(cf_multisequence_cache * cache, uint64_t page_id) {
    auto it = cache->page_store.find(page_id);
    if (it == cache->page_store.end()) throw std::runtime_error("sequence references an unknown page");
    return it->second;
}

void acquire_page(cf_multisequence_cache * cache, uint64_t page_id) {
    SharedPage & page = resolve_page(cache, page_id);
    if (page.retired) throw std::runtime_error("cannot acquire a retired page");
    if (page.refcount == std::numeric_limits<uint32_t>::max()) throw std::runtime_error("page reference count overflow");
    ++page.refcount;
    ++cache->counters.page_ref_increments;
    ++cache->counters.shared_page_references;
    if (page.refcount == 2) ++cache->counters.shared_pages;
}

void release_page(cf_multisequence_cache * cache, uint64_t page_id) {
    SharedPage & page = resolve_page(cache, page_id);
    if (page.refcount == 0) throw std::runtime_error("page reference count underflow");
    if (page.refcount == 2 && cache->counters.shared_pages > 0) --cache->counters.shared_pages;
    --page.refcount;
    ++cache->counters.page_ref_decrements;
    if (cache->counters.shared_page_references > 0) --cache->counters.shared_page_references;
    if (page.refcount == 0) {
        page.retired = true;
        cache->retired.push_back({page.page_id, page.resident_bytes, cache->global_generation});
        cache->counters.pages_pending_reclamation = cache->retired.size();
        cache->counters.pending_reclamation_bytes += page.resident_bytes;
        cache->counters.pending_reclamation_bytes_peak = std::max(
            cache->counters.pending_reclamation_bytes_peak,
            cache->counters.pending_reclamation_bytes);
    }
}

void release_sequence_pages(cf_multisequence_cache * cache, const SequenceState & sequence) {
    for (const auto & layer : sequence.layers) {
        for (const auto & ref : layer.sealed_pages) release_page(cache, ref.page_id);
    }
}

void validate_page_order(const LayerSequenceState & layer) {
    uint64_t end = 0;
    bool first = true;
    for (const auto & ref : layer.sealed_pages) {
        if (ref.token_count == 0) throw std::runtime_error("zero-length sealed page reference");
        if (!first && ref.logical_token_begin < end) throw std::runtime_error("overlapping sequence page ranges");
        end = static_cast<uint64_t>(ref.logical_token_begin) + ref.token_count;
        first = false;
    }
    if (!first && layer.active_token_count > 0 && layer.active_token_begin < end)
        throw std::runtime_error("active tail overlaps sealed history");
}

int audit_locked(const cf_multisequence_cache * cache) {
    std::unordered_map<uint64_t, uint64_t> observed;
    try {
        for (const auto & [sequence_id, sequence] : cache->sequences) {
            if (sequence_id != sequence.id) throw std::runtime_error("sequence map key mismatch");
            if (sequence.layers.size() != cache->config.layer_count) throw std::runtime_error("sequence layer count mismatch");
            for (const auto & layer : sequence.layers) {
                validate_page_order(layer);
                for (const auto & ref : layer.sealed_pages) {
                    auto page = cache->page_store.find(ref.page_id);
                    if (page == cache->page_store.end()) throw std::runtime_error("page reference does not resolve");
                    if (page->second.retired) throw std::runtime_error("live sequence references retired page");
                    ++observed[ref.page_id];
                }
            }
        }
        for (const auto & [page_id, page] : cache->page_store) {
            const uint64_t expected = observed[page_id];
            if (!page.retired && page.refcount != expected) throw std::runtime_error("page reference count mismatch");
            if (page.retired && (page.refcount != 0 || expected != 0)) throw std::runtime_error("invalid retired page state");
        }
        return 0;
    } catch (...) {
        return -1;
    }
}

void maybe_audit(cf_multisequence_cache * cache) {
    if (!cache->config.enable_invariant_auditor) return;
    ++cache->counters.invariant_audits;
    if (audit_locked(cache) != 0) {
        ++cache->counters.invariant_failures;
        throw std::runtime_error("multisequence invariant audit failed");
    }
}

} // namespace

extern "C" cf_multisequence_cache * cf_multisequence_create(const cf_multisequence_config * config) {
    if (!config || !valid_config(*config)) return nullptr;
    try {
        auto cache = std::make_unique<cf_multisequence_cache>();
        cache->config = *config;
        return cache.release();
    } catch (...) {
        return nullptr;
    }
}

extern "C" void cf_multisequence_destroy(cf_multisequence_cache * cache, void *) {
    if (!cache) return;
    std::lock_guard<std::mutex> guard(cache->mutex);
    try {
        for (const auto & [_, sequence] : cache->sequences) release_sequence_pages(cache, sequence);
        cache->sequences.clear();
        for (const auto & retired : cache->retired) cache->page_store.erase(retired.page_id);
        cache->retired.clear();
    } catch (...) {
    }
    delete cache;
}

extern "C" int cf_sequence_create(cf_multisequence_cache * cache, int64_t sequence_id) {
    if (!cache) return -1;
    std::lock_guard<std::mutex> guard(cache->mutex);
    try {
        if (cache->sequences.count(sequence_id)) throw std::runtime_error("sequence already exists");
        SequenceState sequence;
        sequence.id = sequence_id;
        sequence.layers.resize(cache->config.layer_count);
        sequence.topology_generation = ++cache->global_generation;
        cache->sequences.emplace(sequence_id, std::move(sequence));
        ++cache->counters.sequences_created;
        update_active_peak(cache);
        maybe_audit(cache);
        cache->last_error.clear();
        return 0;
    } catch (const std::exception & error) {
        set_error(cache, error.what());
        return -1;
    }
}

extern "C" int cf_sequence_copy_async(cf_multisequence_cache * cache,
                                         int64_t source_id,
                                         int64_t destination_id,
                                         uint32_t token_begin,
                                         uint32_t token_end,
                                         void *) {
    if (!cache || source_id == destination_id || token_end < token_begin) return -1;
    std::lock_guard<std::mutex> guard(cache->mutex);
    try {
        auto source_it = cache->sequences.find(source_id);
        if (source_it == cache->sequences.end()) throw std::runtime_error("source sequence does not exist");
        SequenceState candidate;
        candidate.id = destination_id;
        candidate.layers.resize(cache->config.layer_count);
        candidate.topology_generation = cache->global_generation + 1;

        std::vector<uint64_t> acquired;
        try {
            for (uint32_t layer_index = 0; layer_index < cache->config.layer_count; ++layer_index) {
                const auto & source_layer = source_it->second.layers[layer_index];
                auto & destination_layer = candidate.layers[layer_index];
                for (const auto & ref : source_layer.sealed_pages) {
                    const uint64_t ref_end = static_cast<uint64_t>(ref.logical_token_begin) + ref.token_count;
                    if (ref.logical_token_begin < token_begin || ref_end > token_end) continue;
                    acquire_page(cache, ref.page_id);
                    acquired.push_back(ref.page_id);
                    destination_layer.sealed_pages.push_back(ref);
                }
                const uint64_t active_end = static_cast<uint64_t>(source_layer.active_token_begin) + source_layer.active_token_count;
                if (source_layer.active_token_count && source_layer.active_token_begin >= token_begin && active_end <= token_end) {
                    destination_layer.active_token_begin = source_layer.active_token_begin;
                    destination_layer.active_token_count = source_layer.active_token_count;
                    destination_layer.active_generation = source_layer.active_generation + 1;
                }
                destination_layer.descriptor_generation = source_layer.descriptor_generation + 1;
            }
        } catch (...) {
            for (uint64_t page_id : acquired) release_page(cache, page_id);
            throw;
        }

        auto old = cache->sequences.find(destination_id);
        if (old != cache->sequences.end()) {
            release_sequence_pages(cache, old->second);
            cache->sequences.erase(old);
        }
        cache->sequences.emplace(destination_id, std::move(candidate));
        ++cache->global_generation;
        ++cache->counters.sequences_copied;
        update_active_peak(cache);
        maybe_audit(cache);
        cache->last_error.clear();
        return 0;
    } catch (const std::exception & error) {
        set_error(cache, error.what());
        return -1;
    }
}

extern "C" int cf_sequence_remove_async(cf_multisequence_cache * cache,
                                           int64_t sequence_id,
                                           int64_t position_begin,
                                           int64_t position_end,
                                           void *) {
    if (!cache || position_end < position_begin) return -1;
    std::lock_guard<std::mutex> guard(cache->mutex);
    try {
        auto it = cache->sequences.find(sequence_id);
        if (it == cache->sequences.end()) throw std::runtime_error("sequence does not exist");
        if (position_begin < 0 && position_end < 0) {
            release_sequence_pages(cache, it->second);
            cache->sequences.erase(it);
            ++cache->counters.sequences_removed;
            ++cache->global_generation;
            update_active_peak(cache);
            maybe_audit(cache);
            cache->last_error.clear();
            return 0;
        }

        if (position_begin < 0) position_begin = 0;
        for (auto & layer : it->second.layers) {
            std::vector<PageRef> retained;
            retained.reserve(layer.sealed_pages.size());
            for (const auto & ref : layer.sealed_pages) {
                const int64_t ref_begin = ref.logical_token_begin;
                const int64_t ref_end = ref_begin + ref.token_count;
                if (ref_end <= position_begin || ref_begin >= position_end) {
                    retained.push_back(ref);
                    continue;
                }
                if (position_begin > ref_begin || position_end < ref_end) {
                    ++cache->counters.unsupported_operations;
                    throw std::runtime_error("removal inside immutable page is not supported");
                }
                release_page(cache, ref.page_id);
            }
            layer.sealed_pages = std::move(retained);
            const int64_t active_begin = layer.active_token_begin;
            const int64_t active_end = active_begin + layer.active_token_count;
            if (position_begin <= active_begin && position_end >= active_end) {
                layer.active_token_count = 0;
            } else if (position_begin >= active_begin && position_begin < active_end && position_end >= active_end) {
                layer.active_token_count = static_cast<uint32_t>(position_begin - active_begin);
            } else if (position_end > active_begin && position_begin < active_end) {
                ++cache->counters.unsupported_operations;
                throw std::runtime_error("non-suffix active-tail removal is not supported");
            }
            ++layer.descriptor_generation;
        }
        ++it->second.topology_generation;
        ++cache->global_generation;
        ++cache->counters.sequences_removed;
        maybe_audit(cache);
        cache->last_error.clear();
        return 0;
    } catch (const std::exception & error) {
        set_error(cache, error.what());
        return -1;
    }
}

extern "C" int cf_sequence_shift_async(cf_multisequence_cache * cache,
                                          int64_t sequence_id,
                                          uint32_t position_begin,
                                          uint32_t position_end,
                                          int32_t delta,
                                          void *) {
    if (!cache || position_end < position_begin) return -1;
    std::lock_guard<std::mutex> guard(cache->mutex);
    try {
        auto it = cache->sequences.find(sequence_id);
        if (it == cache->sequences.end()) throw std::runtime_error("sequence does not exist");
        for (auto & layer : it->second.layers) {
            for (auto & ref : layer.sealed_pages) {
                const uint64_t ref_end = static_cast<uint64_t>(ref.logical_token_begin) + ref.token_count;
                if (ref.logical_token_begin >= position_begin && ref_end <= position_end) {
                    const int64_t shifted = static_cast<int64_t>(ref.logical_token_begin) + delta;
                    if (shifted < 0) throw std::runtime_error("context shift produces negative token position");
                    ref.logical_token_begin = static_cast<uint32_t>(shifted);
                } else if (ref.logical_token_begin < position_end && ref_end > position_begin) {
                    ++cache->counters.unsupported_operations;
                    throw std::runtime_error("context shift must align with immutable page boundaries");
                }
            }
            const uint64_t active_end = static_cast<uint64_t>(layer.active_token_begin) + layer.active_token_count;
            if (layer.active_token_count && layer.active_token_begin >= position_begin && active_end <= position_end) {
                const int64_t shifted = static_cast<int64_t>(layer.active_token_begin) + delta;
                if (shifted < 0) throw std::runtime_error("context shift produces negative active position");
                layer.active_token_begin = static_cast<uint32_t>(shifted);
            }
            validate_page_order(layer);
            ++layer.descriptor_generation;
        }
        it->second.logical_shift += delta;
        ++it->second.topology_generation;
        ++cache->global_generation;
        ++cache->counters.sequences_shifted;
        maybe_audit(cache);
        cache->last_error.clear();
        return 0;
    } catch (const std::exception & error) {
        set_error(cache, error.what());
        return -1;
    }
}

extern "C" int cf_sequence_append_async(cf_multisequence_cache * cache,
                                           int64_t sequence_id,
                                           uint32_t layer_index,
                                           uint32_t token_begin,
                                           const cf_device_tensor_view * keys,
                                           const cf_device_tensor_view * values,
                                           void *) {
    if (!cache || !keys || !values || layer_index >= cache->config.layer_count ||
        cf_device_tensor_validate(keys) != 0 || cf_device_tensor_validate(values) != 0) return -1;
    std::lock_guard<std::mutex> guard(cache->mutex);
    try {
        auto it = cache->sequences.find(sequence_id);
        if (it == cache->sequences.end()) throw std::runtime_error("sequence does not exist");
        if (keys->token_count != values->token_count || keys->head_count != values->head_count ||
            keys->head_dim != values->head_dim || keys->head_count != cache->config.kv_head_count ||
            keys->head_dim != cache->config.head_dim) throw std::runtime_error("append tensor topology mismatch");
        auto & layer = it->second.layers[layer_index];
        if (layer.active_token_count && token_begin != layer.active_token_begin + layer.active_token_count)
            throw std::runtime_error("non-contiguous sequence append");
        if (layer.active_token_count + keys->token_count > cache->config.page_size)
            throw std::runtime_error("append crosses sequence active-page boundary");
        if (!layer.active_token_count) layer.active_token_begin = token_begin;
        layer.active_token_count += keys->token_count;
        ++layer.active_generation;
        ++it->second.topology_generation;
        ++cache->global_generation;
        maybe_audit(cache);
        cache->last_error.clear();
        return 0;
    } catch (const std::exception & error) {
        set_error(cache, error.what());
        return -1;
    }
}

extern "C" int cf_multisequence_attention_async(cf_multisequence_cache * cache,
                                                   const cf_sequence_attention_item * items,
                                                   uint32_t item_count,
                                                   void *) {
    if (!cache || !items || item_count == 0) return -1;
    std::lock_guard<std::mutex> guard(cache->mutex);
    try {
        std::set<int64_t> observed;
        for (uint32_t i = 0; i < item_count; ++i) {
            const auto & item = items[i];
            if (cache->sequences.find(item.sequence_id) == cache->sequences.end())
                throw std::runtime_error("attention item references unknown sequence");
            if (item.layer >= cache->config.layer_count || cf_device_tensor_validate(&item.query) != 0 || !item.output)
                throw std::runtime_error("invalid sequence attention item");
            observed.insert(item.sequence_id);
        }
        ++cache->counters.batches_dispatched;
        cache->counters.sequences_per_batch_peak = std::max<uint64_t>(
            cache->counters.sequences_per_batch_peak,
            observed.size());
        ++cache->counters.unsupported_operations;
        throw std::runtime_error("multi-sequence CUDA descriptor resolver is not connected yet; dense fallback is forbidden");
    } catch (const std::exception & error) {
        set_error(cache, error.what());
        return -1;
    }
}

extern "C" cf_speculation_id cf_sequence_speculation_begin(cf_multisequence_cache * cache, int64_t sequence_id) {
    if (!cache) return 0;
    std::lock_guard<std::mutex> guard(cache->mutex);
    try {
        auto it = cache->sequences.find(sequence_id);
        if (it == cache->sequences.end()) throw std::runtime_error("sequence does not exist");
        if (it->second.speculation.active) throw std::runtime_error("speculation already active");
        uint32_t baseline = 0;
        for (const auto & layer : it->second.layers) baseline = std::max(baseline, layer.active_token_count);
        const cf_speculation_id id = cache->next_speculation_id++;
        it->second.speculation = {id, baseline, it->second.topology_generation, true};
        ++cache->counters.speculative_transactions_started;
        cache->last_error.clear();
        return id;
    } catch (const std::exception & error) {
        set_error(cache, error.what());
        return 0;
    }
}

extern "C" int cf_sequence_speculation_commit(cf_multisequence_cache * cache,
                                                 int64_t sequence_id,
                                                 cf_speculation_id speculation_id,
                                                 uint32_t accepted_tokens,
                                                 void *) {
    if (!cache || speculation_id == 0) return -1;
    std::lock_guard<std::mutex> guard(cache->mutex);
    try {
        auto it = cache->sequences.find(sequence_id);
        if (it == cache->sequences.end() || !it->second.speculation.active || it->second.speculation.id != speculation_id)
            throw std::runtime_error("invalid speculative transaction");
        for (auto & layer : it->second.layers) {
            const uint32_t baseline = it->second.speculation.baseline_token_count;
            if (layer.active_token_count < baseline) throw std::runtime_error("active tail predates speculative baseline");
            const uint32_t provisional = layer.active_token_count - baseline;
            if (accepted_tokens > provisional) throw std::runtime_error("accepted token count exceeds provisional tail");
            layer.active_token_count = baseline + accepted_tokens;
            ++layer.active_generation;
        }
        it->second.speculation = {};
        ++it->second.topology_generation;
        ++cache->global_generation;
        ++cache->counters.speculative_transactions_committed;
        maybe_audit(cache);
        cache->last_error.clear();
        return 0;
    } catch (const std::exception & error) {
        set_error(cache, error.what());
        return -1;
    }
}

extern "C" int cf_sequence_speculation_rollback(cf_multisequence_cache * cache,
                                                   int64_t sequence_id,
                                                   cf_speculation_id speculation_id,
                                                   void *) {
    if (!cache || speculation_id == 0) return -1;
    std::lock_guard<std::mutex> guard(cache->mutex);
    try {
        auto it = cache->sequences.find(sequence_id);
        if (it == cache->sequences.end() || !it->second.speculation.active || it->second.speculation.id != speculation_id)
            throw std::runtime_error("invalid speculative transaction");
        for (auto & layer : it->second.layers) {
            layer.active_token_count = it->second.speculation.baseline_token_count;
            ++layer.active_generation;
        }
        it->second.topology_generation = it->second.speculation.baseline_generation;
        it->second.speculation = {};
        ++cache->global_generation;
        ++cache->counters.speculative_transactions_rolled_back;
        maybe_audit(cache);
        cache->last_error.clear();
        return 0;
    } catch (const std::exception & error) {
        set_error(cache, error.what());
        return -1;
    }
}

extern "C" int cf_multisequence_reclaim_async(cf_multisequence_cache * cache, void *) {
    if (!cache) return -1;
    std::lock_guard<std::mutex> guard(cache->mutex);
    try {
        for (const auto & retired : cache->retired) {
            auto page = cache->page_store.find(retired.page_id);
            if (page == cache->page_store.end() || page->second.refcount != 0)
                throw std::runtime_error("invalid page in reclamation queue");
            cache->page_store.erase(page);
            ++cache->counters.pages_reclaimed;
            cache->counters.pending_reclamation_bytes -= retired.resident_bytes;
        }
        cache->retired.clear();
        cache->counters.pages_pending_reclamation = 0;
        maybe_audit(cache);
        cache->last_error.clear();
        return 0;
    } catch (const std::exception & error) {
        set_error(cache, error.what());
        return -1;
    }
}

extern "C" int cf_multisequence_audit(const cf_multisequence_cache * cache) {
    if (!cache) return -1;
    std::lock_guard<std::mutex> guard(cache->mutex);
    return audit_locked(cache);
}

extern "C" int cf_multisequence_get_counters(const cf_multisequence_cache * cache, cf_multisequence_counters * counters) {
    if (!cache || !counters) return -1;
    std::lock_guard<std::mutex> guard(cache->mutex);
    *counters = cache->counters;
    return 0;
}

extern "C" const char * cf_multisequence_last_error(const cf_multisequence_cache * cache) {
    return cache ? cache->last_error.c_str() : "null multisequence cache";
}
