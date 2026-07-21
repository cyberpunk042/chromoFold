#include "chromofold/kv_reference.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <tuple>

namespace chromofold::reference {
namespace {

constexpr int kQMin = -7;
constexpr int kQMax = 7;

uint8_t encode_nibble(int value) {
    if (value < kQMin || value > kQMax) throw std::invalid_argument("int4 value out of range");
    return static_cast<uint8_t>(value & 0x0f);
}

int decode_nibble(uint8_t value) {
    value &= 0x0f;
    return value >= 8 ? static_cast<int>(value) - 16 : static_cast<int>(value);
}

void pack_value(std::vector<uint8_t> *packed, std::size_t index, int value) {
    const std::size_t byte = index / 2;
    const uint8_t nibble = encode_nibble(value);
    if ((index & 1u) == 0) {
        (*packed)[byte] = static_cast<uint8_t>(((*packed)[byte] & 0xf0u) | nibble);
    } else {
        (*packed)[byte] = static_cast<uint8_t>(((*packed)[byte] & 0x0fu) | (nibble << 4u));
    }
}

int unpack_value(const std::vector<uint8_t> &packed, std::size_t index) {
    const uint8_t byte = packed[index / 2];
    return decode_nibble((index & 1u) == 0 ? byte : static_cast<uint8_t>(byte >> 4u));
}

float safe_scale(float maximum) {
    return maximum > 0.0f && std::isfinite(maximum) ? maximum / static_cast<float>(kQMax) : 1.0f;
}

int quantize(float value, float scale) {
    if (!std::isfinite(value)) throw std::invalid_argument("non-finite KV value");
    const long rounded = std::lround(static_cast<double>(value / scale));
    return static_cast<int>(std::clamp(rounded, static_cast<long>(kQMin), static_cast<long>(kQMax)));
}

struct OnlineState {
    double maximum = -std::numeric_limits<double>::infinity();
    double denominator = 0.0;
    std::vector<double> weighted;

    explicit OnlineState(uint32_t dim) : weighted(dim, 0.0) {}

    void add(double score, const float *value) {
        const double next_max = std::max(maximum, score);
        const double old_scale = denominator == 0.0 ? 0.0 : std::exp(maximum - next_max);
        const double new_scale = std::exp(score - next_max);
        denominator = denominator * old_scale + new_scale;
        for (std::size_t d = 0; d < weighted.size(); ++d) {
            weighted[d] = weighted[d] * old_scale + static_cast<double>(value[d]) * new_scale;
        }
        maximum = next_max;
    }
};

uint32_t map_query_to_kv_head(uint32_t query_head, const AttentionShape &shape) {
    return static_cast<uint32_t>((static_cast<uint64_t>(query_head) * shape.kv_head_count) /
                                 shape.query_head_count);
}

void validate_shape(const AttentionShape &shape, const float *queries, std::vector<float> *output) {
    if (queries == nullptr || output == nullptr || shape.query_count == 0 || shape.query_head_count == 0 ||
        shape.kv_head_count == 0 || shape.head_dim == 0 || shape.query_head_count % shape.kv_head_count != 0 ||
        !std::isfinite(shape.score_scale) || shape.score_scale <= 0.0f) {
        throw std::invalid_argument("invalid attention shape");
    }
}

} // namespace

std::size_t SealedKvPage::value_count() const noexcept {
    return static_cast<std::size_t>(token_count) * head_dim;
}

std::size_t SealedKvPage::resident_bytes() const noexcept {
    return packed_k.size() + packed_v.size() + k_scales.size() * sizeof(float) +
           v_scales.size() * sizeof(float);
}

bool SealedKvPage::valid() const noexcept {
    const std::size_t values = value_count();
    const std::size_t packed = (values + 1u) / 2u;
    return token_count != 0 && head_dim != 0 && packed_k.size() == packed && packed_v.size() == packed &&
           k_scales.size() == head_dim && v_scales.size() == token_count &&
           token_begin <= std::numeric_limits<uint32_t>::max() - token_count;
}

bool ActiveKvTail::valid() const noexcept {
    return token_count != 0 && head_dim != 0 && k != nullptr && v != nullptr &&
           token_begin <= std::numeric_limits<uint32_t>::max() - token_count;
}

SealedKvPage seal_int4_page(const float *k, const float *v,
                            uint32_t token_begin, uint32_t token_count,
                            uint32_t layer, uint32_t kv_head, uint32_t head_dim) {
    if (k == nullptr || v == nullptr || token_count == 0 || head_dim == 0 ||
        token_begin > std::numeric_limits<uint32_t>::max() - token_count) {
        throw std::invalid_argument("invalid page input");
    }

    SealedKvPage page;
    page.token_begin = token_begin;
    page.token_count = token_count;
    page.layer = layer;
    page.kv_head = kv_head;
    page.head_dim = head_dim;
    const std::size_t values = page.value_count();
    page.packed_k.assign((values + 1u) / 2u, 0u);
    page.packed_v.assign((values + 1u) / 2u, 0u);
    page.k_scales.resize(head_dim);
    page.v_scales.resize(token_count);

    for (uint32_t d = 0; d < head_dim; ++d) {
        float maximum = 0.0f;
        for (uint32_t t = 0; t < token_count; ++t) {
            maximum = std::max(maximum, std::abs(k[static_cast<std::size_t>(t) * head_dim + d]));
        }
        page.k_scales[d] = safe_scale(maximum);
    }
    for (uint32_t t = 0; t < token_count; ++t) {
        float maximum = 0.0f;
        for (uint32_t d = 0; d < head_dim; ++d) {
            maximum = std::max(maximum, std::abs(v[static_cast<std::size_t>(t) * head_dim + d]));
        }
        page.v_scales[t] = safe_scale(maximum);
    }

    for (uint32_t t = 0; t < token_count; ++t) {
        for (uint32_t d = 0; d < head_dim; ++d) {
            const std::size_t index = static_cast<std::size_t>(t) * head_dim + d;
            pack_value(&page.packed_k, index, quantize(k[index], page.k_scales[d]));
            pack_value(&page.packed_v, index, quantize(v[index], page.v_scales[t]));
        }
    }
    return page;
}

void dequantize_page(const SealedKvPage &page, std::vector<float> *k, std::vector<float> *v) {
    if (!page.valid() || k == nullptr || v == nullptr) throw std::invalid_argument("invalid sealed page");
    k->resize(page.value_count());
    v->resize(page.value_count());
    for (uint32_t t = 0; t < page.token_count; ++t) {
        for (uint32_t d = 0; d < page.head_dim; ++d) {
            const std::size_t index = static_cast<std::size_t>(t) * page.head_dim + d;
            (*k)[index] = static_cast<float>(unpack_value(page.packed_k, index)) * page.k_scales[d];
            (*v)[index] = static_cast<float>(unpack_value(page.packed_v, index)) * page.v_scales[t];
        }
    }
}

void paged_attention(const std::vector<SealedKvPage> &pages,
                     const std::vector<ActiveKvTail> &active_tails,
                     const float *queries, const AttentionShape &shape,
                     std::vector<float> *output, AttentionDiagnostics *diagnostics) {
    validate_shape(shape, queries, output);
    for (const auto &page : pages) {
        if (!page.valid() || page.layer != 0 || page.head_dim != shape.head_dim || page.kv_head >= shape.kv_head_count) {
            throw std::invalid_argument("invalid page topology");
        }
    }
    for (const auto &tail : active_tails) {
        if (!tail.valid() || tail.layer != 0 || tail.head_dim != shape.head_dim || tail.kv_head >= shape.kv_head_count) {
            throw std::invalid_argument("invalid active-tail topology");
        }
    }

    for (std::size_t i = 0; i < pages.size(); ++i) {
        for (std::size_t j = i + 1; j < pages.size(); ++j) {
            if (pages[i].kv_head != pages[j].kv_head) continue;
            const uint64_t a_end = static_cast<uint64_t>(pages[i].token_begin) + pages[i].token_count;
            const uint64_t b_end = static_cast<uint64_t>(pages[j].token_begin) + pages[j].token_count;
            if (pages[i].token_begin < b_end && pages[j].token_begin < a_end) {
                throw std::invalid_argument("overlapping sealed pages");
            }
        }
    }

    output->assign(static_cast<std::size_t>(shape.query_count) * shape.query_head_count * shape.head_dim, 0.0f);
    AttentionDiagnostics local{};
    for (const auto &page : pages) local.sealed_payload_bytes += page.resident_bytes();

    for (uint32_t q = 0; q < shape.query_count; ++q) {
        const uint32_t query_token = shape.query_token_begin + q;
        const uint32_t earliest = shape.causal_window == 0 || query_token + 1u <= shape.causal_window
                                      ? 0u
                                      : query_token + 1u - shape.causal_window;
        for (uint32_t qh = 0; qh < shape.query_head_count; ++qh) {
            const uint32_t kvh = map_query_to_kv_head(qh, shape);
            const float *query = queries + (static_cast<std::size_t>(q) * shape.query_head_count + qh) * shape.head_dim;
            OnlineState state(shape.head_dim);

            auto consume = [&](uint32_t token, const float *key, const float *value, bool sealed) {
                if (token < earliest || token > query_token) return;
                double dot = 0.0;
                for (uint32_t d = 0; d < shape.head_dim; ++d) dot += static_cast<double>(query[d]) * key[d];
                state.add(dot * shape.score_scale, value);
                local.attended_values += shape.head_dim;
                if (sealed) local.sealed_values += shape.head_dim;
                else local.active_values += shape.head_dim;
            };

            for (const auto &page : pages) {
                if (page.kv_head != kvh) continue;
                std::vector<float> dense_k;
                std::vector<float> dense_v;
                dequantize_page(page, &dense_k, &dense_v);
                for (uint32_t t = 0; t < page.token_count; ++t) {
                    consume(page.token_begin + t,
                            dense_k.data() + static_cast<std::size_t>(t) * shape.head_dim,
                            dense_v.data() + static_cast<std::size_t>(t) * shape.head_dim, true);
                }
            }
            for (const auto &tail : active_tails) {
                if (tail.kv_head != kvh) continue;
                for (uint32_t t = 0; t < tail.token_count; ++t) {
                    consume(tail.token_begin + t,
                            tail.k + static_cast<std::size_t>(t) * shape.head_dim,
                            tail.v + static_cast<std::size_t>(t) * shape.head_dim, false);
                }
            }

            if (state.denominator == 0.0) throw std::invalid_argument("query has no visible KV tokens");
            float *dst = output->data() + (static_cast<std::size_t>(q) * shape.query_head_count + qh) * shape.head_dim;
            for (uint32_t d = 0; d < shape.head_dim; ++d) {
                dst[d] = static_cast<float>(state.weighted[d] / state.denominator);
            }
        }
    }
    if (diagnostics != nullptr) *diagnostics = local;
}

} // namespace chromofold::reference
