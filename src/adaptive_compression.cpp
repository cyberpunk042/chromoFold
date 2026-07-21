#include "chromofold/adaptive_compression.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

struct cf_adaptive_runtime {
    cf_adaptive_config config{};
    cf_adaptive_counters counters{};
    std::string error;
};

namespace {

uint64_t checksum64(const uint8_t * data, size_t size) {
    uint64_t hash = 1469598103934665603ull;
    for (size_t index = 0; index < size; ++index) {
        hash ^= data[index];
        hash *= 1099511628211ull;
    }
    return hash;
}

int bits_for(uint32_t codec) {
    if (codec == CF_PAGE_INT2_BLOCKWISE) return 2;
    if (codec == CF_PAGE_INT4_BLOCKWISE) return 4;
    if (codec == CF_PAGE_INT8_BLOCKWISE) return 8;
    return 16;
}

float qmax(uint32_t codec) {
    if (codec == CF_PAGE_INT2_BLOCKWISE) return 1.0f;
    if (codec == CF_PAGE_INT4_BLOCKWISE) return 7.0f;
    if (codec == CF_PAGE_INT8_BLOCKWISE) return 127.0f;
    return 1.0f;
}

uint64_t packed_bytes(uint32_t codec, uint32_t count) {
    if (codec == CF_PAGE_INT2_BLOCKWISE) return (count + 3u) / 4u;
    if (codec == CF_PAGE_INT4_BLOCKWISE) return (count + 1u) / 2u;
    if (codec == CF_PAGE_INT8_BLOCKWISE) return count;
    return static_cast<uint64_t>(count) * 2u;
}

uint32_t choose(const cf_adaptive_runtime * runtime, const cf_page_analysis & analysis) {
    const uint32_t candidates[] = {
        CF_PAGE_INT2_BLOCKWISE,
        CF_PAGE_INT4_BLOCKWISE,
        CF_PAGE_INT8_BLOCKWISE,
        CF_PAGE_FP16_RAW,
    };

    for (uint32_t codec : candidates) {
        const int bits = bits_for(codec);
        if (bits < static_cast<int>(runtime->config.min_bits) ||
            bits > static_cast<int>(runtime->config.max_bits)) {
            continue;
        }

        float error = 0.0f;
        if (codec == CF_PAGE_INT2_BLOCKWISE) error = analysis.estimated_int2_error;
        else if (codec == CF_PAGE_INT4_BLOCKWISE) error = analysis.estimated_int4_error;
        else if (codec == CF_PAGE_INT8_BLOCKWISE) error = analysis.estimated_int8_error;

        if (error <= runtime->config.max_page_error) return codec;
    }

    return CF_PAGE_FP16_RAW;
}

void pack(const float * input,
          uint32_t count,
          uint32_t codec,
          uint32_t block_size,
          std::vector<uint8_t> & payload,
          std::vector<float> & scales) {
    payload.assign(packed_bytes(codec, count), 0);
    scales.resize((count + block_size - 1u) / block_size);

    for (size_t block_index = 0; block_index < scales.size(); ++block_index) {
        const uint32_t start = static_cast<uint32_t>(block_index) * block_size;
        const uint32_t end = std::min(count, start + block_size);
        float maximum = 0.0f;
        for (uint32_t index = start; index < end; ++index) {
            maximum = std::max(maximum, std::fabs(input[index]));
        }

        const float scale = maximum > 0.0f ? maximum / qmax(codec) : 1.0f;
        scales[block_index] = scale;

        for (uint32_t index = start; index < end; ++index) {
            if (codec == CF_PAGE_FP16_RAW) {
                const int16_t stored = static_cast<int16_t>(
                    std::lrint(std::clamp(input[index], -32768.0f, 32767.0f)));
                std::memcpy(payload.data() + static_cast<size_t>(index) * 2u, &stored, sizeof(stored));
                continue;
            }

            int quantized = static_cast<int>(std::lrint(input[index] / scale));
            if (codec == CF_PAGE_INT2_BLOCKWISE) {
                quantized = std::clamp(quantized, -2, 1);
                payload[index / 4u] |= static_cast<uint8_t>(quantized & 3) << ((index % 4u) * 2u);
            } else if (codec == CF_PAGE_INT4_BLOCKWISE) {
                quantized = std::clamp(quantized, -8, 7);
                payload[index / 2u] |= static_cast<uint8_t>(quantized & 15) << ((index % 2u) * 4u);
            } else {
                payload[index] = static_cast<uint8_t>(
                    static_cast<int8_t>(std::clamp(quantized, -128, 127)));
            }
        }
    }
}

void unpack(const uint8_t * payload,
            const float * scales,
            uint32_t count,
            uint32_t codec,
            uint32_t block_size,
            float * output) {
    for (uint32_t index = 0; index < count; ++index) {
        if (codec == CF_PAGE_FP16_RAW) {
            int16_t stored = 0;
            std::memcpy(&stored, payload + static_cast<size_t>(index) * 2u, sizeof(stored));
            output[index] = static_cast<float>(stored);
            continue;
        }

        int quantized = 0;
        if (codec == CF_PAGE_INT2_BLOCKWISE) {
            quantized = (payload[index / 4u] >> ((index % 4u) * 2u)) & 3;
            if (quantized >= 2) quantized -= 4;
        } else if (codec == CF_PAGE_INT4_BLOCKWISE) {
            quantized = (payload[index / 2u] >> ((index % 2u) * 4u)) & 15;
            if (quantized >= 8) quantized -= 16;
        } else {
            quantized = static_cast<int8_t>(payload[index]);
        }
        output[index] = static_cast<float>(quantized) * scales[index / block_size];
    }
}

int encode_one(const float * input,
               uint32_t count,
               uint32_t codec,
               uint32_t block_size,
               uint8_t ** payload,
               float ** scales,
               uint64_t * payload_bytes) {
    std::vector<uint8_t> encoded_payload;
    std::vector<float> encoded_scales;
    pack(input, count, codec, block_size, encoded_payload, encoded_scales);

    *payload = new uint8_t[encoded_payload.size()];
    *scales = new float[encoded_scales.size()];
    std::copy(encoded_payload.begin(), encoded_payload.end(), *payload);
    std::copy(encoded_scales.begin(), encoded_scales.end(), *scales);
    *payload_bytes = encoded_payload.size();
    return 0;
}

} // namespace

extern "C" cf_adaptive_runtime * cf_adaptive_create(const cf_adaptive_config * config) {
    if (!config || !config->block_size || config->min_bits > config->max_bits) return nullptr;
    auto * runtime = new cf_adaptive_runtime;
    runtime->config = *config;
    return runtime;
}

extern "C" void cf_adaptive_destroy(cf_adaptive_runtime * runtime) {
    delete runtime;
}

extern "C" int cf_adaptive_analyze(const float * values, uint32_t count, cf_page_analysis * out) {
    if (!values || !count || !out) return -1;
    *out = {};

    double sum = 0.0;
    double square_sum = 0.0;
    float maximum = 0.0f;
    uint64_t outliers = 0;

    for (uint32_t index = 0; index < count; ++index) {
        if (std::isnan(values[index])) ++out->nan_values;
        if (std::isinf(values[index])) ++out->inf_values;
        maximum = std::max(maximum, std::fabs(values[index]));
        sum += values[index];
        square_sum += static_cast<double>(values[index]) * values[index];
    }

    const double mean = sum / count;
    out->abs_max = maximum;
    out->variance = static_cast<float>(square_sum / count - mean * mean);
    for (uint32_t index = 0; index < count; ++index) {
        if (std::fabs(values[index]) > maximum * 0.75f) ++outliers;
    }
    out->outlier_ratio = static_cast<float>(outliers) / count;
    out->estimated_int2_error = maximum / 3.0f;
    out->estimated_int4_error = maximum / 15.0f;
    out->estimated_int8_error = maximum / 255.0f;
    return (out->nan_values || out->inf_values) ? -1 : 0;
}

extern "C" int cf_adaptive_encode(cf_adaptive_runtime * runtime,
                                    const float * keys,
                                    const float * values,
                                    uint32_t count,
                                    cf_encoded_page * out) {
    if (!runtime || !keys || !values || !count || !out) return -1;
    *out = {};

    cf_page_analysis key_analysis{};
    cf_page_analysis value_analysis{};
    if (cf_adaptive_analyze(keys, count, &key_analysis) != 0 ||
        cf_adaptive_analyze(values, count, &value_analysis) != 0) {
        return -1;
    }

    const uint32_t key_codec = runtime->config.policy == CF_POLICY_FIXED_INT4
        ? CF_PAGE_INT4_BLOCKWISE
        : choose(runtime, key_analysis);
    const uint32_t value_codec = runtime->config.policy == CF_POLICY_FIXED_INT4
        ? CF_PAGE_INT4_BLOCKWISE
        : choose(runtime, value_analysis);

    out->key_codec = {1, key_codec, runtime->config.block_size, 0};
    out->value_codec = {1, value_codec, runtime->config.block_size, 0};
    encode_one(keys, count, key_codec, runtime->config.block_size,
               &out->key_payload, &out->key_scales, &out->key_payload_bytes);
    encode_one(values, count, value_codec, runtime->config.block_size,
               &out->value_payload, &out->value_scales, &out->value_payload_bytes);
    out->value_count = count;
    out->checksum = checksum64(out->key_payload, out->key_payload_bytes) ^
                    checksum64(out->value_payload, out->value_payload_bytes);

    for (uint32_t codec : {key_codec, value_codec}) {
        if (codec == CF_PAGE_INT2_BLOCKWISE) ++runtime->counters.int2_pages;
        else if (codec == CF_PAGE_INT4_BLOCKWISE) ++runtime->counters.int4_pages;
        else if (codec == CF_PAGE_INT8_BLOCKWISE) ++runtime->counters.int8_pages;
        else ++runtime->counters.fp16_pages;
    }

    const uint64_t raw_bytes = static_cast<uint64_t>(count) * sizeof(float) * 2u;
    const uint64_t encoded_bytes = out->key_payload_bytes + out->value_payload_bytes;
    if (encoded_bytes < raw_bytes) runtime->counters.bytes_saved += raw_bytes - encoded_bytes;
    return 0;
}

extern "C" int cf_adaptive_decode(const cf_encoded_page * page,
                                    float * keys,
                                    float * values,
                                    uint32_t count) {
    if (!page || !keys || !values || count != page->value_count) return -1;
    unpack(page->key_payload, page->key_scales, count,
           page->key_codec.codec, page->key_codec.block_size, keys);
    unpack(page->value_payload, page->value_scales, count,
           page->value_codec.codec, page->value_codec.block_size, values);
    return 0;
}

extern "C" int cf_adaptive_recompress(cf_adaptive_runtime * runtime,
                                        const cf_encoded_page * source,
                                        uint32_t codec,
                                        cf_encoded_page * out) {
    if (!runtime || !source || !out) return -1;
    ++runtime->counters.recompression_attempts;

    std::vector<float> keys(source->value_count);
    std::vector<float> values(source->value_count);
    if (cf_adaptive_decode(source, keys.data(), values.data(), source->value_count) != 0) return -1;

    const cf_adaptive_config previous = runtime->config;
    runtime->config.policy = CF_POLICY_QUALITY_BUDGET;
    runtime->config.min_bits = static_cast<uint32_t>(bits_for(codec));
    runtime->config.max_bits = static_cast<uint32_t>(bits_for(codec));
    const int result = cf_adaptive_encode(runtime, keys.data(), values.data(), source->value_count, out);
    runtime->config = previous;
    if (result == 0) ++runtime->counters.recompression_successes;
    return result;
}

extern "C" void cf_encoded_page_release(cf_encoded_page * page) {
    if (!page) return;
    delete[] page->key_payload;
    delete[] page->value_payload;
    delete[] page->key_scales;
    delete[] page->value_scales;
    *page = {};
}

extern "C" int cf_adaptive_get_counters(const cf_adaptive_runtime * runtime,
                                          cf_adaptive_counters * out) {
    if (!runtime || !out) return -1;
    *out = runtime->counters;
    return 0;
}

extern "C" const char * cf_adaptive_last_error(const cf_adaptive_runtime * runtime) {
    return runtime ? runtime->error.c_str() : "runtime is null";
}
