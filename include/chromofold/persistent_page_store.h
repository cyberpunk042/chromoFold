#pragma once
#include <stdint.h>
#include "chromofold/adaptive_compression.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct cf_persistent_store_config {
    const char * path;
    const char * model_sha256;
    uint64_t maximum_bytes;
    uint32_t read_enabled;
    uint32_t write_enabled;
    uint32_t verify_checksums;
} cf_persistent_store_config;

typedef struct cf_persistent_page_key {
    uint32_t layer;
    uint32_t page_size;
    uint32_t kv_head_count;
    uint32_t head_dim;
    uint64_t source_hash;
} cf_persistent_page_key;

typedef struct cf_persistent_store_counters {
    uint64_t records_written;
    uint64_t records_loaded;
    uint64_t warm_hits;
    uint64_t cold_misses;
    uint64_t corrupted_records_rejected;
    uint64_t incompatible_records_rejected;
    uint64_t bytes_written;
    uint64_t bytes_loaded;
} cf_persistent_store_counters;

typedef struct cf_persistent_store cf_persistent_store;
cf_persistent_store * cf_persistent_store_open(const cf_persistent_store_config * config);
void cf_persistent_store_close(cf_persistent_store * store);
int cf_persistent_store_put(cf_persistent_store * store, const cf_persistent_page_key * key, const cf_encoded_page * page);
int cf_persistent_store_get(cf_persistent_store * store, const cf_persistent_page_key * key, cf_encoded_page * page);
int cf_persistent_store_verify(cf_persistent_store * store);
int cf_persistent_store_compact(cf_persistent_store * store);
int cf_persistent_store_get_counters(const cf_persistent_store * store, cf_persistent_store_counters * out);
const char * cf_persistent_store_last_error(const cf_persistent_store * store);
#ifdef __cplusplus
}
#endif
