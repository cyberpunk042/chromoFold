#include "chromofold/llama_server_runtime.h"

#include <cassert>
#include <iostream>

int main() {
    cf_llama_server_config config{};
    config.layer_count = 2;
    config.query_head_count = 4;
    config.kv_head_count = 2;
    config.head_dim = 64;
    config.page_size = 64;
    config.block_size = 32;
    config.context_window = 4096;
    config.gqa_group_size = 2;
    config.strict_mode = 1;
    config.llama_commit = "76f46ad29d61fd8c1401e8221842934bf62a6064";
    config.model_sha256 = "0000000000000000000000000000000000000000000000000000000000000000";

    cf_llama_server_runtime * runtime = cf_llama_server_runtime_create(&config);
    assert(runtime != nullptr);
    assert(cf_llama_server_slot_create(runtime, 100) == 0);
    assert(cf_llama_server_slot_create(runtime, 101) == 0);
    assert(cf_llama_server_sequence_copy_async(runtime, 100, 102, 0, 0, nullptr) == 0);
    assert(cf_llama_server_cancel_async(runtime, 101, nullptr) == 0);
    assert(cf_llama_server_slot_reset_async(runtime, 101, nullptr) == 0);
    assert(cf_llama_server_sequence_shift_async(runtime, 100, 0, 0, 0, nullptr) == 0);
    assert(cf_llama_server_shutdown_async(runtime, nullptr) == 0);

    cf_llama_server_stats stats{};
    assert(cf_llama_server_get_stats(runtime, &stats) == 0);
    assert(stats.http_requests == 2);
    assert(stats.cancelled_requests == 1);
    assert(stats.slot_reuses == 1);
    assert(stats.shutdown_audits == 1);
    assert(stats.shutdown_failures == 0);
    assert(stats.dense_nodes_scheduled == 0);

    cf_llama_server_runtime_destroy(runtime, nullptr);
    std::cout << "M11 pinned server lifecycle contract passed\n";
    return 0;
}
