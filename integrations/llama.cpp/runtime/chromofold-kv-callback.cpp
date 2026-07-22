#include "chromofold-kv-callback.h"

#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <string>

namespace {
struct cf_map_state {
    std::string path;
    std::set<std::string> seen;
    std::mutex m;
};

bool cf_matches(const char * n) {
    // attention / KV-cache related node names, across llama.cpp naming variants
    static const char * keys[] = {"cache_k", "cache_v", "k_cache", "v_cache",
                                  "Kcur", "Vcur", "Qcur", "k_cur", "v_cur", "q_cur",
                                  "kq", "kqv", "attn"};
    for (const char * k : keys) {
        if (std::strstr(n, k) != nullptr) return true;
    }
    return false;
}
}  // namespace

extern "C" void * chromofold_kv_map_state_create(void) {
    const char * p = std::getenv("CHROMOFOLD_KV_MAP_PATH");
    if (p == nullptr || p[0] == '\0') return nullptr;
    std::remove(p);
    auto * st = new cf_map_state();
    st->path = p;
    return st;
}

extern "C" bool chromofold_kv_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    if (ask) return true;  // deliver the computed tensor
    auto * st = static_cast<cf_map_state *>(user_data);
    if (st == nullptr || t == nullptr || t->name[0] == '\0' || !cf_matches(t->name)) return true;
    char line[320];
    std::snprintf(line, sizeof line, "%-30s | %-16s | [%lld,%lld,%lld,%lld] | %s\n",
                  t->name, ggml_op_desc(t),
                  (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3],
                  ggml_type_name(t->type));
    std::lock_guard<std::mutex> lk(st->m);
    if (st->seen.insert(line).second) {
        FILE * f = std::fopen(st->path.c_str(), "a");
        if (f != nullptr) {
            std::fputs(line, f);
            std::fclose(f);
        }
    }
    return true;  // never alters the graph in this increment
}
