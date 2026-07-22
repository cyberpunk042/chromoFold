#include "chromofold-kv-callback.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#ifdef GGML_CHROMOFOLD
#include "chromofold_kv_adapter.h"
#include <cmath>
#endif

namespace {

bool cf_matches_map(const char * n) {
    static const char * keys[] = {"cache_k", "cache_v", "k_cache", "v_cache",
                                  "Kcur", "Vcur", "Qcur", "kq", "kqv", "attn"};
    for (const char * k : keys) if (std::strstr(n, k) != nullptr) return true;
    return false;
}

// Parse the layer index from a node named "<prefix>N" (e.g. "Kcur-10" -> 10). Returns -1 if not that shape.
int cf_layer_of(const char * name, const char * prefix) {
    size_t plen = std::strlen(prefix);
    if (std::strncmp(name, prefix, plen) != 0) return -1;
    const char * p = name + plen;
    if (*p < '0' || *p > '9') return -1;
    int v = 0;
    for (; *p >= '0' && *p <= '9'; ++p) v = v * 10 + (*p - '0');
    if (*p != '\0') return -1;  // must be exactly "<prefix>N" (excludes "Kcur-0 (view)")
    return v;
}

struct cf_cb_state {
    // node-map mode (CHROMOFOLD_KV_MAP_PATH)
    std::string map_path;
    std::set<std::string> seen;

#ifdef GGML_CHROMOFOLD
    // append mode: ingest the live model K/V into the ChromoFold compressed cache (single sequence)
    bool append_on = false, tried = false, disabled = false;
    cf_llama_kv_adapter * adapter = nullptr;
    uint32_t head_dim = 0, kv_heads = 0, page_size = 32;
    static const int MAXL = 128;
    std::vector<float> pend_k[MAXL];
    uint32_t pend_k_T[MAXL] = {0};
    uint32_t tok_begin[MAXL] = {0};
    unsigned long long append_errors = 0, nonfinite = 0;
    std::string stats_path;
#endif
    std::mutex m;
};

#ifdef GGML_CHROMOFOLD
bool cf_extract(const ggml_tensor * t, std::vector<float> & out, unsigned long long & nonfinite) {
    if (t->type != GGML_TYPE_F32) return false;
    size_t n = ggml_nelements(t);
    out.resize(n);
    ggml_backend_tensor_get(t, out.data(), 0, n * sizeof(float));
    for (size_t i = 0; i < n; ++i) if (!std::isfinite(out[i])) { ++nonfinite; return false; }
    return true;
}

void cf_write_stats(cf_cb_state * s) {
    if (s->adapter == nullptr || s->stats_path.empty()) return;
    cf_llama_kv_stats st{};
    if (cf_llama_kv_get_stats(s->adapter, &st) != 0) return;
    FILE * f = std::fopen(s->stats_path.c_str(), "w");
    if (f == nullptr) return;
    std::fprintf(f,
        "{\"appended_tokens\":%llu,\"sealed_tokens\":%llu,\"sealed_pages\":%llu,\"active_tokens\":%llu,"
        "\"dense_active_bytes\":%llu,\"compressed_bytes\":%llu,\"descriptor_bytes\":%llu,"
        "\"append_errors\":%llu,\"nonfinite\":%llu,\"head_dim\":%u,\"kv_heads\":%u}\n",
        (unsigned long long) st.appended_tokens, (unsigned long long) st.sealed_tokens,
        (unsigned long long) st.sealed_pages, (unsigned long long) st.active_tokens,
        (unsigned long long) st.dense_active_bytes, (unsigned long long) st.compressed_bytes,
        (unsigned long long) st.descriptor_bytes, s->append_errors, s->nonfinite, s->head_dim, s->kv_heads);
    std::fclose(f);
}

// Append one layer's paired K,V (raw contiguous [dim, kv_head, token]) into the adapter, per kv head.
void cf_append_layer(cf_cb_state * s, int L, const std::vector<float> & k, const std::vector<float> & v, uint32_t T) {
    const uint32_t hd = s->head_dim, kvh = s->kv_heads, wide = hd * kvh;
    std::vector<float> kh(T * hd), vh(T * hd);
    for (uint32_t h = 0; h < kvh; ++h) {
        for (uint32_t tok = 0; tok < T; ++tok) {
            const float * ks = &k[(size_t) tok * wide + (size_t) h * hd];
            const float * vs = &v[(size_t) tok * wide + (size_t) h * hd];
            std::memcpy(&kh[(size_t) tok * hd], ks, hd * sizeof(float));
            std::memcpy(&vh[(size_t) tok * hd], vs, hd * sizeof(float));
        }
        if (cf_llama_kv_append(s->adapter, (uint32_t) L, h, s->tok_begin[L], kh.data(), vh.data(), T, nullptr) != 0) {
            ++s->append_errors;
        }
    }
    s->tok_begin[L] += T;
    cf_write_stats(s);
}
#endif

}  // namespace

extern "C" void * chromofold_kv_map_state_create(void) {
    const char * map = std::getenv("CHROMOFOLD_KV_MAP_PATH");
    bool append_on = false;
#ifdef GGML_CHROMOFOLD
    const char * be = std::getenv("CHROMOFOLD_KV_BACKEND");
    append_on = (be != nullptr && std::strcmp(be, "chromofold") == 0);
#endif
    if ((map == nullptr || map[0] == '\0') && !append_on) return nullptr;
    auto * s = new cf_cb_state();
    if (map != nullptr && map[0] != '\0') { s->map_path = map; std::remove(map); }
#ifdef GGML_CHROMOFOLD
    s->append_on = append_on;
    const char * ps = std::getenv("CHROMOFOLD_PAGE_SIZE");
    if (ps != nullptr && ps[0] != '\0') { int p = std::atoi(ps); if (p > 0) s->page_size = (uint32_t) p; }
    const char * sp = std::getenv("CHROMOFOLD_KV_STATS_PATH");
    if (sp != nullptr && sp[0] != '\0') { s->stats_path = sp; std::remove(sp); }
#endif
    return s;
}

extern "C" bool chromofold_kv_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    if (ask) return true;
    auto * s = static_cast<cf_cb_state *>(user_data);
    if (s == nullptr || t == nullptr || t->name[0] == '\0') return true;

    // node-map mode
    if (!s->map_path.empty() && cf_matches_map(t->name)) {
        char line[320];
        std::snprintf(line, sizeof line, "%-30s | %-16s | [%lld,%lld,%lld,%lld] | %s\n",
                      t->name, ggml_op_desc(t),
                      (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3],
                      ggml_type_name(t->type));
        std::lock_guard<std::mutex> lk(s->m);
        if (s->seen.insert(line).second) {
            FILE * f = std::fopen(s->map_path.c_str(), "a");
            if (f != nullptr) { std::fputs(line, f); std::fclose(f); }
        }
    }

#ifdef GGML_CHROMOFOLD
    if (s->append_on && !s->disabled) {
        // Match the final per-layer K (RoPE'd) and V (reshaped): [head_dim, kv_heads, tokens] f32, contiguous.
        int kL = (t->op == GGML_OP_ROPE)    ? cf_layer_of(t->name, "Kcur-") : -1;
        int vL = (t->op == GGML_OP_RESHAPE) ? cf_layer_of(t->name, "Vcur-") : -1;
        if ((kL >= 0 || vL >= 0) && t->type == GGML_TYPE_F32 && t->ne[3] == 1 && t->ne[0] > 0 && t->ne[1] > 0) {
            std::lock_guard<std::mutex> lk(s->m);
            int L = kL >= 0 ? kL : vL;
            if (L < cf_cb_state::MAXL) {
                uint32_t hd = (uint32_t) t->ne[0], kvh = (uint32_t) t->ne[1], T = (uint32_t) t->ne[2];
                if (!s->tried) {  // lazily create the adapter from the observed K/V shape (append-only: gqa=1)
                    s->tried = true;
                    s->head_dim = hd; s->kv_heads = kvh;
                    cf_llama_kv_options o{};
                    o.struct_size = sizeof(o); o.backend = CF_LLAMA_KV_BACKEND_CHROMOFOLD;
                    o.layer_count = cf_cb_state::MAXL; o.kv_head_count = kvh; o.query_head_count = kvh;
                    o.head_dim = hd; o.page_size = s->page_size; o.gqa_group_size = 1;
                    s->adapter = cf_llama_kv_create(&o);
                    if (s->adapter == nullptr) s->disabled = true;
                }
                if (!s->disabled && hd == s->head_dim && kvh == s->kv_heads) {
                    std::vector<float> buf;
                    if (cf_extract(t, buf, s->nonfinite)) {
                        if (kL >= 0) { s->pend_k[L] = std::move(buf); s->pend_k_T[L] = T; }
                        else if (s->pend_k_T[L] == T && !s->pend_k[L].empty()) {
                            cf_append_layer(s, L, s->pend_k[L], buf, T);
                            s->pend_k[L].clear(); s->pend_k_T[L] = 0;
                        }
                    }
                }
            }
        }
    }
#endif
    return true;
}
