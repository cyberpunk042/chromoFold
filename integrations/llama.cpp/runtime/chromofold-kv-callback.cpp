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
    uint32_t head_dim = 0, kv_heads = 0, page_size = 32, kv_bits = 4;
    static const int MAXL = 128;
    std::vector<float> pend_k[MAXL];
    std::vector<float> pend_v[MAXL];
    uint32_t pend_k_T[MAXL] = {0};
    uint32_t pend_v_T[MAXL] = {0};
    uint32_t tok_begin[MAXL] = {0};
    unsigned long long append_errors = 0, nonfinite = 0;
    std::string stats_path;
    // replace mode: serve attention from the compressed cache and overwrite kqv_out
    bool replace_on = false;
    uint32_t query_heads = 0;
    std::vector<float> q_host[MAXL];
    uint32_t q_count[MAXL] = {0};
    unsigned long long attn_launches = 0, attn_errors = 0;
    uint32_t max_append_T = 0;
    bool compare_on = false;
    double attn_max_diff = 0.0, attn_sum_diff = 0.0;
    unsigned long long attn_diff_count = 0;
    std::string debug_path;
    unsigned dbg = 0;
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
        "\"append_errors\":%llu,\"nonfinite\":%llu,\"head_dim\":%u,\"kv_heads\":%u,"
        "\"query_heads\":%u,\"max_append_T\":%u,\"compressed_attention_launches\":%llu,\"attn_errors\":%llu,"
        "\"attn_max_diff\":%g,\"attn_mean_diff\":%g}\n",
        (unsigned long long) st.appended_tokens, (unsigned long long) st.sealed_tokens,
        (unsigned long long) st.sealed_pages, (unsigned long long) st.active_tokens,
        (unsigned long long) st.dense_active_bytes, (unsigned long long) st.compressed_bytes,
        (unsigned long long) st.descriptor_bytes, s->append_errors, s->nonfinite, s->head_dim, s->kv_heads,
        s->query_heads, s->max_append_T, s->attn_launches, s->attn_errors,
        s->attn_max_diff, s->attn_diff_count ? s->attn_sum_diff / (double) s->attn_diff_count : 0.0);
    std::fclose(f);
}

// Append one layer's paired K,V (raw contiguous [dim, kv_head, token]) into the adapter, per kv head.
void cf_append_layer(cf_cb_state * s, int L, const std::vector<float> & k, const std::vector<float> & v, uint32_t T) {
    // A prefill pass (T>1) at layer 0 marks a fresh sequence: llama re-evaluates from position 0, so reset the
    // shadow cache to stay position-aligned (drops any phantom/warmup passes). Single-ubatch prompts only
    // (initial_support scope); a multi-ubatch prompt would need llama's real positions instead.
    if (T > 1 && L == 0) {
        cf_llama_kv_clear(s->adapter);
        for (int i = 0; i < cf_cb_state::MAXL; ++i) s->tok_begin[i] = 0;
    }
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
    if (T > s->max_append_T) s->max_append_T = T;
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
    const char * kb = std::getenv("CHROMOFOLD_KV_BITS");
    if (kb != nullptr && std::atoi(kb) == 8) s->kv_bits = 8;
    const char * sp = std::getenv("CHROMOFOLD_KV_STATS_PATH");
    if (sp != nullptr && sp[0] != '\0') { s->stats_path = sp; std::remove(sp); }
    const char * rp = std::getenv("CHROMOFOLD_KV_REPLACE");
    s->replace_on = append_on && rp != nullptr && rp[0] == '1';
    const char * cp = std::getenv("CHROMOFOLD_KV_COMPARE");
    s->compare_on = append_on && cp != nullptr && cp[0] == '1';
    const char * dp = std::getenv("CHROMOFOLD_KV_DEBUG");
    if (dp != nullptr && dp[0] != '\0') { s->debug_path = dp; std::remove(dp); }
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
        if (!s->debug_path.empty() && s->dbg < 4000) {
            const char * nm = t->name;
            const bool k0 = (t->op == GGML_OP_ROPE && cf_layer_of(nm, "Kcur-") == 0);
            const bool o0 = (cf_layer_of(nm, "kqv_out-") == 0);
            if (k0 || o0) {
                std::lock_guard<std::mutex> lk(s->m);
                ++s->dbg;
                FILE * f = std::fopen(s->debug_path.c_str(), "a");
                if (f != nullptr) {
                    std::fprintf(f, "%-12s op=%-8s ne=[%lld,%lld,%lld,%lld] tok_begin=%u\n", nm, ggml_op_desc(t),
                                 (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3],
                                 s->tok_begin[0]);
                    std::fclose(f);
                }
            }
        }
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
                    o.head_dim = hd; o.page_size = s->page_size; o.gqa_group_size = 1; o.kv_bits = s->kv_bits;
                    s->adapter = cf_llama_kv_create(&o);
                    if (s->adapter == nullptr) s->disabled = true;
                }
                if (!s->disabled && hd == s->head_dim && kvh == s->kv_heads) {
                    std::vector<float> buf;
                    if (cf_extract(t, buf, s->nonfinite)) {
                        // Kcur (ROPE) and Vcur (RESHAPE) fire in either order per pass; buffer both and append
                        // the pair once both are present with matching T, so K and V come from the SAME pass.
                        if (kL >= 0) { s->pend_k[L] = std::move(buf); s->pend_k_T[L] = T; }
                        else { s->pend_v[L] = std::move(buf); s->pend_v_T[L] = T; }
                        if (!s->pend_k[L].empty() && !s->pend_v[L].empty() && s->pend_k_T[L] == s->pend_v_T[L]) {
                            cf_append_layer(s, L, s->pend_k[L], s->pend_v[L], s->pend_k_T[L]);
                            s->pend_k[L].clear(); s->pend_v[L].clear();
                            s->pend_k_T[L] = 0; s->pend_v_T[L] = 0;
                        }
                    }
                }
            }
        }

        // Replace step: capture the RoPE'd query, then overwrite kqv_out with compressed-cache attention.
        if ((s->replace_on || s->compare_on) && t->op == GGML_OP_ROPE && t->type == GGML_TYPE_F32 &&
            t->ne[3] == 1 && t->ne[0] > 0) {
            int qL = cf_layer_of(t->name, "Qcur-");
            if (qL >= 0 && qL < cf_cb_state::MAXL) {
                std::lock_guard<std::mutex> lk(s->m);
                s->query_heads = (uint32_t) t->ne[1];
                s->q_count[qL] = (uint32_t) t->ne[2];
                s->q_host[qL].resize(static_cast<std::size_t>(t->ne[0]) * t->ne[1] * t->ne[2]);
                ggml_backend_tensor_get(t, s->q_host[qL].data(), 0, s->q_host[qL].size() * sizeof(float));
            }
        }
        if ((s->replace_on || s->compare_on) && s->adapter != nullptr && s->query_heads > 0 && s->head_dim > 0 &&
            s->kv_heads > 0 && s->query_heads % s->kv_heads == 0 && t->type == GGML_TYPE_F32 &&
            t->ne[3] == 1 && t->ne[2] == 1) {
            int oL = cf_layer_of(t->name, "kqv_out-");
            if (oL >= 0 && oL < cf_cb_state::MAXL) {
                const uint32_t T = (uint32_t) t->ne[1];
                std::lock_guard<std::mutex> lk(s->m);
                if (s->q_count[oL] == T && !s->q_host[oL].empty() && s->tok_begin[oL] >= T &&
                    (uint32_t) t->ne[0] == s->query_heads * s->head_dim) {
                    const uint32_t gqa = s->query_heads / s->kv_heads;
                    const uint32_t qtb = s->tok_begin[oL] - T;
                    const float scale = 1.0f / std::sqrt((float) s->head_dim);
                    std::vector<float> out(static_cast<std::size_t>(T) * s->query_heads * s->head_dim);
                    if (cf_llama_kv_attention_host(s->adapter, (uint32_t) oL, s->q_host[oL].data(), out.data(),
                                                   T, s->query_heads, gqa, qtb, scale, 0, nullptr) == 0) {
                        ++s->attn_launches;
                        if (s->compare_on) {  // measure vs llama's own (dense) kqv_out before any overwrite
                            std::vector<float> ref(out.size());
                            ggml_backend_tensor_get(t, ref.data(), 0, ref.size() * sizeof(float));
                            for (std::size_t i = 0; i < out.size(); ++i) {
                                const double e = std::fabs((double) out[i] - (double) ref[i]);
                                if (e > s->attn_max_diff) s->attn_max_diff = e;
                                s->attn_sum_diff += e;
                            }
                            s->attn_diff_count += out.size();
                        }
                        if (s->replace_on) ggml_backend_tensor_set(t, out.data(), 0, out.size() * sizeof(float));
                    } else {
                        ++s->attn_errors;
                    }
                    cf_write_stats(s);
                }
            }
        }
    }
#endif
    return true;
}
