// chromofold-kv-callback.h — the ChromoFold KV interception seam for llama.cpp, via the ggml graph eval callback
// (cparams.cb_eval). Installed env-gated from common_context_params_to_llama when CHROMOFOLD_KV_BACKEND=chromofold.
// This is the plumbing for the maintained downstream patch's layer 2 (route attention through the compressed KV
// cache). First increment: it maps the model's attention/KV graph nodes (name/op/shape) so the append/replace
// step can hook the exact tensors. No graph-builder surgery — a single callback registration.
#pragma once

struct ggml_tensor;

#ifdef __cplusplus
extern "C" {
#endif

// Per-node graph eval callback. On ask==true returns true (deliver the computed tensor); on ask==false records
// attention/KV nodes (name, op, shape) to CHROMOFOLD_KV_MAP_PATH. Always returns true (never alters the graph yet).
bool chromofold_kv_cb_eval(struct ggml_tensor * t, bool ask, void * user_data);

// Create the callback's state from CHROMOFOLD_KV_MAP_PATH. Returns nullptr if the env var is unset (=> the
// callback is not installed and llama runs exactly as before).
void * chromofold_kv_map_state_create(void);

#ifdef __cplusplus
}
#endif
