const DATA = window.CHROMOFOLD_SITE_DATA || {};
const $ = (id) => document.getElementById(id);
const esc = (value) => String(value ?? "").replace(/[&<>'"]/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[c]));
const list = (value) => Array.isArray(value) ? value : [];

function setupNav() {
  const button = document.querySelector(".menu");
  const nav = $("navlinks");
  if (!button || !nav) return;
  button.addEventListener("click", () => {
    const open = button.getAttribute("aria-expanded") === "true";
    button.setAttribute("aria-expanded", String(!open));
    nav.classList.toggle("open", !open);
  });
}

function releaseStatus() {
  const release = DATA.release_channel || {};
  const target = $("release-status");
  if (!target) return;
  target.innerHTML = `<span class="release-pill">${esc(release.channel || "candidate")}</span><strong>${esc(release.latest_version || "No public candidate")}</strong><span>${esc(release.release_status || "not-published")}</span><small>${esc(release.public_message || "Exact-digest qualification is required before production claims.")}</small>`;
}

function claimCard(claim) {
  return `<article class="card evidence-card"><div class="card-top"><span class="level level-${esc(claim.evidence_level)}">${esc(claim.evidence_level)}</span><strong>${esc(claim.result)}</strong></div><h3>${esc(claim.name)}</h3><p>${esc(claim.meaning)}</p><dl><div><dt>Hardware</dt><dd>${esc(claim.hardware)}</dd></div><div><dt>Scope</dt><dd>${esc(claim.scope)}</dd></div></dl></article>`;
}

function renderEvidence() {
  const target = $("claims");
  if (!target) return;
  const level = $("evidence-level")?.value || $("evidence-filter")?.value || "all";
  const query = ($("evidence-search")?.value || "").trim().toLowerCase();
  const claims = list(DATA.claims).filter((claim) => (level === "all" || claim.evidence_level === level) && (!query || JSON.stringify(claim).toLowerCase().includes(query)));
  target.innerHTML = claims.length ? claims.map(claimCard).join("") : '<p class="empty">No claims match these filters.</p>';
  const summary = $("evidence-summary");
  if (summary) summary.textContent = `${claims.length} of ${list(DATA.claims).length} claims shown · ${DATA.claim_boundary || "Evidence remains scope-bound."}`;
}

function renderEvidenceLadder() {
  const target = $("evidence-ladder");
  if (!target) return;
  const labels = [
    ["estimate", "A planning hypothesis. No measured workload result."],
    ["measured", "A repository benchmark with hardware and scope."],
    ["qualified", "Defined gates passed for an immutable target."],
    ["independently-reproduced", "An external party published auditable artifacts."],
  ];
  target.innerHTML = labels.map(([name, text], i) => `<article><span>${i + 1}</span><div><h3>${esc(name)}</h3><p>${esc(text)}</p></div></article>`).join("");
}

function renderCompatibility() {
  const target = $("compatibility-grid");
  if (!target) return;
  const entries = list(DATA.compatibility?.entries);
  const runtime = $("compat-runtime"); const os = $("compat-os"); const status = $("compat-status");
  const fill = (select, values) => { if (select && select.options.length === 1) [...new Set(values)].sort().forEach((v) => select.add(new Option(v, v))); };
  fill(runtime, entries.map((e) => e.runtime)); fill(os, entries.map((e) => e.operating_system)); fill(status, entries.map((e) => e.status));
  const filtered = entries.filter((e) => (!runtime || runtime.value === "all" || e.runtime === runtime.value) && (!os || os.value === "all" || e.operating_system === os.value) && (!status || status.value === "all" || e.status === status.value));
  target.innerHTML = filtered.map((e) => `<article class="compat-row"><div><span class="level ${esc(e.status)}">${esc(e.status)}</span><h3>${esc(e.runtime)} · ${esc(e.model_format)}</h3><p>${esc(e.gpu_vendor)} · ${esc(e.operating_system)}</p></div><div><strong>Evidence</strong><p>${esc(e.evidence)}</p></div><div><strong>Limitations</strong><ul>${list(e.limitations).map((x) => `<li>${esc(x)}</li>`).join("")}</ul></div></article>`).join("") || '<p class="empty">No compatibility entries match.</p>';
}

function profileFor(goal) { return ({"longer-context":"maximum-context","more-users":"high-concurrency","shared-prefix":"shared-prefix","lowest-risk":"safe","balanced":"balanced"})[goal] || "balanced"; }
function planFor(goal, vram, context, concurrency, runtime, os) {
  const profile = profileFor(goal); const active = context * concurrency;
  const entry = list(DATA.compatibility?.entries).find((e) => e.runtime === runtime && e.operating_system === os);
  return {schema:"chromofold.portal-plan.v1", evidence_level:"estimate", profile, advisory_pressure:active >= 262144 ? "critical" : active >= 131072 ? "high" : active >= 65536 ? "moderate" : "low-or-unknown", compatibility:entry ? {status:entry.status,evidence:entry.evidence,limitations:entry.limitations} : {status:"unknown",evidence:"No matching registry entry"}, inputs:{vram_gb:vram,context,concurrency,total_active_tokens:active,runtime,operating_system:os}, qualification_required:true, next_steps:["Download and verify the candidate archive.","Run local machine inspection.",`Generate the ${profile} profile with silent fallback disabled.`,"Run matched baseline and candidate workloads.","Run exact-digest qualification before publishing a win."]};
}

function setupPlanner() {
  const form = $("onboarding-form") || $("estimator"); if (!form) return;
  form.addEventListener("submit", (event) => {
    event.preventDefault(); const result = planFor($("goal").value, Number($("vram").value), Number($("context").value), Number($("concurrency").value), $("runtime")?.value || "llama.cpp", $("os")?.value || "Linux");
    $("plan").innerHTML = `<div class="plan-head"><div><span class="level">estimate</span><h2>${esc(result.profile)}</h2></div><button id="copy-plan" type="button" class="small-button">Copy JSON</button></div><p><strong>Compatibility:</strong> ${esc(result.compatibility.status)}</p><p>${esc(result.compatibility.evidence)}</p><pre>${esc(JSON.stringify(result, null, 2))}</pre>`;
    $("copy-plan").addEventListener("click", async () => { await navigator.clipboard.writeText(JSON.stringify(result, null, 2)); $("copy-plan").textContent = "Copied"; });
  });
}

function renderProfiles() {
  const target = $("profiles"); if (!target) return;
  target.innerHTML = Object.entries(DATA.profiles?.profiles || {}).map(([name,p]) => `<article class="card"><span class="level">${esc(name)}</span><h3>${Number(p.context_target).toLocaleString()} token target</h3><p>${esc(p.why)}</p><strong>Risks</strong><ul>${list(p.risks).map((r) => `<li>${esc(r)}</li>`).join("")}</ul><small>Active tail: ${esc(p.kv_cache?.active_tail_tokens)} · block: ${esc(p.kv_cache?.block_tokens)}</small></article>`).join("");
}

function renderDownloads() { const target = $("downloads-list"); if (!target) return; target.innerHTML = list(DATA.downloads).map((item) => `<article class="card download-card"><span class="level">${esc(item.status)}</span><h3>${esc(item.name)}</h3><p>${esc(item.description)}</p><a class="secondary" href="${esc(item.href)}">${esc(item.action)} ↗</a></article>`).join(""); }
function renderReleaseGates() { const target = $("release-gates"); if (!target) return; target.innerHTML = list(DATA.release_channel?.promotion_gates).map((gate,i) => `<article><span>${i+1}</span><div><p>${esc(gate)}</p></div></article>`).join(""); }
function renderContribute() { const fields = $("reproduction-fields"); if (fields) fields.innerHTML = list(DATA.portal?.reproduction_fields).map((f) => `<div>✓ ${esc(f)}</div>`).join(""); const paths = $("contribution-paths"); if (paths) paths.innerHTML = list(DATA.portal?.contribution_paths).map((p,i) => `<article><span>${i+1}</span><div><h3>${esc(p.name)}</h3><p>${esc(p.description)}</p></div></article>`).join(""); }

function chartSvg(labels, first, second, firstLabel, secondLabel, unit) {
  const width = 900, height = 390, left = 64, right = 28, top = 26, bottom = 58;
  const max = Math.max(...first, ...second) * 1.08;
  const x = (i) => left + (width - left - right) * i / (labels.length - 1);
  const y = (value) => height - bottom - (height - top - bottom) * value / max;
  const points = (values) => values.map((value, i) => `${x(i)},${y(value)}`).join(" ");
  const grid = [0, .25, .5, .75, 1].map((fraction) => {
    const value = max * fraction, py = y(value);
    return `<line x1="${left}" y1="${py}" x2="${width-right}" y2="${py}"/><text x="${left-10}" y="${py+4}" text-anchor="end">${value >= 10 ? value.toFixed(0) : value.toFixed(1)}</text>`;
  }).join("");
  const ticks = labels.map((label, i) => `<text x="${x(i)}" y="${height-bottom+24}" text-anchor="middle">${label >= 1024 ? `${label/1024}K` : label}</text>`).join("");
  return `<svg viewBox="0 0 ${width} ${height}" aria-hidden="true"><g class="chart-grid">${grid}${ticks}<text x="${left}" y="${height-12}">context tokens</text></g><polygon class="series-area" points="${points(first)} ${x(labels.length-1)},${height-bottom} ${x(0)},${height-bottom}"/><polyline class="series first" points="${points(first)}"/><polyline class="series second" points="${points(second)}"/>${first.map((v,i)=>`<circle class="point first-point" cx="${x(i)}" cy="${y(v)}" r="4"><title>${labels[i].toLocaleString()} tokens · ${firstLabel}: ${v} ${unit}</title></circle>`).join("")}${second.map((v,i)=>`<circle class="point second-point" cx="${x(i)}" cy="${y(v)}" r="4"><title>${labels[i].toLocaleString()} tokens · ${secondLabel}: ${v} ${unit}</title></circle>`).join("")}</svg>`;
}

function renderKvScaling(mode = "latency") {
  const target = $("kv-chart");
  const data = DATA.kv_scaling;
  if (!target || !data) return;
  const latency = mode === "latency";
  const first = latency ? data.latency_ms.decode_all : data.kv_memory_mib.fp16;
  const second = latency ? data.latency_ms.window_read : data.kv_memory_mib.chromofold;
  const firstLabel = latency ? "decode all" : "fp16 KV";
  const secondLabel = latency ? "fixed-window read" : "ChromoFold KV";
  const unit = latency ? "ms" : "MiB";
  target.innerHTML = chartSvg(data.context_tokens, first, second, firstLabel, secondLabel, unit);
  $("kv-chart-title").textContent = latency ? "Latency versus context length" : "KV memory versus context length";
  $("kv-chart-sub").textContent = latency ? `A fixed ${data.window_tokens}-token window decodes only the blocks it reads; decode-all expands the stored context.` : "Both grow with context, but ChromoFold retains the measured lower KV footprint at every sampled length.";
  $("kv-legend").innerHTML = `<span><i class="legend-line first"></i>${esc(firstLabel)}</span><span><i class="legend-line second"></i>${esc(secondLabel)}</span>`;
  $("chart-latency")?.setAttribute("aria-pressed", String(latency));
  $("chart-memory")?.setAttribute("aria-pressed", String(!latency));
}

function renderKvEvidence() {
  const data = DATA.kv_scaling;
  if (!data || !$("kv-chart")) return;
  $("kv-scope").textContent = `${data.hardware} · ${data.scope}`;
  $("kv-boundary").textContent = data.boundary;
  $("kv-kpis").innerHTML = `<article><strong>~${esc(data.headline.window_latency_ms)} ms</strong><span>fixed-window latency across 1K–64K in the prototype sweep</span></article><article><strong>${Number(data.headline.speedup_at_64k).toFixed(0)}×</strong><span>decode-all / window-read at 64K</span></article><article><strong>${esc(data.headline.kv_memory_reduction)}</strong><span>native fused-attention KV memory reduction</span></article>`;
  const tbody = document.querySelector("#quality-table tbody");
  if (tbody) tbody.innerHTML = list(data.quality).map((q) => `<tr class="${q.recommended ? "recommended" : ""}"><td>${esc(q.scheme)}${q.recommended ? ' <span class="level">usable point</span>' : ""}</td><td>${q.bits_per_value}</td><td>${q.compression_vs_fp16}×</td><td>${Number(q.attention_mse).toExponential(2)}</td></tr>`).join("");
  const status = $("native-status");
  if (status) status.innerHTML = list(data.native_status).map((item) => `<article class="card"><h3>${esc(item.stage)}</h3><strong>${esc(item.result)}</strong><p>${esc(item.correctness)}</p></article>`).join("");
  renderKvScaling();
  $("chart-latency")?.addEventListener("click", () => renderKvScaling("latency"));
  $("chart-memory")?.addEventListener("click", () => renderKvScaling("memory"));
}

setupNav(); releaseStatus(); renderEvidence(); renderEvidenceLadder(); renderCompatibility(); setupPlanner(); renderProfiles(); renderDownloads(); renderReleaseGates(); renderContribute(); renderKvEvidence();
["evidence-level","evidence-search","evidence-filter"].forEach((id) => $(id)?.addEventListener(id === "evidence-search" ? "input" : "change", renderEvidence));
["compat-runtime","compat-os","compat-status"].forEach((id) => $(id)?.addEventListener("change", renderCompatibility));
