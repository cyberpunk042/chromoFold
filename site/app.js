const DATA = window.CHROMOFOLD_SITE_DATA || {};
const $ = (id) => document.getElementById(id);

function esc(value) {
  return String(value ?? "").replace(/[&<>'"]/g, (char) => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[char]));
}

function renderRelease() {
  const release = DATA.release_channel || {};
  const status = release.release_status || "not-published";
  const version = release.latest_version || "No public candidate yet";
  $("release-status").innerHTML = `
    <span class="release-pill">${esc(release.channel || "candidate")}</span>
    <strong>${esc(version)}</strong>
    <p>${esc(status)}</p>
    <small>Production claims require exact-digest PASS evidence.</small>`;
}

function renderClaims(filter = "all") {
  const claims = (DATA.claims || []).filter((claim) => filter === "all" || claim.evidence_level === filter);
  $("claims").innerHTML = claims.length ? claims.map((claim) => `
    <article class="card evidence-card">
      <div class="card-top"><span class="level level-${esc(claim.evidence_level)}">${esc(claim.evidence_level)}</span><strong>${esc(claim.result)}</strong></div>
      <h3>${esc(claim.name)}</h3>
      <p>${esc(claim.meaning)}</p>
      <dl><div><dt>Hardware</dt><dd>${esc(claim.hardware)}</dd></div><div><dt>Scope</dt><dd>${esc(claim.scope)}</dd></div></dl>
    </article>`).join("") : '<p class="empty">No claims exist at this evidence level yet.</p>';
}

function renderDownloads() {
  const releases = DATA.downloads || [];
  $("downloads-list").innerHTML = releases.map((item) => `
    <article class="card download-card">
      <span class="level">${esc(item.status)}</span>
      <h3>${esc(item.name)}</h3>
      <p>${esc(item.description)}</p>
      <a class="secondary" href="${esc(item.href)}">${esc(item.action)} ↗</a>
    </article>`).join("");
}

function planFor(goal, vram, context, concurrency) {
  const mapping = {
    "longer-context": ["maximum-context", "KV-cache capacity"],
    "more-users": ["high-concurrency", "KV-cache capacity and scheduling"],
    "shared-prefix": ["shared-prefix", "duplicate prompt state"],
    "lowest-risk": ["safe", "compatibility and correctness"],
    "balanced": ["balanced", "workload-dependent memory pressure"],
  };
  const [profile, bottleneck] = mapping[goal];
  const tokenLoad = context * concurrency;
  const pressure = tokenLoad >= 262144 ? "critical" : tokenLoad >= 131072 ? "high" : tokenLoad >= 65536 ? "moderate" : "unknown";
  return {
    evidence_level: "estimate",
    profile,
    likely_bottleneck: bottleneck,
    advisory_pressure: pressure,
    inputs: {vram_gb: vram, context, concurrency, total_active_tokens: tokenLoad},
    qualification_required: true,
    next_steps: [
      "Download and verify the candidate archive.",
      "Run local machine inspection; browser inputs do not establish compatibility.",
      `Generate the ${profile} profile with silent fallback disabled.`,
      "Measure an identical baseline and candidate workload.",
      "Run exact-digest hardware qualification before publishing a win.",
    ],
  };
}

$("estimator").addEventListener("submit", (event) => {
  event.preventDefault();
  const result = planFor($("goal").value, Number($("vram").value), Number($("context").value), Number($("concurrency").value));
  $("plan").innerHTML = `<div class="plan-head"><strong>Advisory estimate — not a benchmark</strong><button id="copy-plan" type="button" class="small-button">Copy JSON</button></div><pre>${esc(JSON.stringify(result, null, 2))}</pre>`;
  $("copy-plan").addEventListener("click", async () => {
    await navigator.clipboard.writeText(JSON.stringify(result, null, 2));
    $("copy-plan").textContent = "Copied";
  });
});

$("evidence-filter").addEventListener("change", (event) => renderClaims(event.target.value));
renderRelease();
renderClaims();
renderDownloads();
