const DATA = window.CHROMOFOLD_SITE_DATA || {};
const $ = (id) => document.getElementById(id);

function renderClaims() {
  const claims = DATA.claims || [];
  $("claims").innerHTML = claims.map((claim) => `
    <article class="card">
      <span class="level">${claim.evidence_level}</span>
      <strong>${claim.result}</strong>
      <h3>${claim.name}</h3>
      <p>${claim.meaning}</p>
      <small>${claim.hardware} · ${claim.scope}</small>
    </article>`).join("");
}

function renderDownloads() {
  const releases = DATA.downloads || [];
  $("downloads-list").innerHTML = releases.map((item) => `
    <article class="card">
      <span class="level">${item.status}</span>
      <h3>${item.name}</h3>
      <p>${item.description}</p>
      <a class="secondary" href="${item.href}">${item.action}</a>
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
  const pressure = context * concurrency >= 131072 ? "high" : context * concurrency >= 65536 ? "moderate" : "unknown";
  return {
    evidence_level: "estimate",
    profile,
    likely_bottleneck: bottleneck,
    advisory_pressure: pressure,
    inputs: {vram_gb: vram, context, concurrency},
    next_steps: [
      "Download the local bundle.",
      "Run machine inspection; browser inputs do not establish compatibility.",
      `Generate the ${profile} profile with silent fallback disabled.`,
      "Measure an identical baseline and candidate workload.",
      "Run exact-digest hardware qualification before publishing a win.",
    ],
  };
}

$("estimator").addEventListener("submit", (event) => {
  event.preventDefault();
  const result = planFor(
    $("goal").value,
    Number($("vram").value),
    Number($("context").value),
    Number($("concurrency").value),
  );
  $("plan").innerHTML = `<strong>Advisory plan — not a benchmark</strong><pre>${JSON.stringify(result, null, 2)}</pre>`;
});

renderClaims();
renderDownloads();