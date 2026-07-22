const $ = (id) => document.getElementById(id);
let selectedProfile = null;

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: {"Content-Type": "application/json"},
    ...options,
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || JSON.stringify(data));
  return data;
}

function pretty(value) {
  return JSON.stringify(value, null, 2);
}

function profileName(result) {
  return result.profile || result.recommended_profile || result.name || null;
}

async function loadCatalog() {
  const data = await api("/api/catalog");
  const profiles = data.profiles.profiles || data.profiles;
  $("catalog").innerHTML = Object.entries(profiles).map(([name, profile]) => `
    <article class="card">
      <h3>${name}</h3>
      <p>${profile.description || profile.goal || "ChromoFold profile"}</p>
      <small>${profile.qualification_required === false ? "Measured use still recommended" : "Hardware qualification required"}</small>
    </article>`).join("");
  const levels = data.evidence.levels || data.evidence;
  $("evidence").innerHTML = Object.entries(levels).map(([name, level]) => `
    <article class="card"><h3>${name}</h3><p>${level.description || level.requirement || pretty(level)}</p></article>`).join("");
}

$("inspect").addEventListener("click", async () => {
  $("machine").textContent = "Inspecting…";
  try { $("machine").textContent = pretty(await api("/api/inspect")); }
  catch (error) { $("machine").textContent = String(error); }
});

$("recommend").addEventListener("click", async () => {
  $("recommendation").textContent = "Analyzing workload…";
  $("configure").disabled = true;
  try {
    const result = await api("/api/recommend", {
      method: "POST",
      body: JSON.stringify({
        goal: $("goal").value,
        model: $("model").value,
        context: Number($("context").value),
        concurrency: Number($("concurrency").value),
      }),
    });
    selectedProfile = profileName(result);
    $("recommendation").innerHTML = `<div class="status">Estimate — qualification required</div><pre>${pretty(result)}</pre>`;
    $("configure").disabled = !selectedProfile;
  } catch (error) {
    $("recommendation").textContent = String(error);
  }
});

$("configure").addEventListener("click", async () => {
  $("bundle").textContent = "Generating…";
  try {
    const result = await api("/api/configure", {
      method: "POST",
      body: JSON.stringify({profile: selectedProfile, model: $("model").value}),
    });
    $("bundle").textContent = pretty(result);
  } catch (error) {
    $("bundle").textContent = String(error);
  }
});

loadCatalog().catch((error) => { $("catalog").textContent = String(error); });
