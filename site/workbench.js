(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const esc = (v) => String(v ?? "").replace(/[&<>'"]/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[c]));
  const MATCH = ["model", "runtime", "hardware", "workload"];
  const REQUIRED = ["throughput_tokens_s", "latency_p50_ms", "latency_p95_ms", "peak_vram_mib", "correctness"];
  const EXAMPLE_BASELINE = {schema:"chromofold.evidence-result.v1",role:"baseline",fingerprint:{model:"Qwen2.5-0.5B",runtime:"llama.cpp@example",hardware:"NVIDIA RTX 2080 Ti",workload:"context-32768-concurrency-1",release_digest:null},metrics:{throughput_tokens_s:42,latency_p50_ms:24,latency_p95_ms:31,peak_vram_mib:8210,capacity_tokens:32768,correctness:true}};
  const EXAMPLE_CANDIDATE = {schema:"chromofold.evidence-result.v1",role:"candidate",fingerprint:{model:"Qwen2.5-0.5B",runtime:"llama.cpp@example",hardware:"NVIDIA RTX 2080 Ti",workload:"context-32768-concurrency-1",release_digest:"sha256:example-only"},metrics:{throughput_tokens_s:44.1,latency_p50_ms:25.2,latency_p95_ms:33,peak_vram_mib:4920,capacity_tokens:65536,correctness:true}};

  function validate(value, role) {
    if (!value || value.schema !== "chromofold.evidence-result.v1" || value.role !== role) throw new Error(`Expected ${role} chromofold.evidence-result.v1`);
    if (!value.fingerprint || !value.metrics) throw new Error(`${role}: fingerprint and metrics are required`);
    const missing = MATCH.filter((k) => !value.fingerprint[k]).concat(REQUIRED.filter((k) => !(k in value.metrics)));
    if (missing.length) throw new Error(`${role}: missing ${missing.join(", ")}`);
    return value;
  }
  function pct(c, b) { if (Number(b) === 0) throw new Error("Baseline metrics cannot be zero"); return (Number(c) - Number(b)) / Number(b) * 100; }
  function analyze(b, c) {
    validate(b, "baseline"); validate(c, "candidate");
    const mismatches = MATCH.filter((k) => b.fingerprint[k] !== c.fingerprint[k]);
    const deltas = {throughput_pct:pct(c.metrics.throughput_tokens_s,b.metrics.throughput_tokens_s),latency_p50_pct:pct(c.metrics.latency_p50_ms,b.metrics.latency_p50_ms),latency_p95_pct:pct(c.metrics.latency_p95_ms,b.metrics.latency_p95_ms),peak_vram_pct:pct(c.metrics.peak_vram_mib,b.metrics.peak_vram_mib)};
    if ("capacity_tokens" in b.metrics && "capacity_tokens" in c.metrics) deltas.capacity_pct = pct(c.metrics.capacity_tokens,b.metrics.capacity_tokens);
    const gates = {fingerprints_match:mismatches.length===0,baseline_correct:Boolean(b.metrics.correctness),candidate_correct:Boolean(c.metrics.correctness),release_digest_present:Boolean(c.fingerprint.release_digest),capacity_improved:(deltas.capacity_pct||0)>0||deltas.peak_vram_pct<0};
    const state = (!gates.fingerprints_match||!gates.baseline_correct||!gates.candidate_correct) ? "FAIL" : !gates.release_digest_present ? "INCOMPLETE" : gates.capacity_improved ? "PASS" : "FAIL";
    return {schema:"chromofold.evidence-analysis.v1",state,mismatched_fingerprint_fields:mismatches,gates,deltas,baseline:b.metrics,candidate:c.metrics,boundary:"PASS means the portable comparison gates passed. It is not maintainer qualification or independent reproduction."};
  }
  function metric(name, baseline, candidate, delta, better) {
    const good = better === "higher" ? delta >= 0 : delta <= 0;
    return `<article class="metric-card ${good ? "metric-good" : "metric-bad"}"><span>${esc(name)}</span><strong>${delta >= 0 ? "+" : ""}${delta.toFixed(2)}%</strong><small>${esc(baseline)} → ${esc(candidate)}</small></article>`;
  }
  function render(result) {
    const d=result.deltas,b=result.baseline,c=result.candidate;
    const metrics=[metric("Throughput",b.throughput_tokens_s,c.throughput_tokens_s,d.throughput_pct,"higher"),metric("Latency p50",b.latency_p50_ms,c.latency_p50_ms,d.latency_p50_pct,"lower"),metric("Latency p95",b.latency_p95_ms,c.latency_p95_ms,d.latency_p95_pct,"lower"),metric("Peak VRAM",b.peak_vram_mib,c.peak_vram_mib,d.peak_vram_pct,"lower")];
    if ("capacity_pct" in d) metrics.push(metric("Capacity",b.capacity_tokens,c.capacity_tokens,d.capacity_pct,"higher"));
    $("workbench-result").innerHTML=`<div class="plan-head"><div><span class="state state-${result.state.toLowerCase()}">${result.state}</span><h2>Portable comparison</h2></div><div class="actions"><button id="download-analysis" type="button" class="small-button">Download JSON</button><button id="download-report" type="button" class="small-button">Download report</button></div></div><div class="metric-grid">${metrics.join("")}</div><h3>Gates</h3><div class="gate-list">${Object.entries(result.gates).map(([k,v])=>`<div><span>${v?"PASS":"FAIL"}</span><code>${esc(k)}</code></div>`).join("")}</div>${result.mismatched_fingerprint_fields.length?`<p class="note">Mismatched fingerprints: ${esc(result.mismatched_fingerprint_fields.join(", "))}</p>`:""}<p class="note">${esc(result.boundary)}</p>`;
    const save=(name,type,text)=>{const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([text],{type}));a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);};
    $("download-analysis").onclick=()=>save("chromofold-evidence-analysis.json","application/json",JSON.stringify(result,null,2));
    $("download-report").onclick=()=>save("chromofold-evidence-report.md","text/markdown",`# ChromoFold evidence comparison\n\n**State:** ${result.state}\n\n${Object.entries(d).map(([k,v])=>`- ${k}: ${v>=0?"+":""}${v.toFixed(3)}%`).join("\n")}\n\n${result.boundary}\n`);
  }
  async function readFile(input) { const file=input.files?.[0]; if(!file) throw new Error("Select both result files"); return JSON.parse(await file.text()); }
  $("workbench-form")?.addEventListener("submit",async(e)=>{e.preventDefault();try{render(analyze(await readFile($("baseline-file")),await readFile($("candidate-file"))))}catch(err){$("workbench-result").innerHTML=`<p class="note"><strong>Cannot analyze:</strong> ${esc(err.message)}</p>`;}});
  $("load-example")?.addEventListener("click",()=>render(analyze(EXAMPLE_BASELINE,EXAMPLE_CANDIDATE)));
})();
