(() => {
  "use strict";
  const DATA = window.CHROMOFOLD_SITE_DATA || {};
  const $ = (id) => document.getElementById(id);
  const esc = (value) => String(value ?? "").replace(/[&<>'"]/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[c]));

  function historyChart(history) {
    const width = 900, height = 390, left = 72, right = 34, top = 28, bottom = 86;
    const values = history.map((item) => Number(item.compressed_latency_us));
    const max = Math.max(...values) * 1.08;
    const x = (index) => left + (width - left - right) * index / Math.max(1, history.length - 1);
    const y = (value) => height - bottom - (height - top - bottom) * value / max;
    const points = values.map((value, index) => `${x(index)},${y(value)}`).join(" ");
    const grid = [0, .25, .5, .75, 1].map((fraction) => {
      const value = max * fraction, py = y(value);
      return `<line x1="${left}" y1="${py}" x2="${width-right}" y2="${py}"/><text x="${left-10}" y="${py+4}" text-anchor="end">${Math.round(value/1000)} ms</text>`;
    }).join("");
    const labels = history.map((item, index) => `<text x="${x(index)}" y="${height-bottom+24}" text-anchor="middle"><tspan x="${x(index)}">${index + 1}</tspan><tspan x="${x(index)}" dy="16">${esc(item.gap_vs_dense)}× dense</tspan></text>`).join("");
    const dots = history.map((item, index) => `<circle class="point second-point" cx="${x(index)}" cy="${y(item.compressed_latency_us)}" r="5"><title>${esc(item.stage)} · ${(item.compressed_latency_us/1000).toFixed(3)} ms · ${item.gap_vs_dense}× fair dense</title></circle>`).join("");
    return `<svg viewBox="0 0 ${width} ${height}" aria-hidden="true"><g class="chart-grid">${grid}${labels}<text x="${left}" y="${height-12}">optimization stage</text></g><polyline class="series second" points="${points}"/>${dots}</svg>`;
  }

  function renderM11() {
    const data = DATA.native_kv_performance;
    if (!data || !$('m11-history-chart')) return;
    $('m11-scope').textContent = `${data.hardware} · ${data.model_shape} · ${data.context_tokens.toLocaleString()} tokens`;
    $('m11-kpis').innerHTML = `<article><strong>${esc(data.headline.speedup_over_original)}×</strong><span>faster than the original bit-by-bit compressed decoder</span></article><article><strong>${esc(data.headline.current_gap_vs_fair_dense)}</strong><span>against an equally warp-cooperative dense-f16 baseline</span></article><article><strong>NOT MET</strong><span>equal-or-better latency ship criterion on this measured workload</span></article>`;
    $('m11-history-chart').innerHTML = historyChart(data.history);
    $('m11-history-legend').innerHTML = data.history.map((item, index) => `<span><i class="legend-line ${index === data.history.length - 1 ? 'second' : 'first'}"></i>${index + 1}. ${esc(item.stage)}</span>`).join('');
    $('m11-boundary').textContent = `${data.crossover_hypothesis} ${data.limitations.join(' ')}`;
    const tbody = document.querySelector('#fair-performance-table tbody');
    if (tbody) tbody.innerHTML = data.fair_comparisons.map((row) => `<tr><td>${esc(row.workload)}</td><td>${row.compressed_us.toLocaleString()} µs</td><td>${row.dense_f16_us.toLocaleString()} µs</td><td>${row.compressed_over_dense.toFixed(2)}× slower</td></tr>`).join('');
  }

  function renderCampaign() {
    const data = DATA.kv_crossover_campaign;
    if (!data || !$('campaign-summary')) return;
    $('campaign-status').textContent = data.status;
    const combinations = Object.values(data.dimensions).reduce((total, values) => total * values.length, 1);
    $('campaign-summary').innerHTML = `<article class="card"><h3>${combinations.toLocaleString()} configurations</h3><p>Context, head dimension, query count, KV heads and code width are varied systematically.</p></article><article class="card"><h3>${data.required_repetitions} repetitions minimum</h3><p>Both compressed and fair dense latency must remain at or below ${data.maximum_cv_pct}% coefficient of variation.</p></article><article class="card"><h3>Four publication states</h3><p>${data.publication_states.map(esc).join(' · ')}</p></article><article class="card"><h3>Fairness locked</h3><p>${esc(data.fairness[0])}</p></article>`;
    $('campaign-boundary').textContent = `${data.parity_rule} ${data.win_rule} ${data.boundary}`;
  }

  renderM11();
  renderCampaign();
})();
