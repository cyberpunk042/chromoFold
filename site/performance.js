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
    const grid = [0, .25, .5, .75, 1].map((fraction) => { const value=max*fraction, py=y(value); return `<line x1="${left}" y1="${py}" x2="${width-right}" y2="${py}"/><text x="${left-10}" y="${py+4}" text-anchor="end">${Math.round(value/1000)} ms</text>`; }).join("");
    const labels = history.map((item,index)=>`<text x="${x(index)}" y="${height-bottom+24}" text-anchor="middle"><tspan x="${x(index)}">${index+1}</tspan><tspan x="${x(index)}" dy="16">${esc(item.gap_vs_dense)}× dense</tspan></text>`).join("");
    const dots = history.map((item,index)=>`<circle class="point second-point" cx="${x(index)}" cy="${y(item.compressed_latency_us)}" r="5"><title>${esc(item.stage)} · ${(item.compressed_latency_us/1000).toFixed(3)} ms · ${item.gap_vs_dense}× fair dense</title></circle>`).join("");
    return `<svg viewBox="0 0 ${width} ${height}" aria-hidden="true"><g class="chart-grid">${grid}${labels}<text x="${left}" y="${height-12}">optimization stage</text></g><polyline class="series second" points="${points}"/>${dots}</svg>`;
  }

  function renderM11() {
    const data=DATA.native_kv_performance;
    if(!data||!$('m11-history-chart')) return;
    $('m11-scope').textContent=`${data.hardware} · ${data.model_shape}`;
    $('m11-kpis').innerHTML=`<article><strong>${esc(data.headline.speedup_over_original)}×</strong><span>faster than the original compressed decoder</span></article><article><strong>${esc(data.headline.best_measured_latency_win)}</strong><span>only at head_dim 64 and long context</span></article><article><strong>${esc(data.headline.capacity_multiplier)}</strong><span>the primary measured product value</span></article>`;
    $('m11-history-chart').innerHTML=historyChart(data.history);
    $('m11-history-legend').innerHTML=data.history.map((item,index)=>`<span><i class="legend-line ${index===data.history.length-1?'second':'first'}"></i>${index+1}. ${esc(item.stage)}</span>`).join('');
    $('m11-boundary').textContent=`${data.crossover.interpretation} ${data.limitations.join(' ')}`;
    const tbody=document.querySelector('#fair-performance-table tbody');
    if(tbody) tbody.innerHTML=data.fair_comparisons.map((row)=>`<tr><td>${esc(row.workload)}</td><td colspan="2">${esc(row.verdict)}</td><td>${row.compressed_over_dense<1?'WIN':'NO CROSSOVER'}</td></tr>`).join('');
  }

  function renderCampaign() {
    const data=DATA.kv_crossover_campaign;
    if(!data||!$('campaign-summary')) return;
    $('campaign-status').textContent=data.status;
    const sweep=data.initial_sweep;
    const wins=(sweep?.results||[]).filter((r)=>r.state==='WIN').length;
    $('campaign-summary').innerHTML=`<article class="card"><h3>${wins} measured wins</h3><p>${esc(sweep?.finding||'No measured sweep published.')}</p></article><article class="card"><h3>head_dim 64</h3><p>Compressed overtakes dense from about 16K context and plateaus near a four-percent win.</p></article><article class="card"><h3>head_dim 128</h3><p>No crossover through 64K; decode arithmetic cancels the memory-read advantage.</p></article><article class="card"><h3>Capacity first</h3><p>${esc(sweep?.primary_value||'Capacity remains the primary value.')}</p></article>`;
    $('campaign-boundary').textContent=`${data.parity_rule} ${data.win_rule} ${data.boundary}`;
  }

  renderM11();
  renderCampaign();
})();
