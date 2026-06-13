const $ = id => document.getElementById(id);
const logEl = $('log'), dot = $('dot'), statusEl = $('status'), stopBtn = $('stop');
const runBtns = document.querySelectorAll('button.run');
let cursor = 0, polling = false;

// ── Custom dropdowns (single + multi-select), styled like the deals page ──────
function csValue(name) {
  const cs = document.querySelector('.cs[data-name="' + name + '"]');
  if (cs.classList.contains('multi'))
    return [...cs.querySelectorAll('.cs-menu li[data-value].sel')].map(li => li.dataset.value);
  const sel = cs.querySelector('.cs-menu li[data-value].sel');
  return sel ? sel.dataset.value : '';
}

function updateAreasLabel() {
  const cs = document.querySelector('.cs[data-name="areas"]');
  const all = cs.querySelectorAll('.cs-menu li[data-value]');
  const sel = cs.querySelectorAll('.cs-menu li[data-value].sel');
  const label = $('areas-label');
  if (sel.length === all.length) label.textContent = 'All areas (' + all.length + ')';
  else if (sel.length === 0) label.textContent = 'No areas selected';
  else label.textContent = sel.length + ' of ' + all.length + ' areas';
}

function onProviderChange(v) {
  $('c_model').value = v === 'anthropic' ? 'claude-haiku-4-5' : 'gemini-2.5-flash';
}

document.querySelectorAll('.cs').forEach(cs => {
  const btn = cs.querySelector('.cs-btn');
  btn.addEventListener('click', e => {
    e.stopPropagation();
    document.querySelectorAll('.cs.open').forEach(o => { if (o !== cs) o.classList.remove('open'); });
    cs.classList.toggle('open');
  });

  cs.querySelectorAll('.cs-menu li[data-value]').forEach(li => {
    li.addEventListener('click', e => {
      e.stopPropagation();
      if (cs.classList.contains('multi')) {
        li.classList.toggle('sel');         // multi: toggle, keep menu open
        updateAreasLabel();
      } else {
        cs.querySelectorAll('.cs-menu li[data-value]').forEach(x => x.classList.remove('sel'));
        li.classList.add('sel');
        cs.querySelector('.cs-btn span').textContent = li.dataset.value;
        cs.classList.remove('open');
        if (cs.dataset.name === 'provider') onProviderChange(li.dataset.value);
      }
    });
  });

  // Select all / Clear tools (multi-select menus only).
  const allBtn = cs.querySelector('[data-all]'), noneBtn = cs.querySelector('[data-none]');
  if (allBtn) allBtn.addEventListener('click', e => {
    e.stopPropagation();
    cs.querySelectorAll('.cs-menu li[data-value]').forEach(li => li.classList.add('sel'));
    updateAreasLabel();
  });
  if (noneBtn) noneBtn.addEventListener('click', e => {
    e.stopPropagation();
    cs.querySelectorAll('.cs-menu li[data-value]').forEach(li => li.classList.remove('sel'));
    updateAreasLabel();
  });
});
document.addEventListener('click', () => document.querySelectorAll('.cs.open').forEach(o => o.classList.remove('open')));

// ── Parameter collection per stage ────────────────────────────────────────────
const PARAMS = {
  scrape: () => ({ areas: csValue('areas').join(','), price_min: $('s_price_min').value,
    price_max: $('s_price_max').value, max_listings: $('s_max_listings').value,
    max_pages: $('s_max_pages').value }),
  backfill: () => ({ impersonate: $('b_impersonate').value, min_delay: $('b_min_delay').value,
    max_delay: $('b_max_delay').value, max_listings: $('b_max_listings').value }),
  score: () => ({ provider: csValue('provider'), model: $('c_model').value,
    max_listings: $('c_max_listings').value }),
};

// ── Job control + live log polling ────────────────────────────────────────────
function setStatus(s) {
  dot.className = 'dot ' + (s || '');
  statusEl.textContent = s || 'idle';
  const running = s === 'running';
  runBtns.forEach(b => b.disabled = running);
  stopBtn.disabled = !running;
}

async function poll() {
  let d;
  try { d = await (await fetch('/api/run/log?cursor=' + cursor)).json(); }
  catch (e) { setTimeout(poll, 1200); return; }
  if (d.lines && d.lines.length) {
    logEl.textContent += (logEl.textContent ? '\n' : '') + d.lines.join('\n');
    logEl.scrollTop = logEl.scrollHeight;
    cursor = d.cursor;
  }
  setStatus(d.status);
  if (d.status === 'running') setTimeout(poll, 800);
  else polling = false;
}
function ensurePolling() { if (!polling) { polling = true; poll(); } }

runBtns.forEach(b => b.addEventListener('click', async () => {
  const stage = b.dataset.stage;
  if (stage === 'scrape' && csValue('areas').length === 0) {
    logEl.textContent = '⚠ Select at least one area to scrape.';
    return;
  }
  logEl.textContent = ''; cursor = 0;
  const d = await (await fetch('/api/run/' + stage, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(PARAMS[stage]())
  })).json();
  if (!d.ok) { logEl.textContent = '⚠ ' + d.error; return; }
  setStatus('running'); ensurePolling();
}));

stopBtn.addEventListener('click', () => fetch('/api/run/stop', { method: 'POST' }));

updateAreasLabel();

// Re-attach to any job already in flight on load.
(async () => {
  const d = await (await fetch('/api/run/log?cursor=0')).json();
  if (d.lines && d.lines.length) { logEl.textContent = d.lines.join('\n'); cursor = d.cursor; }
  setStatus(d.status);
  if (d.status === 'running') ensurePolling();
})();
