const DECK = window.__DECK || [];

// ── Navigation helper (preserves other query params) ──
function go(param, value) {
  const u = new URL(window.location);
  u.searchParams.set(param, value);
  u.searchParams.delete('refresh');
  window.location = u;
}

// ── Custom dropdowns ──
document.querySelectorAll('.cs').forEach(cs => {
  const btn = cs.querySelector('.cs-btn');
  btn.addEventListener('click', e => {
    e.stopPropagation();
    document.querySelectorAll('.cs.open').forEach(o => { if (o !== cs) o.classList.remove('open'); });
    cs.classList.toggle('open');
  });
  // Multi-select area filter: toggle items (menu stays open), apply on demand.
  if (cs.classList.contains('multi')) {
    const label = cs.querySelector('.cs-btn span');
    const items = () => [...cs.querySelectorAll('.cs-menu li[data-value]')];
    const relabel = () => {
      const n = items().filter(li => li.classList.contains('sel')).length;
      label.textContent = (n === 0 || n === items().length) ? 'All areas' : n + ' area' + (n === 1 ? '' : 's');
    };
    const apply = () => {
      const sel = items().filter(li => li.classList.contains('sel'));
      const val = (sel.length === 0 || sel.length === items().length)
        ? '' : sel.map(li => li.dataset.value).join(',');
      go('nh', val);
    };
    items().forEach(li => li.addEventListener('click', e => {
      e.stopPropagation(); li.classList.toggle('sel'); relabel();
    }));
    const tools = cs.querySelector('.cs-tools');
    if (tools) {
      tools.addEventListener('click', e => e.stopPropagation());
      tools.querySelector('[data-all]')?.addEventListener('click', () => { items().forEach(li => li.classList.add('sel')); relabel(); });
      tools.querySelector('[data-none]')?.addEventListener('click', () => { items().forEach(li => li.classList.remove('sel')); relabel(); });
      tools.querySelector('[data-apply]')?.addEventListener('click', apply);
    }
    relabel();
    return;   // skip single-select navigation wiring
  }

  cs.querySelectorAll('.cs-menu li').forEach(li => {
    li.addEventListener('click', () => go(cs.dataset.param, li.dataset.value));
  });
});
document.addEventListener('click', () => document.querySelectorAll('.cs.open').forEach(o => o.classList.remove('open')));

// ── Building grouping: toggle grouped/flat, and expand a building's units ──
const grpToggle = document.getElementById('grp-toggle');
if (grpToggle) grpToggle.addEventListener('click', () => go('group', grpToggle.dataset.group));

document.querySelectorAll('.bldg-toggle').forEach(btn => {
  btn.addEventListener('click', e => {
    e.preventDefault(); e.stopPropagation();
    const panel = document.getElementById(btn.dataset.target);
    if (!panel) return;
    panel.hidden = !panel.hidden;
    btn.classList.toggle('open', !panel.hidden);
  });
});

// ── Type-to-filter: a search box inside each list dropdown ──
(function () {
  document.querySelectorAll('.cs').forEach(cs => {
    const menu = cs.querySelector('.cs-menu');
    if (!menu) return;
    const opts = [...menu.querySelectorAll('li[data-value]')];
    if (!opts.length) return;                         // skip the price slider (no options)

    const search = document.createElement('input');
    search.type = 'text';
    search.className = 'cs-search';
    search.placeholder = 'Type to filter…';
    const tools = menu.querySelector('.cs-tools');
    if (tools) tools.after(search); else menu.prepend(search);

    const filter = () => {
      const q = search.value.trim().toLowerCase();
      opts.forEach(li => { li.style.display = li.textContent.toLowerCase().includes(q) ? '' : 'none'; });
    };
    search.addEventListener('click', e => e.stopPropagation());
    search.addEventListener('input', filter);
    search.addEventListener('keydown', e => {
      if (e.key === 'Enter') {                        // Enter = act on the first match
        e.preventDefault();
        const first = opts.find(li => li.style.display !== 'none');
        if (first) first.click();
      } else if (e.key === 'Escape') {
        cs.classList.remove('open');
      }
    });
    // Focus the box and clear any prior filter each time the dropdown opens.
    cs.querySelector('.cs-btn').addEventListener('click', () => {
      setTimeout(() => {
        if (cs.classList.contains('open')) { search.value = ''; filter(); search.focus(); }
      }, 0);
    });
  });
})();

// ── Price dual-range slider ──
(function () {
  const lo = document.getElementById('p-min');
  const hi = document.getElementById('p-max');
  if (!lo || !hi) return;
  const range = document.getElementById('p-range');
  const vMin = document.getElementById('pv-min');
  const vMax = document.getElementById('pv-max');
  const PMIN = +lo.min, PMAX = +lo.max;
  const fmt = v => (+v >= PMAX) ? '$' + (PMAX / 1000) + 'k+' : '$' + (+v).toLocaleString();
  function paint() {
    let a = +lo.value, b = +hi.value;
    if (a > b) { const t = a; a = b; b = t; }
    range.style.left = ((a - PMIN) / (PMAX - PMIN) * 100) + '%';
    range.style.right = (100 - (b - PMIN) / (PMAX - PMIN) * 100) + '%';
    vMin.textContent = fmt(a); vMax.textContent = fmt(b);
  }
  // Keep the thumbs from crossing, and repaint as they move.
  lo.addEventListener('input', () => { if (+lo.value > +hi.value) hi.value = lo.value; paint(); });
  hi.addEventListener('input', () => { if (+hi.value < +lo.value) lo.value = hi.value; paint(); });
  paint();
  // Don't let interacting with the slider close the dropdown.
  const menu = document.querySelector('#price-cs .cs-menu');
  if (menu) menu.addEventListener('click', e => e.stopPropagation());
  document.getElementById('p-reset').addEventListener('click', () => { lo.value = PMIN; hi.value = PMAX; paint(); });
  document.getElementById('p-apply').addEventListener('click', () => {
    const a = Math.min(+lo.value, +hi.value), b = Math.max(+lo.value, +hi.value);
    const u = new URL(window.location);
    if (a <= PMIN) u.searchParams.delete('pmin'); else u.searchParams.set('pmin', a);
    if (b >= PMAX) u.searchParams.delete('pmax'); else u.searchParams.set('pmax', b);
    u.searchParams.delete('refresh');
    window.location = u;
  });
})();

// ── Collapse / summarize filters (persisted) ──
(function () {
  const tools = document.querySelector('.tools');
  const toggle = document.getElementById('filters-toggle');
  const summary = document.getElementById('filter-summary');
  if (!tools || !toggle) return;
  if (localStorage.getItem('filtersCollapsed') === '1') tools.classList.add('collapsed');
  const setCollapsed = c => {
    tools.classList.toggle('collapsed', c);
    localStorage.setItem('filtersCollapsed', c ? '1' : '0');
  };
  toggle.addEventListener('click', () => setCollapsed(!tools.classList.contains('collapsed')));
  if (summary) summary.addEventListener('click', () => setCollapsed(false));
})();

// ── Reactions (hearts + swipes) ──
async function react(id, reaction) {
  const res = await fetch('/api/react', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({listing_id: id, reaction}),
  });
  return res.json();
}
function paintHeart(id, liked) {
  const h = document.querySelector('.heart[data-id="' + CSS.escape(id) + '"]');
  if (h) h.classList.toggle('on', liked);
}
document.querySelectorAll('.heart').forEach(h => {
  h.addEventListener('click', async e => {
    e.preventDefault(); e.stopPropagation();
    const liked = h.classList.contains('on');
    const out = await react(h.dataset.id, liked ? 'none' : 'liked');
    h.classList.toggle('on', out.reaction === 'liked');
  });
});
// Undo (restore) a passed listing — clears the reaction and drops the card.
document.querySelectorAll('.undo').forEach(b => {
  b.addEventListener('click', async e => {
    e.preventDefault(); e.stopPropagation();
    await react(b.dataset.id, 'none');
    b.closest('.card').remove();
  });
});
// Pass (✕) a listing straight from the grid — record it and slide the card out
// (passed listings are hidden from every non-"passed" view, so it shouldn't
// linger here).
document.querySelectorAll('.card-pass').forEach(b => {
  b.addEventListener('click', async e => {
    e.preventDefault(); e.stopPropagation();
    react(b.dataset.id, 'passed');                 // fire-and-forget persist
    const card = b.closest('.card');
    if (!card) return;
    card.style.transition = 'opacity .2s, transform .2s';
    card.style.opacity = '0';
    card.style.transform = 'scale(.97)';
    setTimeout(() => card.remove(), 190);
  });
});

// ── Click a card to open its detail page ──
// Ignore clicks that land on a button or link inside the card (heart, ✨ similar,
// view ↗, tags) — those have their own behavior.
document.querySelectorAll('.card[data-href]').forEach(card => {
  card.addEventListener('click', e => {
    if (e.target.closest('a, button')) return;
    window.location = card.dataset.href;
  });
});

// ── Grid photo paging (same feel as Tinder) ──
// Tapping the left/right third of a card photo flips through that listing's
// photos; the center third (and the body below) still opens the detail page.
const DECK_BY_ID = {};
DECK.forEach(r => { DECK_BY_ID[r.listing_id] = r; });

document.querySelectorAll('.card[data-href] .photo[data-id]').forEach(photo => {
  const r = DECK_BY_ID[photo.dataset.id];
  const positions = r && r.img_positions;
  if (!positions || positions.length < 2) return;     // nothing to page through

  let gi = Math.max(0, positions.indexOf(r.img_pos));
  const count = document.createElement('span');
  count.className = 'photo-count';
  const left = document.createElement('span');
  left.className = 'photo-nav left'; left.textContent = '‹';
  const right = document.createElement('span');
  right.className = 'photo-nav right'; right.textContent = '›';
  photo.append(count, left, right);

  const paint = () => {
    photo.style.backgroundImage = "url('/img/" + r.listing_id + "/" + positions[gi] + "')";
    count.textContent = (gi + 1) + ' / ' + positions.length;
  };
  paint();

  photo.addEventListener('click', e => {
    if (e.target.closest('.heart, .undo, .card-pass')) return;  // let those buttons work
    const rect = photo.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    if (x < 0.34) { e.stopPropagation(); gi = (gi - 1 + positions.length) % positions.length; paint(); }
    else if (x > 0.66) { e.stopPropagation(); gi = (gi + 1) % positions.length; paint(); }
    // center third: let the click bubble to the card → opens the detail page
  });
});

const refInput = document.getElementById('ref');
const refChip = document.getElementById('ref-chip');
const refClear = document.getElementById('ref-clear');
if (refClear) refClear.addEventListener('click', () => {
  refInput.value = '';
  refChip.classList.add('hidden');
});

// ── Tinder mode ──
const tinder = document.getElementById('tinder');
const stage  = document.getElementById('t-stage');
const counter = document.getElementById('t-count');
const doneEl = document.getElementById('t-done');
const undoBtn = document.getElementById('t-undo');
const toast = document.getElementById('t-toast');
const toastMsg = document.getElementById('t-toast-msg');
let ti = 0;
let history = [];   // {idx, id, prev} per decision, for undo
let toastTimer = null;

// Photo paging: each row carries `img_positions` (ordered photo indices) and we
// track which one is showing in `r._pi`. It starts on the representative photo
// (`img_pos`) so the first thing you see is the same hero shot as the grid.
function photoCount(r) { return (r.img_positions && r.img_positions.length) || 0; }
function curPhotoIdx(r) {
  if (r._pi === undefined) {
    const k = r.img_positions ? r.img_positions.indexOf(r.img_pos) : -1;
    r._pi = k >= 0 ? k : 0;
  }
  return r._pi;
}
function curPhotoPos(r) {
  const n = photoCount(r);
  if (!n) return r.img_pos;
  return r.img_positions[curPhotoIdx(r)];
}

function cardMarkup(r) {
  const pos = curPhotoPos(r);
  const url = (pos !== null && pos !== undefined) ? "/img/" + r.listing_id + "/" + pos : "";
  // Two layers: a blurred "cover" fill so the frame is never empty, and the
  // full photo "contain"-ed on top so nothing important gets cropped.
  const layers = url
    ? '<div class="t-photo-fill" style="background-image:url(\'' + url + '\')"></div>' +
      '<div class="t-photo-img" style="background-image:url(\'' + url + '\')"></div>'
    : '';
  const n = photoCount(r);
  const counter = n > 1
    ? '<span class="t-photo-count">' + (curPhotoIdx(r) + 1) + ' / ' + n + '</span>'
    : '';
  const badge = r.pct_diff < 0
    ? '<span class="t-badge good">' + Math.round(-r.pct_diff*100) + '% below model · save $' + r.delta.toLocaleString() + '/mo</span>'
    : '<span class="t-badge bad">' + Math.round(r.pct_diff*100) + '% above model</span>';
  const sqft = r.sqft ? ' · ' + r.sqft.toLocaleString() + ' ft²' : '';
  const beds = (r.beds === null || r.beds === undefined) ? '?' : Math.round(r.beds);
  const baths = (r.baths === null || r.baths === undefined) ? '?' : r.baths;
  const tags = (r.tags && r.tags.length)
    ? '<div class="t-tags">' + r.tags.slice(0, 6).map(
        t => '<span class="t-tag">' + t.replace(/_/g, ' ') + '</span>').join('') + '</div>'
    : '';
  return '<div class="t-photo">' + layers + counter +
      '<span class="t-nav-hint left">‹</span><span class="t-nav-hint right">›</span>' +
    '</div>' +
    '<span class="stamp like">LIKE</span><span class="stamp pass">NOPE</span>' +
    '<div class="t-info">' + badge +
      '<div class="t-addr">' + r.name + '</div>' +
      '<div class="t-meta">' + r.neighborhood + ' · ' + beds + ' bd / ' + baths + ' ba' + sqft + '</div>' + tags +
      '<div class="t-price">$' + r.rent.toLocaleString() + ' <span>model $' + r.predicted.toLocaleString() + '/mo</span></div>' +
      ((r.lat != null && r.lng != null)
        ? '<div class="t-map-wrap"><div class="t-map"></div>' +
          '<button class="t-map-expand" type="button" aria-label="Expand map" title="Expand map">⤢</button></div>'
        : '') +
    '</div>';
}

// One tiny Leaflet map, mounted on the top card only and torn down each render.
// Display-only (pointer-events:none in CSS) so dragging across it still swipes.
let tMap = null;
function clearMiniMap() { if (tMap) { tMap.remove(); tMap = null; } }
function mountMiniMap(cardEl, r) {
  const el = cardEl.querySelector('.t-map');
  if (!el || !window.L || r.lat == null || r.lng == null) return;
  tMap = L.map(el, {
    zoomControl: false, attributionControl: false, dragging: false,
    scrollWheelZoom: false, doubleClickZoom: false, boxZoom: false,
    keyboard: false, touchZoom: false, tap: false,
  }).setView([r.lat, r.lng], 15);
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 19 }).addTo(tMap);
  L.marker([r.lat, r.lng]).addTo(tMap);
}

function renderDeck() {
  clearMiniMap();
  stage.innerHTML = '';
  counter.textContent = ti < DECK.length ? (ti + 1) + ' / ' + DECK.length : DECK.length + ' / ' + DECK.length;
  if (ti >= DECK.length) { doneEl.classList.remove('hidden'); return; }
  doneEl.classList.add('hidden');

  // Card behind (depth), if present.
  if (ti + 1 < DECK.length) {
    const behind = document.createElement('div');
    behind.className = 't-card behind';
    behind.innerHTML = cardMarkup(DECK[ti + 1]);
    stage.appendChild(behind);
  }
  // Top card.
  const top = document.createElement('div');
  top.className = 't-card';
  top.innerHTML = cardMarkup(DECK[ti]);
  stage.appendChild(top);
  attachDrag(top);
  mountMiniMap(top, DECK[ti]);
}

function decide(reaction) {
  const r = DECK[ti];
  history.push({idx: ti, id: r.listing_id, prev: r.reaction || null});
  r.reaction = reaction;
  react(r.listing_id, reaction);                 // persist (fire and forget)
  paintHeart(r.listing_id, reaction === 'liked'); // reflect in grid behind
  ti += 1;
  showToast(reaction);
  updateUndo();
  renderDeck();
}

function undoLast() {
  const last = history.pop();
  if (!last) return;
  ti = last.idx;
  DECK[ti].reaction = last.prev;
  react(last.id, last.prev || 'none');           // persist the restored state
  paintHeart(last.id, last.prev === 'liked');
  hideToast();
  updateUndo();
  renderDeck();
}

function updateUndo() { undoBtn.disabled = history.length === 0; }

function showToast(reaction) {
  toastMsg.textContent = reaction === 'liked' ? 'Liked ❤' : 'Passed ✕';
  toast.classList.remove('hidden');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(hideToast, 2600);
}
function hideToast() { clearTimeout(toastTimer); toast.classList.add('hidden'); }

function flyTop(dir) {                            // dir: 1 = like/right, -1 = pass/left
  const top = stage.querySelector('.t-card:not(.behind)');
  if (!top) return;
  top.style.transition = 'transform .3s, opacity .3s';
  top.style.transform = 'translate(' + (dir * 700) + 'px,0) rotate(' + (dir * 28) + 'deg)';
  top.style.opacity = '0';
  setTimeout(() => decide(dir > 0 ? 'liked' : 'passed'), 230);
}

// Page to another photo of the SAME listing and update the card in place
// (no full re-render, so the drag handlers on the card stay attached).
function navPhoto(el, r, dir) {
  const n = photoCount(r);
  if (n < 2) return;
  const next = curPhotoIdx(r) + dir;
  if (next < 0 || next >= n) return;             // clamp at the ends, no wrap
  r._pi = next;
  const url = "url('/img/" + r.listing_id + "/" + r.img_positions[r._pi] + "')";
  el.querySelectorAll('.t-photo-fill, .t-photo-img').forEach(d => d.style.backgroundImage = url);
  const cnt = el.querySelector('.t-photo-count');
  if (cnt) cnt.textContent = (r._pi + 1) + ' / ' + n;
}

function attachDrag(el) {
  let sx = 0, sy = 0, dx = 0, dragging = false, moved = 0;
  const like = el.querySelector('.stamp.like');
  const pass = el.querySelector('.stamp.pass');
  el.addEventListener('pointerdown', e => {
    if (e.target.closest('.t-map-expand')) return;   // let the expand button get its own click
    dragging = true; sx = e.clientX; sy = e.clientY; dx = 0; moved = 0;
    el.setPointerCapture(e.pointerId); el.style.transition = 'none';
  });
  el.addEventListener('pointermove', e => {
    if (!dragging) return;
    dx = e.clientX - sx; const dy = e.clientY - sy;
    moved = Math.max(moved, Math.hypot(dx, dy));
    el.style.transform = 'translate(' + dx + 'px,' + dy + 'px) rotate(' + (dx / 18) + 'deg)';
    if (like) like.style.opacity = Math.max(0, Math.min(1, dx / 110));
    if (pass) pass.style.opacity = Math.max(0, Math.min(1, -dx / 110));
  });
  function snapBack() {
    el.style.transition = 'transform .25s';
    el.style.transform = '';
    if (like) like.style.opacity = 0;
    if (pass) pass.style.opacity = 0;
  }
  function end(e) {
    if (!dragging) return;
    dragging = false;
    if (dx > 110) { flyTop(1); }
    else if (dx < -110) { flyTop(-1); }
    else if (moved < 8 && e) {
      // A tap (not a drag): left half → previous photo, right half → next.
      const rect = el.getBoundingClientRect();
      navPhoto(el, DECK[ti], (e.clientX - rect.left) < rect.width / 2 ? -1 : 1);
      snapBack();
    } else {
      snapBack();
    }
  }
  el.addEventListener('pointerup', end);
  el.addEventListener('pointercancel', () => end(null));
}

function openTinder() {
  ti = 0; history = []; hideToast(); updateUndo();
  tinder.classList.remove('hidden'); renderDeck();
}
function closeTinder() { hideToast(); hideHint(); clearMiniMap(); closeFullMap(); tinder.classList.add('hidden'); }

// ── Fullscreen map (the ⤢ expand button on a card's mini-map) ──
let fullMap = null;
function openFullMap(r) {
  if (!window.L || r.lat == null || r.lng == null) return;
  document.getElementById('t-map-full').classList.remove('hidden');
  if (fullMap) { fullMap.remove(); fullMap = null; }
  fullMap = L.map('t-map-full-canvas').setView([r.lat, r.lng], 16);
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
    { maxZoom: 19, attribution: '© OpenStreetMap' }).addTo(fullMap);
  L.marker([r.lat, r.lng]).addTo(fullMap).bindPopup(r.name || '').openPopup();
  setTimeout(() => { if (fullMap) fullMap.invalidateSize(); }, 30);  // container was display:none
}
function closeFullMap() {
  if (fullMap) { fullMap.remove(); fullMap = null; }
  document.getElementById('t-map-full').classList.add('hidden');
}
// The expand button lives on a re-rendered card, so delegate from the stage.
stage.addEventListener('click', e => {
  if (e.target.closest('.t-map-expand')) { e.stopPropagation(); openFullMap(DECK[ti]); }
});
document.getElementById('t-map-full-close').addEventListener('click', closeFullMap);

// "?" help popover
const hintEl = document.getElementById('t-hint');
function hideHint() { if (hintEl) hintEl.classList.add('hidden'); }

const openBtn = document.getElementById('tinder-open');
if (openBtn) openBtn.addEventListener('click', openTinder);
document.getElementById('t-close').addEventListener('click', closeTinder);
document.getElementById('t-back').addEventListener('click', closeTinder);
document.getElementById('t-help').addEventListener('click', e => {
  e.stopPropagation();                       // don't trigger the click-out handler
  if (hintEl) hintEl.classList.toggle('hidden');
});
// Click on the backdrop (the dark whitespace around the card) closes the modal;
// a click anywhere else just dismisses the help popover.
tinder.addEventListener('click', e => {
  if (e.target === tinder) { closeTinder(); return; }
  if (hintEl && !hintEl.contains(e.target)) hideHint();
});
document.getElementById('t-like').addEventListener('click', () => flyTop(1));
document.getElementById('t-pass').addEventListener('click', () => flyTop(-1));
undoBtn.addEventListener('click', undoLast);
document.getElementById('t-toast-undo').addEventListener('click', undoLast);
document.addEventListener('keydown', e => {
  if (tinder.classList.contains('hidden')) return;
  if (fullMap) { if (e.key === 'Escape') closeFullMap(); return; }   // map open: don't swipe/undo
  if (e.key === 'ArrowRight') flyTop(1);
  else if (e.key === 'ArrowLeft') flyTop(-1);
  else if (e.key === 'z' || e.key === 'Backspace') { e.preventDefault(); undoLast(); }
  else if (e.key === 'Escape') closeTinder();
});
