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
  cs.querySelectorAll('.cs-menu li').forEach(li => {
    li.addEventListener('click', () => go(cs.dataset.param, li.dataset.value));
  });
});
document.addEventListener('click', () => document.querySelectorAll('.cs.open').forEach(o => o.classList.remove('open')));

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

// ── AI "✨ similar": set this listing as the reference, focus the search box ──
const refInput = document.getElementById('ref');
const refChip = document.getElementById('ref-chip');
document.querySelectorAll('.similar').forEach(b => {
  b.addEventListener('click', e => {
    e.preventDefault(); e.stopPropagation();
    refInput.value = b.dataset.id;
    document.getElementById('ref-name').textContent = b.dataset.name;
    refChip.classList.remove('hidden');
    const q = document.getElementById('q');
    q.focus();
    if (!q.value.trim()) q.value = 'like this but ';
    q.setSelectionRange(q.value.length, q.value.length);
  });
});
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

function cardMarkup(r) {
  const bg = r.img_pos !== null ? "background-image:url('/img/" + r.listing_id + "/" + r.img_pos + "')" : "";
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
  return '<div class="t-photo" style="' + bg + '"></div>' +
    '<span class="stamp like">LIKE</span><span class="stamp pass">NOPE</span>' +
    '<div class="t-info">' + badge +
      '<div class="t-addr">' + r.name + '</div>' +
      '<div class="t-meta">' + r.neighborhood + ' · ' + beds + ' bd / ' + baths + ' ba' + sqft + '</div>' + tags +
      '<div class="t-price">$' + r.rent.toLocaleString() + ' <span>model $' + r.predicted.toLocaleString() + '/mo</span></div>' +
    '</div>';
}

function renderDeck() {
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

function attachDrag(el) {
  let sx = 0, sy = 0, dx = 0, dragging = false;
  const like = el.querySelector('.stamp.like');
  const pass = el.querySelector('.stamp.pass');
  el.addEventListener('pointerdown', e => {
    dragging = true; sx = e.clientX; sy = e.clientY; dx = 0;
    el.setPointerCapture(e.pointerId); el.style.transition = 'none';
  });
  el.addEventListener('pointermove', e => {
    if (!dragging) return;
    dx = e.clientX - sx; const dy = e.clientY - sy;
    el.style.transform = 'translate(' + dx + 'px,' + dy + 'px) rotate(' + (dx / 18) + 'deg)';
    if (like) like.style.opacity = Math.max(0, Math.min(1, dx / 110));
    if (pass) pass.style.opacity = Math.max(0, Math.min(1, -dx / 110));
  });
  function end() {
    if (!dragging) return;
    dragging = false;
    if (dx > 110) { flyTop(1); }
    else if (dx < -110) { flyTop(-1); }
    else {
      el.style.transition = 'transform .25s';
      el.style.transform = '';
      if (like) like.style.opacity = 0;
      if (pass) pass.style.opacity = 0;
    }
  }
  el.addEventListener('pointerup', end);
  el.addEventListener('pointercancel', end);
}

function openTinder() {
  ti = 0; history = []; hideToast(); updateUndo();
  tinder.classList.remove('hidden'); renderDeck();
}
function closeTinder() { hideToast(); tinder.classList.add('hidden'); }

const openBtn = document.getElementById('tinder-open');
if (openBtn) openBtn.addEventListener('click', openTinder);
document.getElementById('t-close').addEventListener('click', closeTinder);
document.getElementById('t-back').addEventListener('click', closeTinder);
document.getElementById('t-like').addEventListener('click', () => flyTop(1));
document.getElementById('t-pass').addEventListener('click', () => flyTop(-1));
undoBtn.addEventListener('click', undoLast);
document.getElementById('t-toast-undo').addEventListener('click', undoLast);
document.addEventListener('keydown', e => {
  if (tinder.classList.contains('hidden')) return;
  if (e.key === 'ArrowRight') flyTop(1);
  else if (e.key === 'ArrowLeft') flyTop(-1);
  else if (e.key === 'z' || e.key === 'Backspace') { e.preventDefault(); undoLast(); }
  else if (e.key === 'Escape') closeTinder();
});
