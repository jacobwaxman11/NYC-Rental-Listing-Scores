// Detail page: photo gallery, Leaflet map, and the save (heart) button.
const G = window.__GALLERY || { id: '', positions: [] };

// ── Photo gallery ──
const main = document.getElementById('g-main');
const idxEl = document.getElementById('g-idx');
const thumbs = document.getElementById('g-thumbs');
let gi = 0;

function showPhoto(i) {
  if (!G.positions.length) return;
  gi = (i + G.positions.length) % G.positions.length;   // wrap around
  if (main) main.style.backgroundImage =
    "url('/img/" + G.id + "/" + G.positions[gi] + "')";
  if (idxEl) idxEl.textContent = gi + 1;
  if (thumbs) thumbs.querySelectorAll('.g-thumb').forEach((t, k) =>
    t.classList.toggle('sel', k === gi));
}

const prev = document.getElementById('g-prev');
const next = document.getElementById('g-next');
if (prev) prev.addEventListener('click', () => showPhoto(gi - 1));
if (next) next.addEventListener('click', () => showPhoto(gi + 1));
if (thumbs) thumbs.querySelectorAll('.g-thumb').forEach(t =>
  t.addEventListener('click', () => showPhoto(parseInt(t.dataset.i, 10))));
// Tap the photo halves (mobile) + arrow keys.
if (main) main.addEventListener('click', e => {
  if (e.target.closest('.g-nav')) return;             // buttons handle themselves
  const rect = main.getBoundingClientRect();
  showPhoto(gi + ((e.clientX - rect.left) < rect.width / 2 ? -1 : 1));
});
document.addEventListener('keydown', e => {
  if (e.key === 'ArrowLeft') showPhoto(gi - 1);
  else if (e.key === 'ArrowRight') showPhoto(gi + 1);
});

// ── Map (Leaflet + OpenStreetMap) ──
const mapEl = document.getElementById('map');
if (mapEl && window.L) {
  const lat = parseFloat(mapEl.dataset.lat);
  const lng = parseFloat(mapEl.dataset.lng);
  if (!isNaN(lat) && !isNaN(lng)) {
    const map = L.map(mapEl, { scrollWheelZoom: false }).setView([lat, lng], 15);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19,
      attribution: '© OpenStreetMap contributors',
    }).addTo(map);
    L.marker([lat, lng]).addTo(map).bindPopup(mapEl.dataset.label || 'Listing');
  }
}

// ── Building units: reveal the capped overflow ──
const unitsMore = document.getElementById('units-more');
if (unitsMore) unitsMore.addEventListener('click', () => {
  document.querySelectorAll('#unit-list .unit.extra').forEach(u => { u.hidden = false; });
  unitsMore.remove();
});

// ── Like / Pass ──
// Like and pass are mutually exclusive — the backend stores a single reaction,
// so we just repaint both buttons from whatever it returns.
const heart = document.getElementById('d-heart');
const passBtn = document.getElementById('d-pass');

function paintActions(reaction) {
  if (heart) {
    const on = reaction === 'liked';
    heart.classList.toggle('on', on);
    const s = heart.querySelector('span'); if (s) s.textContent = on ? 'Saved' : 'Save';
  }
  if (passBtn) {
    const on = reaction === 'passed';
    passBtn.classList.toggle('on', on);
    const s = passBtn.querySelector('span'); if (s) s.textContent = on ? 'Passed' : 'Pass';
  }
}

async function setReaction(id, reaction) {
  const res = await fetch('/api/react', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ listing_id: id, reaction }),
  });
  const out = await res.json();
  paintActions(out.reaction);
}

if (heart) heart.addEventListener('click', () =>
  setReaction(heart.dataset.id, heart.classList.contains('on') ? 'none' : 'liked'));
if (passBtn) passBtn.addEventListener('click', () =>
  setReaction(passBtn.dataset.id, passBtn.classList.contains('on') ? 'none' : 'passed'));
