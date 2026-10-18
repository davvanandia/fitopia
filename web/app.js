const $ = s => document.querySelector(s);
const cam = $('#cam'), msg = $('#msg');
const say = (t, err) => { msg.textContent = t; msg.className = 'msg' + (err ? ' err' : ''); };

async function api(url, body) {
  let r;
  try { r = await fetch(url, body ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) } : {}); }
  catch { throw new Error('Server Python belum menyala. Jalankan: python server.py'); }
  const j = await r.json(); if (!r.ok) throw new Error(j.error || 'Terjadi kesalahan'); return j;
}

async function startCamera() {
  try { cam.srcObject = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } }); }
  catch { say('Kamera tidak bisa dibuka. Izinkan akses kamera di browser, atau pakai "Hitung manual".', true); }
}

const wait = ms => new Promise(r => setTimeout(r, ms));
async function countdown(n = 5) { for (let i = n; i > 0; i--) { $('#count').textContent = i; await wait(1000); } $('#count').textContent = ''; }
async function frames(n = 10) {
  const c = document.createElement('canvas'); c.width = cam.videoWidth; c.height = cam.videoHeight; const out = [];
  for (let i = 0; i < n; i++) { c.getContext('2d').drawImage(cam, 0, 0); out.push(c.toDataURL('image/jpeg', .8)); await wait(150); }
  return out;
}
const form = () => ({ tb: $('#tb').value, bb: $('#bb').value, usia: $('#usia').value, aktivitas: $('#akt').value });

async function jalankan(mode) {
  const tombol = document.querySelectorAll('.btn'); tombol.forEach(b => b.disabled = true);
  try {
    if (mode !== 'man') {
      if (!cam.srcObject) throw new Error('Kamera belum aktif.');
      say('Berdiri tegak, tangan sedikit menjauh dari badan…'); await countdown(5);
      say('Mengukur… jangan bergerak.'); var fr = await frames();
      say('Menganalisis di server Python…');
    }
    if (mode === 'kal') {
      const j = await api('/api/kalibrasi', { ...form(), frames: fr });
      say(j.estimasi_bb ? 'Kalibrasi tersimpan. Berat badan kini diestimasi otomatis.' : 'Kalibrasi tinggi tersimpan. Siluet belum terbaca, berat tetap diisi manual.'); status();
    } else if (mode === 'scan') {
      const j = await api('/api/scan', { ...form(), frames: fr });
      if (j.perlu_bb) say(`Tinggi terukur ${j.tb} cm. Isi berat badan dari timbangan (atau lakukan kalibrasi), lalu scan lagi.`, true);
      else { say(`Selesai. Berat dari ${j.sumber}.`); tampil(j); riwayat(); }
    } else {
      const j = await api('/api/analisis', form()); say('Selesai (input manual).'); tampil(j); riwayat();
    }
  } catch (e) { say(e.message, true); }
  tombol.forEach(b => b.disabled = false);
}

function tampil(p) {
  $('#hasil').hidden = false;
  $('#stats').innerHTML = [['Tinggi', p.tb + ' cm'], ['Berat', p.bb + ' kg'], ['BMI', `${p.bmi} (${p.kategori})`], ['BB ideal', p.ideal.join('–') + ' kg'], ['TDEE', p.tdee + ' kkal'], ['Target menu', p.target + ' kkal']]
    .map(([l, v]) => `<div><span>${l}</span><b>${v}</b></div>`).join('');
  $('#mark').style.left = Math.min(Math.max((p.bmi - 14) / 20 * 100, 1), 98) + '%';
  $('#olah').innerHTML = `<b>${p.olahraga.jenis}</b><br>${p.olahraga.detail}`;
  $('#menu').innerHTML = p.menu.map(m => `<li><span><b>${m.nama}</b><br>${m.desk}</span><em>${m.kkal} kkal</em></li>`).join('');
  $('#hasil').scrollIntoView({ behavior: 'smooth' });
}

async function riwayat() {
  try {
    const d = await api('/api/riwayat'); $('#rows').innerHTML = d.slice().reverse().map(r => `<tr><td>${r.waktu}</td><td>${r.tb_cm}</td><td>${r.bb_kg}</td><td>${r.bmi}</td><td>${r.sumber_bb || '-'}</td></tr>`).join('') || '<tr><td colspan="5">Belum ada data. Lakukan scan pertamamu.</td></tr>';
    const v = d.map(r => +r.bb_kg); if (v.length < 2) { $('#grafik').innerHTML = ''; return; }
    const lo = Math.min(...v) - 1, hi = Math.max(...v) + 1, x = i => 20 + i * 560 / (v.length - 1), y = n => 120 - (n - lo) / (hi - lo) * 100;
    $('#grafik').innerHTML = `<polyline fill="none" stroke="var(--fo)" stroke-width="3" points="${v.map((n, i) => x(i) + ',' + y(n)).join(' ')}"/>` + v.map((n, i) => `<circle cx="${x(i)}" cy="${y(n)}" r="4" fill="var(--kr)"/>`).join('');
  } catch { /* server belum menyala */ }
}

async function status() {
  const s = $('#srv');
  try { const j = await api('/api/status'); s.textContent = 'Server: tersambung' + (j.kalibrasi ? ' · terkalibrasi' : ''); s.className = 'pill'; }
  catch { s.textContent = 'Server: belum menyala'; s.className = 'pill off'; }
}

$('#scan').onclick = () => jalankan('scan'); $('#kal').onclick = () => jalankan('kal'); $('#man').onclick = () => jalankan('man');
startCamera(); status(); riwayat();