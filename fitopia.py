"""
FITOPIA - Scanner BB/TB dengan kamera (satu file)

Install : py -m pip install -U opencv-python mediapipe matplotlib numpy
Tes tanpa kamera : py fitopia.py --demo --tb 160 --bb 62 --usia 28
Pakai kamera     : py fitopia.py
Grafik progres   : py fitopia.py --riwayat

Tombol: SPASI = foto (timer lalu ukur) | T = ganti timer 3/5/10 dtk | C = kalibrasi
        F = layar penuh | Q = keluar        (semua input diketik di jendela kamera)

Cara kerja:
- Tinggi : panjang tubuh 3D dari landmark 'world' MediaPipe (tidak bergantung jarak ke kamera),
           lalu dikoreksi faktor kalibrasi.
- Berat  : ESTIMASI dari lebar siluet badan relatif tinggi, dikalibrasi sekali dengan timbangan.
           Ini perkiraan kasar, bukan hasil timbang. Hasil edukasi, bukan diagnosis medis.
"""
import argparse, csv, json, math, os, statistics, sys, time, urllib.request
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
from collections import deque
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyBboxPatch

FAKTOR = {"sedentari": 1.2, "ringan": 1.375, "sedang": 1.55, "aktif": 1.725}
CAL, LOG, WIN = "kalibrasi_v2.json", "riwayat_fitopia.csv", "Fitopia Scanner"
KOLOM = ["waktu", "tb_cm", "bb_kg", "bmi", "kategori", "tdee", "target_kkal", "sumber_bb"]
MODEL = "pose_landmarker_full.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_full/float16/latest/pose_landmarker_full.task")
OFFSET_KEPALA = 14.0   # cm: hidung -> puncak kepala + tumit -> lantai (dikoreksi lagi oleh kalibrasi)
KUNCI = (0, 11, 12, 23, 24, 25, 26, 27, 28, 29, 30)

# ---------------------------------------------------------------- analisis
def analisis(tb, bb, usia, akt):
    if not (100 <= tb <= 220 and 25 <= bb <= 200 and 15 <= usia <= 90):
        raise ValueError("Nilai di luar rentang wajar (TB 100-220, BB 25-200, usia 15-90).")
    if akt not in FAKTOR:
        raise ValueError(f"Aktivitas harus salah satu dari: {', '.join(FAKTOR)}")
    bmi = bb / (tb / 100) ** 2                      # ambang Asia-Pasifik
    kat = "Kurus" if bmi < 18.5 else "Normal" if bmi < 23 else "Overweight" if bmi < 25 else "Obesitas"
    bmr = 10 * bb + 6.25 * tb - 5 * usia - 161      # Mifflin-St Jeor (wanita)
    tdee = bmr * FAKTOR[akt]
    faktor = {"Kurus": 1.10, "Normal": 1.0}.get(kat, 0.85)
    return dict(bmi=bmi, kategori=kat, bmr=bmr, tdee=tdee, target=round(tdee * faktor / 10) * 10,
                ideal=(18.5 * (tb / 100) ** 2, 22.9 * (tb / 100) ** 2))

def olahraga(kat):
    return {"Kurus": ("Pilates Mat + Strength ringan", "3x/minggu, 40 menit"),
            "Normal": ("Yoga Flow + Pilates", "3x/minggu, 45 menit"),
            "Overweight": ("Zumba + Pilates Core", "4x/minggu, 40 menit"),
            "Obesitas": ("Jalan cepat + Yoga low-impact", "4x/minggu, 30 menit")}[kat]

def menu(target):
    m = [("Makan siang", .35, "Ayam panggang herbal, nasi merah, tumis buncis"),
         ("Makan malam", .30, "Salmon teriyaki, quinoa, brokoli kukus"),
         ("Snack kantor", .10, "Greek yogurt, almond, blueberry")]
    return [(n, round(target * p), d) for n, p, d in m]

# ------------------------------------------------------------- visualisasi
def visualisasi(tb, bb, usia, h, path, sumber="timbangan"):
    BG, INK, GREEN, KRAFT = "#F2F5F0", "#1F2B24", "#2E5E4A", "#B98B5E"
    fig = plt.figure(figsize=(12, 6.5), facecolor=BG)
    ax = fig.add_axes([0.03, 0.05, 0.3, 0.88]); ax.set_xlim(-1, 1); ax.set_ylim(0, 10); ax.axis("off")
    lebar = 0.55 + min(max(h["bmi"] - 18.5, 0), 12) * 0.03
    ax.add_patch(Ellipse((0, 8.6), 0.9, 1.1, color=KRAFT))
    ax.add_patch(FancyBboxPatch((-lebar, 4.6), lebar * 2, 3.5, boxstyle="round,pad=0.1", color=GREEN))
    ax.add_patch(FancyBboxPatch((-lebar * .8, 0.4), lebar * 1.6, 4.4, boxstyle="round,pad=0.1", color=GREEN, alpha=.85))
    for y in (1.5, 3.5, 5.5, 7.5, 9.0): ax.plot([-0.95, 0.95], [y, y], color=KRAFT, lw=1.2, ls="--")
    ax.annotate("", xy=(0.92, 9.6), xytext=(0.92, 0.3), arrowprops=dict(arrowstyle="<->", color=INK))
    ax.text(0.8, 5, f"TB {tb:.0f} cm", rotation=90, ha="right", va="center", color=INK, fontsize=11)
    ax.set_title("Scan tubuh", color=INK, loc="left", fontsize=13, fontweight="bold")

    a2 = fig.add_axes([0.40, 0.58, 0.56, 0.3], facecolor=BG)
    for a, b, c, n in [(14, 18.5, "#E6D58A", "Kurus"), (18.5, 23, "#8DB89F", "Normal"),
                       (23, 25, "#E6D58A", "Over"), (25, 34, "#D9958A", "Obesitas")]:
        a2.barh(0, b - a, left=a, color=c, height=.5); a2.text((a + b) / 2, -0.45, n, ha="center", fontsize=9, color=INK)
    x = min(max(h["bmi"], 14), 34)
    a2.axvline(x, color=INK, lw=3); a2.text(x, 0.45, f"BMI {h['bmi']:.1f}", ha="center", fontsize=12, fontweight="bold", color=INK)
    a2.set_xlim(14, 34); a2.set_ylim(-.7, .9); a2.axis("off")
    a2.set_title(f"BB {bb:.0f} kg ({sumber}) | {h['kategori']} | BB ideal {h['ideal'][0]:.0f}-{h['ideal'][1]:.0f} kg",
                 color=INK, loc="left", fontsize=12, fontweight="bold")

    a3 = fig.add_axes([0.40, 0.30, 0.56, 0.17], facecolor=BG)
    nilai = [h["bmr"], h["tdee"], h["target"]]
    a3.barh(["BMR", "TDEE", "Target menu"], nilai, color=[KRAFT, GREEN, "#5B2A4A"])
    for i, v in enumerate(nilai): a3.text(v + 20, i, f"{v:.0f} kkal", va="center", color=INK)
    a3.set_xlim(0, max(nilai) * 1.25); a3.invert_yaxis()
    for s in ("top", "right", "bottom"): a3.spines[s].set_visible(False)
    a3.tick_params(colors=INK, bottom=False, labelbottom=False)

    jenis, detail = olahraga(h["kategori"])
    teks = f"Olahraga: {jenis} ({detail})\n\n" + "\n".join(f"{n} ({k} kkal): {d}" for n, k, d in menu(h["target"]))
    fig.text(0.40, 0.04, teks, fontsize=10, color=INK, va="bottom", linespacing=1.6)
    fig.savefig(path, dpi=150, facecolor=BG); plt.close(fig)

def simpan(tb, bb, usia, akt, sumber="timbangan"):
    h = analisis(tb, bb, usia, akt)
    baru = not os.path.exists(LOG)
    with open(LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if baru: w.writerow(KOLOM)
        w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M"), round(tb, 1), round(bb, 1), round(h["bmi"], 1),
                    h["kategori"], round(h["tdee"]), h["target"], sumber])
    nama = f"hasil_{datetime.now():%Y%m%d_%H%M%S}.png"
    visualisasi(tb, bb, usia, h, nama, sumber)
    print(f"TB {tb:.1f} cm | BB {bb:.1f} kg ({sumber}) | BMI {h['bmi']:.1f} ({h['kategori']}) | "
          f"BMR {h['bmr']:.0f} | TDEE {h['tdee']:.0f} | Target menu {h['target']} kkal\nGambar tersimpan: {nama}")
    return nama

# ------------------------------------------------------------ pengukuran
def panjang_tubuh(w):
    """Panjang tubuh (cm) = jumlah segmen 3D hidung-bahu-pinggul-lutut-mata kaki-tumit (landmark world)."""
    def p(i): return (w[i].x, w[i].y, w[i].z)
    def mid(i, j): return tuple((a + b) / 2 for a, b in zip(p(i), p(j)))
    rangka = [p(0), mid(11, 12), mid(23, 24), mid(25, 26), mid(27, 28), mid(29, 30)]
    return sum(math.dist(a, b) for a, b in zip(rangka, rangka[1:])) * 100

def lebar_siluet(mask, cx, y1, y2):
    """Median lebar siluet badan (piksel) pada baris y1..y2, hanya bagian yang tersambung ke garis tengah badan."""
    m = mask > 0.5
    H, W = m.shape; hasil = []
    for y in range(max(y1, 0), min(y2, H), 4):
        row = m[y]
        if not (0 <= cx < W and row[cx]): continue
        l = r = cx
        while l > 0 and row[l - 1]: l -= 1
        while r < W - 1 and row[r + 1]: r += 1
        hasil.append(r - l + 1)
    return statistics.median(hasil) if len(hasil) >= 5 else None

def ukur(res, W, H):
    """Return (panjang_tubuh_cm, rasio_lebar) untuk satu frame, atau None jika tubuh belum utuh."""
    if not (res.pose_landmarks and res.pose_world_landmarks): return None
    lm = res.pose_landmarks[0]
    if not all(lm[i].visibility > .6 for i in KUNCI): return None
    span = panjang_tubuh(res.pose_world_landmarks[0]); a = None
    if res.segmentation_masks:
        mask = np.squeeze(res.segmentation_masks[0].numpy_view())
        sy, hy = (lm[11].y + lm[12].y) / 2 * H, (lm[23].y + lm[24].y) / 2 * H
        lebar = lebar_siluet(mask, int((lm[23].x + lm[24].x) / 2 * W), int(sy), int(hy + 0.5 * (hy - sy)))
        px = (max(lm[29].y, lm[30].y) - lm[0].y) * H
        if lebar and px > 0: a = lebar / px
    return span, a

def tinggi_cm(span, prof): return (span + OFFSET_KEPALA) * prof.get("k", 1.0)

def berat_estimasi(a, tb, prof):
    """BMI ~ (lebar/tinggi)^2, diskalakan dari satu titik kalibrasi (BB timbangan). None jika belum kalibrasi."""
    if not a or "a_ref" not in prof: return None
    bmi_ref = prof["bb_ref"] / (prof["tb_ref"] / 100) ** 2
    return min(max(bmi_ref * (a / prof["a_ref"]) ** 2 * (tb / 100) ** 2, 30), 150)

def muat_profil(): return json.load(open(CAL, encoding="utf-8")) if os.path.exists(CAL) else {}
def simpan_profil(p): json.dump(p, open(CAL, "w", encoding="utf-8"))

# ------------------------------------------------------------------ kamera
def siapkan_model():
    if not os.path.exists(MODEL):
        print("Mengunduh model pose (sekali saja, ~9 MB)...")
        try: urllib.request.urlretrieve(MODEL_URL, MODEL)
        except Exception as e:
            sys.exit(f"Gagal mengunduh: {e}\nUnduh manual dari:\n{MODEL_URL}\nsimpan sebagai {MODEL} di folder ini.")
    return open(MODEL, "rb").read()

def tulis(cv2, f, teks, xy, skala=.7, warna=(255, 255, 255), tebal=2):
    s = f.shape[0] / 480; org = (int(xy[0] * s), int(xy[1] * s))
    cv2.putText(f, teks, org, cv2.FONT_HERSHEY_SIMPLEX, skala * s, (0, 0, 0), int(tebal * s) + 3)
    cv2.putText(f, teks, org, cv2.FONT_HERSHEY_SIMPLEX, skala * s, warna, max(1, int(tebal * s)))

def input_jendela(cv2, frame, label, tipe):
    """Ketik angka di jendela kamera. Enter = OK, Backspace = hapus, Esc = batal."""
    H = frame.shape[0]; teks = ""
    while True:
        t = frame.copy()
        cv2.rectangle(t, (0, H // 2 - int(H * .12)), (frame.shape[1], H // 2 + int(H * .1)), (0, 0, 0), -1)
        tulis(cv2, t, label, (16, 225), .65)
        tulis(cv2, t, teks + "_", (16, 262), .9, (120, 220, 150))
        cv2.imshow(WIN, t)
        k = cv2.waitKey(50) & 0xFF
        if k == 27: return None
        if k == 13:
            try: return tipe(teks)
            except ValueError: teks = ""
        elif k == 8: teks = teks[:-1]
        elif k != 255 and chr(k) in "0123456789.": teks += chr(k)

def pilih_aktivitas(cv2, frame):
    n = input_jendela(cv2, frame, "Aktivitas: 1=duduk 2=ringan 3=sedang 4=aktif", int)
    return list(FAKTOR)[n - 1] if n and 1 <= n <= 4 else None

def layar_penuh(cv2, ui):
    ui["penuh"] = not ui["penuh"]
    cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN if ui["penuh"] else cv2.WINDOW_NORMAL)

def proses_sampel(cv2, frame, sampel, prof):
    span = statistics.median(x[0] for x in sampel)
    al = [x[1] for x in sampel if x[1]]
    a = statistics.median(al) if len(al) >= 5 else None
    tb = tinggi_cm(span, prof); bb = berat_estimasi(a, tb, prof); sumber = "estimasi kamera"
    if bb is None:
        bb = input_jendela(cv2, frame, "BB belum bisa diestimasi. Isi dari timbangan (kg):", float); sumber = "timbangan"
        if not bb: return None
    if not prof.get("usia"):
        u = input_jendela(cv2, frame, "Usia (tahun), lalu Enter:", int)
        if not u: return None
        prof["usia"] = u
    if not prof.get("akt"):
        prof["akt"] = pilih_aktivitas(cv2, frame)
        if not prof["akt"]: return None
    simpan_profil(prof)
    return tb, bb, prof["usia"], prof["akt"], sumber

def tampil_hasil(cv2, png, W, H, ui):
    img = cv2.imread(png)
    sc = min(W / img.shape[1], H / img.shape[0])
    img = cv2.resize(img, (int(img.shape[1] * sc), int(img.shape[0] * sc)))
    kv = np.full((H, W, 3), 245, np.uint8)
    y, x = (H - img.shape[0]) // 2, (W - img.shape[1]) // 2
    kv[y:y + img.shape[0], x:x + img.shape[1]] = img
    tulis(cv2, kv, "SPASI = kembali ke kamera | Q = keluar", (16, 470), .6, (60, 60, 60))
    while True:
        cv2.imshow(WIN, kv); k = cv2.waitKey(50) & 0xFF
        if k in (32, 13, 27): return False
        if k == ord("q"): return True
        if k == ord("f"): layar_penuh(cv2, ui)

def kamera(idx):
    import cv2, mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    GARIS = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
             (23, 25), (25, 27), (24, 26), (26, 28), (27, 29), (28, 30), (29, 31), (30, 32)]
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(idx)
    if not cap.isOpened(): sys.exit("Kamera tidak bisa dibuka. Tutup aplikasi lain yang memakai kamera, atau coba --kamera 1.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    opsi = vision.PoseLandmarkerOptions(base_options=BaseOptions(model_asset_buffer=siapkan_model()),
                                        running_mode=vision.RunningMode.VIDEO, output_segmentation_masks=True,
                                        min_pose_detection_confidence=.6, min_tracking_confidence=.6)
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO); cv2.resizeWindow(WIN, 1100, 700)
    ui, prof, timer = {"penuh": False}, muat_profil(), 5
    bs, ba = deque(maxlen=30), deque(maxlen=30)
    state, t0, sampel, catatan = "live", 0, [], ("", 0)
    mulai, ts_lalu = time.time(), -1
    print("Kamera aktif. Berdiri tegak 2-3 m dari kamera, seluruh tubuh terlihat.")

    with vision.PoseLandmarker.create_from_options(opsi) as pose:
        while True:
            ok, frame = cap.read()
            if not ok: break
            frame = cv2.flip(frame, 1); mentah = frame.copy(); H, W = frame.shape[:2]
            ts = max(int((time.time() - mulai) * 1000), ts_lalu + 1); ts_lalu = ts
            res = pose.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                                                 data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), ts)
            now, u = time.time(), ukur(res, W, H)
            if res.pose_landmarks:
                lm = res.pose_landmarks[0]
                for a, b in GARIS:
                    if lm[a].visibility > .5 and lm[b].visibility > .5:
                        cv2.line(frame, (int(lm[a].x * W), int(lm[a].y * H)), (int(lm[b].x * W), int(lm[b].y * H)), (80, 200, 120), 2)
            if u:
                bs.append(u[0])
                if u[1]: ba.append(u[1])
            else: bs.clear(); ba.clear()

            span = statistics.median(bs) if bs else None
            stabil = bool(span) and len(bs) == bs.maxlen and statistics.pstdev(bs) < 0.015 * span
            tb = tinggi_cm(span, prof) if span else None
            a_med = statistics.median(ba) if len(ba) >= 10 else None
            bb = berat_estimasi(a_med, tb, prof) if (tb and a_med) else None

            tulis(cv2, frame, ("Stabil - siap" if stabil else "Tahan posisi...") if u else "Tubuh belum terlihat utuh",
                  (16, 30), .8, (120, 220, 150))
            if tb: tulis(cv2, frame, f"TB ~ {tb:.1f} cm" + ("" if "k" in prof else " (belum kalibrasi)"), (16, 64), .8)
            if bb: tulis(cv2, frame, f"BB ~ {bb:.1f} kg (estimasi)", (16, 96), .8)
            tulis(cv2, frame, f"SPASI=foto  T=timer {timer}s  C=kalibrasi  F=layar penuh  Q=keluar", (16, 470), .5)
            if now < catatan[1]: tulis(cv2, frame, catatan[0], (16, 440), .65, (0, 200, 255))

            if state == "countdown":
                sisa = timer - (now - t0)
                if sisa <= 0: state, t0, sampel = "capture", now, []
                else:
                    s = H / 480; angka = str(int(math.ceil(sisa)))
                    (tw, th), _ = cv2.getTextSize(angka, cv2.FONT_HERSHEY_SIMPLEX, 7 * s, int(12 * s))
                    pos = ((W - tw) // 2, (H + th) // 2)
                    cv2.putText(frame, angka, pos, cv2.FONT_HERSHEY_SIMPLEX, 7 * s, (0, 0, 0), int(12 * s) + 6)
                    cv2.putText(frame, angka, pos, cv2.FONT_HERSHEY_SIMPLEX, 7 * s, (255, 255, 255), int(12 * s))
                    tulis(cv2, frame, "Berdiri tegak, tangan sedikit menjauh dari badan", (16, 410), .65)
            elif state == "capture":
                if u: sampel.append(u)
                if now - t0 < .15:
                    al = .7 * (1 - (now - t0) / .15)
                    frame = cv2.addWeighted(frame, 1 - al, np.full_like(frame, 255), al, 0)
                tulis(cv2, frame, "Mengukur... jangan bergerak", (16, 410), .8, (120, 220, 150))
                if now - t0 >= 1.5:
                    state = "live"
                    if len(sampel) < 8:
                        catatan = ("Gagal: tubuh tidak terlihat utuh, ulangi", now + 4)
                    else:
                        hasil = proses_sampel(cv2, mentah, sampel, prof)
                        if hasil:
                            cv2.imwrite(f"foto_{datetime.now():%Y%m%d_%H%M%S}.jpg", mentah)
                            try:
                                png = simpan(*hasil)
                                if tampil_hasil(cv2, png, W, H, ui): break
                            except ValueError as e: catatan = (f"Gagal: {e}", time.time() + 5)
                        bs.clear(); ba.clear()

            cv2.imshow(WIN, frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"): break
            elif k == ord("f"): layar_penuh(cv2, ui)
            elif k == ord("t"): timer = {3: 5, 5: 10, 10: 3}[timer]
            elif k == 32:
                if state != "live": state = "live"
                elif u: state, t0 = "countdown", now
                else: catatan = ("Tubuh belum terlihat utuh", now + 3)
            elif k == ord("c") and stabil:
                asli = input_jendela(cv2, mentah, "Tinggi badan asli Anda (cm), lalu Enter:", float)
                berat = input_jendela(cv2, mentah, "Berat badan asli dari timbangan (kg):", float) if asli else None
                usia = input_jendela(cv2, mentah, "Usia (tahun):", int) if berat else None
                akt = pilih_aktivitas(cv2, mentah) if usia else None
                if akt and 100 <= asli <= 220 and 25 <= berat <= 200:
                    prof.update(k=asli / (span + OFFSET_KEPALA), tb_ref=asli, bb_ref=berat, usia=usia, akt=akt)
                    if a_med: prof["a_ref"] = a_med
                    simpan_profil(prof); catatan = ("Kalibrasi tersimpan", time.time() + 3)
                bs.clear(); ba.clear()
    cap.release(); cv2.destroyAllWindows()

# ----------------------------------------------------------------- riwayat
def riwayat():
    if not os.path.exists(LOG): sys.exit("Belum ada data. Ambil foto dulu (tombol SPASI).")
    baris = list(csv.DictReader(open(LOG, encoding="utf-8")))
    t = [b["waktu"] for b in baris]
    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor="#F2F5F0")
    ax.plot(t, [float(b["bb_kg"]) for b in baris], marker="o", color="#2E5E4A")
    ax2 = ax.twinx(); ax2.plot(t, [float(b["bmi"]) for b in baris], marker="s", color="#B98B5E")
    ax.set_ylabel("BB (kg)"); ax2.set_ylabel("BMI"); ax.set_title("Progres Fitopia")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right"); fig.tight_layout(); fig.savefig("riwayat_progress.png", dpi=150)
    for b in baris: print(b["waktu"], f"TB {b['tb_cm']} | BB {b['bb_kg']} | BMI {b['bmi']} {b['kategori']}")
    if len(baris) > 1:
        print(f"\nPerubahan BB sejak awal: {float(baris[-1]['bb_kg']) - float(baris[0]['bb_kg']):+.1f} kg. Grafik: riwayat_progress.png")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Fitopia scanner BB/TB")
    p.add_argument("--demo", action="store_true", help="tanpa kamera, input manual")
    p.add_argument("--riwayat", action="store_true")
    p.add_argument("--kamera", type=int, default=0)
    p.add_argument("--tb", type=float, default=160); p.add_argument("--bb", type=float, default=62)
    p.add_argument("--usia", type=int, default=28); p.add_argument("--aktivitas", choices=FAKTOR, default="ringan")
    a = p.parse_args()
    if a.riwayat: riwayat()
    elif a.demo: simpan(a.tb, a.bb, a.usia, a.aktivitas)
    else: kamera(a.kamera)