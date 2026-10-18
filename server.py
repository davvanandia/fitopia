"""Server Fitopia: menyambungkan website (web/) ke fitopia.py.
Jalankan: python server.py  ->  buka http://127.0.0.1:5000"""
import base64, csv, os, statistics, threading
from datetime import datetime
import numpy as np
from flask import Flask, jsonify, request
import fitopia as f

app = Flask(__name__, static_folder="web", static_url_path="")
_lm, _kunci = None, threading.Lock()

def landmarker():
    global _lm
    if _lm is None:
        from mediapipe.tasks.python import BaseOptions, vision
        _lm = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=f.siapkan_model()),
            running_mode=vision.RunningMode.IMAGE, output_segmentation_masks=True,
            min_pose_detection_confidence=.6))
    return _lm

def ukur_frames(frames):
    """Ukur banyak frame dari browser -> (panjang_tubuh_cm, rasio_lebar, jumlah) atau None."""
    import cv2, mediapipe as mp
    hasil = []
    with _kunci:
        lm = landmarker()
        for d in frames:
            img = cv2.imdecode(np.frombuffer(base64.b64decode(d.split(",")[-1]), np.uint8), cv2.IMREAD_COLOR)
            if img is None: continue
            H, W = img.shape[:2]
            res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
            u = f.ukur(res, W, H)
            if u: hasil.append(u)
    if len(hasil) < 5: return None
    al = [x[1] for x in hasil if x[1]]
    return statistics.median(x[0] for x in hasil), (statistics.median(al) if len(al) >= 5 else None), len(hasil)

def paket(tb, bb, usia, akt):
    h = f.analisis(tb, bb, usia, akt); jenis, detail = f.olahraga(h["kategori"])
    return dict(tb=round(tb, 1), bb=round(bb, 1), bmi=round(h["bmi"], 1), kategori=h["kategori"],
                bmr=round(h["bmr"]), tdee=round(h["tdee"]), target=h["target"], ideal=[round(x) for x in h["ideal"]],
                olahraga=dict(jenis=jenis, detail=detail),
                menu=[dict(nama=n, kkal=k, desk=d) for n, k, d in f.menu(h["target"])])

def catat(p, sumber):
    baru = not os.path.exists(f.LOG)
    with open(f.LOG, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if baru: w.writerow(f.KOLOM)
        w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M"), p["tb"], p["bb"], p["bmi"], p["kategori"],
                    p["tdee"], p["target"], sumber])

@app.errorhandler(Exception)
def galat(e):
    kode = 400 if isinstance(e, (ValueError, KeyError, TypeError)) else 500
    return jsonify(error=str(e)), kode

@app.get("/")
def index(): return app.send_static_file("index.html")

@app.get("/api/status")
def status(): return jsonify(ok=True, kalibrasi="k" in f.muat_profil())

@app.post("/api/analisis")                      # input manual TB/BB
def analisis_manual():
    d = request.json; p = paket(float(d["tb"]), float(d["bb"]), int(d["usia"]), d["aktivitas"])
    catat(p, "manual"); return jsonify(p)

@app.post("/api/scan")                          # frame kamera -> TB, estimasi BB, analisis
def scan():
    d, prof = request.json, f.muat_profil()
    r = ukur_frames(d["frames"])
    if not r: return jsonify(error="Tubuh belum terlihat utuh. Mundur sampai kepala hingga kaki masuk bingkai."), 422
    tb = f.tinggi_cm(r[0], prof); est = f.berat_estimasi(r[1], tb, prof)
    bb = est or (float(d["bb"]) if d.get("bb") else None)
    if bb is None: return jsonify(perlu_bb=True, tb=round(tb, 1))
    usia = int(d.get("usia") or prof.get("usia") or 0); akt = d.get("aktivitas") or prof.get("akt")
    p = paket(tb, bb, usia, akt); sumber = "estimasi kamera" if est else "timbangan"
    catat(p, sumber); return jsonify({**p, "sumber": sumber})

@app.post("/api/kalibrasi")
def kalibrasi():
    d, prof = request.json, f.muat_profil()
    asli, berat = float(d["tb"]), float(d["bb"])
    f.analisis(asli, berat, int(d["usia"]), d["aktivitas"])         # validasi rentang
    r = ukur_frames(d["frames"])
    if not r: return jsonify(error="Tubuh belum terlihat utuh. Ulangi kalibrasi."), 422
    prof.update(k=asli / (r[0] + f.OFFSET_KEPALA), tb_ref=asli, bb_ref=berat, usia=int(d["usia"]), akt=d["aktivitas"])
    if r[1]: prof["a_ref"] = r[1]
    f.simpan_profil(prof); return jsonify(ok=True, estimasi_bb="a_ref" in prof)

@app.get("/api/riwayat")
def riwayat():
    if not os.path.exists(f.LOG): return jsonify([])
    return jsonify(list(csv.DictReader(open(f.LOG, encoding="utf-8"))))

if __name__ == "__main__":
    print("Fitopia berjalan di http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, threaded=True)