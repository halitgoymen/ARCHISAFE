"""
╔══════════════════════════════════════════════════════════════════════╗
║         ARCHISAFE — 5. Hafta Canlı Kamera Tespiti (GELİŞMİŞ)       ║
║         Kullanıcıyı HER ZAMAN kutu içinde izler                     ║
╠══════════════════════════════════════════════════════════════════════╣
║  YENİ ÖZELLİKLER (Hafta-4'e göre):                                 ║
║    ✅ Kişi kutusu HER ZAMAN görünür (kaybolsa bile tahmin devam)    ║
║    ✅ Eksik KKD listesi ekranda gösterilir                          ║
║    ✅ Sınıf bazında doğruluk/güven oranları gösterilir             ║
║    ✅ Maske/gözlük/yelek tanıma sorunları düzeltildi               ║
║    ✅ Alarm sistemi geliştirildi                                     ║
║    ✅ Kişi takibi (ByteTrack benzeri basit IoU tracker)             ║
╠══════════════════════════════════════════════════════════════════════╣
║  Kullanım:                                                           ║
║    python test_live_v5.py                    # Varsayılan kamera 0  ║
║    python test_live_v5.py --camera 1         # 2. kamera             ║
║    python test_live_v5.py --conf 0.3         # Güven eşiği           ║
║    python test_live_v5.py --record           # Video kaydet          ║
║    python test_live_v5.py --model best.pt    # Manuel model          ║
║                                                                      ║
║  Kontroller (pencere açıkken):                                       ║
║    Q / ESC  → Çıkış                                                  ║
║    S        → Anlık ekran görüntüsü                                  ║
║    R        → Kayıt başlat/durdur                                    ║
║    +/-      → Güven eşiği artır/azalt                               ║
║    H        → HUD aç/kapat                                          ║
║    P        → Duraklat/Devam                                         ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import argparse
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ══════════════════════════════════════════════════════
#                    KONFİGÜRASYON
# ══════════════════════════════════════════════════════

_SCRIPT_DIR   = Path(__file__).resolve().parent
_HAFTA5_DIR   = _SCRIPT_DIR.parent

# Hafta-5 eğitim çıktısı (train_v5.py tarafından oluşturulur)
DEFAULT_MODEL = _HAFTA5_DIR / "results" / "ARCHISAFE_v5" / "yolo26n_200ep" / "weights" / "best.pt"
OUTPUT_DIR    = _HAFTA5_DIR / "results" / "live"

DEFAULT_CONF  = 0.30    # Hafta-4'te 0.35'ti, 0.30'a düşürüldü (daha fazla tespit)
DEFAULT_IOU   = 0.45
CAMERA_INDEX  = 0
FRAME_WIDTH   = 1280
FRAME_HEIGHT  = 720
FPS_SMOOTH    = 30

# Kişi kutusunun kaybolması durumunda kaç frame bekle
PERSON_BOX_HOLD_FRAMES = 10   # Bu kadar frame sonra kutu kaybolur

# ══════════════════════════════════════════════════════
#              SINIF TANIMLARI
# ══════════════════════════════════════════════════════

# Güvenli KKD sınıfları (var olması GEREKENler)
REQUIRED_PPE = {
    "hardhat":        "Baret",
    "vest":           "Yelek",
    "mask":           "Maske",
    "gloves":         "Eldiven",
    "Ear-protection": "Kulak Koruyucu",
    "shoes":          "Bota/Ayakkabı",
}

# İhlal sınıfları (var olmaması GEREKENler)
VIOLATION_CLASSES = {
    "unhardhat":    "Baretsiz ⚠",
    "unvest":       "Yeleksiz ⚠",
    "unmask":       "Maskesiz ⚠",
    "ungloves":     "Eldivensiz ⚠",
    "no_arm_sleeve":"Kolsuz ⚠",
    "Fall Detection - v4 resized640_aug3x-ACCURATE": "DÜŞME! 🚨",
    "Fall Detection": "DÜŞME! 🚨",
}

# Tüm alarm verilen sınıflar
ALARM_CLASSES = set(VIOLATION_CLASSES.keys())

# Kutu renkleri (BGR)
CLASS_COLORS = {
    # Güvenli (yeşil tonları)
    "hardhat":        (30,  200,  60),
    "vest":           (30,  200, 200),
    "mask":           (200,  50, 200),
    "gloves":         (255, 165,  30),
    "Ear-protection": (200, 200,  30),
    "shoes":          (100, 160,  80),
    # İhlaller (kırmızı/turuncu tonları)
    "unhardhat":      (0,    0,  255),
    "unvest":         (0,   80,  255),
    "unmask":         (80,   0,  200),
    "ungloves":       (0,  100,  230),
    "no_arm_sleeve":  (100, 50,  200),
    "Fall Detection - v4 resized640_aug3x-ACCURATE": (0, 0, 255),
    "Fall Detection": (0,   0,  255),
    # Kişi — mavi
    "person":         (255, 100,   0),
}
DEFAULT_COLOR = (160, 160, 160)

# ══════════════════════════════════════════════════════
#              KİŞİ TAKİBİ (Persistent Box)
# ══════════════════════════════════════════════════════

class PersonTracker:
    """
    Kamerada kişiyi takip eder. Kişi geçici olarak kaybolsa bile
    (örn. model düşük güvenle tespit edemedi) son bilinen konumunu
    tutar ve kutuyu PERSON_BOX_HOLD_FRAMES kadar göstermeye devam eder.
    """
    def __init__(self, hold_frames: int = PERSON_BOX_HOLD_FRAMES):
        self.hold_frames   = hold_frames
        self.persons       = {}   # track_id → {box, miss_count, last_conf}
        self._next_id      = 0

    def _iou(self, a, b):
        """İki kutu arasındaki IoU hesapla."""
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - inter
        return inter / (union + 1e-6)

    def update(self, detected_boxes: list) -> list:
        """
        detected_boxes: [(x1,y1,x2,y2,conf), ...]
        Returns: [(x1,y1,x2,y2,conf,track_id,is_real), ...]
        """
        # Eşleme: greedy IoU matching
        matched_ids = set()
        used_dets   = set()

        for tid, info in self.persons.items():
            best_iou, best_di = 0.0, -1
            for di, det in enumerate(detected_boxes):
                if di in used_dets:
                    continue
                iou = self._iou(info["box"], det[:4])
                if iou > best_iou:
                    best_iou, best_di = iou, di

            if best_iou > 0.3 and best_di >= 0:
                det = detected_boxes[best_di]
                info["box"]       = det[:4]
                info["last_conf"] = det[4]
                info["miss_count"] = 0
                matched_ids.add(tid)
                used_dets.add(best_di)

        # Yeni kişiler ekle
        for di, det in enumerate(detected_boxes):
            if di not in used_dets:
                self.persons[self._next_id] = {
                    "box": det[:4], "last_conf": det[4], "miss_count": 0
                }
                self._next_id += 1

        # Miss count artır veya sil
        to_delete = []
        for tid in list(self.persons.keys()):
            if tid not in matched_ids:
                self.persons[tid]["miss_count"] += 1
                if self.persons[tid]["miss_count"] > self.hold_frames:
                    to_delete.append(tid)
        for tid in to_delete:
            del self.persons[tid]

        # Sonuçları döndür
        output = []
        for tid, info in self.persons.items():
            is_real = info["miss_count"] == 0  # False = tahmin (hold durumu)
            output.append((*info["box"], info["last_conf"], tid, is_real))
        return output

# ══════════════════════════════════════════════════════
#              ÇİZİM YARDIMCILARI
# ══════════════════════════════════════════════════════

def draw_rounded_rect(img, pt1, pt2, color, radius=8, thickness=2):
    """Köşeleri yuvarlak dikdörtgen çiz."""
    x1, y1 = pt1
    x2, y2 = pt2
    r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)

    if thickness < 0:
        # Dolu
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        for cx, cy in [(x1+r, y1+r), (x2-r, y1+r), (x1+r, y2-r), (x2-r, y2-r)]:
            cv2.circle(img, (cx, cy), r, color, -1)
    else:
        cv2.line(img, (x1+r, y1), (x2-r, y1), color, thickness)
        cv2.line(img, (x1+r, y2), (x2-r, y2), color, thickness)
        cv2.line(img, (x1, y1+r), (x1, y2-r), color, thickness)
        cv2.line(img, (x2, y1+r), (x2, y2-r), color, thickness)
        for angles, cx, cy in [
            (180, x1+r, y1+r), (270, x2-r, y1+r),
            (90,  x1+r, y2-r), (0,   x2-r, y2-r),
        ]:
            cv2.ellipse(img, (cx, cy), (r, r), angles, 0, 90, color, thickness)


def put_text_bg(img, text, pos, font_scale=0.55, color=(255,255,255),
                bg_color=(0,0,0), thickness=1, alpha=0.7, padding=4):
    """Arka planlı metin yaz."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), bl = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    # Arka plan
    overlay = img.copy()
    cv2.rectangle(overlay,
                  (x - padding, y - th - padding),
                  (x + tw + padding, y + bl + padding),
                  bg_color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


# ══════════════════════════════════════════════════════
#              ANA ÇIZIM FONKSİYONU
# ══════════════════════════════════════════════════════

def draw_frame(frame, results, model_names, fps, conf_thr, is_recording,
               tracker: PersonTracker, show_hud=True):
    """
    Frame üzerine tüm tespitleri, persistent kişi kutusunu,
    eksik KKD listesini ve sınıf güven oranlarını çizer.
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # ─── Tespitleri ayır ───────────────────────────────
    person_boxes   = []   # (x1,y1,x2,y2,conf)
    ppe_detections = {}   # label → conf
    violations     = {}   # label → conf
    has_alarm      = False

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf  = float(box.conf[0])
        label = model_names[int(box.cls[0])]

        if label == "person":
            person_boxes.append((x1, y1, x2, y2, conf))
            continue

        color = CLASS_COLORS.get(label, DEFAULT_COLOR)

        if label in ALARM_CLASSES:
            has_alarm = True
            violations[label] = max(violations.get(label, 0), conf)
        elif label in REQUIRED_PPE:
            ppe_detections[label] = max(ppe_detections.get(label, 0), conf)

        # Nesne kutusu
        draw_rounded_rect(overlay, (x1, y1), (x2, y2), color, radius=6, thickness=2)

        # Etiket
        disp = label.split(" ")[0] if len(label) > 15 else label
        text = f"{disp}  {conf:.0%}"
        (tw, th_), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        label_y = max(y1 - 2, th_ + 4)
        draw_rounded_rect(overlay, (x1, label_y - th_ - 4), (x1 + tw + 6, label_y + 2),
                          color, radius=4, thickness=-1)
        cv2.putText(overlay, text, (x1 + 3, label_y - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    # ─── Kişi takibi — Persistent Box ─────────────────
    tracked = tracker.update(person_boxes)

    for (x1, y1, x2, y2, conf, tid, is_real) in tracked:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        if is_real:
            # Gerçek tespit — mavi solid kutu
            box_color = (255, 120, 30)
            line_th   = 2
        else:
            # Tahmin (hold) — kesik çizgili görünüm
            box_color = (180, 80,  20)
            line_th   = 1
            # Köşelerde L şekli
            ln = 20
            for (sx, sy, dx, dy) in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
                cv2.line(overlay, (sx, sy), (sx + dx*ln, sy), box_color, 2)
                cv2.line(overlay, (sx, sy), (sx, sy + dy*ln), box_color, 2)

        draw_rounded_rect(overlay, (x1, y1), (x2, y2), box_color,
                          radius=8, thickness=line_th)

        # Kişi ID ve güven
        hold_label = "" if is_real else " [TAHM.]"
        p_text = f"Kisi #{tid}  {conf:.0%}{hold_label}"
        put_text_bg(overlay, p_text, (x1, max(y1 - 8, 16)),
                    bg_color=box_color, color=(255, 255, 255), font_scale=0.5)

        # ─── Eksik KKD Listesi (kişi kutusunun sağına) ─
        missing = []
        for cls_key, cls_tr in REQUIRED_PPE.items():
            if cls_key not in ppe_detections:
                missing.append(cls_tr)

        if missing:
            mx = x2 + 8
            my = y1
            put_text_bg(overlay, "✗ Eksik KKD:", (mx, my + 18),
                        bg_color=(20, 20, 100), color=(100, 180, 255),
                        font_scale=0.48, padding=3)
            for i, m in enumerate(missing):
                put_text_bg(overlay, f"  • {m}", (mx, my + 40 + i * 20),
                            bg_color=(20, 20, 80), color=(0, 140, 255),
                            font_scale=0.44, padding=2)

    # ─── Alarm overlay ────────────────────────────────
    if has_alarm:
        alarm_layer = overlay.copy()
        cv2.rectangle(alarm_layer, (0, 0), (w, h), (0, 0, 160), -1)
        cv2.addWeighted(alarm_layer, 0.15, overlay, 0.85, 0, overlay)
        # Alarm banner
        alarm_text = " ⚠  ALARM: " + " | ".join(
            f"{VIOLATION_CLASSES.get(k, k)} {v:.0%}"
            for k, v in violations.items()
        )
        put_text_bg(overlay, alarm_text, (10, h - 18),
                    font_scale=0.7, bg_color=(0, 0, 200),
                    color=(255, 220, 0), thickness=2, alpha=0.85)

    # ─── HUD (Üst bar) ────────────────────────────────
    if show_hud:
        cv2.rectangle(overlay, (0, 0), (w, 52), (15, 15, 15), -1)
        cv2.line(overlay, (0, 52), (w, 52), (60, 60, 60), 1)

        # FPS
        fps_color = (0, 220, 80) if fps >= 20 else (0, 180, 255) if fps >= 10 else (0, 80, 255)
        cv2.putText(overlay, f"FPS: {fps:.1f}", (10, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, fps_color, 2, cv2.LINE_AA)

        # Tespit özeti
        n_total = len(results[0].boxes)
        n_ppe   = len(ppe_detections)
        n_viol  = len(violations)
        cv2.putText(overlay, f"Tespit: {n_total}", (140, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(overlay, f"KKD: {n_ppe}", (280, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 220, 80), 2, cv2.LINE_AA)
        cv2.putText(overlay, f"İhlal: {n_viol}", (380, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 60, 255) if n_viol > 0 else (120, 120, 120), 2, cv2.LINE_AA)

        # Conf eşiği
        cv2.putText(overlay, f"Conf: {conf_thr:.2f}", (490, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (180, 180, 180), 1, cv2.LINE_AA)

        # Saat
        ts = datetime.now().strftime("%H:%M:%S")
        cv2.putText(overlay, ts, (w - 100, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1, cv2.LINE_AA)

        # Kayıt
        if is_recording:
            cv2.circle(overlay, (w - 120, 26), 9, (0, 0, 255), -1)
            cv2.putText(overlay, "REC", (w - 160, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)

        # ─── Sağda Güven Oranı Paneli ──────────────────
        # Tüm tespit edilen sınıfların güven oranlarını göster
        panel_x    = w - 230
        panel_y    = 65
        panel_items = []
        for label, conf_val in ppe_detections.items():
            panel_items.append((label, conf_val, (50, 200, 80)))
        for label, conf_val in violations.items():
            disp = VIOLATION_CLASSES.get(label, label)
            panel_items.append((disp, conf_val, (0, 80, 255)))

        if panel_items:
            # Arka plan
            ph = 20 + len(panel_items) * 22
            bg = overlay.copy()
            cv2.rectangle(bg, (panel_x - 6, panel_y - 14),
                          (w - 6, panel_y + ph), (15, 15, 15), -1)
            cv2.addWeighted(bg, 0.75, overlay, 0.25, 0, overlay)
            cv2.putText(overlay, "Güven Oranları:", (panel_x, panel_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)
            for i, (lbl, c, col) in enumerate(panel_items):
                y_ = panel_y + 22 + i * 22
                # Bar arka plan
                bar_w = 200
                cv2.rectangle(overlay, (panel_x, y_ - 12),
                              (panel_x + bar_w, y_ + 4), (40, 40, 40), -1)
                # Bar dolgu
                fill_w = int(bar_w * c)
                bar_color = col if c >= 0.5 else (0, 120, 255)
                cv2.rectangle(overlay, (panel_x, y_ - 12),
                              (panel_x + fill_w, y_ + 4), bar_color, -1)
                # Etiket + yüzde
                short = (lbl[:14] + "…") if len(lbl) > 15 else lbl
                cv2.putText(overlay, f"{short} {c:.0%}", (panel_x + 4, y_),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

    return overlay, has_alarm


# ══════════════════════════════════════════════════════
#                    ANA DÖNGÜ
# ══════════════════════════════════════════════════════

def run_live(model_path: Path, camera_index: int, conf: float, iou: float, auto_record: bool):
    print("\n" + "═" * 65)
    print("      ARCHISAFE — 5. Hafta Canlı Tespit (Gelişmiş)")
    print("═" * 65)

    # Model yolu kontrol
    if not model_path.exists():
        print(f"\n❌ Model bulunamadı: {model_path}")
        print("   Lütfen önce train_v5.py scriptini çalıştır:")
        print("   cd Hafta-5/scripts && python train_v5.py")
        return

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"\n🤖 Model  : {model_path}")
    print(f"📷 Kamera : {camera_index}")
    print(f"⚙️  Conf   : {conf}  (daha düşük = daha fazla tespit)")
    print(f"💻 Device : {'GPU 🚀' if device == 0 else 'CPU (yavaş)'}")
    print(f"\nKontroller: Q/ESC=Çıkış | S=Screenshot | R=Kayıt | +/-=Conf | H=HUD | P=Duraklat")

    # Model yükle
    model = YOLO(str(model_path))
    print(f"✅ Model yüklendi ({len(model.names)} sınıf)")
    for i, name in model.names.items():
        print(f"   [{i:2d}] {name}")

    # Kamera aç
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # Gecikmeyi azalt

    if not cap.isOpened():
        print(f"❌ Kamera {camera_index} açılamadı!")
        return

    # FPS hesaplama
    fps_deque = deque(maxlen=FPS_SMOOTH)
    prev_time = time.time()

    # Video kaydı
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    writer      = None
    is_recording = auto_record
    if auto_record:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        vp = OUTPUT_DIR / f"live_v5_{ts}.mp4"
        writer = cv2.VideoWriter(str(vp), cv2.VideoWriter_fourcc(*"mp4v"),
                                 20, (FRAME_WIDTH, FRAME_HEIGHT))
        print(f"🎥 Kayıt: {vp}")

    tracker          = PersonTracker(hold_frames=PERSON_BOX_HOLD_FRAMES)
    screenshot_count = 0
    conf_threshold   = conf
    show_hud         = True
    paused           = False

    print("\n" + "═" * 65)
    print("Kamera başlatılıyor...")

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("⚠️  Frame okunamadı, bağlantı kesiliyor.")
                    break

                # FPS
                cur_time = time.time()
                fps_deque.append(1.0 / (cur_time - prev_time + 1e-9))
                prev_time = cur_time
                fps = sum(fps_deque) / len(fps_deque)

                # Çıkarım
                results = model.predict(
                    source      = frame,
                    conf        = conf_threshold,
                    iou         = iou,
                    device      = device,
                    stream      = False,
                    verbose     = False,
                    max_det     = 50,
                )

                # Çiz
                annotated, has_alarm = draw_frame(
                    frame, results, model.names, fps,
                    conf_threshold, is_recording, tracker, show_hud
                )

                if is_recording and writer is not None:
                    writer.write(annotated)

                cv2.imshow("ARCHISAFE v5 - Canli Tespit", annotated)
                last_annotated = annotated
            else:
                # Duraklat ekranı
                pause_frame = last_annotated.copy()
                put_text_bg(pause_frame, "  ⏸  DURAKLATILDI  ",
                            (FRAME_WIDTH // 2 - 100, FRAME_HEIGHT // 2),
                            font_scale=1.0, bg_color=(20, 20, 80),
                            color=(255, 255, 0), thickness=2, alpha=0.85)
                cv2.imshow("ARCHISAFE v5 - Canli Tespit", pause_frame)

            # Klavye kontrolleri
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):     # Çıkış
                break
            elif key == ord("s"):          # Screenshot
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                sp = OUTPUT_DIR / f"screenshot_v5_{ts}.jpg"
                cv2.imwrite(str(sp), last_annotated)
                screenshot_count += 1
                print(f"📸 Screenshot: {sp}")
            elif key == ord("r"):          # Kayıt toggle
                is_recording = not is_recording
                if is_recording:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    vp = OUTPUT_DIR / f"live_v5_{ts}.mp4"
                    writer = cv2.VideoWriter(str(vp), cv2.VideoWriter_fourcc(*"mp4v"),
                                            20, (FRAME_WIDTH, FRAME_HEIGHT))
                    print(f"🎥 Kayıt başladı: {vp}")
                else:
                    if writer:
                        writer.release()
                        writer = None
                    print("⏹  Kayıt durduruldu.")
            elif key == ord("+") or key == ord("="):
                conf_threshold = min(0.95, conf_threshold + 0.05)
                print(f"⬆️  Conf: {conf_threshold:.2f}")
            elif key == ord("-"):
                conf_threshold = max(0.05, conf_threshold - 0.05)
                print(f"⬇️  Conf: {conf_threshold:.2f}")
            elif key == ord("h"):          # HUD toggle
                show_hud = not show_hud
                print(f"HUD: {'açık' if show_hud else 'kapalı'}")
            elif key == ord("p"):          # Duraklat / Devam
                paused = not paused
                if not paused:
                    last_annotated = frame.copy()  # reset
                print(f"{'⏸  Duraklatıldı' if paused else '▶  Devam'}")

    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print(f"\n✅ Kapatıldı. Toplam screenshot: {screenshot_count}")
        print("═" * 65)


# ══════════════════════════════════════════════════════
#                    CLI
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ARCHISAFE v5 - Canlı Kamera (Persistent Person Box + Eksik KKD)"
    )
    parser.add_argument("--model",  type=str, default=str(DEFAULT_MODEL),
                        help="Model ağırlığı (.pt)")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX,
                        help="Kamera index (varsayılan: 0)")
    parser.add_argument("--conf",   type=float, default=DEFAULT_CONF,
                        help=f"Güven eşiği (varsayılan: {DEFAULT_CONF})")
    parser.add_argument("--iou",    type=float, default=DEFAULT_IOU,
                        help=f"IoU eşiği (varsayılan: {DEFAULT_IOU})")
    parser.add_argument("--record", action="store_true",
                        help="Başlatılınca hemen kayda al")
    args = parser.parse_args()

    run_live(
        model_path   = Path(args.model),
        camera_index = args.camera,
        conf         = args.conf,
        iou          = args.iou,
        auto_record  = args.record,
    )
