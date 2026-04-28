"""
╔══════════════════════════════════════════════════════════════════════╗
║         ARCHISAFE — 6. Hafta Canlı Kamera Tespiti                   ║
║         YOLOv12 Nano | PPE + Fall Detection AYRI FONKSİYONLAR       ║
╠══════════════════════════════════════════════════════════════════════╣
║  FONKSİYONEL YAPI:                                                   ║
║    detect_ppe(boxes, names)   → PPE tespiti (gözlük,baret,yelek..)  ║
║    detect_fall(boxes, names)  → Düşme tespiti (kritik alarm!)        ║
║    draw_frame(...)            → Her ikisini entegre çizer            ║
║                                                                      ║
║  13 SINIF:                                                           ║
║    0: Ear-protection           7: shoes                              ║
║    1: Fall Detection (ACCURTE) 8: ungloves                           ║
║    2: gloves                   9: unhardhat                          ║
║    3: hardhat                 10: unmask                             ║
║    4: mask                    11: unvest                             ║
║    5: no_arm_sleeve           12: vest                               ║
║    6: person                                                         ║
╠══════════════════════════════════════════════════════════════════════╣
║  Kullanım:                                                           ║
║    python test_live_v6.py                     # Kamera 0             ║
║    python test_live_v6.py --camera 1          # 2. kamera            ║
║    python test_live_v6.py --conf 0.30         # Güven eşiği          ║
║    python test_live_v6.py --record            # Kayıt                ║
║    python test_live_v6.py --model path/to/best.pt                   ║
║    python test_live_v6.py --source video.mp4  # Video dosyası       ║
║                                                                      ║
║  Kontroller:                                                         ║
║    Q / ESC  → Çıkış       S → Screenshot     R → Kayıt toggle      ║
║    +/-      → Conf ±0.05  H → HUD aç/kapat   P → Duraklat          ║
║    F        → Fall-Only modu (sadece düşme alarmı)                  ║
║    D        → Demo modu (istatistik ekranı)                         ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import argparse
import time
from collections import deque, defaultdict
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ══════════════════════════════════════════════════════
#                    KONFİGÜRASYON
# ══════════════════════════════════════════════════════

_SCRIPT_DIR   = Path(__file__).resolve().parent
_HAFTA6_DIR   = _SCRIPT_DIR.parent

DEFAULT_MODEL = (
    _HAFTA6_DIR / "results" / "ARCHISAFE_v6" / "yolo12n_150ep" / "weights" / "best.pt"
)
OUTPUT_DIR    = _HAFTA6_DIR / "results" / "live"

DEFAULT_CONF   = 0.30
DEFAULT_IOU    = 0.45
CAMERA_INDEX   = 0
FRAME_WIDTH    = 1280
FRAME_HEIGHT   = 720
FPS_SMOOTH     = 30

# Kişi kutusunu kaç frame tut (kısa kayıplar için)
PERSON_HOLD_FRAMES = 8

# Fall detection sesi veya alarm kaç frame tetiklensin
FALL_ALARM_FRAMES  = 3   # Ardışık bu kadar frame fall görülürse alarm

# ══════════════════════════════════════════════════════
#                 SINIF HARİTALARI
# ══════════════════════════════════════════════════════

# dataset'teki tam sınıf isimleri (data.yaml ile eşleşmeli)
FALL_CLASS_NAMES = {
    "Fall Detection - v4 resized640_aug3x-ACCURATE",
    "Fall Detection",           # geçiş uyumluluğu için
}

# Mevcut olması gereken KKD ekipmanları
REQUIRED_PPE: dict[str, str] = {
    "hardhat":          "Baret",
    "vest":             "Yelek",
    "mask":             "Maske",
    "gloves":           "Eldiven",
    "Ear-protection":   "Kulak/Göz Koruyucu",
    "shoes":            "Ayakkabı",
}

# İhlal sınıfları (olması kötü)
VIOLATION_PPE: dict[str, str] = {
    "unhardhat":      "Baretsiz ⚠",
    "unvest":         "Yeleksiz ⚠",
    "unmask":         "Maskesiz ⚠",
    "ungloves":       "Eldivensiz ⚠",
    "no_arm_sleeve":  "Kol-Koruyucusuz ⚠",
}

# Tüm aktif alarm sınıfları
ALL_ALARM_CLASSES = set(VIOLATION_PPE.keys()) | FALL_CLASS_NAMES

# ─── Kutu Renkleri (BGR) ───────────────────────────────
CLASS_COLORS: dict[str, tuple] = {
    # Güvenli KKD → yeşilimsi
    "hardhat":          (40,  210,  70),
    "vest":             (40,  210, 210),
    "mask":             (200,  60, 200),
    "gloves":           (255, 170,  30),
    "Ear-protection":   (210, 210,  40),
    "shoes":            (100, 170,  80),
    # İhlaller → kırmızı/turuncu
    "unhardhat":        (0,    0,  255),
    "unvest":           (0,   90,  255),
    "unmask":           (80,   0,  220),
    "ungloves":         (0,  110,  240),
    "no_arm_sleeve":    (110,  50, 210),
    # Fall → parlak kırmızı
    "Fall Detection - v4 resized640_aug3x-ACCURATE": (0,  0, 255),
    "Fall Detection":   (0,   0,  255),
    # Kişi → turuncu
    "person":           (255, 110,  20),
}
DEFAULT_COLOR = (160, 160, 160)

# ══════════════════════════════════════════════════════
#                    VERİ YAPILARI
# ══════════════════════════════════════════════════════

@dataclass
class PPEResult:
    """detect_ppe() çıktısı."""
    present_ppe:   dict = field(default_factory=dict)   # cls → conf
    violations:    dict = field(default_factory=dict)   # cls → conf
    missing_ppe:   list = field(default_factory=list)   # eksik Türkçe isimler
    has_violation: bool = False
    risk_level:    str  = "OK"                          # OK | WARNING | DANGER

@dataclass
class FallResult:
    """detect_fall() çıktısı."""
    detected:      bool  = False
    confidence:    float = 0.0
    box:           Optional[tuple] = None               # (x1,y1,x2,y2) veya None
    alarm_active:  bool  = False                        # ardışık frame alarmı


# ══════════════════════════════════════════════════════
#       ❶  PPE TESPİT FONKSİYONU (ana işlev)
# ══════════════════════════════════════════════════════

def detect_ppe(boxes, names: dict) -> PPEResult:
    """
    Model çıktısındaki kutuları tarayarak KKD/ihlal tespiti yapar.

    Parametreler
    ------------
    boxes  : results[0].boxes  (ultralytics YOLO çıktısı)
    names  : model.names  (index → sınıf ismi dict)

    Döndürür
    --------
    PPEResult — tüm PPE sonuçları, ihlaller, eksikler ve risk seviyesi.

    Notlar
    ------
    • Fall Detection sınıfları bu fonksiyondan ÇIKARTILIR;
      düşme analizi için detect_fall() kullanılır.
    • risk_level: "OK" / "WARNING" (mevcut kişi ihlal var) / "DANGER" (ciddi ihlal)
    """
    result = PPEResult()

    for box in boxes:
        cls_idx = int(box.cls[0])
        conf    = float(box.conf[0])
        label   = names.get(cls_idx, "unknown")

        # Fall sınıflarını atla → detect_fall() işleyecek
        if label in FALL_CLASS_NAMES or label == "person":
            continue

        if label in REQUIRED_PPE:
            # Daha yüksek conf varsa güncelle (çok tespit durumu)
            result.present_ppe[label] = max(result.present_ppe.get(label, 0.0), conf)

        elif label in VIOLATION_PPE:
            result.violations[label] = max(result.violations.get(label, 0.0), conf)
            result.has_violation = True

    # Eksik KKD hesapla
    result.missing_ppe = [
        tr_name
        for cls_key, tr_name in REQUIRED_PPE.items()
        if cls_key not in result.present_ppe
    ]

    # Risk seviyesi belirle
    if result.violations:
        # Yüksek güven ihlal → DANGER
        max_viol_conf = max(result.violations.values())
        result.risk_level = "DANGER" if max_viol_conf >= 0.55 else "WARNING"
    elif result.missing_ppe:
        result.risk_level = "WARNING"
    else:
        result.risk_level = "OK"

    return result


# ══════════════════════════════════════════════════════
#       ❷  DÜŞME TESPİT FONKSİYONU (ana işlev)
# ══════════════════════════════════════════════════════

class FallDetector:
    """
    detect_fall() metodunu ardışık frame geçmişiyle yönetir.
    Tek seferlik false-positive'leri filtreler.

    Kullanım
    --------
        detector = FallDetector(alarm_frames=3)
        result   = detector.detect(boxes, names)   # Her frame'de çağır
    """

    def __init__(self, alarm_frames: int = FALL_ALARM_FRAMES):
        self.alarm_frames   = alarm_frames
        self._consecutive   = 0             # ardışık fall frame sayacı
        self._last_conf     = 0.0
        self._last_box      = None

    def detect(self, boxes, names: dict) -> FallResult:
        """
        Model çıktısından düşme tespiti yapar.

        Parametreler
        ------------
        boxes  : results[0].boxes
        names  : model.names

        Döndürür
        --------
        FallResult — tespet durumu, güven değeri, kutu ve alarm bayrağı.

        Algoritma
        ---------
        1. Kutular içinde FALL_CLASS_NAMES'e ait herhangi bir sınıf ara.
        2. En yüksek güvenli tespiti seç.
        3. consecutive sayacını güncelle.
        4. alarm_frames ardışık frame fall görülürse alarm_active=True dön.
        """
        best_conf = 0.0
        best_box  = None

        for box in boxes:
            cls_idx = int(box.cls[0])
            conf    = float(box.conf[0])
            label   = names.get(cls_idx, "unknown")

            if label in FALL_CLASS_NAMES and conf > best_conf:
                best_conf = conf
                best_box  = tuple(map(int, box.xyxy[0]))

        # Ardışık sayacı güncelle
        if best_conf > 0.0:
            self._consecutive  += 1
            self._last_conf     = best_conf
            self._last_box      = best_box
        else:
            self._consecutive   = 0
            self._last_conf     = 0.0
            self._last_box      = None

        alarm = self._consecutive >= self.alarm_frames

        return FallResult(
            detected      = best_conf > 0.0,
            confidence    = best_conf,
            box           = best_box,
            alarm_active  = alarm,
        )


# ══════════════════════════════════════════════════════
#              KİŞİ TAKİBİ (Persistent Box)
# ══════════════════════════════════════════════════════

class PersonTracker:
    """IoU tabanlı basit kişi takibi — kısa kayıplarda kutuyu tutar."""

    def __init__(self, hold_frames: int = PERSON_HOLD_FRAMES):
        self.hold_frames = hold_frames
        self.persons     = {}   # tid → {box, last_conf, miss_count}
        self._next_id    = 0

    @staticmethod
    def _iou(a, b) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter  = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return inter / (area_a + area_b - inter + 1e-6)

    def update(self, detected: list) -> list:
        """
        detected: [(x1,y1,x2,y2,conf), ...]
        Returns : [(x1,y1,x2,y2,conf,tid,is_real), ...]
        """
        matched_ids = set()
        used_dets   = set()

        for tid, info in self.persons.items():
            best_iou, best_di = 0.0, -1
            for di, det in enumerate(detected):
                if di in used_dets:
                    continue
                iou = self._iou(info["box"], det[:4])
                if iou > best_iou:
                    best_iou, best_di = iou, di
            if best_iou > 0.25 and best_di >= 0:
                det = detected[best_di]
                info.update(box=det[:4], last_conf=det[4], miss_count=0)
                matched_ids.add(tid)
                used_dets.add(best_di)

        for di, det in enumerate(detected):
            if di not in used_dets:
                self.persons[self._next_id] = {
                    "box": det[:4], "last_conf": det[4], "miss_count": 0
                }
                self._next_id += 1

        to_del = []
        for tid in list(self.persons):
            if tid not in matched_ids:
                self.persons[tid]["miss_count"] += 1
                if self.persons[tid]["miss_count"] > self.hold_frames:
                    to_del.append(tid)
        for tid in to_del:
            del self.persons[tid]

        return [
            (*info["box"], info["last_conf"], tid, info["miss_count"] == 0)
            for tid, info in self.persons.items()
        ]


# ══════════════════════════════════════════════════════
#              ÇİZİM YARDIMCILARI
# ══════════════════════════════════════════════════════

def draw_rounded_rect(img, pt1, pt2, color, radius=8, thickness=2):
    x1, y1 = pt1; x2, y2 = pt2
    r = max(1, min(radius, (x2 - x1) // 3, (y2 - y1) // 3))
    if thickness < 0:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        for cx, cy in [(x1+r,y1+r),(x2-r,y1+r),(x1+r,y2-r),(x2-r,y2-r)]:
            cv2.circle(img, (cx, cy), r, color, -1)
    else:
        cv2.line(img, (x1+r, y1), (x2-r, y1), color, thickness)
        cv2.line(img, (x1+r, y2), (x2-r, y2), color, thickness)
        cv2.line(img, (x1, y1+r), (x1, y2-r), color, thickness)
        cv2.line(img, (x2, y1+r), (x2, y2-r), color, thickness)
        for ang, cx, cy in [(180,x1+r,y1+r),(270,x2-r,y1+r),(90,x1+r,y2-r),(0,x2-r,y2-r)]:
            cv2.ellipse(img, (cx, cy), (r, r), ang, 0, 90, color, thickness)


def put_text_bg(img, text, pos, font_scale=0.52, color=(255, 255, 255),
                bg_color=(0, 0, 0), thickness=1, alpha=0.72, padding=4):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), bl = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    ov = img.copy()
    cv2.rectangle(ov, (x - padding, y - th - padding),
                  (x + tw + padding, y + bl + padding), bg_color, -1)
    cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def draw_risk_badge(img, risk_level: str, x: int, y: int):
    """Risk seviyesini renkli rozet olarak çizer."""
    colors = {"OK": (40, 200, 60), "WARNING": (0, 170, 255), "DANGER": (0, 0, 220)}
    labels = {"OK": "✓ GÜVENLI", "WARNING": "⚠ UYARI", "DANGER": "✖ TEHLİKE"}
    col = colors.get(risk_level, (160, 160, 160))
    lbl = labels.get(risk_level, risk_level)
    put_text_bg(img, lbl, (x, y), font_scale=0.65, color=(255, 255, 255),
                bg_color=col, thickness=2, alpha=0.85, padding=6)


def draw_fall_alarm_overlay(img, fall_result: FallResult):
    """Düşme alarmı olduğunda kırmızı titreşim efekti ve büyük banner çizer."""
    h, w = img.shape[:2]
    if not fall_result.alarm_active:
        return

    # Kırmızı overlay
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 180), -1)
    cv2.addWeighted(overlay, 0.22, img, 0.78, 0, img)

    # Kenarlık
    cv2.rectangle(img, (4, 4), (w - 4, h - 4), (0, 0, 255), 4)

    # Büyük banner — üst merkez
    banner   = f"  🚨 DÜŞME ALAMI! {fall_result.confidence:.0%}  "
    font     = cv2.FONT_HERSHEY_DUPLEX
    fs       = 1.1
    (bw, bh), _ = cv2.getTextSize(banner, font, fs, 2)
    bx = (w - bw) // 2
    by = 90

    # Gölge
    cv2.putText(img, banner, (bx + 2, by + 2), font, fs, (0, 0, 0), 3, cv2.LINE_AA)
    # Ana metin
    cv2.putText(img, banner, (bx,     by),     font, fs, (0, 50, 255), 2, cv2.LINE_AA)

    # Fall kutusu varsa çiz
    if fall_result.box:
        x1, y1, x2, y2 = fall_result.box
        draw_rounded_rect(img, (x1, y1), (x2, y2), (0, 0, 255), radius=8, thickness=3)
        put_text_bg(img, f"DÜŞME! {fall_result.confidence:.0%}",
                    (x1, max(y1 - 8, 20)),
                    font_scale=0.65, bg_color=(0, 0, 200),
                    color=(255, 220, 0), thickness=2, alpha=0.88)


# ══════════════════════════════════════════════════════
#              ANA ÇIZIM FONKSİYONU
# ══════════════════════════════════════════════════════

def draw_frame(
    frame: np.ndarray,
    results,
    model_names:  dict,
    fps:          float,
    conf_thr:     float,
    is_recording: bool,
    tracker:      PersonTracker,
    fall_detector: FallDetector,
    show_hud:     bool = True,
    fall_only:    bool = False,
) -> tuple[np.ndarray, PPEResult, FallResult]:
    """
    Tüm tespitleri, PPE panelini ve düşme alarmını frame üzerine çizer.

    Döndürür: (annotated_frame, ppe_result, fall_result)
    """
    h, w = frame.shape[:2]
    canvas = frame.copy()

    boxes = results[0].boxes

    # ─── ❶ PPE Tespiti ───────────────────────────────
    ppe_result  = detect_ppe(boxes, model_names)

    # ─── ❷ Düşme Tespiti ─────────────────────────────
    fall_result = fall_detector.detect(boxes, model_names)

    # ─── Kişi kutularını topla ────────────────────────
    person_boxes = []
    for box in boxes:
        label = model_names.get(int(box.cls[0]), "unknown")
        if label == "person":
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            person_boxes.append((x1, y1, x2, y2, float(box.conf[0])))

    # ─── Tüm nesne kutularını çiz ─────────────────────
    for box in boxes:
        cls_idx = int(box.cls[0])
        conf    = float(box.conf[0])
        label   = model_names.get(cls_idx, "unknown")

        if label == "person":
            continue  # kişiyi tracker üzerinden çiziyoruz

        if fall_only and label not in FALL_CLASS_NAMES:
            continue   # Fall-Only modda yalnız fall kutularını göster

        color = CLASS_COLORS.get(label, DEFAULT_COLOR)
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        draw_rounded_rect(canvas, (x1, y1), (x2, y2), color, radius=6, thickness=2)

        disp = label.replace("Fall Detection - v4 resized640_aug3x-ACCURATE", "FALL!")
        disp = (disp[:16] + "…") if len(disp) > 17 else disp
        txt  = f"{disp} {conf:.0%}"
        (tw, th_), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        lbl_y = max(y1 - 2, th_ + 6)
        draw_rounded_rect(canvas, (x1, lbl_y - th_ - 4), (x1 + tw + 6, lbl_y + 2),
                          color, radius=4, thickness=-1)
        cv2.putText(canvas, txt, (x1 + 3, lbl_y - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    # ─── Kişi takibi — Persistent Box ────────────────
    tracked = tracker.update(person_boxes)
    for (x1, y1, x2, y2, conf, tid, is_real) in tracked:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        box_color = (255, 120, 30) if is_real else (180, 80, 20)
        line_th   = 2              if is_real else 1

        if not is_real:
            ln = 18
            for sx, sy, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
                cv2.line(canvas, (sx, sy), (sx + dx*ln, sy), box_color, 2)
                cv2.line(canvas, (sx, sy), (sx, sy + dy*ln), box_color, 2)

        draw_rounded_rect(canvas, (x1, y1), (x2, y2), box_color, radius=8, thickness=line_th)
        hold_tag = "" if is_real else " [TAHMİN]"
        put_text_bg(canvas, f"Kisi #{tid}  {conf:.0%}{hold_tag}",
                    (x1, max(y1 - 8, 18)),
                    bg_color=box_color, color=(255, 255, 255), font_scale=0.50)

        # ─── Eksik KKD → kişinin sağında ─────────────
        if ppe_result.missing_ppe and not fall_only:
            mx, my = x2 + 8, y1
            put_text_bg(canvas, "✗ Eksik KKD:", (mx, my + 20),
                        bg_color=(20, 20, 100), color=(120, 190, 255),
                        font_scale=0.48, padding=3)
            for i, m in enumerate(ppe_result.missing_ppe):
                put_text_bg(canvas, f"  • {m}", (mx, my + 42 + i * 20),
                            bg_color=(20, 20, 80), color=(0, 150, 255),
                            font_scale=0.42, padding=2)

    # ─── PPE İhlal banner (alt) ───────────────────────
    if ppe_result.has_violation and not fall_only:
        viol_txt = " ⚠  İHLAL: " + " | ".join(
            f"{VIOLATION_PPE.get(k, k)} {v:.0%}"
            for k, v in ppe_result.violations.items()
        )
        put_text_bg(canvas, viol_txt, (10, h - 18),
                    font_scale=0.65, bg_color=(20, 0, 160),
                    color=(255, 210, 0), thickness=2, alpha=0.85)

    # ─── Düşme alarmı overlay ─────────────────────────
    draw_fall_alarm_overlay(canvas, fall_result)

    # ─── HUD (üst bar) ────────────────────────────────
    if show_hud:
        cv2.rectangle(canvas, (0, 0), (w, 54), (14, 14, 14), -1)
        cv2.line(canvas, (0, 54), (w, 54), (55, 55, 55), 1)

        fps_col = (0, 220, 80) if fps >= 20 else (0, 180, 255) if fps >= 10 else (0, 80, 255)
        cv2.putText(canvas, f"FPS: {fps:.1f}", (10, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, fps_col, 2, cv2.LINE_AA)

        n_total = len(results[0].boxes)
        n_ppe   = len(ppe_result.present_ppe)
        n_viol  = len(ppe_result.violations)
        n_fall  = 1 if fall_result.detected else 0

        cv2.putText(canvas, f"Tespit:{n_total}", (145, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 60), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"KKD:{n_ppe}", (270, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (60, 210, 60), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"İhlal:{n_viol}", (360, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60,
                    (0, 60, 255) if n_viol > 0 else (110, 110, 110), 2, cv2.LINE_AA)
        fall_col = (0, 0, 255) if fall_result.detected else (110, 110, 110)
        cv2.putText(canvas, f"Düşme:{n_fall}", (467, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, fall_col, 2, cv2.LINE_AA)

        cv2.putText(canvas, f"Conf:{conf_thr:.2f}", (578, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (170, 170, 170), 1, cv2.LINE_AA)

        ts_str = datetime.now().strftime("%H:%M:%S")
        cv2.putText(canvas, ts_str, (w - 102, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (160, 160, 160), 1, cv2.LINE_AA)

        if is_recording:
            cv2.circle(canvas, (w - 122, 28), 9, (0, 0, 255), -1)
            cv2.putText(canvas, "REC", (w - 160, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 0, 255), 2, cv2.LINE_AA)

        if fall_only:
            cv2.putText(canvas, "[ FALL ONLY ]", (670, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 120, 255), 2, cv2.LINE_AA)

        # ─── Risk rozeti ─────────────────────────────
        draw_risk_badge(canvas, ppe_result.risk_level, w - 300, 36)

        # ─── Güven Oranları Paneli (sağ) ─────────────
        _draw_confidence_panel(canvas, ppe_result, w, fall_result)

    return canvas, ppe_result, fall_result


def _draw_confidence_panel(img, ppe: PPEResult, frame_w: int, fall: FallResult):
    """Sağ tarafta tespit güven oranı barlarını çizer."""
    items = []
    for lbl, cf in ppe.present_ppe.items():
        items.append((lbl, cf, (50, 200, 80)))
    for lbl, cf in ppe.violations.items():
        items.append((VIOLATION_PPE.get(lbl, lbl), cf, (0, 80, 255)))
    if fall.detected:
        items.append((f"DÜŞME! ({fall.confidence:.0%})", fall.confidence, (0, 0, 220)))

    if not items:
        return

    px, py = frame_w - 230, 65
    ph     = 20 + len(items) * 22
    bg     = img.copy()
    cv2.rectangle(bg, (px - 6, py - 14), (frame_w - 6, py + ph), (14, 14, 14), -1)
    cv2.addWeighted(bg, 0.78, img, 0.22, 0, img)
    cv2.putText(img, "Güven Oranları:", (px, py),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 200, 200), 1, cv2.LINE_AA)

    bar_w = 210
    for i, (lbl, cf, col) in enumerate(items):
        yy = py + 22 + i * 22
        cv2.rectangle(img, (px, yy - 12), (px + bar_w, yy + 4), (40, 40, 40), -1)
        fill = int(bar_w * min(cf, 1.0))
        bar_col = col if cf >= 0.5 else (0, 130, 255)
        cv2.rectangle(img, (px, yy - 12), (px + fill, yy + 4), bar_col, -1)
        short = (lbl[:14] + "…") if len(lbl) > 15 else lbl
        cv2.putText(img, f"{short} {cf:.0%}", (px + 4, yy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)


# ══════════════════════════════════════════════════════
#                 İSTATİSTİK TRACKER
# ══════════════════════════════════════════════════════

class SessionStats:
    """Oturum boyunca fall ve ihlal sayılarını tutar."""

    def __init__(self):
        self.total_frames   = 0
        self.fall_frames    = 0
        self.violation_frames = 0
        self.class_counts   = defaultdict(int)

    def update(self, ppe: PPEResult, fall: FallResult):
        self.total_frames += 1
        if fall.alarm_active:
            self.fall_frames += 1
        if ppe.has_violation:
            self.violation_frames += 1
        for cls in ppe.violations:
            self.class_counts[cls] += 1

    def draw(self, img):
        h, w = img.shape[:2]
        lines = [
            f"Oturum İstatistikleri",
            f"Toplam Frame  : {self.total_frames}",
            f"Düşme Alarmı  : {self.fall_frames} frame",
            f"İhlal Frame   : {self.violation_frames} frame",
        ]
        if self.class_counts:
            lines.append("En Sık İhlaller:")
            for cls, cnt in sorted(self.class_counts.items(), key=lambda x: -x[1])[:4]:
                lines.append(f"  {VIOLATION_PPE.get(cls, cls)}: {cnt}")

        bx, by = 10, 70
        for i, ln in enumerate(lines):
            y_ = by + i * 22
            put_text_bg(img, ln, (bx, y_),
                        font_scale=0.50,
                        bg_color=(20, 20, 60) if i > 0 else (50, 30, 100),
                        color=(240, 240, 240) if i > 0 else (255, 220, 80),
                        alpha=0.80, padding=4)


# ══════════════════════════════════════════════════════
#                    ANA DÖNGÜ
# ══════════════════════════════════════════════════════

def run_live(
    model_path:   Path,
    camera_index: int,
    source:       str,
    conf:         float,
    iou:          float,
    auto_record:  bool,
):
    print("\n" + "═" * 68)
    print("       ARCHISAFE — 6. Hafta Canlı Tespit (YOLOv12 Nano)")
    print("       PPE Detection + Fall Detection — Ayrı Fonksiyonlar")
    print("═" * 68)

    if not model_path.exists():
        print(f"\n❌ Model bulunamadı: {model_path}")
        print("   Lütfen önce eğitimi tamamla: python train_v6.py")
        return

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"\n🤖 Model  : {model_path}")
    print(f"📷 Kaynak : {source if source else f'Kamera {camera_index}'}")
    print(f"⚙️  Conf   : {conf}")
    print(f"💻 Device : {'GPU 🚀' if device == 0 else 'CPU'}")
    print(f"\nKontroller: Q/ESC=Çık  S=Ekranlık  R=Kayıt  +/-=Conf  H=HUD  P=Poz  F=FallOnly  D=Demo")

    model = YOLO(str(model_path))
    print(f"\n✅ Model yüklendi ({len(model.names)} sınıf):")
    for i, name in model.names.items():
        tag = " ← FALL" if name in FALL_CLASS_NAMES else ""
        tag = tag or (" ← PPE" if name in REQUIRED_PPE else "")
        print(f"   [{i:2d}] {name}{tag}")

    # Kaynak aç
    if source:
        cap = cv2.VideoCapture(source)
    else:
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"❌ Kaynak açılamadı: {source or camera_index}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # FPS
    fps_deque = deque(maxlen=FPS_SMOOTH)
    prev_time = time.time()

    # Video kaydı
    writer       = None
    is_recording = auto_record
    if auto_record:
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        vp  = OUTPUT_DIR / f"live_v6_{ts}.mp4"
        writer = cv2.VideoWriter(
            str(vp), cv2.VideoWriter_fourcc(*"mp4v"), 20, (FRAME_WIDTH, FRAME_HEIGHT)
        )
        print(f"🎥 Kayıt başladı: {vp}")

    tracker       = PersonTracker(hold_frames=PERSON_HOLD_FRAMES)
    fall_detector = FallDetector(alarm_frames=FALL_ALARM_FRAMES)
    stats         = SessionStats()

    screenshot_count = 0
    conf_threshold   = conf
    show_hud         = True
    paused           = False
    fall_only_mode   = False
    demo_mode        = False
    last_canvas      = None

    print("\n" + "═" * 68)
    print("🎥 Başlatılıyor...")

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("⚠️  Frame okunamadı, kapatılıyor.")
                    break

                # Kamera için boyutlandır
                if not source:
                    pass  # zaten doğru boyutta
                else:
                    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

                cur  = time.time()
                fps_deque.append(1.0 / (cur - prev_time + 1e-9))
                prev_time = cur
                fps = sum(fps_deque) / len(fps_deque)

                results = model.predict(
                    source   = frame,
                    conf     = conf_threshold,
                    iou      = iou,
                    device   = device,
                    stream   = False,
                    verbose  = False,
                    max_det  = 60,
                )

                canvas, ppe_result, fall_result = draw_frame(
                    frame, results, model.names, fps,
                    conf_threshold, is_recording,
                    tracker, fall_detector,
                    show_hud, fall_only_mode,
                )

                stats.update(ppe_result, fall_result)

                if demo_mode:
                    stats.draw(canvas)

                if is_recording and writer is not None:
                    writer.write(canvas)

                cv2.imshow("ARCHISAFE v6 — YOLOv12 Nano", canvas)
                last_canvas = canvas

            else:
                pc = last_canvas.copy() if last_canvas is not None else np.zeros(
                    (FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8
                )
                put_text_bg(pc, "  ⏸  DURAKLATILDI  ",
                            (FRAME_WIDTH // 2 - 110, FRAME_HEIGHT // 2),
                            font_scale=1.0, bg_color=(20, 20, 80),
                            color=(255, 255, 0), thickness=2, alpha=0.88)
                cv2.imshow("ARCHISAFE v6 — YOLOv12 Nano", pc)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
            elif key == ord("s"):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                sp = OUTPUT_DIR / f"screenshot_v6_{ts}.jpg"
                cv2.imwrite(str(sp), last_canvas)
                screenshot_count += 1
                print(f"📸 Screenshot: {sp}")
            elif key == ord("r"):
                is_recording = not is_recording
                if is_recording:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    vp = OUTPUT_DIR / f"live_v6_{ts}.mp4"
                    writer = cv2.VideoWriter(
                        str(vp), cv2.VideoWriter_fourcc(*"mp4v"), 20, (FRAME_WIDTH, FRAME_HEIGHT)
                    )
                    print(f"🎥 Kayıt başladı: {vp}")
                else:
                    if writer:
                        writer.release(); writer = None
                    print("⏹  Kayıt durduruldu.")
            elif key in (ord("+"), ord("=")):
                conf_threshold = min(0.95, conf_threshold + 0.05)
                print(f"⬆️  Conf: {conf_threshold:.2f}")
            elif key == ord("-"):
                conf_threshold = max(0.05, conf_threshold - 0.05)
                print(f"⬇️  Conf: {conf_threshold:.2f}")
            elif key == ord("h"):
                show_hud = not show_hud
                print(f"HUD: {'açık' if show_hud else 'kapalı'}")
            elif key == ord("p"):
                paused = not paused
                print(f"{'⏸  Duraklatıldı' if paused else '▶  Devam'}")
            elif key == ord("f"):
                fall_only_mode = not fall_only_mode
                print(f"Fall-Only modu: {'✅ AÇIK' if fall_only_mode else '❌ kapalı'}")
            elif key == ord("d"):
                demo_mode = not demo_mode
                print(f"Demo/istatistik: {'✅ AÇIK' if demo_mode else '❌ kapalı'}")

    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print(f"\n✅ Oturum kapatıldı.")
        print(f"   Toplam Frame      : {stats.total_frames}")
        print(f"   Düşme Alarmları   : {stats.fall_frames}")
        print(f"   İhlal Frames      : {stats.violation_frames}")
        print(f"   Screenshots       : {screenshot_count}")
        print("═" * 68)


# ══════════════════════════════════════════════════════
#                    CLI
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ARCHISAFE v6 — YOLOv12 Nano | PPE + Fall Detection"
    )
    parser.add_argument(
        "--model", type=str, default=str(DEFAULT_MODEL),
        help="Model ağırlığı (.pt dosyası)"
    )
    parser.add_argument(
        "--camera", type=int, default=CAMERA_INDEX,
        help="Kamera index (varsayılan: 0)"
    )
    parser.add_argument(
        "--source", type=str, default="",
        help="Video dosyası yolu (boş bırakılırsa kamera kullanılır)"
    )
    parser.add_argument(
        "--conf", type=float, default=DEFAULT_CONF,
        help=f"Güven eşiği (varsayılan: {DEFAULT_CONF})"
    )
    parser.add_argument(
        "--iou", type=float, default=DEFAULT_IOU,
        help=f"IoU eşiği (varsayılan: {DEFAULT_IOU})"
    )
    parser.add_argument(
        "--record", action="store_true",
        help="Başlatılınca hemen video kaydet"
    )
    args = parser.parse_args()

    run_live(
        model_path   = Path(args.model),
        camera_index = args.camera,
        source       = args.source,
        conf         = args.conf,
        iou          = args.iou,
        auto_record  = args.record,
    )
