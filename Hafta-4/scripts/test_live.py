"""
╔══════════════════════════════════════════════════════════════════════╗
║         ARCHISAFE — Canlı Kamera Test Scripti                        ║
║         4. Hafta: Gerçek Zamanlı KKD ve Düşme Tespiti               ║
╚══════════════════════════════════════════════════════════════════════╝

Kullanım:
    python test_live.py                     # Varsayılan webcam (kamera 0)
    python test_live.py --camera 1          # 2. kamera
    python test_live.py --conf 0.4          # Güven eşiği değiştir
    python test_live.py --record            # Videoyu kaydet

Kontroller (pencere açıkken):
    Q / ESC  → Çıkış
    S        → Anlık ekran görüntüsü kaydet
    R        → Kayıt başlat/durdur (toggle)
    +/-      → Güven eşiğini artır/azalt
"""

import argparse
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

# ══════════════════════════════════════════════════════
#                    KONFİGÜRASYON
# ══════════════════════════════════════════════════════

DEFAULT_MODEL = Path(__file__).parent.parent / "results" / "ARCHISAFE_Final" / "YOLO12s_50ep" / "weights" / "best.pt"
OUTPUT_DIR    = Path(__file__).parent.parent / "results" / "live"

DEFAULT_CONF    = 0.35
DEFAULT_IOU     = 0.45
CAMERA_INDEX    = 0
FRAME_WIDTH     = 1280
FRAME_HEIGHT    = 720
FPS_SMOOTH      = 30    # FPS smoothing için frame sayısı

# Alarm sınıfları — bu sınıflar tespit edilince ekran kırmızıya döner
ALARM_CLASSES = {
    "Fall Detection - v4 resized640_aug3x-ACCURATE",
    "unhardhat",
    "unvest",
    "unmask",
    "ungloves",
}

# Sınıf renkleri (BGR)
CLASS_COLORS = {
    "hardhat":      (0,   200,  50),
    "unhardhat":    (0,   0,   255),
    "vest":         (0,   200, 200),
    "unvest":       (0,   100, 255),
    "mask":         (200, 0,   200),
    "unmask":       (100, 0,   255),
    "gloves":       (255, 165, 0),
    "ungloves":     (0,   80,  200),
    "person":       (255, 255, 0),
    "Fall Detection - v4 resized640_aug3x-ACCURATE": (0, 0, 255),
    "Ear-protection": (200, 200, 0),
    "shoes":        (100, 60,  0),
    "no_arm_sleeve": (100, 100, 200),
}

# ══════════════════════════════════════════════════════
#                    YARDIMCI FONKSİYONLAR
# ══════════════════════════════════════════════════════

def draw_frame(frame, results, model_names, fps: float, conf_threshold: float, is_recording: bool):
    """Frame üzerine tespitleri ve HUD bilgilerini çiz."""
    overlay = frame.copy()
    has_alarm = False

    # Her kutu için
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf  = float(box.conf[0])
        label = model_names[int(box.cls[0])]
        color = CLASS_COLORS.get(label, (200, 200, 200))

        if label in ALARM_CLASSES:
            has_alarm = True

        # Kutu
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        # Etiket arka plan
        text  = f"{label}  {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(overlay, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(overlay, text, (x1 + 3, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Alarm: kırmızı overlay
    if has_alarm:
        alarm_overlay = overlay.copy()
        cv2.rectangle(alarm_overlay, (0, 0), (overlay.shape[1], overlay.shape[0]), (0, 0, 180), -1)
        cv2.addWeighted(alarm_overlay, 0.18, overlay, 0.82, 0, overlay)
        cv2.putText(overlay, "⚠ ALARM: Ihlal Tespit Edildi!", (10, overlay.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    # HUD — üst bar
    h, w = overlay.shape[:2]
    cv2.rectangle(overlay, (0, 0), (w, 50), (20, 20, 20), -1)

    # FPS
    cv2.putText(overlay, f"FPS: {fps:.1f}", (10, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 100), 2, cv2.LINE_AA)

    # Tespit sayısı
    n = len(results[0].boxes)
    cv2.putText(overlay, f"Tespit: {n}", (130, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2, cv2.LINE_AA)

    # Conf eşiği
    cv2.putText(overlay, f"Conf: {conf_threshold:.2f}", (280, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1, cv2.LINE_AA)

    # Kayıt indikatörü
    if is_recording:
        cv2.circle(overlay, (w - 25, 25), 10, (0, 0, 255), -1)
        cv2.putText(overlay, "REC", (w - 70, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)

    return overlay, has_alarm


# ══════════════════════════════════════════════════════
#                    ANA DÖNGÜ
# ══════════════════════════════════════════════════════

def run_live(model_path: Path, camera_index: int, conf: float, iou: float, auto_record: bool):
    print("\n" + "═" * 60)
    print("      ARCHISAFE — Canlı Kamera Tespiti")
    print("═" * 60)

    if not model_path.exists():
        print(f"\n❌ Model bulunamadı: {model_path}")
        print("Lütfen önce train_final.py scriptini çalıştırın!")
        return

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"\n🤖 Model  : {model_path}")
    print(f"📷 Kamera : {camera_index}")
    print(f"⚙️  Conf   : {conf} | IoU: {iou}")
    print(f"💻 Device : {'GPU' if device == 0 else 'CPU'}")
    print(f"\nKontroller: Q/ESC=Çıkış | S=Screenshot | R=Kayıt | +/-=Conf\n")

    # Model yükle
    model = YOLO(str(model_path))

    # Kamera aç
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print(f"❌ Kamera {camera_index} açılamadı!")
        return

    # FPS hesaplama
    fps_deque = deque(maxlen=FPS_SMOOTH)
    prev_time = time.time()

    # Video kaydı
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    writer = None
    is_recording = auto_record

    if auto_record:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = OUTPUT_DIR / f"live_{ts}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, 20, (FRAME_WIDTH, FRAME_HEIGHT))
        print(f"🎥 Kayıt: {video_path}")

    screenshot_count = 0
    conf_threshold   = conf

    print("═" * 60)
    print("Kamera başlatılıyor...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️  Frame okunamadı, kamera bağlantısını kontrol et.")
                break

            # FPS
            cur_time = time.time()
            fps_deque.append(1.0 / (cur_time - prev_time + 1e-6))
            prev_time = cur_time
            fps = sum(fps_deque) / len(fps_deque)

            # Model çıkarımı
            results = model.predict(
                source=frame,
                conf=conf_threshold,
                iou=iou,
                device=device,
                stream=False,
                verbose=False,
            )

            # Frame çiz
            annotated, has_alarm = draw_frame(
                frame, results, model.names, fps, conf_threshold, is_recording
            )

            # Video kaydı
            if is_recording and writer is not None:
                writer.write(annotated)

            # Göster
            cv2.imshow("ARCHISAFE - Canli Tespit", annotated)

            # Klavye kontrolleri
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):   # Q veya ESC
                break
            elif key == ord("s"):        # Screenshot
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                shot_path = OUTPUT_DIR / f"screenshot_{ts}.jpg"
                cv2.imwrite(str(shot_path), annotated)
                screenshot_count += 1
                print(f"📸 Screenshot kaydedildi: {shot_path}")
            elif key == ord("r"):        # Kayıt toggle
                is_recording = not is_recording
                if is_recording:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    video_path = OUTPUT_DIR / f"live_{ts}.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(video_path), fourcc, 20, (FRAME_WIDTH, FRAME_HEIGHT))
                    print(f"🎥 Kayıt başladı: {video_path}")
                else:
                    if writer:
                        writer.release()
                        writer = None
                    print("⏹  Kayıt durduruldu.")
            elif key == ord("+"):        # Conf artır
                conf_threshold = min(0.95, conf_threshold + 0.05)
                print(f"⬆️  Conf eşiği: {conf_threshold:.2f}")
            elif key == ord("-"):        # Conf azalt
                conf_threshold = max(0.05, conf_threshold - 0.05)
                print(f"⬇️  Conf eşiği: {conf_threshold:.2f}")

    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print(f"\n✅ Kamera kapatıldı. Toplam screenshot: {screenshot_count}")
        print("═" * 60)


# ══════════════════════════════════════════════════════
#                    CLI
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ARCHISAFE - Canlı Kamera Test Scripti")
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
        model_path=Path(args.model),
        camera_index=args.camera,
        conf=args.conf,
        iou=args.iou,
        auto_record=args.record,
    )
