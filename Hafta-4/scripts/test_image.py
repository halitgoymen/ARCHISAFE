"""
╔══════════════════════════════════════════════════════════════════════╗
║         ARCHISAFE — Resim ile Test Scripti                           ║
║         4. Hafta: Eğitilmiş Modeli Resimde Test Et                  ║
╚══════════════════════════════════════════════════════════════════════╝

Kullanım:
    python test_image.py
    python test_image.py --image ../test_images/ornek.jpg
    python test_image.py --image ../test_images/ornek.jpg --conf 0.4

NOT: Eğitilen modelin en iyi ağırlığını otomatik bulur.
     Manuel belirtmek için: --model ../results/ARCHISAFE_Final/YOLO12s_50ep/weights/best.pt
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

# ══════════════════════════════════════════════════════
#                    KONFİGÜRASYON
# ══════════════════════════════════════════════════════

# Otomatik model yolu (eğitim sonucu)
DEFAULT_MODEL = Path(__file__).parent.parent / "results" / "ARCHISAFE_Final" / "YOLO12s_50ep" / "weights" / "best.pt"

# Test görselleri klasörü
TEST_IMAGES_DIR = Path(__file__).parent.parent / "test_images"

# Tespit eşiği
DEFAULT_CONF = 0.35
DEFAULT_IOU  = 0.45

# Sınıf renkleri (BGR formatı)
CLASS_COLORS = {
    "hardhat":       (0,   200,  0),    # Yeşil
    "unhardhat":     (0,   0,   255),   # Kırmızı
    "vest":          (0,   200, 200),   # Sarı
    "unvest":        (0,   0,   200),   # Turuncu
    "mask":          (200, 0,   200),   # Mor
    "unmask":        (128, 0,   128),   # Koyu mor
    "gloves":        (255, 165, 0),     # Mavi açık
    "ungloves":      (0,   128, 128),   # Koyu sarı
    "person":        (255, 255, 0),     # Cyan
    "Fall Detection - v4 resized640_aug3x-ACCURATE": (0, 0, 255),  # Kırmızı
    "Ear-protection": (200, 200, 0),
    "shoes":         (150, 75,  0),
    "no_arm_sleeve": (100, 100, 200),
}

# ══════════════════════════════════════════════════════
#                    GÖRSEL ÇIZIM
# ══════════════════════════════════════════════════════

def draw_detections(img, results, model_names):
    """Tespit kutularını ve etiketleri görsel üzerine çiz."""
    annotated = img.copy()

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf  = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = model_names[cls_id]

        # Renge göre bul (bilinmiyorsa beyaz)
        color = CLASS_COLORS.get(label, (255, 255, 255))

        # Kutu çiz
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Etiket arka planı
        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)

        # Etiket metni
        cv2.putText(annotated, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # Tespit sayısı
    n = len(results[0].boxes)
    cv2.putText(annotated, f"Tespit: {n}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

    return annotated


# ══════════════════════════════════════════════════════
#                    ANA FONKSİYON
# ══════════════════════════════════════════════════════

def test_image(image_path: Path, model_path: Path, conf: float, iou: float):
    print("\n" + "═" * 60)
    print("      ARCHISAFE — Resim Test Modu")
    print("═" * 60)

    # Model kontrolü
    if not model_path.exists():
        print(f"\n❌ Model bulunamadı: {model_path}")
        print("Lütfen önce train_final.py scriptini çalıştırın!")
        sys.exit(1)

    # Görsel kontrolü
    if not image_path.exists():
        print(f"\n❌ Görsel bulunamadı: {image_path}")
        sys.exit(1)

    print(f"\n🤖 Model   : {model_path}")
    print(f"🖼️  Görsel  : {image_path}")
    print(f"⚙️  Conf    : {conf} | IoU: {iou}")

    # GPU/CPU
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"💻 Device  : {'GPU' if device == 0 else 'CPU'}\n")

    # Model yükle
    model = YOLO(str(model_path))

    # Görseli oku
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"❌ Görsel okunamadı: {image_path}")
        sys.exit(1)

    # Çıkarım yap
    t0 = time.time()
    results = model.predict(
        source=img,
        conf=conf,
        iou=iou,
        device=device,
        verbose=False
    )
    inference_ms = (time.time() - t0) * 1000

    # Tespit sayısı
    n_detections = len(results[0].boxes)
    print(f"✅ Tespit Edilen Nesne: {n_detections} adet")
    print(f"⚡ Çıkarım Süresi    : {inference_ms:.1f} ms")
    print()

    # Sınıf bazında özet
    if n_detections > 0:
        class_counts = {}
        for box in results[0].boxes:
            cls = model.names[int(box.cls[0])]
            class_counts[cls] = class_counts.get(cls, 0) + 1
        print("📊 Sınıf Dağılımı:")
        for cls, count in class_counts.items():
            print(f"   {cls:40s}: {count}")

    # Annotated görseli çiz
    annotated = draw_detections(img, results, model.names)

    # Sonucu kaydet
    output_dir = Path(__file__).parent.parent / "results" / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"result_{image_path.stem}.jpg"
    cv2.imwrite(str(output_path), annotated)
    print(f"\n💾 Sonuç kaydedildi: {output_path}")

    # Pencerede göster
    print("🖼️  Pencere açılıyor... (kapatmak için herhangi bir tuşa bas)")
    cv2.imshow("ARCHISAFE - Tespit Sonucu", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("═" * 60)


# ══════════════════════════════════════════════════════
#                    CLI
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ARCHISAFE - Resim Test Scripti")
    parser.add_argument(
        "--image", type=str,
        default=None,
        help="Test edilecek görsel yolu. Belirtilmezse test_images/ klasöründeki ilk görseli kullanır."
    )
    parser.add_argument(
        "--model", type=str,
        default=str(DEFAULT_MODEL),
        help="Kullanılacak model ağırlığı (.pt dosyası)."
    )
    parser.add_argument(
        "--conf", type=float, default=DEFAULT_CONF,
        help=f"Güven eşiği (varsayılan: {DEFAULT_CONF})"
    )
    parser.add_argument(
        "--iou", type=float, default=DEFAULT_IOU,
        help=f"IoU eşiği NMS için (varsayılan: {DEFAULT_IOU})"
    )
    args = parser.parse_args()

    # Görsel seçimi
    if args.image:
        image_path = Path(args.image)
    else:
        # test_images klasöründe ilk görseli bul
        supported = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
        candidates = [f for f in TEST_IMAGES_DIR.iterdir() if f.suffix.lower() in supported]
        if not candidates:
            print(f"❌ {TEST_IMAGES_DIR} klasöründe görsel bulunamadı.")
            print("   Lütfen bir test görseli koyun veya --image parametresiyle belirtin.")
            sys.exit(1)
        image_path = candidates[0]
        print(f"ℹ️  Otomatik seçilen görsel: {image_path.name}")

    test_image(
        image_path=image_path,
        model_path=Path(args.model),
        conf=args.conf,
        iou=args.iou,
    )
