"""
╔══════════════════════════════════════════════════════════════════════╗
║         ARCHISAFE — YOLO12-Small Final Eğitim Scripti               ║
║         4. Hafta: Final Model Eğitimi                                ║
╚══════════════════════════════════════════════════════════════════════╝

Karşılaştırma çalışmasında kazanan model: YOLO12-Small (mAP50: 0.4744)
Bu script, modeli tam eğitim parametreleriyle (50 epoch) eğitir.

BAŞLATMAK İÇİN:
    cd Hafta-4/scripts
    python train_final.py

Eğitimi durdurmak için: CTRL+C  (en iyi ağırlık zaten kaydedilmiş olur)
"""

import os
import time
import torch
from pathlib import Path
from ultralytics import YOLO

# ══════════════════════════════════════════════════════
#                    KONFİGÜRASYON
# ══════════════════════════════════════════════════════

BASE_DIR   = Path(__file__).resolve().parent.parent.parent   # ARCHISAFE/
DATA_YAML  = BASE_DIR / "PPE" / "Fall-Detecetion-1" / "data.yaml"
MODEL_PATH = BASE_DIR / "models" / "yolo12s.pt"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# ─── Eğitim Parametreleri ────────────────────────────
EPOCHS        = 30      # Maksimum epoch sayısı
IMGSZ         = 640     # Giriş çözünürlüğü
BATCH         = 8       # RTX 5070 Ti için uygun
WORKERS       = 0       # Windows'ta 0 bırak

# ─── Overfit Önleme ──────────────────────────────────
PATIENCE      = 10      # Bu kadar epoch iyileşme olmazsa ERKEN DUR
DROPOUT       = 0.0     # YOLO12 için (0.0–0.3 arası dene, başlangıç 0)
WEIGHT_DECAY  = 0.0005  # L2 regularization — ağırlıkların büyümesini frenler
LABEL_SMOOTHING = 0.1   # Etiket yumuşatma — modeli aşırı emin olmaktan alıkoyar

# ─── Öğrenme Oranı ───────────────────────────────────
LR0           = 0.01    # Başlangıç LR
LRF           = 0.001   # Final LR (cosine decay ile düşer)
MOMENTUM      = 0.937
WARMUP_EPOCHS = 3       # İlk 3 epoch LR yavaş başlar

# ─── Veri Artırma (Augmentation) ─────────────────────
# Overfit önlemenin en etkili yolu: modele her seferinde biraz farklı görsel göstermek
AUGMENT_CONFIG = dict(
    hsv_h      = 0.015,   # Renk tonu değişimi
    hsv_s      = 0.7,     # Doygunluk değişimi
    hsv_v      = 0.4,     # Parlaklık değişimi
    degrees    = 5.0,     # Hafif rotasyon (±5°)
    translate  = 0.1,     # Öteleme
    scale      = 0.5,     # Ölçek değişimi
    shear      = 2.0,     # Kesme dönüşümü
    flipud     = 0.1,     # Dikey çevirme (fabrika sahneleri için düşük)
    fliplr     = 0.5,     # Yatay çevirme
    mosaic     = 1.0,     # Mozaik (4 görsel birleştirme) — küçük nesneler için güçlü
    mixup      = 0.1,     # MixUp — iki görseli karıştırma
    copy_paste = 0.0,     # Copy-paste segmentasyon için, burada kapalı
    close_mosaic = 10,    # Son 10 epoch'ta mozaiği kapat (fine-tuning için)
    erasing    = 0.4,     # Random Erasing — pikselleri siler, overfit engeller
)

PROJECT_NAME  = "ARCHISAFE_Final"
RUN_NAME      = "YOLO12s_50ep"

# ══════════════════════════════════════════════════════
#                    GPU TESPİTİ
# ══════════════════════════════════════════════════════

def setup_device():
    try:
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            capability  = torch.cuda.get_device_capability(0)
            torch.zeros(1).cuda()  # hızlı test
            print(f"✅ GPU Aktif  : {device_name} (sm_{capability[0]}{capability[1]})")
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"   VRAM       : {vram:.1f} GB")
            return 0
        else:
            print("⚠️  CUDA bulunamadı → CPU kullanılacak (çok yavaş olabilir)")
            return "cpu"
    except RuntimeError as e:
        print(f"❌ GPU Hatası: {e} → CPU'ya geçiliyor")
        return "cpu"

# ══════════════════════════════════════════════════════
#                 OVERFIT KONTROLÜ (CALLBACK)
# ══════════════════════════════════════════════════════

class OverfitMonitor:
    """
    Her epoch sonunda train_loss ile val_loss farkını izler.
    Fark çok büyüdüğünde uyarı verir.
    """
    def __init__(self, gap_threshold=0.3):
        self.gap_threshold = gap_threshold
        self.history = []

    def on_fit_epoch_end(self, trainer):
        metrics = trainer.metrics
        train_loss = getattr(trainer, "loss", None)
        val_loss   = metrics.get("val/box_loss", None)

        if train_loss is not None and val_loss is not None:
            gap = float(val_loss) - float(train_loss)
            self.history.append(gap)
            if gap > self.gap_threshold:
                print(f"\n  ⚠️  [OverfitMonitor] Epoch {trainer.epoch+1}: "
                      f"val_loss - train_loss = {gap:.4f}  (>{self.gap_threshold}) "
                      f"→ Overfit belirtisi!")

# ══════════════════════════════════════════════════════
#                    EĞİTİM
# ══════════════════════════════════════════════════════

def train():
    print("\n" + "═" * 65)
    print("        ARCHISAFE — YOLO12-Small Final Eğitimi")
    print("═" * 65)

    # Kontroller
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Veri seti bulunamadı:\n  {DATA_YAML}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model dosyası bulunamadı:\n  {MODEL_PATH}")

    print(f"\n📂 Veri Seti   : {DATA_YAML}")
    print(f"🤖 Model       : {MODEL_PATH}")
    print(f"📁 Çıktı       : {RESULTS_DIR / PROJECT_NAME / RUN_NAME}")
    print(f"\n⚙️  Eğitim Ayarları:")
    print(f"   Epochs          : {EPOCHS} (max)")
    print(f"   Early Stopping  : {PATIENCE} epoch sabır — iyileşme yoksa DURUR")
    print(f"   Weight Decay    : {WEIGHT_DECAY}  (L2 regularization)")
    print(f"   Label Smoothing : {LABEL_SMOOTHING}  (etiket yumuşatma)")
    print(f"   Mosaic Kapatma  : Son {AUGMENT_CONFIG['close_mosaic']} epoch")
    print(f"   Batch / ImgSz   : {BATCH} / {IMGSZ}x{IMGSZ}")
    print(f"   LR              : {LR0} → {LRF} (cosine decay)")
    print(f"   AMP             : ✅ Mixed Precision")
    print("═" * 65)

    device = setup_device()

    # Model yükle
    model = YOLO(str(MODEL_PATH))

    # Overfit monitörü ekle
    monitor = OverfitMonitor(gap_threshold=0.3)
    model.add_callback("on_fit_epoch_end", monitor.on_fit_epoch_end)

    # Eğitim başlat
    print(f"\n🚀 Eğitim başlıyor...\n"
          f"   (CTRL+C ile durdurabilirsin — en iyi ağırlık otomatik kaydedilir)\n")
    start_time = time.time()

    results = model.train(
        data            = str(DATA_YAML),
        epochs          = EPOCHS,
        imgsz           = IMGSZ,
        batch           = BATCH,
        device          = device,
        workers         = WORKERS,
        # Anti-overfitting
        patience        = PATIENCE,
        weight_decay    = WEIGHT_DECAY,
        label_smoothing = LABEL_SMOOTHING,
        dropout         = DROPOUT,
        # Augmentation
        **AUGMENT_CONFIG,
        # LR
        lr0             = LR0,
        lrf             = LRF,
        momentum        = MOMENTUM,
        warmup_epochs   = WARMUP_EPOCHS,
        # Kayıt
        project         = str(RESULTS_DIR / PROJECT_NAME),
        name            = RUN_NAME,
        exist_ok        = True,
        save            = True,
        save_period     = 10,   # Her 10 epoch'ta checkpoint
        plots           = True,
        verbose         = True,
        val             = True,
        amp             = True,
    )

    elapsed = time.time() - start_time
    h, m, s = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)

    print("\n" + "═" * 65)
    print("✅ EĞİTİM TAMAMLANDI!")
    print(f"   Toplam Süre    : {h}s {m}dk {s}sn")
    print(f"   mAP50          : {results.results_dict.get('metrics/mAP50(B)', 0):.4f}")
    print(f"   mAP50-95       : {results.results_dict.get('metrics/mAP50-95(B)', 0):.4f}")

    best = RESULTS_DIR / PROJECT_NAME / RUN_NAME / "weights" / "best.pt"
    print(f"\n💾 En İyi Ağırlık: {best}")
    print("\n🎯 Sonraki adım:")
    print("   python test_image.py   → resimle test")
    print("   python test_live.py    → canlı kamera testi")
    print("═" * 65)

    return best


if __name__ == "__main__":
    train()
