"""
╔══════════════════════════════════════════════════════════════════════╗
║         ARCHISAFE — 5. Hafta Eğitim Scripti                         ║
║         Base: yolo26n.pt | 200 Epoch | RTX 5070 Ti Optimize         ║
╠══════════════════════════════════════════════════════════════════════╣
║  Sınıflar (13 adet):                                                 ║
║    0: Ear-protection       7: shoes                                  ║
║    1: Fall Detection       8: ungloves                               ║
║    2: gloves               9: unhardhat                              ║
║    3: hardhat             10: unmask                                  ║
║    4: mask                11: unvest                                  ║
║    5: no_arm_sleeve       12: vest                                   ║
║    6: person                                                          ║
╠══════════════════════════════════════════════════════════════════════╣
║  Hafta-4 ile BAĞLANTI YOK — tamamen bağımsız çalışır                ║
╚══════════════════════════════════════════════════════════════════════╝

BAŞLATMAK İÇİN:
    cd Hafta-5/scripts
    python train_v5.py

Eğitimi durdurmak için: CTRL+C  (en iyi ağırlık zaten kaydedilmiş olur)
"""

import os
import time
import json
from pathlib import Path

import torch
from ultralytics import YOLO

# ══════════════════════════════════════════════════════
#                    KONFİGÜRASYON
# ══════════════════════════════════════════════════════

# ─── Dizinler ────────────────────────────────────────
# Hafta-4'e bağımlılık yok — her şey Hafta-5 içinden alınır
BASE_DIR    = Path(__file__).resolve().parent.parent.parent   # ARCHISAFE/
DATA_YAML   = BASE_DIR / "PPE" / "Fall-Detecetion-1" / "data.yaml"
MODEL_PATH  = Path(__file__).resolve().parent / "yolo26n.pt"  # Hafta-5/scripts/yolo26n.pt
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# ─── Eğitim Parametreleri — RTX 5070 Ti (16 GB VRAM) ─
EPOCHS        = 125     # ~3.5-4 saat (5070 Ti)
IMGSZ         = 512     # 640→512: epoch başına %36 hız kazanımı, kalite kaybı minimal
BATCH         = 32      # 512px'te batch 32 sığar (16GB VRAM)
WORKERS       = 4       # Windows'ta 8 worker deadlock yapabiliyor; 4 güvenli

# ─── Overfit Önleme ──────────────────────────────────
PATIENCE      = 25      # 125 epoch için uygun sabır
DROPOUT       = 0.1
WEIGHT_DECAY  = 0.0005
LABEL_SMOOTHING = 0.1

# ─── Öğrenme Oranı ───────────────────────────────────
LR0           = 0.01
LRF           = 0.00005 # 200 epoch için çok küçük final LR — hassas ince ayar
MOMENTUM      = 0.937
WARMUP_EPOCHS = 5

# ─── Veri Artırma (Augmentation) ─────────────────────
AUGMENT_CONFIG = dict(
    hsv_h        = 0.02,
    hsv_s        = 0.8,
    hsv_v        = 0.5,
    degrees      = 10.0,
    translate    = 0.15,
    scale        = 0.6,
    shear        = 3.0,
    perspective  = 0.0002,
    flipud       = 0.05,
    fliplr       = 0.5,
    mosaic       = 1.0,
    mixup        = 0.15,
    copy_paste   = 0.1,
    close_mosaic = 12,   # Son 12 epoch'ta mozaiği kapat
    erasing      = 0.4,
    auto_augment = "randaugment",
)

PROJECT_NAME  = "ARCHISAFE_v5"
RUN_NAME      = "yolo26n_200ep"

# ══════════════════════════════════════════════════════
#              SINIF BİLGİLERİ (13 SINIF)
# ══════════════════════════════════════════════════════

CLASS_NAMES = {
    0:  "Ear-protection",
    1:  "Fall Detection",
    2:  "gloves",
    3:  "hardhat",
    4:  "mask",
    5:  "no_arm_sleeve",
    6:  "person",
    7:  "shoes",
    8:  "ungloves",
    9:  "unhardhat",
    10: "unmask",
    11: "unvest",
    12: "vest",
}

# ══════════════════════════════════════════════════════
#                    GPU TESPİTİ
# ══════════════════════════════════════════════════════

def setup_device():
    try:
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            capability  = torch.cuda.get_device_capability(0)
            torch.zeros(1).cuda()
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✅ GPU Aktif  : {device_name} (sm_{capability[0]}{capability[1]})")
            print(f"   VRAM       : {vram:.1f} GB")

            # VRAM'e göre batch boyutunu öner
            if vram >= 16:
                print(f"   💡 İpucu   : {vram:.0f}GB VRAM var, BATCH=16 deneyebilirsin")
            return 0
        else:
            print("⚠️  CUDA bulunamadı → CPU kullanılacak (çok yavaş olabilir)")
            return "cpu"
    except RuntimeError as e:
        print(f"❌ GPU Hatası: {e} → CPU'ya geçiliyor")
        return "cpu"


# ══════════════════════════════════════════════════════
#             PER-CLASS mAP TAKİBİ (CALLBACK)
# ══════════════════════════════════════════════════════

class DetailedMetricsMonitor:
    """
    Her epoch sonunda sınıf bazında metrik takibi yapar.
    Özellikle sorunlu sınıfları (gözlük, maske, yelek) izler.
    """
    WATCH_CLASSES = {
        "Ear-protection", "gloves", "mask", "unmask",
        "ungloves", "unvest", "vest", "unhardhat"
    }

    def __init__(self, gap_threshold=0.4):
        self.gap_threshold = gap_threshold
        self.epoch_history = []
        self.best_map50    = 0.0

    def on_fit_epoch_end(self, trainer):
        metrics = trainer.metrics
        epoch   = trainer.epoch + 1

        map50    = metrics.get("metrics/mAP50(B)", 0)
        map5095  = metrics.get("metrics/mAP50-95(B)", 0)
        prec     = metrics.get("metrics/precision(B)", 0)
        recall   = metrics.get("metrics/recall(B)", 0)
        box_loss = metrics.get("train/box_loss", 0)
        cls_loss = metrics.get("train/cls_loss", 0)

        is_best = map50 > self.best_map50
        if is_best:
            self.best_map50 = map50

        self.epoch_history.append({
            "epoch": epoch,
            "mAP50": map50,
            "mAP50-95": map5095,
            "precision": prec,
            "recall": recall,
        })

        # Her 10 epoch'ta özet yazdır
        if epoch % 10 == 0 or is_best:
            marker = "🏆 YENİ EN İYİ" if is_best else f"Epoch {epoch:3d}"
            print(f"\n  [{marker}] mAP50={map50:.4f} | mAP50-95={map5095:.4f} "
                  f"| P={prec:.3f} | R={recall:.3f} "
                  f"| BoxLoss={box_loss:.4f} | ClsLoss={cls_loss:.4f}")

    def save_history(self, save_dir: Path):
        """Eğitim tarihçesini JSON olarak kaydet."""
        path = save_dir / "metrics_history.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.epoch_history, f, indent=2, ensure_ascii=False)
        print(f"\n📊 Metrik tarihçesi kaydedildi: {path}")


# ══════════════════════════════════════════════════════
#                    EĞİTİM
# ══════════════════════════════════════════════════════

def train():
    print("\n" + "═" * 70)
    print("        ARCHISAFE — 5. Hafta / YOLO12-Small Gelişmiş Eğitimi")
    print("═" * 70)

    # ─── Dosya kontrolleri ───────────────────────────
    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Veri seti bulunamadı:\n  {DATA_YAML}\n"
            f"  PPE/Fall-Detecetion-1/data.yaml dosyası mevcut olmalı!"
        )
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Base model bulunamadı:\n  {MODEL_PATH}\n"
            f"  Hafta-5/scripts/yolo26n.pt dosyası gerekli!"
        )

    # ─── Sınıf bilgisi yazdır ────────────────────────
    print(f"\n📦 Veri Seti  : {DATA_YAML}")
    print(f"🤖 Base Model : {MODEL_PATH}  ← yolo26n (Hafta-4'ten bağımsız)")
    print(f"📁 Çıktı      : {RESULTS_DIR / PROJECT_NAME / RUN_NAME}")

    print(f"\n📋 Eğitilecek Sınıflar ({len(CLASS_NAMES)} adet):")
    for idx, name in CLASS_NAMES.items():
        tag = " ⚠️  (sorunlu)" if name in DetailedMetricsMonitor.WATCH_CLASSES else ""
        print(f"   [{idx:2d}] {name}{tag}")

    print(f"\n⚙️  Eğitim Ayarları (RTX 5070 Ti optimize):")
    print(f"   Base Model      : yolo26n.pt  (Hafta-4'ten BAĞIMSIZ)")
    print(f"   Epochs          : {EPOCHS}")
    print(f"   Early Stopping  : {PATIENCE} epoch sabır")
    print(f"   Batch           : {BATCH}  (5070 Ti 16GB — gerekirse 16'ya düşür)")
    print(f"   Workers         : {WORKERS}  (CPU dataloader thread)")
    print(f"   Dropout         : {DROPOUT}")
    print(f"   Warmup          : {WARMUP_EPOCHS} epoch")
    print(f"   LR              : {LR0} → {LRF} (cosine decay)")
    print(f"   Mosaic kapat    : Son {AUGMENT_CONFIG['close_mosaic']} epoch")
    print(f"   ImgSz           : {IMGSZ}×{IMGSZ}")
    print(f"   AMP             : ✅ Mixed Precision (FP16, hızlı)")
    print("═" * 70)

    device = setup_device()

    # ─── Model yükle ────────────────────────────────────
    model = YOLO(str(MODEL_PATH))

    # ─── Metrik monitörü ekle ────────────────────────────
    monitor = DetailedMetricsMonitor(gap_threshold=0.4)
    model.add_callback("on_fit_epoch_end", monitor.on_fit_epoch_end)

    # ─── Eğitim başlat ──────────────────────────────────
    print(f"\n🚀 Eğitim başlıyor ({EPOCHS} epoch)...\n"
          f"   (CTRL+C ile durdurabilirsin — en iyi ağırlık otomatik kaydedilir)\n")
    start_time = time.time()

    results = model.train(
        data             = str(DATA_YAML),
        epochs           = EPOCHS,
        imgsz            = IMGSZ,
        batch            = BATCH,
        device           = device,
        workers          = WORKERS,
        # Anti-overfitting
        patience         = PATIENCE,
        weight_decay     = WEIGHT_DECAY,
        label_smoothing  = LABEL_SMOOTHING,
        dropout          = DROPOUT,
        # Augmentation
        **AUGMENT_CONFIG,
        # LR
        lr0              = LR0,
        lrf              = LRF,
        momentum         = MOMENTUM,
        warmup_epochs    = WARMUP_EPOCHS,
        # Kayıt
        project          = str(RESULTS_DIR / PROJECT_NAME),
        name             = RUN_NAME,
        exist_ok         = True,
        save             = True,
        save_period      = 10,   # Her 10 epoch'ta checkpoint
        plots            = True,
        verbose          = True,
        val              = True,
        amp              = True,   # FP16 — 5070 Ti'de ~2× hız
        # Hız ve kararlılık
        cos_lr           = True,   # Cosine LR decay
        nbs              = 64,     # Nominal batch normalizasyonu
        cache            = "disk", # 'ram' Windows'ta 8+ worker ile deadlock yapıyor; disk güvenli & deterministik
        rect             = False,  # Dikdörtgen eğitim kapalı (mosaic ile uyumsuz)
        multi_scale      = False,  # Sabit imgsz, stabil eğitim
    )

    # ─── Özet ────────────────────────────────────────────
    elapsed = time.time() - start_time
    h, m, s = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)

    best_run_dir = RESULTS_DIR / PROJECT_NAME / RUN_NAME

    print("\n" + "═" * 70)
    print("✅ EĞİTİM TAMAMLANDI!")
    print(f"   Toplam Süre    : {h}s {m}dk {s}sn")
    print(f"   Base Model     : yolo26n.pt")
    print(f"   mAP50          : {results.results_dict.get('metrics/mAP50(B)', 0):.4f}")
    print(f"   mAP50-95       : {results.results_dict.get('metrics/mAP50-95(B)', 0):.4f}")
    print(f"   Precision      : {results.results_dict.get('metrics/precision(B)', 0):.4f}")
    print(f"   Recall         : {results.results_dict.get('metrics/recall(B)', 0):.4f}")

    # Metrik tarihçesini kaydet
    monitor.save_history(best_run_dir)

    best = best_run_dir / "weights" / "best.pt"
    print(f"\n💾 En İyi Ağırlık: {best}")
    print("\n🎯 Sonraki adım:")
    print("   python test_live_v5.py    → canlı kamera (tam özellikli)")
    print("   python test_live_v5.py --model", best)
    print("═" * 70)

    return best


if __name__ == "__main__":
    train()
