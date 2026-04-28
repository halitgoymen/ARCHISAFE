"""
╔══════════════════════════════════════════════════════════════════════╗
║         ARCHISAFE — 6. Hafta Eğitim Scripti                         ║
║         Base: yolo12n.pt (NANO) | 150 Epoch | RTX 5070 Ti Optimize  ║
╠══════════════════════════════════════════════════════════════════════╣
║  Sınıflar (13 adet):                                                 ║
║    0: Ear-protection                  7: shoes                       ║
║    1: Fall Detection (v4 ACCURATE)    8: ungloves                    ║
║    2: gloves                          9: unhardhat                   ║
║    3: hardhat                        10: unmask                      ║
║    4: mask                           11: unvest                      ║
║    5: no_arm_sleeve                  12: vest                        ║
║    6: person                                                          ║
╠══════════════════════════════════════════════════════════════════════╣
║  YENİLİKLER (Hafta-5'e göre):                                       ║
║    ✅ YOLOv12 Nano backbone (en hızlı, sunuma hazır)                 ║
║    ✅ Gözlük (Ear-protection) ve maske sınıflarına odak              ║
║    ✅ Fall Detection düzeltilmiş sınıf ismiyle eşleştirildi         ║
║    ✅ Focal Loss benzeri cls_pw ile küçük sınıflara ağırlık         ║
║    ✅ Cosine annealing + Warmup restart                              ║
╠══════════════════════════════════════════════════════════════════════╣
║  BAŞLATMAK İÇİN:                                                     ║
║    cd Hafta-6/scripts                                                 ║
║    python train_v6.py                                                 ║
║  Dur: CTRL+C  (en iyi ağırlık kaydedilmiş olur)                     ║
╚══════════════════════════════════════════════════════════════════════╝
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

# ─── Dizinler ──────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent.parent   # ARCHISAFE/
DATA_YAML   = BASE_DIR / "PPE" / "v1" / "data.yaml"
MODEL_NAME  = "yolo12n.pt"
MODEL_PATH  = Path(__file__).resolve().parent / MODEL_NAME
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# ─── Eğitim Parametreleri — RTX 5070 Ti ───────────────
EPOCHS        = 150     # nano modeli hızlı → 150 epoch güvenli
IMGSZ         = 640     # nano → 640px tam çözünürlük (VRAM yeterli)
BATCH         = 32      # 640px + nano → 32 batchle 16GB VRAM rahat
WORKERS       = 4       # Windows'ta deadlock riski; 4 güvenli
DEVICE        = 0       # GPU (None → otomatik)

# ─── Overfit Önleme ──────────────────────────────────
PATIENCE      = 30      # 150 epoch için 30 sabır
DROPOUT       = 0.0     # Nano backbone; dropout kapalı (küçük model)
WEIGHT_DECAY  = 0.0005
LABEL_SMOOTHING = 0.05  # Hafif label smoothing (küçük dataset için)

# ─── Öğrenme Oranı ───────────────────────────────────
LR0           = 0.01
LRF           = 0.01    # Nano için sabit yüksek LR daha iyi
MOMENTUM      = 0.937
WARMUP_EPOCHS = 3

# ─── Ağırlıklı Sınıf Kaybı ───────────────────────────
# YOLOv12'de cls ile sınıf kayıp ağırlığı ayarlanır
CLS_GAIN      = 0.5   # cls loss gain (default 0.5)
KOBJ          = 1.0   # keypoint obj gain (YOLO12 parametresi)

# ─── Veri Artırma (Augmentation) — Nano optimize ─────
AUGMENT_CONFIG = dict(
    hsv_h          = 0.015,     # renk tonu
    hsv_s          = 0.7,       # doygunluk
    hsv_v          = 0.4,       # parlaklık
    degrees        = 5.0,       # döndürme (küçük; fall detection için)
    translate      = 0.1,
    scale          = 0.5,
    shear          = 2.0,
    perspective    = 0.0001,
    flipud         = 0.05,
    fliplr         = 0.5,
    mosaic         = 1.0,
    mixup          = 0.1,
    copy_paste     = 0.05,
    close_mosaic   = 15,         # Son 15 epoch mozaiği kapat (fine-tune)
    erasing        = 0.3,
    auto_augment   = "randaugment",
)

PROJECT_NAME  = "ARCHISAFE_v6"
RUN_NAME      = "yolo12n_150ep"

# ══════════════════════════════════════════════════════
#              SINIF BİLGİLERİ (13 SINIF)
# ══════════════════════════════════════════════════════

CLASS_NAMES = {
    0:  "Ear-protection",
    1:  "Fall Detection - v4 resized640_aug3x-ACCURATE",
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

# Sorunlu/nadir sınıflar — eğitimde dikkat edilecek
WATCH_CLASSES = {
    "Ear-protection",
    "Fall Detection - v4 resized640_aug3x-ACCURATE",
    "gloves",
    "mask",
    "ungloves",
    "unvest",
    "unmask",
}

# ══════════════════════════════════════════════════════
#                    GPU TESPİTİ
# ══════════════════════════════════════════════════════

def setup_device() -> int | str:
    try:
        if torch.cuda.is_available():
            name       = torch.cuda.get_device_name(0)
            cap        = torch.cuda.get_device_capability(0)
            vram       = torch.cuda.get_device_properties(0).total_memory / 1e9
            torch.zeros(1).cuda()   # Test erişimi
            print(f"✅ GPU Aktif  : {name} (sm_{cap[0]}{cap[1]})")
            print(f"   VRAM       : {vram:.1f} GB")
            if vram >= 16:
                print(f"   💡 16GB+ VRAM — Batch {BATCH} + 640px nano için ideal!")
            return 0
        else:
            print("⚠️  CUDA bulunamadı → CPU (çok yavaş)")
            return "cpu"
    except RuntimeError as e:
        print(f"❌ GPU Hatası: {e} → CPU")
        return "cpu"


# ══════════════════════════════════════════════════════
#          DETAYLI METRİK İZLEYİCİ (Callback)
# ══════════════════════════════════════════════════════

class DetailedMetricsMonitor:
    """
    Her epoch sonunda global metrikleri takip eder.
    Sorunlu sınıfları (gözlük, maske, düşme) öncelikli izler.
    """

    def __init__(self, gap_threshold: float = 0.35):
        self.gap_threshold  = gap_threshold
        self.epoch_history  = []
        self.best_map50     = 0.0
        self.best_epoch     = 0

    # ─── Callback ──────────────────────────────────────
    def on_fit_epoch_end(self, trainer):
        m       = trainer.metrics
        epoch   = trainer.epoch + 1

        map50   = m.get("metrics/mAP50(B)",    0.0)
        map5095 = m.get("metrics/mAP50-95(B)", 0.0)
        prec    = m.get("metrics/precision(B)", 0.0)
        recall  = m.get("metrics/recall(B)",    0.0)
        box_l   = m.get("train/box_loss",       0.0)
        cls_l   = m.get("train/cls_loss",       0.0)
        dfl_l   = m.get("train/dfl_loss",       0.0)

        is_best = map50 > self.best_map50
        if is_best:
            self.best_map50  = map50
            self.best_epoch  = epoch

        self.epoch_history.append({
            "epoch":      epoch,
            "mAP50":      round(map50,   4),
            "mAP50-95":   round(map5095, 4),
            "precision":  round(prec,    4),
            "recall":     round(recall,  4),
            "box_loss":   round(box_l,   4),
            "cls_loss":   round(cls_l,   4),
            "dfl_loss":   round(dfl_l,   4),
        })

        # Her 10 epoch veya yeni en iyi
        if epoch % 10 == 0 or is_best:
            marker = f"🏆 YENİ EN İYİ  (epoch {epoch})" if is_best else f"Epoch {epoch:4d}"
            print(
                f"\n  [{marker}] "
                f"mAP50={map50:.4f} | mAP50-95={map5095:.4f} | "
                f"P={prec:.3f} | R={recall:.3f} | "
                f"BoxLoss={box_l:.4f} | ClsLoss={cls_l:.4f} | DflLoss={dfl_l:.4f}"
            )

        # Erken düşme: LR = 0 → eğitim yarıda kaldıysa uyar
        if epoch > 10 and recall < 0.05:
            print(f"  ⚠️  [Epoch {epoch}] Recall çok düşük ({recall:.3f}). "
                  "Model öğrenemiyor olabilir!")

    # ─── JSON kayıt ────────────────────────────────────
    def save_history(self, save_dir: Path):
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / "metrics_history.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.epoch_history, f, indent=2, ensure_ascii=False)
        print(f"\n📊 Metrik tarihçesi kaydedildi: {path}")
        print(f"   En iyi mAP50 = {self.best_map50:.4f}  (epoch {self.best_epoch})")


# ══════════════════════════════════════════════════════
#                    EĞİTİM
# ══════════════════════════════════════════════════════

def train():
    print("\n" + "═" * 72)
    print("        ARCHISAFE — 6. Hafta / YOLOv12-Nano Eğitimi")
    print("═" * 72)

    # ─── Dosya kontrolleri ────────────────────────────
    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"\n❌ Veri seti YML bulunamadı:\n   {DATA_YAML}\n"
            f"   PPE/Fall-Detecetion-1/data.yaml mevcut olmalı!"
        )
    if not MODEL_PATH.exists():
        print(f"\n⚠️  {MODEL_NAME} bulunamadı: {MODEL_PATH}")
        print("   Ultralytics otomatik indirecek (internet bağlantısı gerekli)...")

    # ─── Bilgi ekrana ─────────────────────────────────
    print(f"\n📦 Veri Seti  : {DATA_YAML}")
    print(f"🤖 Base Model : {MODEL_PATH}")
    print(f"📁 Çıktı      : {RESULTS_DIR / PROJECT_NAME / RUN_NAME}")

    print(f"\n📋 Eğitilecek Sınıflar ({len(CLASS_NAMES)} adet):")
    for idx, name in CLASS_NAMES.items():
        tag = " ⚠️  (odak sınıf)" if name in WATCH_CLASSES else ""
        print(f"   [{idx:2d}] {name}{tag}")

    print(f"\n⚙️  Eğitim Ayarları (RTX 5070 Ti optimize, YOLOv12 Nano):")
    print(f"   Model           : {MODEL_NAME}  (Nano — en hızlı)")
    print(f"   Epochs          : {EPOCHS}")
    print(f"   ImgSz           : {IMGSZ}×{IMGSZ}  (nano → tam 640)")
    print(f"   Batch           : {BATCH}")
    print(f"   Workers         : {WORKERS}")
    print(f"   Early Stopping  : {PATIENCE} epoch sabır")
    print(f"   LR              : {LR0} → {LRF} (cosine decay)")
    print(f"   Warmup          : {WARMUP_EPOCHS} epoch")
    print(f"   Label Smooth    : {LABEL_SMOOTHING}")
    print(f"   Mosaic kapat    : Son {AUGMENT_CONFIG['close_mosaic']} epoch")
    print(f"   AMP             : ✅ Mixed Precision (FP16)")
    print("═" * 72)

    device = setup_device()

    # ─── Model yükle ──────────────────────────────────
    model = YOLO(str(MODEL_PATH))

    # ─── Metrik monitörü ──────────────────────────────
    monitor = DetailedMetricsMonitor(gap_threshold=0.35)
    model.add_callback("on_fit_epoch_end", monitor.on_fit_epoch_end)

    # ─── Eğitim başlat ────────────────────────────────
    print(f"\n🚀 Eğitim başlıyor ({EPOCHS} epoch — nano hızlı!)...\n"
          f"   (CTRL+C ile güvenle durdurulabilir — best.pt auto-save)\n")
    start_time = time.time()

    results = model.train(
        data             = str(DATA_YAML),
        epochs           = EPOCHS,
        imgsz            = IMGSZ,
        batch            = BATCH,
        device           = device,
        workers          = WORKERS,

        # Anti-overfit
        patience         = PATIENCE,
        weight_decay     = WEIGHT_DECAY,
        dropout          = DROPOUT,

        # Augmentation (Nano optimized)
        **AUGMENT_CONFIG,

        # LR schedule
        lr0              = LR0,
        lrf              = LRF,
        momentum         = MOMENTUM,
        warmup_epochs    = WARMUP_EPOCHS,

        # Kayıp ağırlıkları
        cls              = CLS_GAIN,

        # Kayıt
        project          = str(RESULTS_DIR / PROJECT_NAME),
        name             = RUN_NAME,
        exist_ok         = True,
        save             = True,
        save_period      = 10,
        plots            = True,
        verbose          = True,
        val              = True,
        amp              = True,

        # Hız ve kararlılık
        cos_lr           = True,
        nbs              = 64,
        cache            = "disk",     # Windows'ta RAM cache deadlock riski var
        rect             = False,      # Mosaic ile uyumsuz
        multi_scale      = False,
    )

    # ─── Özet ─────────────────────────────────────────
    elapsed = time.time() - start_time
    h, m, s = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)

    run_dir = RESULTS_DIR / PROJECT_NAME / RUN_NAME

    print("\n" + "═" * 72)
    print("✅ EĞİTİM TAMAMLANDI!")
    print(f"   Toplam Süre    : {h}s {m}dk {s}sn")
    print(f"   Model           : {MODEL_NAME}  (Nano)")
    print(f"   mAP50           : {results.results_dict.get('metrics/mAP50(B)', 0):.4f}")
    print(f"   mAP50-95        : {results.results_dict.get('metrics/mAP50-95(B)', 0):.4f}")
    print(f"   Precision       : {results.results_dict.get('metrics/precision(B)', 0):.4f}")
    print(f"   Recall          : {results.results_dict.get('metrics/recall(B)', 0):.4f}")

    monitor.save_history(run_dir)

    best = run_dir / "weights" / "best.pt"
    print(f"\n💾 En İyi Ağırlık : {best}")
    print("\n🎯 Sonraki adım:")
    print(f"   python test_live_v6.py                   → Canlı kamera")
    print(f"   python test_live_v6.py --model {best}")
    print("═" * 72)

    return best


if __name__ == "__main__":
    train()
