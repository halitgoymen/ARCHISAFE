"""
╔══════════════════════════════════════════════════════════════════════════╗
║          ARCHISAFE — Final Haftası Eğitim Scripti                       ║
║          Base: yolo12n.pt (NANO) | 200 Epoch | RTX 5070 Ti Optimize    ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Sınıflar (13 adet):                                                     ║
║    0: Ear-protection              7: shoes                               ║
║    1: Fall Detection              8: ungloves                            ║
║    2: gloves                      9: unhardhat                           ║
║    3: hardhat                    10: unmask                              ║
║    4: mask                       11: unvest                              ║
║    5: no_arm_sleeve              12: vest                                ║
║    6: person                                                              ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Başlatmak için:                                                          ║
║    cd Final_Haftasi/scripts                                               ║
║    python train_final.py                                                  ║
║  Dur: CTRL+C  (en iyi ağırlık otomatik kaydedilmiş olur)                ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import os
import time
import json
from pathlib import Path

import torch
from ultralytics import YOLO

# ══════════════════════════════════════════════════════════════════
#                         DİZİNLER
# ══════════════════════════════════════════════════════════════════

SCRIPT_DIR  = Path(__file__).resolve().parent   # Final_Haftasi/scripts/
FINAL_DIR   = SCRIPT_DIR.parent                 # Final_Haftasi/
BASE_DIR    = FINAL_DIR.parent                  # ARCHISAFE/
DATA_YAML   = BASE_DIR / "PPE" / "v1" / "data.yaml"
RESULTS_DIR = FINAL_DIR / "results"

# Model: scriptin yanında yolo12n.pt varsa onu kullan, yoksa ultralytics indirir
_LOCAL = SCRIPT_DIR / "yolo12n.pt"
MODEL_PATH = str(_LOCAL) if _LOCAL.exists() else "yolo12n.pt"

# ══════════════════════════════════════════════════════════════════
#                   EĞİTİM PARAMETRELERİ
# ══════════════════════════════════════════════════════════════════

EPOCHS   = 200      # Maksimum; early stopping zaten keser
IMGSZ    = 640
BATCH    = 32       # 16→32: gradient acc. azalır, GPU daha verimli ama tam dolu değil
WORKERS  = 6        # 4→6: CPU data loading hızlanır, sistem kasılmaz
DEVICE   = 0        # GPU 0

# ── Early Stopping ────────────────────────────────────────────────
# 40 epoch boyunca mAP50 iyileşmezse dur
# 200 epoch içinde genellikle 80-130. epoch'ta en iyi nokta yakalanır
PATIENCE = 40

# ── Overfit Önleme ────────────────────────────────────────────────
DROPOUT         = 0.0      # Nano backbone → dropout kapalı
WEIGHT_DECAY    = 0.0005
LABEL_SMOOTHING = 0.1

# ── Öğrenme Oranı ─────────────────────────────────────────────────
LR0           = 0.01
LRF           = 0.01       # Nano için sabit yüksek LR daha iyi
MOMENTUM      = 0.937
WARMUP_EPOCHS = 3

# ── Augmentation ──────────────────────────────────────────────────
AUGMENT_CONFIG = dict(
    hsv_h        = 0.015,
    hsv_s        = 0.7,
    hsv_v        = 0.4,
    degrees      = 5.0,
    translate    = 0.1,
    scale        = 0.5,
    shear        = 2.0,
    perspective  = 0.0001,
    flipud       = 0.05,
    fliplr       = 0.5,
    mosaic       = 1.0,
    mixup        = 0.1,
    copy_paste   = 0.1,
    close_mosaic = 20,     # son 20 epoch mozaiksiz → ince ayar
    erasing      = 0.3,
    auto_augment = "randaugment",
)

PROJECT_NAME = "ARCHISAFE_Final"
RUN_NAME     = "yolo12n_200ep"

# ══════════════════════════════════════════════════════════════════
#                     SINIF BİLGİLERİ
# ══════════════════════════════════════════════════════════════════

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

WATCH_CLASSES = {
    "Ear-protection",
    "Fall Detection",
    "gloves",
    "mask",
    "ungloves",
    "unvest",
    "unmask",
}

# ══════════════════════════════════════════════════════════════════
#                        GPU TESPİTİ
# ══════════════════════════════════════════════════════════════════

def setup_device() -> int | str:
    try:
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            cap  = torch.cuda.get_device_capability(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            torch.zeros(1).cuda()
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


# ══════════════════════════════════════════════════════════════════
#               DETAYLI METRİK İZLEYİCİ (Callback)
# ══════════════════════════════════════════════════════════════════

class DetailedMetricsMonitor:
    def __init__(self):
        self.epoch_history = []
        self.best_map50    = 0.0
        self.best_epoch    = 0
        self._start        = time.time()

    def on_fit_epoch_end(self, trainer):
        m       = trainer.metrics
        epoch   = trainer.epoch + 1

        map50   = m.get("metrics/mAP50(B)",     0.0)
        map5095 = m.get("metrics/mAP50-95(B)",  0.0)
        prec    = m.get("metrics/precision(B)",  0.0)
        recall  = m.get("metrics/recall(B)",     0.0)
        box_l   = m.get("train/box_loss",        0.0)
        cls_l   = m.get("train/cls_loss",        0.0)
        dfl_l   = m.get("train/dfl_loss",        0.0)

        is_best = map50 > self.best_map50
        if is_best:
            self.best_map50 = map50
            self.best_epoch = epoch

        elapsed    = time.time() - self._start
        eta_sn     = elapsed / epoch * (EPOCHS - epoch)
        h, rem     = divmod(int(eta_sn), 3600)
        mi         = rem // 60

        self.epoch_history.append({
            "epoch":     epoch,
            "mAP50":     round(map50,   4),
            "mAP50-95":  round(map5095, 4),
            "precision": round(prec,    4),
            "recall":    round(recall,  4),
            "box_loss":  round(box_l,   4),
            "cls_loss":  round(cls_l,   4),
            "dfl_loss":  round(dfl_l,   4),
        })

        if epoch % 10 == 0 or is_best:
            marker = f"🏆 YENİ EN İYİ  (epoch {epoch})" if is_best else f"Epoch {epoch:4d}"
            print(
                f"\n  [{marker}] "
                f"mAP50={map50:.4f} | mAP50-95={map5095:.4f} | "
                f"P={prec:.3f} | R={recall:.3f} | "
                f"BoxL={box_l:.4f} | ClsL={cls_l:.4f} | DflL={dfl_l:.4f} | "
                f"ETA {h}s{mi:02d}dk"
            )

        if epoch > 10 and recall < 0.05:
            print(f"  ⚠️  [Epoch {epoch}] Recall çok düşük ({recall:.3f}) — model öğrenemiyor!")

    def save_history(self, save_dir: Path):
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / "metrics_history.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.epoch_history, f, indent=2, ensure_ascii=False)
        print(f"\n📊 Metrik tarihçesi kaydedildi: {path}")
        print(f"   En iyi mAP50 = {self.best_map50:.4f}  (epoch {self.best_epoch})")


# ══════════════════════════════════════════════════════════════════
#                          EĞİTİM
# ══════════════════════════════════════════════════════════════════

def train():
    print("\n" + "═" * 72)
    print("        ARCHISAFE — Final Haftası / YOLOv12-Nano Eğitimi")
    print("═" * 72)

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"\n❌ Veri seti bulunamadı: {DATA_YAML}\n"
            f"   PPE/v1/data.yaml mevcut olmalı!"
        )

    print(f"\n📦 Veri Seti  : {DATA_YAML}")
    print(f"🤖 Base Model : {MODEL_PATH}")
    print(f"📁 Çıktı      : {RESULTS_DIR / PROJECT_NAME / RUN_NAME}")

    print(f"\n📋 Eğitilecek Sınıflar ({len(CLASS_NAMES)} adet):")
    for idx, name in CLASS_NAMES.items():
        tag = " ⚠️  (odak sınıf)" if name in WATCH_CLASSES else ""
        print(f"   [{idx:2d}] {name}{tag}")

    print(f"\n⚙️  Eğitim Ayarları (RTX 5070 Ti optimize, YOLOv12 Nano):")
    print(f"   Model           : yolo12n.pt  (Nano)")
    print(f"   Epochs          : {EPOCHS}  (max)")
    print(f"   Early Stopping  : patience={PATIENCE}  (otomatik durdurma)")
    print(f"   ImgSz           : {IMGSZ}×{IMGSZ}")
    print(f"   Batch           : {BATCH}  (nbs={BATCH}, accumulation yok)")
    print(f"   Workers         : {WORKERS}")
    print(f"   LR              : {LR0} → {LRF}  (cosine decay)")
    print(f"   Warmup          : {WARMUP_EPOCHS} epoch")
    print(f"   Label Smooth    : {LABEL_SMOOTHING}")
    print(f"   Mosaic kapat    : Son {AUGMENT_CONFIG['close_mosaic']} epoch")
    print(f"   AMP             : ✅ Mixed Precision (FP16)")
    print("═" * 72)

    device = setup_device()

    model   = YOLO(MODEL_PATH)
    monitor = DetailedMetricsMonitor()
    model.add_callback("on_fit_epoch_end", monitor.on_fit_epoch_end)

    print(f"\n🚀 Eğitim başlıyor...\n"
          f"   (CTRL+C ile güvenle durdurulabilir — best.pt otomatik kaydedilir)\n")
    start = time.time()

    results = model.train(
        data            = str(DATA_YAML),
        epochs          = EPOCHS,
        imgsz           = IMGSZ,
        batch           = BATCH,
        device          = device,
        workers         = WORKERS,

        patience        = PATIENCE,
        weight_decay    = WEIGHT_DECAY,
        dropout         = DROPOUT,
        label_smoothing = LABEL_SMOOTHING,

        **AUGMENT_CONFIG,

        lr0             = LR0,
        lrf             = LRF,
        momentum        = MOMENTUM,
        warmup_epochs   = WARMUP_EPOCHS,
        cos_lr          = True,

        cls             = 0.5,

        project         = str(RESULTS_DIR / PROJECT_NAME),
        name            = RUN_NAME,
        exist_ok        = True,
        save            = True,
        save_period     = 10,
        plots           = True,
        verbose         = True,
        val             = True,
        amp             = True,

        nbs             = 32,        # BATCH ile eşleşti, gradient accumulation yok
        cache           = "disk",    # RAM yetersiz (21GB > 16GB free), disk cache daha güvenli
        rect            = False,
        multi_scale     = False,
    )

    elapsed = time.time() - start
    h, rem  = divmod(int(elapsed), 3600)
    mi, s   = divmod(rem, 60)

    run_dir = RESULTS_DIR / PROJECT_NAME / RUN_NAME
    best_pt = run_dir / "weights" / "best.pt"

    monitor.save_history(run_dir)

    print("\n" + "═" * 72)
    print("✅ EĞİTİM TAMAMLANDI!")
    print(f"   Toplam Süre  : {h}s {mi:02d}dk {s:02d}sn")
    print(f"   mAP50        : {results.results_dict.get('metrics/mAP50(B)', 0):.4f}")
    print(f"   mAP50-95     : {results.results_dict.get('metrics/mAP50-95(B)', 0):.4f}")
    print(f"   Precision    : {results.results_dict.get('metrics/precision(B)', 0):.4f}")
    print(f"   Recall       : {results.results_dict.get('metrics/recall(B)', 0):.4f}")
    print(f"\n   En İyi Epoch : {monitor.best_epoch}  (mAP50={monitor.best_map50:.4f})")
    print(f"   💾 best.pt   : {best_pt}")
    print("═" * 72)

    return best_pt


if __name__ == "__main__":
    train()
