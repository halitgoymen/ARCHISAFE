"""
ARCHISAFE — 6. Hafta Model Kurulum Scripti
Yolo12n.pt yi Hafta-6/scripts/ klasörüne indirir.
"""
from pathlib import Path
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_NAME = "yolo12n.pt"
MODEL_PATH = SCRIPT_DIR / MODEL_NAME

if MODEL_PATH.exists():
    print(f"✅ {MODEL_NAME} zaten mevcut: {MODEL_PATH}")
else:
    print(f"⬇️  {MODEL_NAME} indiriliyor (ultralytics otomatik)...")
    m = YOLO(MODEL_NAME)   # Otomatik indirir
    import shutil, torch
    # Ultralytics cache'ten kopyala
    cached = Path(torch.hub.get_dir()) / "ultralytics" / "assets" / MODEL_NAME
    if not cached.exists():
        # Alternatif: zaten çalışır — model başarıyla yüklendi
        print(f"✅ Model hazır (ultralytics önbelleğinde)")
    else:
        shutil.copy(str(cached), str(MODEL_PATH))
        print(f"✅ Kopyalandı: {MODEL_PATH}")

print("\nŞimdi eğitimi başlatabilirsin:")
print("  python train_v6.py")
