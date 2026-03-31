# 📁 4. Hafta — YOLO12-Small Final Eğitim ve Canlı Test

## 🎯 Hafta Hedefi

Karşılaştırma çalışmasında **şampiyon** olan `YOLO12-Small` modelini tam kapasite ile eğit,
ardından hem resim hem canlı kamera üzerinde test et.

---

## 📂 Klasör Yapısı

```
Hafta-4/
├── scripts/
│   ├── train_final.py     # 1. ÇALIŞTIR → Final modeli eğit (50 epoch)
│   ├── test_image.py      # 2. ÇALIŞTIR → Resimle test et
│   └── test_live.py       # 3. ÇALIŞTIR → Canlı kamera testi
├── test_images/           # Test etmek istediğin görselleri buraya koy
├── results/
│   ├── images/            # test_image.py çıktıları burada
│   └── live/              # test_live.py screenshot ve video kayıtları
└── README.md
```

---

## 🚀 Kullanım Sırası

### 1️⃣ Modeli Eğit
```bash
cd Hafta-4/scripts
python train_final.py
```
> ⏱️ Yaklaşık süre: RTX 5070 Ti'de ~30-45 dakika (50 epoch)

**Hiperparametreler:**
| Parametre | Değer | Açıklama |
|-----------|-------|----------|
| Epochs | 50 | Karşılaştırmada 3 idi, şimdi tam eğitim |
| Batch | 8 | RTX 5070 Ti için optimize |
| Image Size | 640×640 | Standart YOLO |
| LR | 0.01 → 0.001 | Cosine decay |
| Patience | 15 | Early stopping |
| AMP | ✅ | Mixed precision (hız + VRAM tasarrufu) |

Eğitim tamamlandığında en iyi ağırlık burada olacak:
```
results/ARCHISAFE_Final/YOLO12s_50ep/weights/best.pt
```

---

### 2️⃣ Resimle Test Et
```bash
# test_images/ klasöründeki ilk görseli otomatik kullanır
python test_image.py

# Belirli bir görsel
python test_image.py --image ../test_images/fabrika.jpg

# Güven eşiği ayarla
python test_image.py --image ../test_images/fabrika.jpg --conf 0.4
```

**Çıktı:** `results/images/result_<görsel_adı>.jpg`

---

### 3️⃣ Canlı Kamera Testi
```bash
# Varsayılan webcam
python test_live.py

# 2. kamera
python test_live.py --camera 1

# Başlarken kayda al
python test_live.py --record
```

**Pencere Kontrolleri:**
| Tuş | Eylem |
|-----|-------|
| `Q` / `ESC` | Çıkış |
| `S` | Anlık screenshot kaydet |
| `R` | Video kaydı başlat/durdur |
| `+` | Conf eşiğini artır (+0.05) |
| `-` | Conf eşiğini azalt (-0.05) |

**Çıktılar:** `results/live/` klasörüne kaydedilir.

---

## ⚠️ Uyarılar

- Eğitimden önce `train_final.py` çalıştırılmalı, ardından test scriptleri kullanılabilir.
- `test_images/` klasörüne `.jpg`, `.png`, `.jpeg` formatında görsel ekleyebilirsiniz.
- Windows'ta kamera açılamazsa `--camera 1` veya `--camera 2` deneyin.
