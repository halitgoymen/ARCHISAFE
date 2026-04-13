# ARCHISAFE — 5. Hafta README
## 🏗️ Genel Bakış

5. Hafta, Hafta-4'teki sorunları gidermek ve sistemi gerçek sahaya hazır hale getirmek için geliştirilmiştir.

---

## 🆕 Hafta-4'e Göre Yenilikler

### `train_v5.py` (Eğitim)
| Parametre | Hafta-4 | Hafta-5 | Açıklama |
|-----------|---------|---------|----------|
| Epoch | 30 | **150** | Tam öğrenme için 5× artırıldı |
| Patience | 10 | **25** | Erken durmayı engeller |
| Dropout | 0.0 | **0.1** | Overfit azaltma |
| Warmup | 3 ep | **5 ep** | Daha stabil başlangıç |
| Final LR | 0.001 | **0.0001** | Daha hassas ince ayar |
| Copy-Paste Aug | ❌ | ✅ 0.1 | Küçük nesne tespiti |
| Perspective Aug | ❌ | ✅ 0.0002 | Kamera açısı çeşitlendirme |
| AutoAugment | ❌ | ✅ randaugment | Otomatik augmentation |
| Cosine LR | ❌ | ✅ | Daha stabil LR düşüşü |

### `test_live_v5.py` (Canlı Kamera)
| Özellik | Hafta-4 | Hafta-5 |
|---------|---------|---------|
| Kişi kutusu | Tespit edilince | **Her zaman kalır** (PERSISTENT) |
| Eksik KKD listesi | ❌ | ✅ Ekranda gösterilir |
| Güven oranı paneli | ❌ | ✅ Sağda bar grafik |
| Duraklat (P tuşu) | ❌ | ✅ |
| Kişi takibi | ❌ | ✅ IoU tabanlı tracker |
| Tahmin modu | ❌ | ✅ Köşe L-kutusu |
| HUD toggle (H) | ❌ | ✅ |

---

## 🚀 Kullanım

### 1. Eğitim (Önce Çalıştır)
```bash
cd Hafta-5/scripts
python train_v5.py
```
> ⏱ Tahmini süre: RTX 5070 Ti ile ~3–6 saat (150 epoch)

### 2. Canlı Kamera Testi
```bash
python test_live_v5.py
# veya
python test_live_v5.py --conf 0.25   # Daha fazla tespit
python test_live_v5.py --record      # Video kaydet
```

> 💡 Eğitim bitmeden de Hafta-4 modelini kullanır (otomatik fallback)

---

## ⌨️ Klavye Kontrolleri (Live)

| Tuş | İşlev |
|-----|-------|
| Q / ESC | Çıkış |
| S | Screenshot kaydet |
| R | Kayıt başlat/durdur |
| + / - | Güven eşiği artır/azalt |
| H | HUD göster/gizle |
| P | Duraklat / Devam |

---

## 📊 Sınıflar (13 adet)

| ID | Sınıf | Tip |
|----|-------|-----|
| 0 | Ear-protection | ✅ KKD |
| 1 | Fall Detection | 🚨 Alarm |
| 2 | gloves | ✅ KKD |
| 3 | hardhat | ✅ KKD |
| 4 | mask | ✅ KKD |
| 5 | no_arm_sleeve | ⚠️ İhlal |
| 6 | person | 👤 Takip |
| 7 | shoes | ✅ KKD |
| 8 | ungloves | ⚠️ İhlal |
| 9 | unhardhat | ⚠️ İhlal |
| 10 | unmask | ⚠️ İhlal |
| 11 | unvest | ⚠️ İhlal |
| 12 | vest | ✅ KKD |

---

## 📁 Klasör Yapısı

```
Hafta-5/
├── scripts/
│   ├── train_v5.py        # Eğitim (150 epoch)
│   └── test_live_v5.py    # Canlı kamera (gelişmiş)
├── results/
│   ├── ARCHISAFE_v5/
│   │   └── YOLO12s_150ep_full/
│   │       ├── weights/
│   │       │   ├── best.pt      ← Ana model
│   │       │   └── last.pt
│   │       └── metrics_history.json
│   └── live/              # Screenshot ve video kayıtları
└── README.md
```
