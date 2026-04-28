# ARCHISAFE — 6. Hafta

## 🎯 Hedef
**YOLOv12 Nano** modeliyle PPE (Kişisel Koruyucu Ekipman) ve **Düşme Tespiti**

## 📋 Sınıflar (13 adet)
| ID | Sınıf | Tür |
|----|-------|-----|
| 0 | Ear-protection | ✅ KKD |
| 1 | Fall Detection - v4 resized640_aug3x-ACCURATE | 🚨 Düşme |
| 2 | gloves | ✅ KKD |
| 3 | hardhat | ✅ KKD |
| 4 | mask | ✅ KKD |
| 5 | no_arm_sleeve | ⚠️ İhlal |
| 6 | person | 👤 Kişi |
| 7 | shoes | ✅ KKD |
| 8 | ungloves | ⚠️ İhlal |
| 9 | unhardhat | ⚠️ İhlal |
| 10 | unmask | ⚠️ İhlal |
| 11 | unvest | ⚠️ İhlal |
| 12 | vest | ✅ KKD |

## 🚀 Hızlı Başlangıç

### 1. Model İndir
```bash
cd Hafta-6/scripts
python setup_v6.py
```

### 2. Eğit
```bash
python train_v6.py
```
> ⏱ RTX 5070 Ti: ~45-60 dk (150 epoch, nano hızlı!)

### 3. Canlı Test
```bash
python test_live_v6.py
python test_live_v6.py --model results/ARCHISAFE_v6/yolo12n_150ep/weights/best.pt
```

## 🏗️ Mimari — Ayrı Fonksiyonlar
```
detect_ppe(boxes, names)   → PPEResult
    .present_ppe   : {sınıf: güven}
    .violations    : {sınıf: güven}
    .missing_ppe   : [eksik Türkçe isimler]
    .risk_level    : "OK" | "WARNING" | "DANGER"

FallDetector.detect(boxes, names)  → FallResult
    .detected      : bool
    .confidence    : float
    .box           : (x1,y1,x2,y2)
    .alarm_active  : bool  ← 3 ardışık frame düşme = gerçek alarm
```

## ⌨️ Klavye Kısayolları
| Tuş | Eylem |
|-----|-------|
| Q / ESC | Çıkış |
| S | Screenshot al |
| R | Kayıt başlat/durdur |
| +/- | Güven eşiği ±0.05 |
| H | HUD aç/kapat |
| P | Duraklat/Devam |
| F | Fall-Only modu |
| D | Demo/İstatistik paneli |

## 📁 Dizin Yapısı
```
Hafta-6/
├── scripts/
│   ├── train_v6.py       ← EĞİTİM
│   ├── test_live_v6.py   ← CANLI TEST
│   ├── setup_v6.py       ← Model indir
│   └── yolo12n.pt        ← Base model (setup ile indirilir)
└── results/
    └── ARCHISAFE_v6/
        └── yolo12n_150ep/
            └── weights/
                ├── best.pt    ← KULLAN
                └── last.pt
```
