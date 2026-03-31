# 📁 1. Hafta — Veri Seti Hazırlığı ve Problem Tanımı

## Yapılanlar

### 🎯 Proje Odak Noktası
ARCHISAFE projesinin temel problemi tanımlandı: Endüstriyel ortamlarda KKD (Kişisel Koruyucu Donanım) tespit ve düşme tespiti.

### 📦 Veri Seti
- **Platform:** Roboflow
- **Workspace:** `muhammets-workspace-bufij`
- **Proje:** `ppe-fall-detecetion-owg28` (Version 1)
- **Yerel Konum:** `../../PPE/Fall-Detecetion-1/`
- **Format:** YOLOv8

### 🏷️ Tespit Sınıfları
Veri setindeki sınıflar `data.yaml` dosyasında tanımlanmıştır:
- Kask (Helmet)
- Güvenlik Yeleği (Safety Vest)
- Maske (Mask)
- Gözlük (Goggles)
- Düşme (Fall Detection)

### 📊 Veri Seti Yapısı
```
PPE/Fall-Detecetion-1/
├── train/        # Eğitim görüntüleri ve etiketleri
├── valid/        # Doğrulama görüntüleri ve etiketleri
├── test/         # Test görüntüleri ve etiketleri
└── data.yaml     # YOLO format veri seti konfigürasyonu
```

### ✅ Kazanımlar
- Veri seti projeye entegre edildi
- Sınıflandırma standartları belirlendi
- İlk temel model denemeleri yapıldı
