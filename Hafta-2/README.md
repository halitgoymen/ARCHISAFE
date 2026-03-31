# 📁 2. Hafta — Model Karşılaştırması ve Teknik Analiz

## Yapılanlar

### 🎯 Hafta Odak Noktası
10 farklı YOLO modelinin (v8, v11, v12, v26) aynı veri seti üzerinde benchmark testleri yapıldı ve en uygun model seçildi.

### ⚙️ Kullanılan Script
`train_comparison.py` — RTX 5070 Ti üzerinde otomatik karşılaştırma scripti.

**Konfigürasyon:**
- Epochs: 3 (hızlı karşılaştırma için)
- Image Size: 640
- Batch: 8
- Device: CUDA (GPU)

### 🏆 Sonuçlar (Özet)

| Model | mAP50 | Süre (s) |
|-------|-------|----------|
| **YOLO12-Small** ✅ | **0.4744** | 1620.72 |
| YOLO11-Small | 0.4658 | 1426.64 |
| YOLOv8-Medium | 0.4642 | 1411.83 |
| YOLOv8-Nano | 0.4231 | 1157.90 ⚡ |

→ Detaylı sonuçlar: `comparison_results.csv`  
→ Analiz raporu: `comparison_report.md`

### ✅ Sonuç
**YOLO12-Small** en yüksek mAP50 skoruna ulaştı ve final model olarak seçildi.
