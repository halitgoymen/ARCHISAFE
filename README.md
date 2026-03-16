# ARCHISAFE - KKD ve Düşme Tespit Sistemi

ARCHISAFE, endüstriyel ortamlarda iş güvenliğini artırmak amacıyla geliştirilmiş, Yapay Zeka destekli bir görüntü işleme projesidir. Proje, çalışanların Kişisel Koruyucu Donanım (KKD) kullanımını denetler ve olası düşme vakalarını anlık olarak tespit eder.

## Proje Gelişim Süreci (Haftalık)

### 1. Hafta: Veri Seti Hazırlığı ve Problem Tanımı
- **Odak Noktası:** KKD (Kask, yelek, maske vb.) ve insan duruşu (düşme tespiti) üzerine odaklanan Roboflow tabanlı veri setinin oluşturulması.
- **Yapılanlar:** 
  - PPE/Fall-Detection veri setinin projeye entegrasyonu.
  - Sınıflandırma ve etiketleme standartlarının belirlenmesi.
  - İlk test görselleri üzerinde temel model denemelerinin yapılması.

### 2. Hafta: Model Karşılaştırma ve Benchmark Çalışmaları
- **Odak Noktası:** Farklı YOLO sürümlerinin (v8, v11, v12, v26) performanslarını aynı veri seti üzerinde test etmek.
- **Yapılanlar:**
  - `train_comparison.py` otomasyon script'i geliştirildi.
  - RTX 5070 Ti GPU ortamında Nano, Small ve Medium modeller için benchmark testleri yapıldı.
  - Eğitim süreleri ve mAP50 skorları kaydedildi.

### 3. Hafta: Teknik Analiz ve Model Seçimi
- **Odak Noktası:** Elde edilen verilerin profesyonel bir raporla analiz edilmesi ve final model kararı.
- **Yapılanlar:**
  - [Karşılaştırma Raporu](comparison_report.md) oluşturuldu.
  - Doğruluk ve hız dengesi açısından **YOLO12-Small** modelinin proje için en uygun seçenek olduğuna karar verildi.
  - Gereksiz runs ve model dosyaları temizlenerek projenin ana yapısı GitHub'a hazır hale getirildi.

## Teknik Detaylar

### Karşılaştırma Analizi
Eğitilen modellerin detaylı performans analizi ve derinlemesine teknik çıkarımlarına [buradan](comparison_report.md) ulaşabilirsiniz.

### Kullanım
Benchmark sürecini veya eğitimi tekrar başlatmak için:
```bash
python train_comparison.py
```
