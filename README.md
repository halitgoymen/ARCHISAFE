# ARCHISAFE - KKD ve Düşme Tespit Sistemi

ARCHISAFE, endüstriyel ortamlarda iş güvenliğini artırmak amacıyla geliştirilmiş, Yapay Zeka destekli bir görüntü işleme projesidir. Proje, çalışanların Kişisel Koruyucu Donanım (KKD) kullanımını denetler ve olası düşme vakalarını anlık olarak tespit eder.

## Proje Gelişim Süreci (Haftalık)

### 1. Hafta: Veri Seti Hazırlığı ve Problem Tanımı
- **Odak Noktası:** KKD (Kask, yelek, maske vb.) ve insan duruşu (düşme tespiti) üzerine odaklanan Roboflow tabanlı veri setinin oluşturulması.
- **Yapılanlar:** 
  - PPE/Fall-Detection veri setinin projeye entegrasyonu.
  - Sınıflandırma ve etiketleme standartlarının belirlenmesi.
  - İlk test görselleri üzerinde temel model denemelerinin yapılması.

### 2. Hafta: Model Karşılaştırma ve Teknik Analiz
- **Odak Noktası:** Farklı YOLO sürümlerinin (v8, v11, v12, v26) benchmark testleri ve en uygun modelin seçimi.
- **Yapılanlar:**
  - `train_comparison.py` ile RTX 5070 Ti üzerinde otomatik benchmark testleri yapıldı.
  - mAP50 skorları ve eğitim süreleri analiz edildi.
  - [Karşılaştırma Raporu](comparison_report.md) oluşturularak **YOLO12-Small** modelinin final seçim olduğu teknik detaylarla açıklandı.
  - Proje yapısı temizlenerek GitHub entegrasyonu sağlandı.

### 3. Hafta: Gelecek Planı 
- **Odak Noktası:** Seçilen modelin canlı sisteme entegrasyonu ve diğer projeler ile kıyaslanması.
- **Planlananlar:**
  -Proje eklenecek-çıkartılacak teknolojiler hakkında planlama.
  - Sistem performansının optimize edilmesi.
  - Veri setinde iyileştirmeler.
  - 
### 4. Hafta: Gelecek Planı (Gelecek Hafta)
- **Odak Noktası:** Seçilen modelin canlı sisteme entegrasyonu ve saha testleri.
- **Planlananlar:**
  - Real-time (gerçek zamanlı) çıkarım script'lerinin yazılması.
  - Uyarı mekanizmalarının ve bildirim sisteminin kodlanması.
  - Sistem performansının optimize edilmesi.
## Teknik Detaylar

### Karşılaştırma Analizi
Eğitilen modellerin detaylı performans analizi ve derinlemesine teknik çıkarımlarına [buradan](comparison_report.md) ulaşabilirsiniz.

### Kullanım
Benchmark sürecini veya eğitimi tekrar başlatmak için:
```bash
python train_comparison.py
```
