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
  - Karşılaştırma Raporu oluşturularak **YOLO12-Small** modelinin final seçim olduğu teknik detaylarla açıklandı.
  - Proje yapısı temizlenerek GitHub entegrasyonu sağlandı.

### 3. Hafta: Gelecek Planı
- **Odak Noktası:** Seçilen modelin canlı sisteme entegrasyonu ve diğer projeler ile kıyaslanması.
- **Planlananlar:**
  - Proje eklenecek-çıkartılacak teknolojiler hakkında planlama.
  - Sistem performansının optimize edilmesi.
  - Veri setinde iyileştirmeler.
  - Modelin kararlaştırılması v12-v8.

### 4. Hafta: Final Model Eğitimi ve Canlı Test
- **Odak Noktası:** YOLO12-Small modelinin tam kapasite eğitimi (50 epoch) ve gerçek ortamda test edilmesi.
- **Yapılanlar:**
  - `train_final.py` ile YOLO12-Small modeli 50 epoch eğitildi.
  - Overfit önleme: Early stopping (10 epoch sabır), label smoothing, weight decay, augmentation.
  - `test_image.py` ile statik görsel üzerinde tespit testi yapıldı.
  - `test_live.py` ile gerçek zamanlı webcam testi gerçekleştirildi.
- **Eğitimi Başlatmak İçin:**
  ```bash
  cd Hafta-4/scripts
  python train_final.py
  ```
  > Eğitim 10 epoch boyunca iyileşme olmazsa **otomatik durur** (early stopping).

  ```bash
  # Bittikten sonra:
  python test_image.py   # resimle test
  python test_live.py    # canlı kamera
  ```

### 5. Hafta: Uyarı Sistemi ve Bildirimler (Gelecek Hafta)
- **Odak Noktası:** Tespit edilen ihlallerin (kaskasız, yeleğsiz, düşme) anlık bildirime dönüştürülmesi.
- **Planlananlar:**
  - İhlal tespit edildiğinde sesli/görsel alarm mekanizması.
  - E-posta / SMS / Telegram bot bildirimi entegrasyonu.
  - İhlal loglarının veritabanına kaydedilmesi.
  - Web arayüzü ile canlı izleme paneli (opsiyonel).
  - Sistem performansının optimize edilmesi ve saha testleri.

## Teknik Detaylar

### Karşılaştırma Analizi
Eğitilen modellerin detaylı performans analizi ve derinlemesine teknik çıkarımlarına [buradan](Hafta-2/comparison_report.md) ulaşabilirsiniz.

### Kullanım
Benchmark sürecini veya eğitimi tekrar başlatmak için:
```bash
python train_comparison.py
```
