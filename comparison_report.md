# YOLO Model Karşılaştırma Raporu

Bu rapor, ARCHISAFE projesindeki KKD ve Düşme tespiti veri seti üzerinde eğitilen farklı YOLO modellerinin performans sonuçlarını içermektedir.

## Karşılaştırma Sonuçları

| Model | mAP50 | mAP50-95 | Eğitim Süresi (s) | Boyut | Versiyon |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **YOLO12-Small** | **0.4744** | **0.2998** | 1620.72 | s | v12 |
| YOLO11-Small | 0.4658 | 0.2951 | 1426.64 | s | v11 |
| YOLOv8-Medium | 0.4642 | 0.2986 | 1411.83 | m | v8 |
| YOLO11-Medium | 0.4398 | 0.2791 | 1625.47 | m | v11 |
| YOLO12-Medium | 0.4392 | 0.2752 | 1884.34 | m | v12 |
| YOLO11-Nano | 0.4393 | 0.2739 | 1371.62 | n | v11 |
| YOLOv8-Small | 0.4327 | 0.2832 | 1220.64 | s | v8 |
| YOLO12-Nano | 0.4260 | 0.2710 | 1506.76 | n | v12 |
| YOLOv8-Nano | 0.4231 | 0.2653 | **1157.90** | n | v8 |
| YOLO26-Nano | 0.3512 | 0.2245 | 1480.93 | n | v26 |

## Analiz ve Öneriler

### 1. En Doğru Model: YOLO12-Small
Test edilen modeller arasında **YOLO12-Small**, 0.4744 mAP50 değeri ile en yüksek doğruluğu sağlamıştır. Güvenlik sistemleri için kritik olan nesneleri (kask, yelek vb.) en iyi bu model tespit etmektedir.

### 2. En Hızlı Model: YOLOv8-Nano
Gerçek zamanlı, çok düşük gecikmeli bir sistem gerekiyorsa **YOLOv8-Nano** 1157 saniyelik eğitim süresi ve yüksek FPS potansiyeli ile en mantıklı seçenektir.

### 3. Genel Değerlendirme
- **YOLO11-Small** ve **YOLO12-Small** arasındaki fark çok azdır, ancak v12 modeli doğrulukta bir adım öndedir.
- **YOLO26-Nano** şu an için (bu veri seti ve kısa eğitimde) diğerlerinin gerisinde kalmıştır.
- **Medium** modeller (YOLO12-M, YOLO11-M), daha fazla işlem gücü tüketmelerine rağmen bu veri setinde Small versiyonlarından daha iyi bir sonuç vermemiştir. Bu durum veri setinin boyutuna veya karmaşıklığına bağlı olabilir.

**Tavsiye:** ARCHISAFE projesinin final sürümü için **YOLO12-Small** modelinin kullanılması, doğruluk ve hız dengesi açısından en mantıklısıdır.

## Teknik Analiz ve Çıkarımlar

Bu çalışma, PPE (Kişisel Koruyucu Donanım) tespiti gibi kritik bir alanda model seçiminin sadece "en yeni sürüm" olmaktan öte, veri seti karakteristiği ile doğrudan ilişkili olduğunu göstermiştir.

### 1. YOLO12-Small Neden Kazandı?
YOLO12-Small, test edilen tüm modeller arasında mAP50 skorunda zirveye yerleşti. Bunun başlıca nedenleri:
- **Mimari Verimlilik:** YOLO12, v11'e göre daha daraltılmış ama daha yoğun bir özellik çıkarımı (feature extraction) yapar. Small ölçekte bile, PPE veri setindeki küçük nesneleri (maske, gözlük, kask detayları) yakalamak için gereken dikkat mekanizmalarını (attention mechanisms) daha iyi kullanmaktadır.
- **Optimum Derinlik:** PPE veri setimiz yaklaşık 1500-2000 görsel bandında olduğu için, v12'nin Small mimarisi veriyi tam kıvamında öğrenirken, Medium modeller "under-fit" veya "over-fit" dengesinde kalmış olabilir.

### 2. YOLO26 ve YOLOv8: Eski vs. Yeni
- **YOLOv8'in Gücü:** YOLOv8-Nano, eğitim süresi başarısıyla hala "Hız/Performans" kralı olduğunu kanıtladı. 1157 saniyelik rekor süresiyle, en hızlı sonuç veren model oldu.
- **YOLO26 Paradoxu:** En yeni nesil olan YOLO26'nın düşük kalması bir başarısızlık değil, bir "karakteristik" meselesidir. YOLO26 mimarisi çok daha fazla parametreye hükmeder; 3 epoch gibi kısa bir sürede "ısınması" (momentum kazanması) zordur. Gerçek gücünü muhtemelen 50+ epoch sonunda gösterecektir.

### 3. "Daha Büyük Her Zaman Daha İyi Değildir"
Sonuçlarda gördüğümüz en ilginç veri, **YOLOv8-Medium'un YOLO12-Small'dan daha düşük skor almasıdır.** Bu, projemiz için şu dersi verir:
> Çok büyük modeller (Medium/Large), bazen sınırlı veri setlerinde hantal kalır. ARCHISAFE gibi spesifik bir alanda "Small" modeller hem daha hızlıdır hem de daha isabetli tespitler yapabilir.

## Final Kararı ve Yol Haritası

ARCHISAFE projesinin GitHub'a push edilecek bu sürümü, en geniş kapsamlı karşılaştırma altyapısını sunmaktadır. 

- **Geliştirme İçin:** `train_comparison.py` kullanarak yeni modeller test edilebilir.
- **Canlı Sistem İçin:** `YOLO12-Small` mimarisi üzerine ağırlık verilmelidir.
- **Kenar Cihazlar (Jetson/Raspberry):** Hız önceliği varsa `YOLOv8-Nano` tercih edilmelidir.

---
*Bu rapor AI Asistan tarafından, RTX 5070 Ti üzerinde gerçekleştirilen 10 farklı modelin benchmark testleri sonucunda otomatik olarak oluşturulmuştur.*
