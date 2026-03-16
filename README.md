# ARCHISAFE

## Yapay Zeka Destekli Fabrika İş Güvenliği ve Otonom Düşme Tespit Sistemi

ARCHISAFE, fabrika ve endüstriyel alanlarda iş güvenliğini artırmak amacıyla geliştirilmiş, yapay zeka tabanlı bir görüntü işleme sistemidir.

### Temel Özellikler
- **Kişisel Koruyucu Donanım (KKD) Tespiti**: Kask, yelek, eldiven gibi güvenlik ekipmanlarının anlık kontrolü.
- **Otonom Düşme Tespit Sistemi**: Çalışanların düşme veya kaza durumlarını gerçek zamanlı olarak algılama ve uyarı mekanizması.
- **Çoklu Model Karşılaştırması**: En güncel YOLO modelleri (v8, v11, v12, v26) ile farklı ölçeklerde (Nano, Small, Medium) performans analizi.

### Kullanım
Modeller arası karşılaştırmalı eğitimi başlatmak için:
```bash
python train_comparison.py
```

### Karşılaştırma Sonuçları
Eğitilen modellerin detaylı performans analizi ve karşılaştırma sonuçlarına [buradan](comparison_report.md) ulaşabilirsiniz.
