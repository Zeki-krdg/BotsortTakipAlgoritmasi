# BlackBox - Sabit Kanatlı İHA Takip Sistemi

Bu proje, sabit kanatlı İHA'lar (Kamikaze dronlar) için geliştirilmiş, yüksek performanslı bir nesne takip sistemidir. **YOLO** tespit gücünü **BOT-SORT** (Bag of Tricks for SORT) algoritmasının Kalman filtresi, kamera hareketi telafisi (CMC) ve akıllı eşleştirme yeteneğiyle birleştirir.

## 🚀 Özellikler (BOT-SORT)
- **BOT-SORT Takip:** Kalman filtresi + IoU eşleştirme + Kamera Hareketi Telafisi (CMC) ile güçlü takip.
- **YOLO Entegrasyonu:** Her karede bağımsız tespit — template drift sorunu yok.
- **Nesne ID Koruması:** Aynı hedef, kareler boyunca aynı ID ile takip edilir.
- **Track Buffer:** Hedef geçici olarak kaybolduğunda (tıkanma, parlama) iz korunur ve yeniden yakalanır.
- **Kalman Filtresi:** 8-boyutlu durum vektörü ile pozisyon + hız + boyut tahmini.
- **Kamera Hareketi Telafisi (CMC):** İHA'nın kendi hareketinden kaynaklanan arka plan kaymasını otomatik düzeltir.

## 📦 Gereksinimler

Projenin çalışması için Python 3.8+ gereklidir. Gerekli kütüphaneleri aşağıdaki komutla yükleyebilirsiniz:

```bash
pip install torch torchvision torchaudio
pip install ultralytics
pip install opencv-python
pip install numpy
```

## 🛠️ Kurulum ve Hazırlık

1. **Depoyu klonlayın:**
   ```bash
   git clone https://github.com/ferhat-yldz/BlackBox.git
   cd BlackBox
   ```

2. **Ağırlık Dosyaları:**
   - YOLO modelinizi (`best.pt`) şu dizine yerleştirin: `runs/detect/yolov8_kamikaze_960/weights/`
   - BOT-SORT Ultralytics içinde hazır gelir, ekstra ağırlık dosyası gerekmez.

## 🏃 Nasıl Çalıştırılır?

### 1. Ana BOT-SORT Takip Sistemi
YOLO tespit + BOT-SORT takip sistemini başlatmak için:
```bash
python botsort_tracker.py
```

### 2. Detaylı BOT-SORT Testi
BOT-SORT performansını detaylı istatistiklerle test etmek için:
```bash
python botsort_test.py
```

### 3. (Eski) SiamCAR Hibrit Sistemi
Eski SiamCAR tabanlı sistemi çalıştırmak için:
```bash
python siamcar.py
```

## 📂 Proje Yapısı

### BOT-SORT (Aktif Sistem)
- `botsort_tracker.py`: YOLO + BOT-SORT takip sisteminin ana giriş noktası.
- `botsort_test.py`: Detaylı performans analizi ve istatistik raporu.
- `botsort_kamikaze.yaml`: Kamikaze İHA takibine özel BOT-SORT konfigürasyonu.

### SiamCAR (Eski Sistem — Referans)
- `siamcar_model.py`: SiamCAR mimarisi — MobileNetV3-Small backbone, DW-XCorr, cls/ctr/reg head'leri.
- `siamcarAlgorithm.py`: SiamCAR takip mantığı, template caching ve model yönetimi.
- `siamcar.py`: YOLO + SiamCAR hibrit sisteminin eski giriş noktası.
- `siamcar_test.py`: SiamCAR test betiği.
- `train_siamcar.py`: SiamCAR head eğitimi.

## 🏗️ Mimari (BOT-SORT)

```
Her Kare ──→ YOLO Tespiti ──→ Kalman Filtresi Tahmini ──→ IoU Eşleştirme ──→ İz Güncelleme
                                       │                         │
                                       └── CMC (Kamera Telafi) ──┘
```

| Bileşen | Açıklama |
|---------|----------|
| YOLO Dedektör | Her karede bağımsız nesne tespiti |
| Kalman Filtresi | [cx, cy, w, h, ẋ, ẏ, ẇ, ḣ] durum vektörü |
| IoU Eşleştirme | 2 aşamalı (yüksek + düşük güvenli) |
| CMC | sparseOptFlow ile kamera hareketi telafisi |
| Track Buffer | 30 kare kayıp toleransı (~1 saniye) |

## 📝 Lisans
Bu proje eğitim ve araştırma amaçlı geliştirilmiştir.
