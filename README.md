# BlackBox - Sabit Kanatlı İHA Takip Sistemi

Bu proje, sabit kanatlı İHA'lar (Kamikaze dronlar) için geliştirilmiş, yüksek performanslı bir hibrit nesne takip sistemidir. **YOLOv8**'in tespit gücünü, **OSTrack (One-Stream Tracking)** algoritmasının gelişmiş Transformer tabanlı takip yeteneğiyle birleştirir.

## 🚀 Özellikler
- **OSTrack Implementation:** Makaledeki (Ye et al.) orijinal Vision Transformer (ViT) mimarisi üzerine kurulu tam sürüm takip algoritması.
- **YOLOv8 Entegrasyonu:** Hedefin ilk tespiti ve takip kaybı durumunda otomatik yeniden yakalama.
- **Hibrit Takip (Handshake):** YOLO (Tespit) ve OSTrack (Takip) arasında güven skoruna dayalı akıllı geçiş mekanizması.
- **FailSafe Protokolü:** Takip kalitesi düştüğünde sistemin çökmesini engelleyen ani kurtarma sistemi.
- **Hareket Tutarlılığı Filtresi:** Koordinat sıçramalarını engelleyerek drift (sapma) sorununu minimize eder.

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
   - YOLOv8 modelinizi (`best.pt`) şu dizine yerleştirin: `runs/detect/yolov8_kamikaze_960/weights/`
   - (Opsiyonel) OSTrack transformer ağırlıklarını (`.pth`) ana dizine ekleyip ilgili scriptlerden yollarını güncelleyebilirsiniz.

## 🏃 Nasıl Çalıştırılır?

### 1. Ana Hibrit Takip Sistemi
YOLO ile tespit yapan ve OSTrack ile takip eden ana sistemi başlatmak için:
```bash
python ostracktry.py
```

### 2. Standalone OSTrack Testi
OSTrack modelini manuel alan seçimi (ROI) ile bağımsız olarak test etmek için:
```bash
python ostrack_test.py
```

## 📂 Proje Yapısı
- `ostrack_model.py`: Vision Transformer (ViT) tabanlı takip mimarisinin PyTorch kodları.
- `ostrackAlgorithm.py`: Takip mantığını, ön işlemeyi ve model yönetimini sağlayan arayüz sınıfı.
- `ostracktry.py`: YOLO + OSTrack hibrit sisteminin ana giriş noktası.
- `ostrack_test.py`: Manuel takip doğrulaması için kullanılan test betiği.

## 📝 Lisans
Bu proje eğitim ve araştırma amaçlı geliştirilmiştir.
