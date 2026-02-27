# Fixed Wing UAV - Kamikaze Detection

Bu proje, YOLOv8 kullanılarak sabit kanatlı İHA'ların (Kamikaze) tespiti için geliştirilmiştir.

## Proje Yapısı
- `train.py`: Modeli eğitmek için kullanılan script.
- `predict_video.py`: Eğitilmiş model ile video üzerinde test yapmak için kullanılır.
- `runs/detect/yolov8_kamikaze_960/weights/best.pt`: Eğitilmiş en iyi model ağırlıkları.

## Kurulum
1. Repoyu klonlayın:
   ```bash
   git clone https://github.com/ferhat-yldz/fixedWingUav.git
   cd fixedWingUav
   ```
2. Gerekli kütüphaneleri kurun:
   ```bash
   pip install ultralytics
   ```

## Dataset
Dataset boyutu büyük olduğu için GitHub'a yüklenmemiştir. Projeyi yeniden eğitmek isterseniz dataseti [BURAYA LİNK GELECEK] adresinden indirip `dataset/` klasörüne yerleştirmeniz gerekmektedir.

## Kullanım
Video üzerinde tahmin yürütmek için:
```bash
python predict_video.py
```
