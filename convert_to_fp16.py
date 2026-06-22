import os
import shutil
from ultralytics import YOLO

def convert_to_fp16(model_path="runs/detect/yolov8_kamikaze_960/weights/best.pt"):
    """
    YOLO modelini FP16 formatında ONNX'e çevirir ve exported_models klasörüne taşır.
    Orijinal .pt dosyasını ASLA silmez.
    """
    output_dir = "exported_models"
    
    # 1. Çıkış klasörünü hazırla
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"[BiLGi] '{output_dir}' klasörü oluşturuldu.")

    # 2. Model dosyasının varlığını kontrol et
    if not os.path.exists(model_path):
        print(f"[HATA] Model bulunamadı: {model_path}")
        print("Lütfen önce eğitimi tamamlayın veya doğru dosya yolunu verin.")
        return

    try:
        print(f"[İŞLEM] {model_path} yükleniyor...")
        model = YOLO(model_path)

        # 3. FP16 - ONNX Dışa Aktarımı
        # half=True: FP16 (Half Precision) sağlar
        # simplify=True: ONNX grafiğini optimize eder
        print("[İŞLEM] Model FP16 ONNX formatına çevriliyor (Bu işlem biraz vakit alabilir)...")
        exported_path = model.export(
            format="onnx", 
            half=True, 
            simplify=True,
            opset=12
        )

        # 4. Dosyayı yeni klasöre taşı
        if exported_path and os.path.exists(exported_path):
            file_name = os.path.basename(exported_path)
            final_dest = os.path.join(output_dir, file_name)
            
            # Eğer hedefte varsa üzerine yaz
            shutil.move(exported_path, final_dest)
            
            print("\n" + "="*50)
            print("🚀 İŞLEM BAŞARIYLA TAMAMLANDI")
            print("="*50)
            print(f"Orijinal Model: {model_path} (KORUNDU)")
            print(f"Yeni FP16 Model: {final_dest}")
            print("\nKullanım Örneği:")
            print(f"model = YOLO('{final_dest.replace(os.sep, '/')}')")
            print("="*50)
        else:
            print("[HATA] Dışa aktarma dosyası oluşturulamadı.")

    except Exception as e:
        print(f"[HATA] Çevirme sırasında bir sorun oluştu: {e}")

if __name__ == "__main__":
    # Sen yolo11'e geçince buradaki yolu "runs/detect/train6/weights/best.pt" gibi güncelleyebilirsin
    convert_to_fp16()
