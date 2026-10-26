import os
import yaml
from ultralytics import YOLO
from pathlib import Path

# 1. YOLO_DATASET Yollarının Belirlenmesi
# Lütfen KENDİ BİLGİSAYARINIZDAKİ TAM YOLU (Absolute Path) buraya yazın
DATASET_DIR = r"C:\Users\Seyma\Desktop\YOLO_DATASET"

# 2. data.yaml Dosyasının Otomatik Oluşturulması
yaml_icerigi = {
    'path': DATASET_DIR,
    'train': 'images/train',
    'val': 'images/val',
    'test': 'images/test',
    'nc': 1, # Aranacak sınıf sayısı
    'names': ['Lezyon'] # Kendi sınıf adınıza göre güncelleyin
}

yaml_path = os.path.join(DATASET_DIR, 'data.yaml')
with open(yaml_path, 'w') as f:
    yaml.dump(yaml_icerigi, f, default_flow_style=False)

print(f"data.yaml dosyası oluşturuldu: {yaml_path}")

# 3. Modelin Yüklenmesi ve Eğitimin Başlatılması
if __name__ == '__main__':
    # YOLOv11 veya YOLOv8 Nano Segmentasyon modelini yükle (Hafif ve hızlı)
    model = YOLO('yolo11n-seg.pt')

    # Eğitimi Başlat
    results = model.train(
        data=yaml_path,
        epochs=70,          # Veri setinin kaç kez taranacağı
        imgsz=640,          # Modele girecek resim boyutu (YOLO standardı)
        batch=8,           # GPU VRAM'ine göre 8 veya 32 yapılabilir
        device='0',         # NVIDIA CUDA GPU kullanımı
        project='Medikal_Vision', # Sonuçların kaydedileceği ana klasör
        name='yolo_lezyon_seg',   # Bu eğitim turunun adı
        patience=10         # 10 epoch boyunca iyileşme olmazsa eğitimi durdur (Overfitting önlemi)
    )
    print("YOLO Eğitimi başarıyla tamamlandı!")