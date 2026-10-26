from ultralytics import YOLO

if __name__ == '__main__':
    # Eğitilen en iyi ağırlıkları yükle
    model = YOLO(r"C:\Users\Seyma\Desktop\Bitirme_veri_hazirlk\runs\segment\Medikal_Vision\yolo_lezyon_seg-2\weights\best.pt")

    # Veri seti yolunu kendi sisteminize göre ayarlayın
    yaml_path = r"C:\Users\Seyma\Desktop\YOLO_DATASET\data.yaml"

    print("YOLO modeli test verisi üzerinde değerlendiriliyor...")
    
    # split='test' parametresi ile sadece test klasörü hedef alınır
    metrics = model.val(data=yaml_path, split='test')

    print("\n" + "="*40)
    print("YOLO TEST SONUÇLARI (Hiç Görülmemiş Veri)")
    print("="*40)
    print(f"Sınır Kutusu mAP50 Skoru: {metrics.box.map50:.4f}")
    print(f"Bölütleme (Mask) mAP50 Skoru: {metrics.seg.map50:.4f}")
    print("Detaylı grafikler 'runs/segment/val' klasörüne kaydedildi.")