import os
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

# ==========================================
# AYARLAR VE YOLLAR
# ==========================================
TEST_IMAGES = r"C:\Users\Seyma\Desktop\YOLO_DATASET\images\test"
TEST_MASKS = r"C:\Users\Seyma\Desktop\YOLO_DATASET\masks\test"  

yolo_model = YOLO(r"C:\Users\Seyma\Desktop\Bitirme_veri_hazirlk\runs\segment\Medikal_Vision\yolo_lezyon_seg-2\weights\best.pt")

def dice_skoru_hesapla(pred_mask, true_mask):
    pred = pred_mask.astype(bool)
    true = true_mask.astype(bool)
    
    kesisim = np.logical_and(pred, true).sum()
    toplam_alan = pred.sum() + true.sum()
    
    if toplam_alan == 0:
        return 1.0 
    return 2. * kesisim / toplam_alan

if __name__ == "__main__":
    resimler = [f for f in os.listdir(TEST_IMAGES) if f.endswith('.jpg') or f.endswith('.png')]
    toplam_dice = 0
    gecerli_resim = 0
    
    print("YOLO Segmentasyon Performansı (Dice Skoru) Hesaplanıyor...")
    
    for img_adi in tqdm(resimler):
        img_yolu = os.path.join(TEST_IMAGES, img_adi)
        
        # İŞTE BURAYI DÜZELTTİK: "-mask.jpg" veya "-mask.png" ekini arıyoruz
        mask_adi_jpg = img_adi.replace('.jpg', '-mask.jpg')
        mask_adi_png = img_adi.replace('.jpg', '-mask.png')
        
        mask_yolu_jpg = os.path.join(TEST_MASKS, mask_adi_jpg)
        mask_yolu_png = os.path.join(TEST_MASKS, mask_adi_png)
        
        if os.path.exists(mask_yolu_jpg):
            mask_yolu = mask_yolu_jpg
        elif os.path.exists(mask_yolu_png):
            mask_yolu = mask_yolu_png
        else:
            continue
            
        true_mask = cv2.imread(mask_yolu, cv2.IMREAD_GRAYSCALE)
        true_mask = (cv2.resize(true_mask, (512, 512)) > 127).astype(np.uint8)
        
        img = cv2.imread(img_yolu)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (512, 512))
        
        results = yolo_model(img_resized, verbose=False)
        
        if results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            pred_mask = np.any(masks, axis=0).astype(np.uint8)
            pred_mask = cv2.resize(pred_mask, (512, 512))
        else:
            pred_mask = np.zeros((512, 512), dtype=np.uint8)
            
        dice = dice_skoru_hesapla(pred_mask, true_mask)
        toplam_dice += dice
        gecerli_resim += 1
        
    if gecerli_resim > 0:
        ortalama_dice = (toplam_dice / gecerli_resim) * 100
        print(f"\nToplam Test Edilen Maske: {gecerli_resim}")
        print(f"ORTALAMA DICE SKORU (Segmentasyon Başarısı): %{ortalama_dice:.2f}")
    else:
        print("\nHATA: Hiç geçerli maske bulunamadı. Lütfen maske klasörünü kontrol et.")