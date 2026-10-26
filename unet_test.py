import os
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Modeli Yükle
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1).to(device)
    model.load_state_dict(torch.load(r"C:\Users\Seyma\Desktop\Bitirme_veri_hazirlk\best_unet_akciger.pth", weights_only=True))
    model.eval()

    TEST_DIR = r"C:\Users\Seyma\Desktop\UNET_DATASET\images\test"
    OUTPUT_DIR = "unet_test_ciktilari"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    test_images = [f for f in os.listdir(TEST_DIR) if f.endswith('.jpg')][:10] # İlk 10 resmi test et

    print("U-Net test ediliyor ve sonuçlar kaydediliyor...")
    with torch.no_grad():
        for img_name in test_images:
            img_path = os.path.join(TEST_DIR, img_name)
            
            raw_img = cv2.imread(img_path)
            raw_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
            
            # Tensör Dönüşümü
            img_resized = cv2.resize(raw_rgb, (512, 512))
            input_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            input_tensor = input_tensor.to(device)
            
            # Maske Tahmini
            pred_mask = model(input_tensor)
            pred_mask = torch.sigmoid(pred_mask).cpu().numpy()[0, 0]
            binary_mask = (pred_mask > 0.5).astype(np.uint8)
            
            # Maskeyi Orijinal Boyuta Getir ve Uygula
            binary_mask_resized = cv2.resize(binary_mask, (raw_img.shape[1], raw_img.shape[0]))
            cleaned_img = cv2.bitwise_and(raw_img, raw_img, mask=binary_mask_resized)
            
            # Görselleştirme ve Kaydetme
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            ax[0].imshow(raw_rgb)
            ax[0].set_title("Orijinal Ham Veri")
            ax[0].axis("off")
            
            ax[1].imshow(binary_mask_resized, cmap="gray")
            ax[1].set_title("U-Net Maskesi")
            ax[1].axis("off")
            
            cleaned_rgb = cv2.cvtColor(cleaned_img, cv2.COLOR_BGR2RGB)
            ax[2].imshow(cleaned_rgb)
            ax[2].set_title("Temizlenmiş Çıktı")
            ax[2].axis("off")
            
            plt.savefig(os.path.join(OUTPUT_DIR, f"test_sonuc_{img_name}"))
            plt.close()

    print(f"Test tamamlandı! Çıktılar '{OUTPUT_DIR}' klasöründe incelenebilir.")