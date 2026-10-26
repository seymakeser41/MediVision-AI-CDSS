import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm

# ==========================================
# 1. VERİ SETİ SINIFI (DATASET CLASS)
# ==========================================
class AkcigerDataset(Dataset):
    def __init__(self, images_dir, masks_dir):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.images = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img_name = self.images[i]
        mask_name = img_name.replace('.jpg', '-mask.jpg') 
        
        img_path = self.images_dir / img_name
        mask_path = self.masks_dir / mask_name

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (512, 512))
        
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (512, 512))
        mask = mask / 255.0

        image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        mask = torch.from_numpy(mask).float().unsqueeze(0)

        return image, mask

# ==========================================
# 2. EĞİTİME KALDIĞI YERDEN DEVAM ETME
# ==========================================
if __name__ == '__main__':
    # Yollar (Kendi bilgisayarınıza göre güncelleyin)
    UNET_DIR = r"C:\Users\Seyma\Desktop\UNET_DATASET"
    TRAIN_IMG = os.path.join(UNET_DIR, "images", "train")
    TRAIN_MASK = os.path.join(UNET_DIR, "masks", "train")
    VAL_IMG = os.path.join(UNET_DIR, "images", "val")
    VAL_MASK = os.path.join(UNET_DIR, "masks", "val")

    train_dataset = AkcigerDataset(TRAIN_IMG, TRAIN_MASK)
    val_dataset = AkcigerDataset(VAL_IMG, VAL_MASK)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2) # Batch_size 4 önerilir
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Modeli Yeniden İnşa Et
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    ).to(device)

    # 2. ESKİ AĞIRLIKLARI (WEIGHTS) YÜKLE
    # Eğer daha önce last_unet_akciger.pth oluşmuşsa oradan, yoksa best'ten başla
    if os.path.exists("last_unet_akciger.pth"):
        model.load_state_dict(torch.load("last_unet_akciger.pth"))
        print("--> 'last_unet_akciger.pth' bulundu. Kaldığı yerden devam ediliyor...")
    elif os.path.exists("best_unet_akciger.pth"):
        model.load_state_dict(torch.load("best_unet_akciger.pth"))
        print("--> 'best_unet_akciger.pth' bulundu. Kaldığı yerden devam ediliyor...")
    else:
        print("--> Kayıtlı model bulunamadı, sıfırdan başlanıyor.")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # 3. YENİ EĞİTİM DÖNGÜSÜ
    KALAN_EPOCH = 30 # Devam etmek istediğin epoch sayısını buradan ayarlayabilirsin
    best_loss = float("inf")

    for epoch in range(KALAN_EPOCH):
        model.train()
        train_loss = 0.0
        
        loop = tqdm(train_loader, desc=f"Ekstra Epoch {epoch+1}/{KALAN_EPOCH} [Eğitim]")
        for images, masks in loop:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        # Doğrulama Aşaması
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")

        # GÜNCELLEME: Her epoch bittiğinde kaldığın yeri kaydet (Sonraki durdurmalar için)
        torch.save(model.state_dict(), "last_unet_akciger.pth")
        print("--> Kaldığın yer 'last_unet_akciger.pth' olarak güncellendi.")

        # Eğer model kendini geliştirmişse best dosyasını da güncelle
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), "best_unet_akciger.pth")
            print("--> Yeni en iyi model 'best_unet_akciger.pth' kaydedildi!")

    print("U-Net Devam Eğitimi Tamamlandı.")