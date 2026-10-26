import os
import glob
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from veri_yukleyici import AkcigerRaporDataset, transform, MAX_KELIME_UZUNLUGU
from model_mimarisi_vlm import AkcigerRaporModeli

# ==========================================
# 1. AYARLAR VE YOLLAR
# ==========================================
TRAIN_IMG_DIR = r"C:\Users\Seyma\Desktop\UNET_DATASET\images\train"
TRAIN_JSON_DIR = r"C:\Users\Seyma\Desktop\doktor_raporlari\train"

with open("medikal_sozluk.json", "r", encoding="utf-8") as f:
    SOZLUK = json.load(f)

VOCAB_SIZE = len(SOZLUK)
EMBED_SIZE = 256
HIDDEN_SIZE = 512
BATCH_SIZE = 4
NUM_EPOCHS = 20
LEARNING_RATE = 0.0003

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("1. Veriler yükleniyor...")
train_dataset = AkcigerRaporDataset(TRAIN_IMG_DIR, TRAIN_JSON_DIR, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

model = AkcigerRaporModeli(EMBED_SIZE, HIDDEN_SIZE, VOCAB_SIZE).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=SOZLUK["<PAD>"])
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ==========================================
# 2. EN SON KAYDEDİLEN MODELİ BULMA (RESUME MANTIĞI)
# ==========================================
kayitli_modeller = glob.glob("vlm_rapor_model_epoch_*.pth")
baslangic_epoch = 0

if kayitli_modeller:
    # Dosya isimlerinden en büyük sayıyı (en son tamamlanan epoch'u) bulur
    en_son_model = max(kayitli_modeller, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    baslangic_epoch = int(en_son_model.split('_')[-1].split('.')[0])
    
    # Ağırlıkları modele yükle
    model.load_state_dict(torch.load(en_son_model, weights_only=True))
    print(f"\n--> {en_son_model} dosyası bulundu!")
    print(f"--> Eğitime {baslangic_epoch + 1}. Epoch'tan devam edilecek.\n")
else:
    print("\n--> Kayıtlı model bulunamadı, eğitim 1. epoch'tan başlayacak.\n")

# ==========================================
# 3. KALDIĞI YERDEN EĞİTİM DÖNGÜSÜ
# ==========================================
if __name__ == "__main__":
    # range() fonksiyonu 0'dan değil, kalınan epoch'tan başlar
    for epoch in range(baslangic_epoch, NUM_EPOCHS):
        model.train()
        toplam_loss = 0
        
        loop = tqdm(train_loader, total=len(train_loader), desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
        
        for batch_idx, (images, captions) in enumerate(loop):
            images = images.to(device)
            captions = captions.to(device)
            
            optimizer.zero_grad()
            outputs = model(images, captions)
            
            outputs_flat = outputs.view(-1, VOCAB_SIZE)
            captions_flat = captions.view(-1)
            
            loss = criterion(outputs_flat, captions_flat)
            loss.backward()
            optimizer.step()
            
            toplam_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        ortalama_loss = toplam_loss / len(train_loader)
        print(f"--> Epoch {epoch+1} Tamamlandı. Ortalama Loss: {ortalama_loss:.4f}")
        
        torch.save(model.state_dict(), f"vlm_rapor_model_epoch_{epoch+1}.pth")

    print("\nDevam Eğitimi Başarıyla Tamamlandı!")