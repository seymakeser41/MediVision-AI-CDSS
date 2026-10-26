import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Daha önce yazdığımız dosyalardan gerekli sınıfları içe aktarıyoruz
from veri_yukleyici import AkcigerRaporDataset, transform, MAX_KELIME_UZUNLUGU
from model_mimarisi_vlm import AkcigerRaporModeli

# ==========================================
# 1. AYARLAR VE YOLLAR
# ==========================================
# Gerçek verilerinin bulunduğu yollar
TRAIN_IMG_DIR = r"C:\Users\Seyma\Desktop\UNET_DATASET\images\train"
TRAIN_JSON_DIR = r"C:\Users\Seyma\Desktop\doktor_raporlari\train"

# Sözlüğü yükleyip boyutunu öğreniyoruz
with open("medikal_sozluk.json", "r", encoding="utf-8") as f:
    SOZLUK = json.load(f)

VOCAB_SIZE = len(SOZLUK)
EMBED_SIZE = 256
HIDDEN_SIZE = 512
BATCH_SIZE = 4 # RTX 4050'nin 6GB VRAM'ini taşırmamak için 4'te tutulmalı
NUM_EPOCHS = 20 # Veri setini kaç kez baştan sona göreceği
LEARNING_RATE = 0.0003

# ==========================================
# 2. HAZIRLIK (DATASET, MODEL VE DONANIM)
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("1. Veri seti yükleniyor...")
train_dataset = AkcigerRaporDataset(TRAIN_IMG_DIR, TRAIN_JSON_DIR, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

print("2. Model belleğe (GPU) aktarılıyor...")
model = AkcigerRaporModeli(EMBED_SIZE, HIDDEN_SIZE, VOCAB_SIZE).to(device)

# Hata fonksiyonu (CrossEntropy)
# DİKKAT: <PAD> tokenlerini (boşlukları) hata hesaplamasına katmıyoruz (ignore_index=0)
criterion = nn.CrossEntropyLoss(ignore_index=SOZLUK["<PAD>"])
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ==========================================
# 3. EĞİTİM DÖNGÜSÜ (TRAINING LOOP)
# ==========================================
print(f"3. Eğitim {device} üzerinde başlatılıyor...\n")

if __name__ == "__main__":
    for epoch in range(NUM_EPOCHS):
        model.train()
        toplam_loss = 0
        
        # Terminalde ilerleme çubuğu (tqdm)
        loop = tqdm(train_loader, total=len(train_loader), desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
        
        for batch_idx, (images, captions) in enumerate(loop):
            # Verileri ekran kartına gönder
            images = images.to(device)
            captions = captions.to(device)
            
            # Gradiyanları sıfırla
            optimizer.zero_grad()
            
            # İleri yayılım (Tahmin üret)
            outputs = model(images, captions)
            
            # Loss hesaplama (Matrisleri tek boyutlu düz listeye çevirmemiz gerekir)
            # outputs boyutu: (Batch, UZUNLUK, VOCAB) -> (Batch * UZUNLUK, VOCAB)
            # captions boyutu: (Batch, UZUNLUK) -> (Batch * UZUNLUK)
            outputs_flat = outputs.view(-1, VOCAB_SIZE)
            captions_flat = captions.view(-1)
            
            loss = criterion(outputs_flat, captions_flat)
            
            # Geri yayılım (Ağırlıkları güncelle)
            loss.backward()
            optimizer.step()
            
            toplam_loss += loss.item()
            
            # İlerleme çubuğunda anlık hatayı (loss) göster
            loop.set_postfix(loss=loss.item())
            
        # Her epoch sonunda ortalama hatayı yazdır
        ortalama_loss = toplam_loss / len(train_loader)
        print(f"--> Epoch {epoch+1} Tamamlandı. Ortalama Loss: {ortalama_loss:.4f}")
        
        # Her epoch sonunda modeli kaydet (Çökme riskine karşı)
        torch.save(model.state_dict(), f"vlm_rapor_model_epoch_{epoch+1}.pth")

    print("\nEğitim Başarıyla Tamamlandı! En son ağırlıklar kaydedildi.")