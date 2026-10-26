import os
import glob
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from veri_yukleyici import AkcigerRaporDataset, transform
from model_mimarisi_v2 import YeniNesilAkcigerModeli

# ==========================================
# 1. AYARLAR VE YOLLAR
# ==========================================
# DİKKAT: YOLO ile temizlenmiş verilerin yolunu buraya kendi klasörüne göre yaz!
TRAIN_IMG_DIR = r"C:\Users\Seyma\Desktop\YOLO_DATASET\images\train" 
TRAIN_JSON_DIR = r"C:\Users\Seyma\Desktop\doktor_raporlari\train"

with open("medikal_sozluk.json", "r", encoding="utf-8") as f:
    SOZLUK = json.load(f)

VOCAB_SIZE = len(SOZLUK)
EMBED_SIZE = 256
BATCH_SIZE = 4 
NUM_EPOCHS = 20
LEARNING_RATE = 0.0001 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("1. Veriler yükleniyor...")
train_dataset = AkcigerRaporDataset(TRAIN_IMG_DIR, TRAIN_JSON_DIR, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

model = YeniNesilAkcigerModeli(embed_size=EMBED_SIZE, vocab_size=VOCAB_SIZE).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=SOZLUK["<PAD>"])
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ==========================================
# 2. EN SON KAYDEDİLEN TRANSFORMER MODELİNİ BULMA 
# ==========================================
# Kaydedilen dosyaların ismine (vlm_transformer_epoch_) göre arama yapıyoruz
kayitli_modeller = glob.glob("vlm_transformer_epoch_*.pth")
baslangic_epoch = 0

if kayitli_modeller:
    # Dosya adındaki sayıya bakarak en büyük (en son biten) epoch'u bul
    en_son_model = max(kayitli_modeller, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    baslangic_epoch = int(en_son_model.split('_')[-1].split('.')[0])
    
    # Ağırlıkları belleğe al
    model.load_state_dict(torch.load(en_son_model, weights_only=True))
    print(f"\n--> {en_son_model} dosyası başarıyla bulundu!")
    print(f"--> Eğitime {baslangic_epoch + 1}. Epoch'tan devam edilecek.\n")
else:
    print("\n--> Kayıtlı Transformer modeli bulunamadı, eğitim 1. epoch'tan baştan başlayacak.\n")

# ==========================================
# 3. KALDIĞI YERDEN TRANSFORMER EĞİTİM DÖNGÜSÜ
# ==========================================
if __name__ == "__main__":
    for epoch in range(baslangic_epoch, NUM_EPOCHS):
        model.train()
        toplam_loss = 0
        
        loop = tqdm(train_loader, total=len(train_loader), desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
        
        for batch_idx, (images, captions) in enumerate(loop):
            images = images.to(device)
            captions = captions.to(device)
            
            # Transformer Öğrenme Mantığı (Teacher Forcing)
            decoder_input = captions[:, :-1] 
            hedef_etiketler = captions[:, 1:]
            
            optimizer.zero_grad()
            outputs = model(images, decoder_input)
            
            outputs_flat = outputs.contiguous().view(-1, VOCAB_SIZE)
            hedef_flat = hedef_etiketler.contiguous().view(-1)
            
            loss = criterion(outputs_flat, hedef_flat)
            loss.backward()
            
            # Gradiyan Patlamasını (Gradient Explosion) Önleme Kalkanı
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            toplam_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        ortalama_loss = toplam_loss / len(train_loader)
        print(f"--> Epoch {epoch+1} Tamamlandı. Ortalama Loss: {ortalama_loss:.4f}")
        
        torch.save(model.state_dict(), f"vlm_transformer_epoch_{epoch+1}.pth")

    print("\nTransformer Devam Eğitimi Başarıyla Tamamlandı!")