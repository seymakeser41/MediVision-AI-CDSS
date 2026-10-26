import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Eski veri yükleyiciyi değiştirmiyoruz, veri aynı veri
from veri_yukleyici import AkcigerRaporDataset, transform
# YENİ mimariyi çağırıyoruz
from model_mimarisi_v2 import YeniNesilAkcigerModeli

# ==========================================
# 1. AYARLAR VE YOLLAR
# ==========================================
TRAIN_IMG_DIR = r"C:\Users\Seyma\Desktop\YOLO_DATASET\images\train"
TRAIN_JSON_DIR = r"C:\Users\Seyma\Desktop\doktor_raporlari\train"

with open("medikal_sozluk.json", "r", encoding="utf-8") as f:
    SOZLUK = json.load(f)

VOCAB_SIZE = len(SOZLUK)
EMBED_SIZE = 256
BATCH_SIZE = 4 
NUM_EPOCHS = 20
# Transformer'lar LSTM'lere göre çok daha hassastır, bu yüzden Öğrenme Oranı (LR) biraz daha düşük tutulur
LEARNING_RATE = 0.0001 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("1. Veriler yükleniyor...")
train_dataset = AkcigerRaporDataset(TRAIN_IMG_DIR, TRAIN_JSON_DIR, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

print("2. Yeni Nesil Model (DenseNet + Transformer) GPU'ya aktarılıyor...")
model = YeniNesilAkcigerModeli(embed_size=EMBED_SIZE, vocab_size=VOCAB_SIZE).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=SOZLUK["<PAD>"])
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ==========================================
# 3. YENİ NESİL EĞİTİM DÖNGÜSÜ
# ==========================================
if __name__ == "__main__":
    print(f"3. Eğitim {device} üzerinde başlatılıyor...\n")
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        toplam_loss = 0
        
        loop = tqdm(train_loader, total=len(train_loader), desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
        
        for batch_idx, (images, captions) in enumerate(loop):
            images = images.to(device)
            captions = captions.to(device)
            
            # ----------------------------------------------------
            # TRANSFORMER ÖĞRENME MANTIĞI (TEACHER FORCING)
            # Girdi  : <SOS> word1 word2 ... (Son token hariç)
            # Etiket : word1 word2 ... <EOS> (İlk token hariç)
            # ----------------------------------------------------
            decoder_input = captions[:, :-1] 
            hedef_etiketler = captions[:, 1:]
            
            optimizer.zero_grad()
            
            # İleri Yayılım
            outputs = model(images, decoder_input)
            
            # Loss Hesaplama (Boyutları matris çarpımı için düzleştiriyoruz)
            outputs_flat = outputs.contiguous().view(-1, VOCAB_SIZE)
            hedef_flat = hedef_etiketler.contiguous().view(-1)
            
            loss = criterion(outputs_flat, hedef_flat)
            
            # Geri Yayılım
            loss.backward()
            
            # Transformer kalkanı: Gradiyan patlamasını (Gradient Clipping) önler
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            toplam_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        ortalama_loss = toplam_loss / len(train_loader)
        print(f"--> Epoch {epoch+1} Tamamlandı. Ortalama Loss: {ortalama_loss:.4f}")
        
        # Karışmaması için yeni modelleri "vlm_transformer_epoch_..." adıyla kaydediyoruz
        torch.save(model.state_dict(), f"vlm_transformer_epoch_{epoch+1}.pth")

    print("\nTransformer Eğitimi Başarıyla Tamamlandı!")