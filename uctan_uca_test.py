import os
import json
import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
from ultralytics import YOLO
from PIL import Image

from model_mimarisi_vlm import AkcigerRaporModeli

# ==========================================
# 1. AYARLAR VE SÖZLÜK
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("medikal_sozluk.json", "r", encoding="utf-8") as f:
    SOZLUK = json.load(f)
SOZLUK_TERS = {int(v): k for k, v in SOZLUK.items()}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 2. MODELLERİN YÜKLENMESİ
# ==========================================
print("Modeller belleğe (GPU) yükleniyor...")

# A. YOLO Yükleme (U-Net yerine geçen hafif arka plan temizleyicimiz)
# Kendi YOLO ağırlıklarının yolunu buraya yaz
yolo_model = YOLO(r"C:\Users\Seyma\Desktop\Bitirme_veri_hazirlk\runs\segment\Medikal_Vision\yolo_lezyon_seg-2\weights\best.pt") 

# B. Encoder-Decoder Yükleme (LSTM - Yazar)
VOCAB_SIZE = len(SOZLUK)
vlm_model = AkcigerRaporModeli(embed_size=256, hidden_size=512, vocab_size=VOCAB_SIZE).to(device)
vlm_model.load_state_dict(torch.load("vlm_rapor_model_epoch_20.pth", weights_only=True))
vlm_model.eval()

# ==========================================
# 3. UÇTAN UCA ÇIKARIM (INFERENCE) FONKSİYONU
# ==========================================
def sistemi_calistir(ham_resim_yolu, max_kelime=100):
    with torch.no_grad():
        # --- ADIM 1: YOLO İLE ARKA PLAN TEMİZLİĞİ ---
        raw_img = cv2.imread(ham_resim_yolu)
        raw_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        
        # YOLO modelini resim üzerinde çalıştır
        results = yolo_model(raw_rgb, verbose=False)
        
        # YOLO akciğer maskesi bulduysa uygula, bulamadıysa orijinal resmi kullan
        if results[0].masks is not None:
            # Maskeleri al (YOLO sağ ve sol akciğeri 2 ayrı maske veya tek maske olarak bulabilir)
            masks = results[0].masks.data.cpu().numpy()
            # Tüm bulunan maskeleri birleştir (bitwise OR mantığı)
            combined_mask = np.any(masks, axis=0).astype(np.uint8)
            # Maskeyi orijinal resim boyutuna getir
            binary_mask_resized = cv2.resize(combined_mask, (raw_img.shape[1], raw_img.shape[0]))
            
            # Maskeyi orijinal resme uygula
            cleaned_img = cv2.bitwise_and(raw_img, raw_img, mask=binary_mask_resized)
        else:
            print("Uyarı: YOLO akciğer bulamadı, ham resim işleniyor.")
            cleaned_img = raw_img
            
        # --- ADIM 2: RESNET (ENCODER) ---
        cleaned_pil = Image.fromarray(cv2.cvtColor(cleaned_img, cv2.COLOR_BGR2RGB))
        cnn_tensor = transform(cleaned_pil).unsqueeze(0).to(device)
        
        features = vlm_model.encoder(cnn_tensor).unsqueeze(1) # (1, 1, 256)
        
        # --- ADIM 3: LSTM (DECODER - GREEDY SEARCH) ---
        rapor = []
        states = None
        inputs = features # İlk girdi ResNet'in ürettiği görüntü vektörü
        
        for _ in range(max_kelime):
            lstm_out, states = vlm_model.decoder.lstm(inputs, states)
            outputs = vlm_model.decoder.linear(lstm_out.squeeze(1))
            
            tahmin_id = outputs.argmax(1).item()
            
            if tahmin_id == SOZLUK["<EOS>"]:
                break
                
            tahmin_kelime = SOZLUK_TERS[tahmin_id]
            rapor.append(tahmin_kelime)
            
            inputs = vlm_model.decoder.embed(torch.tensor([tahmin_id]).to(device)).unsqueeze(1)
            
        return " ".join(rapor)

# ==========================================
# 4. TEST 
# ==========================================
if __name__ == "__main__":
    TEST_RESMI = r"C:\Users\Seyma\Desktop\UNET_DATASET\images\test\0e00f186-1adf3c96-bd340f40-586820a9-09ef1ed8.jpg"
    
    print("\nYOLO ve LSTM entegre çalıştırılıyor...\n")
    uretilen_rapor = sistemi_calistir(TEST_RESMI)
    
    print("="*50)
    print("AI MODELİNİN ÜRETTİĞİ KLİNİK RAPOR")
    print("="*50)
    print(uretilen_rapor)