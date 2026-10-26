import os
import json
import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
from ultralytics import YOLO
from PIL import Image

# Yeni mimarimizi çağırıyoruz
from model_mimarisi_v2 import YeniNesilAkcigerModeli

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
print("Yeni Nesil Modeller Belleğe (GPU) Yükleniyor...")

# YOLO Yükleme (Testteki ham resmi temizleyip eğitime uygun hale getirmek zorundayız)
yolo_model = YOLO(r"C:\Users\Seyma\Desktop\Bitirme_veri_hazirlk\runs\segment\Medikal_Vision\yolo_lezyon_seg-2\weights\best.pt") 

# Yeni Nesil Transformer Yükleme
VOCAB_SIZE = len(SOZLUK)
vlm_model = YeniNesilAkcigerModeli(embed_size=256, vocab_size=VOCAB_SIZE).to(device)

# DİKKAT: En son kaydedilen transformer weights dosyasının adını buraya yaz!
# Eğer 20. epoch'ta bittiyse vlm_transformer_epoch_20.pth olacaktır
vlm_model.load_state_dict(torch.load("vlm_transformer_epoch_20.pth", weights_only=True))
vlm_model.eval()

# ==========================================
# 3. OTOREGRESİF ÜRETİM (AUTOREGRESSIVE DECODING)
# ==========================================
def transformer_rapor_uret(ham_resim_yolu, max_kelime=100):
    with torch.no_grad():
        # --- ADIM 1: YOLO İLE ARKA PLAN TEMİZLİĞİ ---
        raw_img = cv2.imread(ham_resim_yolu)
        raw_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        
        results = yolo_model(raw_rgb, verbose=False)
        if results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            combined_mask = np.any(masks, axis=0).astype(np.uint8)
            binary_mask_resized = cv2.resize(combined_mask, (raw_img.shape[1], raw_img.shape[0]))
            cleaned_img = cv2.bitwise_and(raw_img, raw_img, mask=binary_mask_resized)
        else:
            cleaned_img = raw_img
            
        # --- ADIM 2: DENSENET (GÖRME MERKEZİ) ---
        cleaned_pil = Image.fromarray(cv2.cvtColor(cleaned_img, cv2.COLOR_BGR2RGB))
        cnn_tensor = transform(cleaned_pil).unsqueeze(0).to(device)
        
        # DenseNet, akciğeri 49 bölgeye ayırıp 256 boyutlu vektörlere çevirir (1, 49, 256)
        features = vlm_model.encoder(cnn_tensor) 
        
        # --- ADIM 3: TRANSFORMER (YAZMA MERKEZİ) ---
        # Üretime sadece <SOS> (Start of Sequence) ile başlıyoruz
        uretilen_id_listesi = [SOZLUK["<SOS>"]]
        
        for _ in range(max_kelime):
            # O ana kadar ürettiğimiz tüm kelimeleri tensöre çevirip modele veriyoruz
            tgt_tensor = torch.tensor([uretilen_id_listesi]).to(device) # (1, Mevcut_Uzunluk)
            
            # Transformer çapraz dikkat (Cross-Attention) yapar
            outputs = vlm_model.decoder(features, tgt_tensor) # (1, Mevcut_Uzunluk, Vocab_Size)
            
            # Sadece EN SON üretilen kelimenin olasılıklarını alıyoruz
            son_kelime_tahmini = outputs[0, -1, :] 
            
            # En yüksek olasılıklı kelimeyi seç
            siradaki_id = son_kelime_tahmini.argmax().item()
            
            # Cümle bittiyse döngüyü kır
            if siradaki_id == SOZLUK["<EOS>"]:
                break
                
            uretilen_id_listesi.append(siradaki_id)
            
        # Token ID'lerini gerçek kelimelere çevir (İlk baştaki <SOS>'i siliyoruz)
        rapor_kelimeleri = [SOZLUK_TERS[token] for token in uretilen_id_listesi[1:]]
        
        return " ".join(rapor_kelimeleri)

# ==========================================
# 4. TEST 
# ==========================================
if __name__ == "__main__":
    # Test klasöründen rastgele bir resim seç (Öncekiyle aynı olursa kıyası daha iyi yaparız)
    TEST_RESMI = r"C:\Users\Seyma\Desktop\UNET_DATASET\images\test\0e00f186-1adf3c96-bd340f40-586820a9-09ef1ed8.jpg"
    
    print("\nYeni Nesil Transformer Sistem Çalıştırılıyor...\n")
    uretilen_rapor = transformer_rapor_uret(TEST_RESMI)
    
    print("="*60)
    print("TRANSFORMER AI MODELİNİN ÜRETTİĞİ KLİNİK RAPOR")
    print("="*60)
    print(uretilen_rapor)