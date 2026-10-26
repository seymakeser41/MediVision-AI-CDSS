import os
import json
import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
from ultralytics import YOLO
from PIL import Image
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

from model_mimarisi_vlm import AkcigerRaporModeli

# NLTK için gerekli paketleri bir kere indirir
nltk.download('punkt', quiet=True)

# ==========================================
# 1. AYARLAR VE YOLLAR
# ==========================================
# Test verilerinin bulunduğu yollar (Kendi yollarını kontrol et)
TEST_IMG_DIR = r"C:\Users\Seyma\Desktop\UNET_DATASET\images\test"
TEST_JSON_DIR = r"C:\Users\Seyma\Desktop\doktor_raporlari\test"

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
print("Değerlendirme için modeller GPU'ya alınıyor...")
yolo_model = YOLO(r"C:\Users\Seyma\Desktop\Bitirme_veri_hazirlk\runs\segment\Medikal_Vision\yolo_lezyon_seg-2\weights\best.pt") 

VOCAB_SIZE = len(SOZLUK)
vlm_model = AkcigerRaporModeli(embed_size=256, hidden_size=512, vocab_size=VOCAB_SIZE).to(device)
vlm_model.load_state_dict(torch.load("vlm_rapor_model_epoch_20.pth", weights_only=True))
vlm_model.eval()

# ROUGE Skoru Hesaplayıcı Başlat
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
smoothie = SmoothingFunction().method4

# ==========================================
# 3. METRİK HESAPLAMA DÖNGÜSÜ
# ==========================================
def rapor_uret(ham_resim_yolu, max_kelime=100):
    with torch.no_grad():
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
            
        cleaned_pil = Image.fromarray(cv2.cvtColor(cleaned_img, cv2.COLOR_BGR2RGB))
        cnn_tensor = transform(cleaned_pil).unsqueeze(0).to(device)
        features = vlm_model.encoder(cnn_tensor).unsqueeze(1) 
        
        rapor = []
        states = None
        inputs = features 
        
        for _ in range(max_kelime):
            lstm_out, states = vlm_model.decoder.lstm(inputs, states)
            outputs = vlm_model.decoder.linear(lstm_out.squeeze(1))
            tahmin_id = outputs.argmax(1).item()
            
            if tahmin_id == SOZLUK["<EOS>"]:
                break
            
            tahmin_kelime = SOZLUK_TERS[tahmin_id]
            # <SOS> veya <PAD> gibi sistem tokenlerini rapora dahil etmiyoruz
            if tahmin_kelime not in ["<SOS>", "<PAD>", "<UNK>"]:
                rapor.append(tahmin_kelime)
            
            inputs = vlm_model.decoder.embed(torch.tensor([tahmin_id]).to(device)).unsqueeze(1)
            
        return " ".join(rapor)

if __name__ == "__main__":
    test_resimleri = [f for f in os.listdir(TEST_IMG_DIR) if f.endswith('.jpg')]
    
    toplam_bleu = 0
    toplam_rouge1 = 0
    toplam_rougeL = 0
    gecerli_test_sayisi = 0
    
    print(f"\nToplam {len(test_resimleri)} test verisi üzerinde akademik analiz başlatılıyor...")
    
    for img_adi in tqdm(test_resimleri, desc="Test Ediliyor"):
        img_yolu = os.path.join(TEST_IMG_DIR, img_adi)
        json_yolu = os.path.join(TEST_JSON_DIR, img_adi.replace('.jpg', '.json'))
        
        if not os.path.exists(json_yolu):
            continue
            
        # Orijinal (Ground Truth) raporu oku
        with open(json_yolu, 'r', encoding='utf-8') as f:
            veri = json.load(f)
            gercek_rapor = veri.get("clinical_report", "").lower()
            
        # Modeli çalıştır ve tahmin üret
        uretilen_rapor = rapor_uret(img_yolu)
        
        # --- METRİK 1: BLEU SKORU ---
        # Referans ve tahmin kelimelerini listeye çeviriyoruz
        referans_tokenler = gercek_rapor.split()
        tahmin_tokenler = uretilen_rapor.split()
        
        # Eğer model boş rapor üretirse hataya düşmemesi için kontrol
        if len(tahmin_tokenler) > 0:
            bleu_score = sentence_bleu([referans_tokenler], tahmin_tokenler, smoothing_function=smoothie)
            toplam_bleu += bleu_score
            
            # --- METRİK 2: ROUGE SKORU ---
            rouge_scores = scorer.score(gercek_rapor, uretilen_rapor)
            toplam_rouge1 += rouge_scores['rouge1'].fmeasure
            toplam_rougeL += rouge_scores['rougeL'].fmeasure
            
            gecerli_test_sayisi += 1

    # Ortalamaları Hesapla ve Yazdır
    ortalama_bleu = (toplam_bleu / gecerli_test_sayisi) * 100
    ortalama_rouge1 = (toplam_rouge1 / gecerli_test_sayisi) * 100
    ortalama_rougeL = (toplam_rougeL / gecerli_test_sayisi) * 100
    
    print("\n" + "="*50)
    print("AKADEMİK TEST SONUÇLARI (NLP METRİKLERİ)")
    print("="*50)
    print(f"Toplam Test Edilen Röntgen Sayısı : {gecerli_test_sayisi}")
    print(f"BLEU-4 Skoru (Kelime Örtüşmesi)   : %{ortalama_bleu:.2f}")
    print(f"ROUGE-1 Skoru (Terim Yakalama)    : %{ortalama_rouge1:.2f}")
    print(f"ROUGE-L Skoru (Cümle Yapısı)      : %{ortalama_rougeL:.2f}")
    print("="*50)