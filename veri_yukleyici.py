import os
import json
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# ==========================================
# 1. AYARLAR VE SÖZLÜK YÜKLEME
# ==========================================
# Görüntü dönüşümleri (ResNet-50 için standart ImageNet ayarları)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# Sözlüğü yükle
with open("C:\\Users\\Seyma\\Desktop\\Bitirme_veri_hazirlk\\medikal_sozluk.json", "r", encoding="utf-8") as f:
    SOZLUK = json.load(f)

# Sözlüğün tersi (ID'den kelimeye dönmek için, test aşamasında lazım olacak)
SOZLUK_TERS = {v: k for k, v in SOZLUK.items()}

MAX_KELIME_UZUNLUGU = 150  # Bir raporun maksimum kelime sayısı (padding için)

# ==========================================
# 2. PYTORCH DATASET SINIFI
# ==========================================
class AkcigerRaporDataset(Dataset):
    def __init__(self, image_dir, json_dir, transform=transform):
        self.image_dir = image_dir
        self.json_dir = json_dir
        self.transform = transform
        
        # Klasördeki sadece .jpg resimleri al (JSON eşleşmesini baştan yapmıştık)
        self.images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.images)
    
    def metni_tokenize_et(self, metin):
        # 1. Metni temizle
        import re
        metin = metin.lower()
        metin = re.sub(r'[^a-z0-9\s]', ' ', metin)
        kelimeler = re.sub(r'\s+', ' ', metin).strip().split()
        
        # 2. Kelimeleri ID'ye çevir (Sözlükte yoksa <UNK> yani 3 yap)
        token_id_listesi = [SOZLUK["<SOS>"]] # Cümleye başlama işareti
        for kelime in kelimeler:
            token_id_listesi.append(SOZLUK.get(kelime, SOZLUK["<UNK>"]))
        token_id_listesi.append(SOZLUK["<EOS>"]) # Cümle bitiş işareti
        
        # 3. Padding (Cümle kısaysa 0 ile doldur, uzunsa kes)
        if len(token_id_listesi) < MAX_KELIME_UZUNLUGU:
            padding_sayisi = MAX_KELIME_UZUNLUGU - len(token_id_listesi)
            token_id_listesi.extend([SOZLUK["<PAD>"]] * padding_sayisi)
        else:
            token_id_listesi = token_id_listesi[:MAX_KELIME_UZUNLUGU]
            token_id_listesi[-1] = SOZLUK["<EOS>"] # Ne olursa olsun sonu EOS ile bitsin
            
        return torch.tensor(token_id_listesi)

    def __getitem__(self, idx):
        # A. Resmi Oku
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # B. Raporu Oku
        json_name = img_name.replace('.jpg', '.json')
        json_path = os.path.join(self.json_dir, json_name)
        
        with open(json_path, 'r', encoding='utf-8') as f:
            veri = json.load(f)
            rapor_metni = veri.get("clinical_report", "")
            
        # C. Metni tensöre (sayılara) çevir
        rapor_tensor = self.metni_tokenize_et(rapor_metni)
        
        return image, rapor_tensor

# ==========================================
# 3. TEST (ÇALIŞIYOR MU DİYE KONTROL EDELİM)
# ==========================================
if __name__ == "__main__":
    # Yolları eğitim (train) klasörüne göre ayarlıyoruz
    TEST_RESIM_YOLU = r"C:\Users\Seyma\Desktop\UNET_DATASET\images\train"
    TEST_JSON_YOLU = r"C:\Users\Seyma\Desktop\doktor_raporlari\train"
    
    # Dataset'i başlat
    dataset = AkcigerRaporDataset(TEST_RESIM_YOLU, TEST_JSON_YOLU)
    
    # Dataloader (Verileri 4'er 4'er getirecek)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Veri yükleyiciden 1 batch çekip neye benzediğine bakalım
    images, raporlar = next(iter(loader))
    
    print("\nPyTorch Veri Yükleyici Başarıyla Çalışıyor!")
    print(f"1 Batch Görüntü Tensör Boyutu: {images.shape} -> (Batch, Kanal, En, Boy)")
    print(f"1 Batch Rapor Tensör Boyutu: {raporlar.shape} -> (Batch, Kelime Sayısı)")
    
    print("\nİlk Resmin Raporunun Tensör Hali (İlk 20 kelimenin ID'leri):")
    print(raporlar[0][:20].tolist())