import os
import json
import re
from collections import Counter
from pathlib import Path

# ==========================================
# AYARLAR 
# ==========================================
# Sadece "train" klasörüne bakıyoruz!
TRAIN_RAPORLAR_DIR = Path(r"C:\Users\Seyma\Desktop\doktor_raporlari\train")
CIKTI_SOZLUK = "medikal_sozluk.json"

def metin_temizle(metin):
    # Metni küçült, noktalama işaretlerini boşlukla değiştir ve temizle
    metin = metin.lower()
    metin = re.sub(r'[^a-z0-9\s]', ' ', metin)
    metin = re.sub(r'\s+', ' ', metin).strip()
    return metin

def sozluk_insasi():
    tum_kelimeler = []
    dosya_sayisi = 0
    
    print("SADECE Eğitim (Train) JSON dosyaları taranıyor...")
    
    for dosya_adi in os.listdir(TRAIN_RAPORLAR_DIR):
        if dosya_adi.endswith(".json"):
            dosya_yolu = TRAIN_RAPORLAR_DIR / dosya_adi
            
            with open(dosya_yolu, 'r', encoding='utf-8') as f:
                veri = json.load(f)
                # JSON içindeki rapor kısmını alıyoruz
                rapor = veri.get("clinical_report", "")
                
                temiz_rapor = metin_temizle(rapor)
                tum_kelimeler.extend(temiz_rapor.split())
                dosya_sayisi += 1
                
    kelime_sayaci = Counter(tum_kelimeler)
    print(f"\nToplam {dosya_sayisi} eğitim raporu işlendi.")
    
    # Derin Öğrenme İçin Özel Token'lar (Zorunlu)
    # <PAD>: Boyut eşitleme, <SOS>: Başla, <EOS>: Bitir, <UNK>: Bilinmeyen
    sozluk = {
        "<PAD>": 0,
        "<SOS>": 1,
        "<EOS>": 2,
        "<UNK>": 3
    }
    
    # Filtreleme: Yalnızca en az 3 kere geçen kelimeleri ekle (Yazım hatalarını çöpe atar)
    idx = 4
    for kelime, frekans in kelime_sayaci.items():
        if frekans >= 3:
            sozluk[kelime] = idx
            idx += 1
            
    with open(CIKTI_SOZLUK, 'w', encoding='utf-8') as f:
        json.dump(sozluk, f, ensure_ascii=False, indent=4)
        
    print("\n" + "="*40)
    print("SÖZLÜK BAŞARIYLA OLUŞTURULDU")
    print("="*40)
    print(f"Modelin Öğreneceği Toplam Kelime Sayısı (Sözlük Boyutu): {len(sozluk)}")
    print(f"Çıktı dosyası: {CIKTI_SOZLUK}")

if __name__ == "__main__":
    sozluk_insasi()