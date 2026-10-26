import os
import json
from pathlib import Path

# 1. Dizin Tanımlamaları (Kendi sisteminize göre güncelleyin)
IMAGES_DIR = Path(r"C:\Users\Seyma\Desktop\p10\images") # İçinde resimlerin hazır bulunduğu klasör
MAPPING_TXT = Path(r"C:\Users\Seyma\Desktop\p10_klasör_eslestirme.txt") # Yolların bulunduğu metin belgesi
REPORTS_SOURCE_DIR = Path(r"C:\Users\Seyma\Downloads\p10-doktor-raporlari") # Raporların p10/ şeklinde başladığı ana kaynak klasör
TARGET_REPORTS_DIR = Path(r"C:\Users\Seyma\Desktop\p10\reports") # JSON dosyalarının kaydedileceği hedef

TARGET_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# 2. Eşleştirme Sözlüğü Oluşturma
path_map = {}
with open(MAPPING_TXT, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        parts = line.split()
        if len(parts) >= 2:
            full_path = parts[-1]  
            
            if "p10" in full_path:
                p10_index = full_path.find("p10")
                relevant_path = Path(full_path[p10_index:]) 
                
                image_name = relevant_path.name
                parent_dir = relevant_path.parent # Örn: p10/p10000032/s50414267
                
                path_map[image_name] = parent_dir

# 3. Resimleri Gezme ve JSON Dönüşümü
basarili = 0
hatali = 0

print("JSON dönüştürme işlemi başlatılıyor...")

for img_path in IMAGES_DIR.glob("*.jpg"):
    img_name = img_path.name
    
    if img_name in path_map:
        # KRİTİK DÜZELTME: Artık s... bir klasör değil, doğrudan .txt dosyası olarak aranıyor.
        # Örn: C:\Users\Seyma\Downloads\p10-doktor-raporlari\p10\p10000032\s50414267.txt
        rapor_dosyasi = REPORTS_SOURCE_DIR / f"{path_map[img_name]}.txt"
        
        # Bazı durumlarda dosya uzantısız inmiş olabilir, onu da kapsayalım:
        rapor_dosyasi_uzantisiz = REPORTS_SOURCE_DIR / str(path_map[img_name])
        
        guncel_rapor_yolu = None
        if rapor_dosyasi.exists():
            guncel_rapor_yolu = rapor_dosyasi
        elif rapor_dosyasi_uzantisiz.exists():
            guncel_rapor_yolu = rapor_dosyasi_uzantisiz
            
        if guncel_rapor_yolu:
            with open(guncel_rapor_yolu, "r", encoding="utf-8") as rf:
                rapor_metni = rf.read().strip()
            
            json_icerik = {
                "image_id": img_path.stem,
                "clinical_report": rapor_metni
            }
            
            json_kayit_yolu = TARGET_REPORTS_DIR / f"{img_path.stem}.json"
            
            with open(json_kayit_yolu, "w", encoding="utf-8") as jf:
                json.dump(json_icerik, jf, ensure_ascii=False, indent=4)
                
            basarili += 1
        else:
            print(f"HATA: Rapor bulunamadı -> {rapor_dosyasi}")
            hatali += 1
    else:
        print(f"HATA: Eşleştirme TXT'sinde bulunamadı -> {img_name}")
        hatali += 1

print("-" * 40)
print(f"İşlem Tamamlandı: {basarili} rapor başarıyla JSON yapısına aktarıldı. ({hatali} hata)")