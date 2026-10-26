import os
import shutil
import random
from pathlib import Path

# ==========================================
# 1. YOLLARIN TANIMLANMASI (KENDİ SİSTEMİNE GÖRE DÜZENLE)
# ==========================================
MAPPING_TXT = Path(r"C:\Users\Seyma\Desktop\p10_klasör_eslestirme.txt") 
KAYNAK_LABELS_DIR = Path(r"C:\Users\Seyma\Desktop\p10\labels") # Etiket (.txt) dosyalarının olduğu yer
KAYNAK_KIRPILMIS_DIR = Path(r"C:\Users\Seyma\Downloads\p10-icerisinde-maskle-var") # Ekran görüntüsündeki ana klasör

HEDEF_DATASET = Path(r"C:\Users\Seyma\Desktop\YOLO_DATASET")

# ==========================================
# 2. HEDEF KLASÖRLERİN (80-10-10) OLUŞTURULMASI
# ==========================================
for split in ['train', 'val', 'test']:
    (HEDEF_DATASET / 'images' / split).mkdir(parents=True, exist_ok=True)
    (HEDEF_DATASET / 'labels' / split).mkdir(parents=True, exist_ok=True)
    (HEDEF_DATASET / 'masks' / split).mkdir(parents=True, exist_ok=True) # Maskeleri de aynı düzende yedekleyelim

# ==========================================
# 3. EŞLEŞTİRME SÖZLÜĞÜNÜN (MAPPING) OLUŞTURULMASI
# ==========================================
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
                image_stem = relevant_path.stem # Örn: fc08b4f9...ebd
                parent_dir = relevant_path.parent
                
                path_map[image_stem] = parent_dir

# ==========================================
# 4. GEÇERLİ VERİLERİN TESPİTİ VE BİRLEŞTİRİLMESİ
# ==========================================
gecerli_veriler = []

print("Veriler eşleştiriliyor, lütfen bekleyin...")
# Sadece elimizde etiketi (.txt) olan dosyalar üzerinden ilerliyoruz
for lbl_path in KAYNAK_LABELS_DIR.glob("*.txt"):
    stem = lbl_path.stem # Örn: fc08b4f9...ebd
    
    if stem in path_map:
        ilgili_klasor_yolu = KAYNAK_KIRPILMIS_DIR / path_map[stem]
        
        # O dizindeki kırpılmış jpg ve maske dosyalarını belirle
        kirpilmis_resim = ilgili_klasor_yolu / f"{stem}.jpg"
        maske_resim = ilgili_klasor_yolu / f"{stem}-mask.jpg"
        
        if kirpilmis_resim.exists():
            gecerli_veriler.append({
                'stem': stem,
                'img_kaynak': kirpilmis_resim,
                'lbl_kaynak': lbl_path,
                'mask_kaynak': maske_resim if maske_resim.exists() else None
            })

print(f"Toplam {len(gecerli_veriler)} adet tam eşleşen (Resim + Etiket) veri bulundu.")

# ==========================================
# 5. VERİLERİ KARIŞTIRMA VE %80-%10-%10 BÖLME
# ==========================================
random.seed(42)
random.shuffle(gecerli_veriler)

toplam = len(gecerli_veriler)
train_idx = int(toplam * 0.80)
val_idx = train_idx + int(toplam * 0.10)

train_data = gecerli_veriler[:train_idx]
val_data = gecerli_veriler[train_idx:val_idx]
test_data = gecerli_veriler[val_idx:]

def verileri_kopyala(veri_listesi, split_adi):
    for veri in veri_listesi:
        # Görüntü, etiket ve maske dosyaları için yeni yollar
        hedef_img = HEDEF_DATASET / 'images' / split_adi / f"{veri['stem']}.jpg"
        hedef_lbl = HEDEF_DATASET / 'labels' / split_adi / f"{veri['stem']}.txt"
        
        shutil.copy2(veri['img_kaynak'], hedef_img)
        shutil.copy2(veri['lbl_kaynak'], hedef_lbl)
        
        # İleride işine yarayabileceği için maskeyi de aynı yapıya kopyalayalım
        if veri['mask_kaynak']:
            hedef_mask = HEDEF_DATASET / 'masks' / split_adi / f"{veri['stem']}-mask.jpg"
            shutil.copy2(veri['mask_kaynak'], hedef_mask)

print("Kopyalama işlemi başladı, bu işlem veri boyutuna göre birkaç dakika sürebilir...")
verileri_kopyala(train_data, 'train')
verileri_kopyala(val_data, 'val')
verileri_kopyala(test_data, 'test')

print("-" * 40)
print("İşlem Başarıyla Tamamlandı!")
print(f"Eğitim (Train) Seti: {len(train_data)}")
print(f"Doğrulama (Val) Seti: {len(val_data)}")
print(f"Test Seti: {len(test_data)}")