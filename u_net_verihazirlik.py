import os
import shutil
from pathlib import Path

# ==========================================
# 1. YOLLARIN TANIMLANMASI
# ==========================================
# Tüm ham verilerin (kırpılmamış) toplu halde bulunduğu klasör
KAYNAK_HAM_RESIMLER = Path(r"C:\Users\Seyma\Desktop\p10\images")

# Maskelerin daha önce 80-10-10 olarak ayrıldığı referans veri seti
REFERANS_YOLO_DATASET = Path(r"C:\Users\Seyma\Desktop\YOLO_DATASET")

# Yeni oluşturulacak U-Net / Akciğer Bulucu Segmentasyon Veri Seti
HEDEF_SEG_DATASET = Path(r"C:\Users\Seyma\Desktop\UNET_DATASET")

# ==========================================
# 2. HEDEF KLASÖRLERİN OLUŞTURULMASI
# ==========================================
for split in ['train', 'val', 'test']:
    (HEDEF_SEG_DATASET / 'images' / split).mkdir(parents=True, exist_ok=True)
    (HEDEF_SEG_DATASET / 'masks' / split).mkdir(parents=True, exist_ok=True)

# ==========================================
# 3. MASKELERİ REFERANS ALARAK HAM GÖRÜNTÜLERİ EŞLEŞTİRME
# ==========================================
islenen_dosya = 0
bulunamayan_ham_resim = 0

print("Ham veriler ve maskeler senkronize ediliyor...")

# train, val, test klasörlerini sırayla geziyoruz
for split in ['train', 'val', 'test']:
    referans_maske_dir = REFERANS_YOLO_DATASET / 'masks' / split
    
    if not referans_maske_dir.exists():
        continue
        
    # İlgili split içindeki maskeleri bul (Örn: fc08b4f9...-mask.jpg)
    for maske_yolu in referans_maske_dir.glob("*-mask.jpg"):
        maske_adi = maske_yolu.name
        
        # Maske adından "-mask" kısmını atarak orijinal resim adını buluyoruz
        # Örn: "fc08b4f9...-mask.jpg" -> "fc08b4f9....jpg"
        orijinal_resim_adi = maske_adi.replace("-mask", "")
        
        ham_resim_kaynak = KAYNAK_HAM_RESIMLER / orijinal_resim_adi
        
        # Hedef yollar
        hedef_ham_resim = HEDEF_SEG_DATASET / 'images' / split / orijinal_resim_adi
        hedef_maske = HEDEF_SEG_DATASET / 'masks' / split / maske_adi
        
        # Eğer ham resim C:\Users\Seyma\Desktop\p10\images içinde varsa kopyala
        if ham_resim_kaynak.exists():
            shutil.copy2(ham_resim_kaynak, hedef_ham_resim)
            shutil.copy2(maske_yolu, hedef_maske)
            islenen_dosya += 1
        else:
            print(f"HATA: Maskesi var ama ham resmi bulunamadı -> {orijinal_resim_adi}")
            bulunamayan_ham_resim += 1

print("-" * 50)
print("Veri Seti Hazırlığı Tamamlandı!")
print(f"Başarıyla Senkronize Edilen Çift Sayısı: {islenen_dosya}")
print(f"Bulunamayan Ham Resim Sayısı: {bulunamayan_ham_resim}")
print(f"Yeni Segmentasyon Klasörünüz: {HEDEF_SEG_DATASET.absolute()}")