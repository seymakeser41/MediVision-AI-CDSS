import os
import shutil
from pathlib import Path

# ==========================================
# AYARLAR (Kendi bilgisayarına göre yolları güncelle)
# ==========================================
# U-Net için daha önce ayırdığın görüntülerin ana klasörü
UNET_IMAGES_DIR = Path(r"C:\Users\Seyma\Desktop\UNET_DATASET\images")

# 23.000 JSON dosyasının toplu halde durduğu klasör
TUM_RAPORLAR_DIR = Path(r"C:\Users\Seyma\Desktop\p10\reports")

# Yeni ayrılmış raporların kaydedileceği hedef klasör
HEDEF_RAPORLAR_DIR = Path(r"C:\Users\Seyma\Desktop\doktor_raporlari")

def raporlari_dagit():
    # Eğitim, doğrulama ve test alt klasörlerini (train, val, test) bul
    split_klasorleri = [d for d in os.listdir(UNET_IMAGES_DIR) if os.path.isdir(UNET_IMAGES_DIR / d)]
    
    kopyalanan_sayisi = 0
    bulunamayanlar = 0

    for split in split_klasorleri:
        resim_klasoru = UNET_IMAGES_DIR / split
        hedef_split_klasoru = HEDEF_RAPORLAR_DIR / split
        
        # Hedef klasörü oluştur (örn: AYRILMIS_RAPORLAR/train)
        os.makedirs(hedef_split_klasoru, exist_ok=True)
        
        resimler = [f for f in os.listdir(resim_klasoru) if f.endswith('.jpg')]
        print(f"\n[{split.upper()}] klasöründe {len(resimler)} resim bulundu, raporlar eşleştiriliyor...")
        
        for resim_adi in resimler:
            # Uzantıyı at, .json ekle (Örn: resim_1.jpg -> resim_1.json)
            dosya_kok_adi = os.path.splitext(resim_adi)[0]
            beklenen_json_adi = dosya_kok_adi + ".json"
            
            kaynak_json_yolu = TUM_RAPORLAR_DIR / beklenen_json_adi
            hedef_json_yolu = hedef_split_klasoru / beklenen_json_adi
            
            if kaynak_json_yolu.exists():
                shutil.copy2(kaynak_json_yolu, hedef_json_yolu)
                kopyalanan_sayisi += 1
            else:
                bulunamayanlar += 1
                # Gerekirse yorum satırını kaldırıp hangi dosyaların eksik olduğunu görebilirsin
                # print(f"UYARI: {beklenen_json_adi} bulunamadı!")
                
    print("\n" + "="*40)
    print("EŞLEŞTİRME VE DAĞITIM TAMAMLANDI")
    print("="*40)
    print(f"Başarıyla Kopyalanan Rapor Sayısı: {kopyalanan_sayisi}")
    print(f"Resmi Olup Raporu Bulunamayan Sayısı: {bulunamayanlar}")

if __name__ == "__main__":
    raporlari_dagit()