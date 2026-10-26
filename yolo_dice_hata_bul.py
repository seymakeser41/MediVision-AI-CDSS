import os
mask_klasoru = r"C:\Users\Seyma\Desktop\UNET_DATASET\masks\test"

if not os.path.exists(mask_klasoru):
    print("HATA: Böyle bir klasör bilgisayarda yok! Yol yanlış.")
else:
    dosyalar = os.listdir(mask_klasoru)
    print(f"Klasör bulundu! İçinde {len(dosyalar)} adet dosya var.")
    if len(dosyalar) > 0:
        print(f"Maske klasöründeki ilk dosyanın tam adı: {dosyalar[0]}")