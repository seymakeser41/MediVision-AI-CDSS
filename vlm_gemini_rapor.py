from google import genai
from PIL import Image
import time # Bekleme süresi için eklenen kütüphane

# ==========================================
# AYARLAR: API ANAHTARINIZI BURAYA YAPIŞTIRIN
# ==========================================
GOOGLE_API_KEY = "BURAYA_KENDI_API_ANAHTARINIZI_YAZIN"

def tıbbi_rapor_olustur_gemini(temizlenmis_goruntu_yolu):
    img = Image.open(temizlenmis_goruntu_yolu)
    client = genai.Client(api_key=GOOGLE_API_KEY)
    
    sistem_promptu = """
    Sen uzman bir radyologsun. Ekteki görüntü, U-Net yapay zeka modelimiz tarafından arka planı (kemik ve dokular) temizlenmiş, sadece 'saf' akciğer yapısının kaldığı bir röntgendir.
    Lütfen bu akciğer yapısını incele. Gerçek bir hastalık olmasa bile, yapısını analiz ederek aşağıdaki JSON formatına birebir uyacak şekilde TÜRKÇE bir rapor yaz. 
    Eğer görüntü tamamen temiz görünüyorsa, 'Normal' veya 'Bulgu rastlanmadı' gibi profesyonel ifadeler kullan.
    
    Lütfen sadece JSON formatında çıktı ver, fazladan bir sohbet veya açıklama ekleme.
    
    {
        "clinical_report": "
            INDICATION: [Hastanın durumuna dair bilgi yok, 'Yapay Zeka Destekli Tarama' yazabilirsin]
            IMPRESSION: [Akciğerdeki yapısal durum, efüzyon, anormallik veya temizlik durumu]
            NOTIFICATION: [Kalp, damar veya diğer detaylar (Görüntüde yoksa 'Değerlendirilemedi' yaz)]
        "
    }
    """
    
    print("Gemini VLM modeli akciğeri inceliyor...")
    
    # Otomatik Yeniden Deneme (Retry) Mekanizması
    max_deneme = 4
    for deneme in range(max_deneme):
        try:
            response = client.models.generate_content(
                model='gemini-3.8-flash',
                contents=[sistem_promptu, img]
            )
            return response.text
        
        except Exception as e:
            hata_metni = str(e)
            if "503" in hata_metni or "UNAVAILABLE" in hata_metni:
                bekleme_suresi = (deneme + 1) * 5
                print(f"Sunucu yoğun (503). {deneme + 1}. deneme başarısız. {bekleme_suresi} saniye bekleniyor...")
                time.sleep(bekleme_suresi)
            else:
                return f"Beklenmeyen API Hatası: {hata_metni}"
                
    return "Maksimum deneme sayısına ulaşıldı. Sunucu yanıt vermiyor."

# ==========================================
# ÇALIŞTIRMA
# ==========================================
if __name__ == "__main__":
    test_resmi_yolu = r"C:\Users\Seyma\Desktop\Bitirme_veri_hazirlk\unet_test_ciktilari\test_sonuc_00e9d3ec-6d12ebc2-453b0def-2bca4ee0-98a0a172.jpg" 
    
    rapor_ciktisi = tıbbi_rapor_olustur_gemini(test_resmi_yolu)
    
    print("\n" + "="*50)
    print("VLM (GEMINI) RADYOLOJİ RAPORU")
    print("="*50)
    print(rapor_ciktisi)