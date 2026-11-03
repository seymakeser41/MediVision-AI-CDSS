# MediVision AI - Klinik Karar Destek Sistemi (CDSS) 🩺🤖

MediVision AI, göğüs radyografilerini (X-ray) analiz ederek uçtan uca İngilizce radyoloji raporu üreten ve doktorlar için Açıklanabilir Yapay Zeka (XAI) tabanlı bir arayüz sunan derin öğrenme projesidir. 

Bu sistem, klinisyenlerin iş yükünü hafifletmek ve teşhis doğruluğunu artırmak amacıyla geliştirilmiş bir **Görsel-Dil Modeli (Vision-Language Model - VLM)** mimarisidir. TEKNOFEST Sağlıkta Yapay Zeka yarışması standartlarına uygun olarak tasarlanmıştır.

---

## 🎯 Projenin Amacı ve Yaklaşımımız
Serbest metin (free-text) medikal rapor üretimi, standart sınıflandırma problemlerinden çok daha karmaşıktır. Bu projede modelin performansını artırmak ve ezberlemeyi (halüsinasyon) önlemek için kapsamlı bir **Ablasyon Çalışması (Ablation Study)** yapılmıştır.

Sistemimizde şu mimariler kıyaslanabilir olarak sunulmaktadır:
* **[V2] DenseNet-121 + Transformer + YOLO (Önerilen):** Görüntüyü 49 lokal bölgeye ayırıp, çapraz dikkat (cross-attention) mekanizması ile kelime tekrarlarını çözen en gelişmiş modelimiz.
* **[V1] ResNet-50 + LSTM + YOLO:** Görüntünün global vektörle işlendiği geleneksel sıralı modelimiz.
* **[V0] Uçtan Uca ResNet + LSTM (YOLO'suz):** Segmentasyon temizliği olmadan, doğrudan ham görüntülerle eğitilmiş taban (baseline) modelimiz.

Sistemin kalbinde, modelin tıbbi cihaz kabloları veya kemik yapıları gibi gürültülere odaklanmasını engellemek için **YOLOv8** ile geliştirilmiş otomatik akciğer parankimi segmentasyonu yer alır.

---

## 📊 Model Performansı ve Metrikler
Hiç görülmemiş **2.354 test radyografisi** üzerinde yapılan otomatik değerlendirme sonuçları:

| Metrik | Skor | Açıklama |
| :--- | :--- | :--- |
| **YOLO Dice Skoru** | `%94.00+` | Akciğer bölgesini (Ground Truth) doğru maskeleme başarısı. |
| **ROUGE-1** | `%36.07` | Hastalık/Bulgu terimlerini doktor raporuyla birebir yakalama oranı. |
| **ROUGE-L** | `%26.72` | Üretilen raporun cümle bütünlüğü ve akıcılık skoru. |
| **BLEU-4** | `%3.72` | 4 kelimelik katı medikal şablonların ardışık eşleşme skoru. |

---

## 💻 Geliştirici Yol Haritası: Kendi Bilgisayarında Çalıştır

Bu projeyi kendi bilgisayarınızda (lokal ortamda) ayağa kaldırmak için aşağıdaki adımları sırasıyla uygulayın.

### 1. Gereksinimler
* Python 3.10 veya üzeri
* Git
* CUDA destekli bir NVIDIA GPU (Önerilen, ancak zorunlu değil. Sistem CPU ile de çalışabilir).

### 2. Projeyi Klonlayın
Terminalinizi açın ve projeyi bilgisayarınıza indirin:
```bash
git clone [https://github.com/seymakeser41/MediVision-AI-CDSS.git](https://github.com/KULLANICI_ADIN/MediVision-AI-CDSS.git)
cd MediVision-AI-CDSS
```

### 3.Gerekli Kütüphaneleri Kurun

```
pip install -r requirements.txt

```
### 4. Model Ağırlıklarını İndirin (.pth ve .pt dosyaları)
GitHub'ın 100 MB dosya sınırı nedeniyle, eğitilmiş ağır model dosyaları bu repoda bulunmamaktadır. Sistemi çalıştırabilmek için aşağıdaki bağlantıdan yapay zeka ağırlıklarını indirmelisiniz:

🔗 Model Ağırlıklarını Buradan İndirin (Google Drive) -> (https://drive.google.com/drive/folders/1tV8YrNpXFPdYoWOnFJzCwPnh625aMW6O?usp=drive_link)

İndirdiğiniz ağırlık dosyalarını (örn: best.pt, vlm_transformer_epoch_20.pth) projenin ana dizinine (app.py ile aynı yere) kopyalayın.

### 5. Sistemi Başlatın
Tüm dosyalar hazır olduğunda arayüzü başlatmak için terminale şu komutu girin:

````
python app.py
````
### 6. Arayüze Erişim
Terminalde sunucunun başladığına dair logları (örn: Uvicorn running on http://0.0.0.0:8080) gördükten sonra tarayıcınızı açın ve adres çubuğuna şunu yazın:
👉 http://localhost:8080 (veya loglarda belirtilen port numarası hangisiyse: 8000)
