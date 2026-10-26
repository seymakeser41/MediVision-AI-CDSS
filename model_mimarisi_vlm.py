import torch
import torch.nn as nn
import torchvision.models as models

# ==========================================
# 1. GÖRME MERKEZİ (ENCODER)
# ==========================================
class CNN_Encoder(nn.Module):
    def __init__(self, embed_size):
        super(CNN_Encoder, self).__init__()
        # Pre-trained ResNet50 yüklüyoruz
        resnet = models.resnet50(weights='IMAGENET1K_V1')
        
        # Sınıflandırma yapan son katmanı (fc) kesip atıyoruz, çünkü biz sınıflandırma değil, öznitelik istiyoruz
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        
        # ResNet'in 2048 boyutlu çıkışını bizim istediğimiz embed_size (256) boyutuna sıkıştıran katman
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        # images boyutu: (Batch, 3, 224, 224)
        features = self.resnet(images) 
        features = features.reshape(features.size(0), -1) # (Batch, 2048)
        features = self.linear(features) # (Batch, 256)
        features = self.bn(features)
        return features

# ==========================================
# 2. YAZMA MERKEZİ (DECODER)
# ==========================================
class RNN_Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(RNN_Decoder, self).__init__()
        # Kelimeleri 256 boyutlu vektörlere dönüştüren katman
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        # Hafıza hücreli LSTM katmanı
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        
        # LSTM çıkışını 4865 kelimelik sözlük boyutuna eşleyen karar katmanı
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        # Captions boyutu: (Batch, 150)
        # Son kelimeyi (EOS) atıyoruz çünkü model zaten bir kelime kaydırarak tahmin yapacak
        embeddings = self.embed(captions[:, :-1]) 
        
        # Resimden gelen özellikleri (features) zaman çizelgesinin EN BAŞINA (t=0) ekliyoruz
        # Model önce resme bakacak, sonra kelimeleri yazmaya başlayacak
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1) 
        
        lstm_out, _ = self.lstm(embeddings)
        outputs = self.linear(lstm_out)
        
        # Çıktı boyutu: (Batch, 150, 4865) -> Her kelime adımı için 4865 ihtimal
        return outputs

# ==========================================
# 3. UÇTAN UCA BİRLEŞTİRİLMİŞ MİMARİ
# ==========================================
class AkcigerRaporModeli(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(AkcigerRaporModeli, self).__init__()
        self.encoder = CNN_Encoder(embed_size)
        self.decoder = RNN_Decoder(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

# ==========================================
# 4. MİMARİ TESTİ (DRY RUN)
# ==========================================
if __name__ == "__main__":
    print("Derin Öğrenme Mimarisi Test Ediliyor...")
    
    # Hiperparametreler (6GB VRAM için optimize edilmiştir)
    EMBED_SIZE = 256
    HIDDEN_SIZE = 512
    VOCAB_SIZE = 4865 # Senin oluşturduğun medikal sözlük boyutu
    BATCH_SIZE = 4
    
    # Sahte (Dummy) Tensörler
    fake_images = torch.randn(BATCH_SIZE, 3, 224, 224)
    fake_captions = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, 150))
    
    # Modeli Başlat
    model = AkcigerRaporModeli(EMBED_SIZE, HIDDEN_SIZE, VOCAB_SIZE)
    
    # İleri Yayılım (Forward Pass)
    outputs = model(fake_images, fake_captions)
    
    print("\nTest Başarılı! Mimari sorunsuz derlendi.")
    print(f"Çıktı Tensörü Boyutu: {outputs.shape}")
    print("Açıklama: Model, 4 resim için 150 adım boyunca sözlükteki 4865 kelimenin ayrı ayrı olasılıklarını hesapladı.")