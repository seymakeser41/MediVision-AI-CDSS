import torch
import torch.nn as nn
import torchvision.models as models
import math

# ==========================================
# 1. POZİSYONEL KODLAMA (Transformer'lar kelime sırasını bilmek için buna ihtiyaç duyar)
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=150):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, max_len, embed_size)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x boyutu: (Batch, Seq_Len, Embed_Size)
        x = x + self.pe[:, :x.size(1), :]
        return x

# ==========================================
# 2. YENİ MEDİKAL GÖRME MERKEZİ (DENSENET-121 + SPATIAL ATTENTION İÇİN HAZIRLIK)
# ==========================================
class MedikalEncoder(nn.Module):
    def __init__(self, embed_size):
        super(MedikalEncoder, self).__init__()
        # Medikal görüntülerde ResNet'ten çok daha başarılı olan DenseNet
        densenet = models.densenet121(weights='DEFAULT')
        
        # Sınıflandırıcıyı atıp sadece özellik çıkaran CNN katmanlarını alıyoruz
        self.features = densenet.features
        
        # DenseNet çıkışı 1024 kanallıdır. Bunu bizim Transformer boyutumuza (256) düşürüyoruz
        self.linear = nn.Linear(1024, embed_size)
        self.bn = nn.LayerNorm(embed_size)

    def forward(self, images):
        # images: (Batch, 3, 224, 224)
        x = self.features(images) # Çıktı: (Batch, 1024, 7, 7)
        
        # Görüntüyü 7x7 = 49 farklı bölgeye (patch) ayırıyoruz.
        # Bu sayede model "sağ alt lob", "sol üst lob" gibi kavramları öğrenebilecek.
        x = x.flatten(2).permute(0, 2, 1) # Çıktı: (Batch, 49 bölgeler, 1024 özellik)
        
        x = self.linear(x) # (Batch, 49, 256)
        x = self.bn(x)
        return x # Bu 49 bölge, Transformer'ın hafızası (Memory) olacak

# ==========================================
# 3. YENİ YAZMA MERKEZİ (TRANSFORMER DECODER)
# ==========================================
class TransformerDecoderYazar(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads=8, num_layers=3, max_len=150):
        super(TransformerDecoderYazar, self).__init__()
        self.embed_size = embed_size
        
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.pos_encoder = PositionalEncoding(embed_size, max_len)
        
        # 6GB VRAM için optimize edilmiş 3 katmanlı, 8 başlı Transformer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size, 
            nhead=num_heads, 
            dim_feedforward=1024, 
            dropout=0.1, 
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Sözlük boyutuna dönüştüren nihai karar katmanı
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def gelecek_kelimeleri_gizle_maskesi(self, size):
        # Modelin eğitim sırasında gelecekteki kelimelere bakıp kopya çekmesini engeller (Causal Mask)
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, features, captions):
        # features: (Batch, 49, 256) -> Resmin 49 bölgesinin özellikleri
        # captions: (Batch, 150) -> Kelime ID'leri
        
        hedef_uzunluk = captions.size(1)
        hedef_maske = self.gelecek_kelimeleri_gizle_maskesi(hedef_uzunluk).to(captions.device)
        
        # Kelimeleri vektöre çevir ve pozisyonlarını ekle
        x = self.embed(captions) * math.sqrt(self.embed_size)
        x = self.pos_encoder(x)
        
        # Çapraz Dikkat (Cross-Attention) burada gerçekleşir: 
        # Yazılar (x) ile Görüntü Bölgeleri (features) birbiriyle konuşur
        output = self.transformer(tgt=x, memory=features, tgt_mask=hedef_maske)
        
        output = self.fc_out(output) # (Batch, 150, Vocab_Size)
        return output

# ==========================================
# 4. YENİ UÇTAN UCA MİMARİ
# ==========================================
class YeniNesilAkcigerModeli(nn.Module):
    def __init__(self, embed_size, vocab_size, num_heads=8, num_layers=3):
        super(YeniNesilAkcigerModeli, self).__init__()
        self.encoder = MedikalEncoder(embed_size)
        self.decoder = TransformerDecoderYazar(vocab_size, embed_size, num_heads, num_layers)

    def forward(self, images, captions):
        # Eğitim sırasında ileri yayılım
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

# ==========================================
# TEST (DRY RUN)
# ==========================================
if __name__ == "__main__":
    print("Yeni Nesil DenseNet + Transformer Mimarisi Test Ediliyor...")
    
    dummy_img = torch.randn(4, 3, 224, 224) # 4 adet resim
    dummy_cap = torch.randint(0, 4865, (4, 150)) # 150 uzunluğunda 4 adet rapor
    
    model = YeniNesilAkcigerModeli(embed_size=256, vocab_size=4865)
    out = model(dummy_img, dummy_cap)
    
    print("Sistem Başarıyla Derlendi!")
    print(f"Çıkış Boyutu: {out.shape} -> (Batch, Kelime Sayısı, Sözlük Boyutu)")
    print("VRAM dostu, dikkat mekanizmalı sistem eğitime hazır.")