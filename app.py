import os
import io
import json
import base64
import time
import re
from typing import Optional, List, Dict, Any

import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from ultralytics import YOLO

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from pydantic import BaseModel

from model_mimarisi_v2 import YeniNesilAkcigerModeli

# ==========================================
# 1. AYARLAR VE DİZİNLER
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_WEIGHTS_PATH = os.path.join(BASE_DIR, "runs", "segment", "Medikal_Vision", "yolo_lezyon_seg-2", "weights", "best.pt")
VLM_WEIGHTS_PATH = os.path.join(BASE_DIR, "vlm_transformer_epoch_20.pth")
SOZLUK_PATH = os.path.join(BASE_DIR, "medikal_sozluk.json")

TEST_IMAGES_DIR = r"C:\Users\Seyma\Desktop\UNET_DATASET\images\test"
TEST_REPORTS_DIR = r"C:\Users\Seyma\Desktop\doktor_raporlari\test"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Sistem Çalışma Birimi: {device} (CUDA Kullanılabilir: {torch.cuda.is_available()})")

# NLTK paket kontrolü
try:
    nltk.download('punkt', quiet=True)
except Exception:
    pass

smoothie = SmoothingFunction().method4
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# ==========================================
# 2. SÖZLÜK VE GÖRÜNTÜ DÖNÜŞÜMLERİ
# ==========================================
with open(SOZLUK_PATH, "r", encoding="utf-8") as f:
    SOZLUK = json.load(f)
SOZLUK_TERS = {int(v): k for k, v in SOZLUK.items()}
VOCAB_SIZE = len(SOZLUK)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 3. MODELLERİN YÜKLENMESİ
# ==========================================
print("[*] Modeller belleğe yükleniyor...")

# YOLO Modeli
if os.path.exists(YOLO_WEIGHTS_PATH):
    yolo_model = YOLO(YOLO_WEIGHTS_PATH)
    print(f"[+] YOLO Modeli Başarıyla Yüklendi: {YOLO_WEIGHTS_PATH}")
else:
    raise FileNotFoundError(f"YOLO ağırlık dosyası bulunamadı: {YOLO_WEIGHTS_PATH}")

# VLM Modeli (DenseNet-121 Encoder + Transformer Decoder)
if os.path.exists(VLM_WEIGHTS_PATH):
    vlm_model = YeniNesilAkcigerModeli(embed_size=256, vocab_size=VOCAB_SIZE).to(device)
    vlm_state = torch.load(VLM_WEIGHTS_PATH, map_location=device, weights_only=True)
    vlm_model.load_state_dict(vlm_state)
    vlm_model.eval()
    print(f"[+] Yeni Nesil Transformer VLM Modeli Başarıyla Yüklendi: {VLM_WEIGHTS_PATH}")
else:
    raise FileNotFoundError(f"VLM ağırlık dosyası bulunamadı: {VLM_WEIGHTS_PATH}")

# ==========================================
# 4. YARDIMCI VE İŞLEME FONKSİYONLARI
# ==========================================
def cv2_to_base64(img_bgr, format="JPEG", quality=88):
    """OpenCV görüntüsünü base64 data URI formatına çevirir."""
    if format == "PNG":
        _, buffer = cv2.imencode(".png", img_bgr)
        mime = "image/png"
    else:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, buffer = cv2.imencode(".jpg", img_bgr, encode_param)
        mime = "image/jpeg"
    encoded = base64.b64encode(buffer).decode("utf-8")
    return f"data:{mime};base64,{encoded}"

def pil_to_base64(img_pil, format="JPEG", quality=88):
    """PIL görüntüsünü base64 data URI formatına çevirir."""
    buffered = io.BytesIO()
    if format == "PNG":
        img_pil.save(buffered, format="PNG")
        mime = "image/png"
    else:
        img_pil.save(buffered, format="JPEG", quality=quality)
        mime = "image/jpeg"
    encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"

def format_medical_report(raw_tokens: List[str]) -> Dict[str, Any]:
    """
    Modelin ürettiği kelime dizisini temizler, noktalama ve büyük harf kurallarını uygular,
    ve standart radyoloji raporu bölümlerine (EXAMINATION, INDICATION, FINDINGS, IMPRESSION) ayırır.
    """
    text = " ".join(raw_tokens).strip()
    
    # Standart temizlik
    text = re.sub(r"\s+", " ", text)
    
    # Bölüm başlıklarını tespit et
    sections = {
        "examination": "CHEST RADIOGRAPH (PA / AP / LAT)",
        "indication": "Yapay zeka destekli tarama ve genel kontrol",
        "findings": "Akciğer alanları ve kardiyovasküler sınırlar değerlendirildi.",
        "impression": "Akut patolojik bulgu saptanmadı veya değerlendirilemedi."
    }
    
    lower_text = text.lower()
    
    # Bölümleri regex ile yakalama
    m_exam = re.search(r"examination\s+(.*?)(?=(indication|technique|findings|impression|$))", lower_text)
    if m_exam:
        sections["examination"] = m_exam.group(1).strip().capitalize()
        
    m_ind = re.search(r"indication\s+(.*?)(?=(technique|comparison|findings|impression|$))", lower_text)
    if m_ind:
        sections["indication"] = m_ind.group(1).strip().capitalize()
        
    m_find = re.search(r"findings\s+(.*?)(?=(impression|$))", lower_text)
    if m_find:
        sections["findings"] = m_find.group(1).strip().capitalize()
        
    m_imp = re.search(r"impression\s+(.*?)$", lower_text)
    if m_imp:
        sections["impression"] = m_imp.group(1).strip().capitalize()
    elif not m_find and len(text) > 0:
        # Eğer özel bölümler ayrılmadıysa tüm metni impression olarak ver
        sections["impression"] = text.capitalize()

    # Cümleleri güzelleştir
    clean_full_report = f"""TETKİK (EXAMINATION):
{sections['examination']}

KLİNİK ENDİKASYON (INDICATION):
{sections['indication']}

RADYOLOJİK BULGULAR (FINDINGS):
{sections['findings']}

SONUÇ VE İZLENİM (IMPRESSION):
{sections['impression']}"""

    return {
        "raw_text": text,
        "clean_text": clean_full_report,
        "sections": sections
    }

def is_negated(term: str, text: str) -> bool:
    """Radyoloji raporundaki olumsuzlama (negation) kalıplarını regex ile tespit eder."""
    pattern = rf"\b(no|without|free of|negative for|neither|rules? out|clear of)\b[\w\s,]{{0,40}}\b{term}\b"
    return bool(re.search(pattern, text))

def analyze_clinical_triage(report_text: str, mask_ratio: float, detection_count: int) -> Dict[str, Any]:
    """
    Klinik Karar Destek Sistemi (CDSS) triyaj motoru:
    Rapor metni ve segmentasyon bulgularını inceleyerek risk seviyesi, 
    tespit edilen klinik etiketler ve ICD-10 tanı önerileri sunar.
    """
    text_lower = report_text.lower()
    tags = []
    urgency = "low"
    urgency_label = "Rutin / Düşük Risk"
    urgency_color = "#10b981" # emerald
    summary_tr = []
    suggested_icd10 = []

    # 1. Pnömotoraks Kontrolü (En kritik)
    if "pneumothorax" in text_lower:
        if is_negated("pneumothorax", text_lower):
            tags.append({"name": "Pnömotoraks Yok", "type": "positive"})
        else:
            tags.append({"name": "Pnömotoraks Şüphesi / Tespit", "type": "critical"})
            urgency = "high"
            urgency_label = "Acil / Yüksek Risk (Pnömotoraks)"
            urgency_color = "#ef4444"
            summary_tr.append("Akciğerde hava sızıntısı (pnömotoraks) bulgusu mevcuttur, acil klinik değerlendirme önerilir.")
            suggested_icd10.append({"code": "J93.9", "desc": "Pnömotoraks, tanımlanmamış"})

    # 2. Plevral Efüzyon (Sıvı) Kontrolü
    if "effusion" in text_lower or "pleural effusion" in text_lower:
        if is_negated("effusion", text_lower) or is_negated("pleural effusion", text_lower):
            tags.append({"name": "Plevral Efüzyon Yok", "type": "positive"})
        else:
            tags.append({"name": "Plevral Efüzyon (Sıvı Birikimi)", "type": "warning"})
            if urgency != "high":
                urgency = "medium"
                urgency_label = "Orta Risk / Takip Gereklidir"
                urgency_color = "#f59e0b"
            summary_tr.append("Plevral aralıkta sıvı toplanması (efüzyon) izlenmektedir.")
            suggested_icd10.append({"code": "J90", "desc": "Plevral efüzyon"})

    # 3. Kardiyomegali (Kalp Büyümesi)
    if "cardiomegaly" in text_lower or "enlarged" in text_lower or "top normal" in text_lower:
        if is_negated("cardiomegaly", text_lower) or "top normal" in text_lower or "heart size is normal" in text_lower:
            tags.append({"name": "Kalp Boyutu Doğal / Sınırda", "type": "positive"})
        else:
            tags.append({"name": "Kardiyomegali (Kalp Büyümesi)", "type": "warning"})
            if urgency != "high":
                urgency = "medium"
                urgency_label = "Orta Risk / Kardiyovasküler Takip"
                urgency_color = "#f59e0b"
            summary_tr.append("Kardiyotorasik oranda belirgin genişleme / kardiyomegali izlenmektedir.")
            suggested_icd10.append({"code": "I51.7", "desc": "Kardiyomegali (Kalp büyümesi)"})

    # 4. Pnömoni / Konsolidasyon / Opasite
    if "consolidation" in text_lower or "opacity" in text_lower or "pneumonia" in text_lower or "infiltrate" in text_lower:
        if is_negated("consolidation", text_lower) or is_negated("opacity", text_lower) or is_negated("pneumonia", text_lower) or "lungs are clear" in text_lower:
            tags.append({"name": "Fokal Konsolidasyon Yok", "type": "positive"})
        else:
            tags.append({"name": "Konsolidasyon / Opasite (Pnömoni)", "type": "warning"})
            if urgency != "high":
                urgency = "medium"
                urgency_label = "Orta Risk / Enfeksiyon Şüphesi"
                urgency_color = "#f59e0b"
            summary_tr.append("Akciğer parankiminde dansite artışı / konsolidasyon (zatürre veya atelektazi ile uyumlu olabilir).")
            suggested_icd10.append({"code": "J18.9", "desc": "Pnömoni, tanımlanmamış organizma"})

    # 5. Kateter / Tüp / Cihaz Takibi
    if "tube" in text_lower or "catheter" in text_lower or "line" in text_lower or "dobhoff" in text_lower:
        tags.append({"name": "Tıbbi Cihaz / Kateter / Tüp", "type": "info"})
        summary_tr.append("Takılı medikal tüp / kateter pozisyonu kontrol edilmiştir.")
        suggested_icd10.append({"code": "Z96.6", "desc": "Diğer internal cihazların varlığı"})

    # 6. Tamamen Normal Bulgular
    if "clear" in text_lower or "normal" in text_lower or "unremarkable" in text_lower:
        if len(tags) == 0:
            tags.append({"name": "Normal Akciğer Parankimi", "type": "positive"})
            tags.append({"name": "Patoloji Saptanmadı", "type": "positive"})
            summary_tr.append("Akciğer parankimi doğal görünümde olup akut patolojik lezyon saptanmamıştır.")
            suggested_icd10.append({"code": "Z00.0", "desc": "Genel tıbbi muayene (Normal bulgular)"})

    # Yedek varsayılanlar ve normal özet
    if len(summary_tr) == 0:
        if urgency == "low":
            summary_tr.append("Akciğer parankimi ve kardiyovasküler sınırlar doğal görünümdedir. Akut kardiyopulmoner patoloji saptanmamıştır.")
            suggested_icd10.append({"code": "Z00.0", "desc": "Genel tıbbi muayene (Normal bulgular)"})
        else:
            summary_tr.append("Radyolojik inceleme tamamlandı. Klinik semptomlarla korelasyon önerilir.")
            suggested_icd10.append({"code": "R91.8", "desc": "Akciğer tanısal görüntülemesinde diğer anormal bulgular"})

    return {
        "urgency": urgency,
        "urgency_label": urgency_label,
        "urgency_color": urgency_color,
        "tags": tags,
        "clinical_summary_tr": " ".join(summary_tr),
        "suggested_icd10": suggested_icd10,
        "recommendations": [
            "Klinik semptomlar ve laboratuvar parametreleri ile korele ediniz.",
            "Şüpheli lezyon veya kalıcı semptom varlığında Toraks BT tetkiki değerlendirilebilir.",
            "Önceki grafi varsa karşılaştırmalı değerlendirme yapınız."
        ]
    }

def compute_nlp_metrics(reference_text: str, candidate_text: str) -> Dict[str, float]:
    """Gerçek doktor raporu ile AI raporu arasındaki BLEU ve ROUGE akademik metriklerini hesaplar."""
    if not reference_text or not candidate_text:
        return {"bleu1": 0.0, "bleu4": 0.0, "rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    
    ref_tokens = reference_text.lower().split()
    cand_tokens = candidate_text.lower().split()
    
    if len(cand_tokens) == 0:
        return {"bleu1": 0.0, "bleu4": 0.0, "rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        
    bleu1 = sentence_bleu([ref_tokens], cand_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothie) * 100
    bleu4 = sentence_bleu([ref_tokens], cand_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie) * 100
    
    scores = scorer.score(reference_text.lower(), candidate_text.lower())
    rouge1 = scores['rouge1'].fmeasure * 100
    rouge2 = scores['rouge2'].fmeasure * 100
    rougeL = scores['rougeL'].fmeasure * 100
    
    return {
        "bleu1": round(bleu1, 2),
        "bleu4": round(bleu4, 2),
        "rouge1": round(rouge1, 2),
        "rouge2": round(rouge2, 2),
        "rougeL": round(rougeL, 2)
    }

# ==========================================
# 5. FASTAPI UYGULAMASI VE API ENDPOINTLERİ
# ==========================================
app = FastAPI(
    title="MediVision AI - Klinik Karar Destek Sistemi (CDSS)",
    description="YOLO Segmentasyon ve VLM Rapor Üretimi Tabanlı Göğüs Radyolojisi Karar Destek Platformu",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Statik dosyalar dizini
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h2>MediVision AI başlatılıyor... Lütfen bekleyiniz.</h2>")

@app.get("/api/status")
async def get_system_status():
    """Sistem ve model sağlık durumu."""
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU Mode"
    vram_alloc = round(torch.cuda.memory_allocated() / (1024**3), 2) if torch.cuda.is_available() else 0.0
    
    test_count = 0
    if os.path.exists(TEST_IMAGES_DIR):
        test_count = len([f for f in os.listdir(TEST_IMAGES_DIR) if f.endswith(".jpg")])
        
    return {
        "status": "ready",
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": gpu_name,
        "vram_gb": vram_alloc,
        "yolo_model": "YOLO11-seg (Lezyon/Akciğer Bölütleme)",
        "vlm_model": "DenseNet-121 + Transformer Decoder (20 Epoch - 4865 Kelime)",
        "test_dataset_size": test_count,
        "version": "2.0 CDSS (Transformer)"
    }

@app.get("/api/samples")
async def get_sample_cases():
    """
    Test setinden önceden seçilmiş zengin ve çeşitli test vakalarını ve ilk 25 vakayı listeler.
    Jüriye ve doktora anında 1 tıkla deneme imkanı sunar.
    """
    featured_samples = [
        {
            "id": "001c3589-7aed3964-f06ba8d5-03882592-d77f222c",
            "title": "Vaka 1: Normal Akciğer (Negatif)",
            "finding": "Akut Kardiyopulmoner Patoloji Saptanmadı",
            "category": "Normal / Rutin",
            "badge_color": "#10b981"
        },
        {
            "id": "00258b8f-48b301d2-0cdf26b5-240d3e63-5a92e789",
            "title": "Vaka 2: Geniş Pnömotoraks",
            "finding": "Sağ Akciğerde Geniş Pnömotoraks Saptandı",
            "category": "Kritik / Pnömotoraks",
            "badge_color": "#ef4444"
        },
        {
            "id": "0063ecf4-dc149727-3da5c5fe-e949e244-373efa3c",
            "title": "Vaka 3: Göğüs Tüpü / Kateter",
            "finding": "Sol Göğüs Tüpü Yerleşimi ve Takibi",
            "category": "Girişimsel / Tüp Takibi",
            "badge_color": "#3b82f6"
        },
        {
            "id": "0112406d-29e1eb61-0d3505c6-946467fc-e8a32bb1",
            "title": "Vaka 4: Pnömoni / Konsolidasyon",
            "finding": "Sağ Bazal Pnömoni veya Atelektazi Şüphesi",
            "category": "Enfeksiyon / Opasite",
            "badge_color": "#f59e0b"
        },
        {
            "id": "01f7e5c9-7582e21b-df31df12-32a28198-444ffb02",
            "title": "Vaka 5: Pulmoner Ödem & Efüzyon",
            "finding": "İnterstisyel Pulmoner Ödem ve Küçük Efüzyon",
            "category": "Orta-Yüksek / Efüzyon",
            "badge_color": "#8b5cf6"
        },
        {
            "id": "02225efc-f8e916a2-e64e5200-a084323c-a931d86f",
            "title": "Vaka 6: Çift Taraflı Plevral Efüzyon",
            "finding": "Sol ve Sağ Plevral Efüzyon, Bibaziler Opasite",
            "category": "Efüzyon / Takip",
            "badge_color": "#ec4899"
        }
    ]
    
    # Ayrıca test klasöründen rastgele veya ilk 20 dosya listesi
    all_test_files = []
    if os.path.exists(TEST_IMAGES_DIR):
        files = [f.replace(".jpg", "") for f in os.listdir(TEST_IMAGES_DIR) if f.endswith(".jpg")]
        all_test_files = files[:30]
        
    return {
        "featured": featured_samples,
        "sample_ids": all_test_files
    }

@app.get("/api/sample-image/{sample_id}")
async def get_sample_image(sample_id: str):
    """Test klasöründeki bir görüntüyü döndürür."""
    img_path = os.path.join(TEST_IMAGES_DIR, f"{sample_id}.jpg")
    if not os.path.exists(img_path):
        raise HTTPException(status_code=404, detail="Örnek görüntü bulunamadı.")
    return FileResponse(img_path, media_type="image/jpeg")

@app.post("/api/analyze")
async def analyze_image(
    file: Optional[UploadFile] = File(None),
    sample_id: Optional[str] = Form(None)
):
    """
    Ana Uçtan Uca Çıkarım (End-to-End Inference) Endpointi:
    1. Görüntüyü alır (Dosya yüklemesi veya test veri seti ID)
    2. YOLO Bölütleme modeli ile lezyon/akciğer maskesini çıkarır
    3. Arka planı temizler ve görselleştirme katmanlarını üretir (Overlay, Cleaned ROI, Split)
    4. VLM ResNet50 + LSTM modeli ile klinik raporu üretir
    5. Klinik Karar Destek motoru ile risk ve triyaj analizini yapar
    6. Varsa orijinal doktor raporu (Ground Truth) ile BLEU/ROUGE metriklerini hesaplar
    """
    start_time = time.time()
    
    # 1. Görüntüyü Oku
    raw_img = None
    ground_truth_report = ""
    is_sample = False
    
    if file is not None and file.filename:
        content = await file.read()
        nparr = np.frombuffer(content, np.uint8)
        raw_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # Dosya adı test verisinden ise ground truth kontrol et
        base_name = os.path.splitext(file.filename)[0]
        gt_path = os.path.join(TEST_REPORTS_DIR, f"{base_name}.json")
        if os.path.exists(gt_path):
            try:
                with open(gt_path, "r", encoding="utf-8") as gf:
                    gt_data = json.load(gf)
                    ground_truth_report = gt_data.get("clinical_report", "")
            except Exception:
                pass
    elif sample_id:
        img_path = os.path.join(TEST_IMAGES_DIR, f"{sample_id}.jpg")
        if not os.path.exists(img_path):
            raise HTTPException(status_code=404, detail=f"Örnek görüntü bulunamadı: {sample_id}")
        raw_img = cv2.imread(img_path)
        is_sample = True
        
        # Ground Truth Oku
        gt_path = os.path.join(TEST_REPORTS_DIR, f"{sample_id}.json")
        if os.path.exists(gt_path):
            try:
                with open(gt_path, "r", encoding="utf-8") as gf:
                    gt_data = json.load(gf)
                    ground_truth_report = gt_data.get("clinical_report", "")
            except Exception:
                pass
    else:
        raise HTTPException(status_code=400, detail="Lütfen bir görüntü yükleyin veya bir test örneği seçin.")

    if raw_img is None:
        raise HTTPException(status_code=400, detail="Görüntü çözümlenemedi veya bozuk.")

    orig_h, orig_w = raw_img.shape[:2]
    
    # Web performansı için çok devasa resimleri görselleştirme öncesi makul boyuta getir (örn max 1280)
    max_dim = 1280
    if max(orig_h, orig_w) > max_dim:
        scale = max_dim / max(orig_h, orig_w)
        proc_img = cv2.resize(raw_img, (int(orig_w * scale), int(orig_h * scale)), interpolation=cv2.INTER_AREA)
    else:
        proc_img = raw_img.copy()

    h, w = proc_img.shape[:2]
    proc_rgb = cv2.cvtColor(proc_img, cv2.COLOR_BGR2RGB)

    # 2. YOLO İLE BÖLÜTLEME (SEGMENTATION)
    yolo_t0 = time.time()
    results = yolo_model(proc_rgb, verbose=False)
    yolo_time_ms = int((time.time() - yolo_t0) * 1000)

    has_mask = False
    combined_mask = np.zeros((h, w), dtype=np.uint8)
    boxes_info = []
    avg_conf = 0.0

    if results[0].masks is not None:
        has_mask = True
        masks_data = results[0].masks.data.cpu().numpy() # (N, H_mask, W_mask)
        # Birden fazla maske varsa birleştir
        raw_comb = np.any(masks_data, axis=0).astype(np.uint8)
        combined_mask = (cv2.resize(raw_comb, (w, h), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8) * 255

        # Kutuları al
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            avg_conf = float(np.mean(confs)) * 100 if len(confs) > 0 else 0.0
            
            for b, c in zip(boxes, confs):
                boxes_info.append({
                    "x1": int(b[0]), "y1": int(b[1]),
                    "x2": int(b[2]), "y2": int(b[3]),
                    "conf": round(float(c) * 100, 1),
                    "label": "Lezyon/Doku"
                })

    # Segmentasyon alanı oranı (%)
    mask_pixels = np.count_nonzero(combined_mask)
    total_pixels = h * w
    mask_ratio = round((mask_pixels / total_pixels) * 100, 2)

    # Temizlenmiş Görüntü (Cleaned / Background Removed)
    if has_mask:
        cleaned_bgr = cv2.bitwise_and(proc_img, proc_img, mask=combined_mask)
    else:
        cleaned_bgr = proc_img.copy()

    # Görselleştirme Katmanları Üret
    # A. Renkli Maske Overlay (Cyan / Turkuaz / Emerald Tıbbi Renk)
    overlay_bgr = proc_img.copy()
    if has_mask:
        # Renkli maske katmanı (Cyan: B=255, G=200, R=0)
        color_layer = np.zeros_like(proc_img)
        color_layer[:, :] = (235, 180, 14) # BGR: Elektrik camgöbeği/mavi
        
        # Yarı saydam birleştir
        alpha = 0.45
        mask_bool = combined_mask > 0
        overlay_bgr[mask_bool] = cv2.addWeighted(proc_img[mask_bool], 1 - alpha, color_layer[mask_bool], alpha, 0)
        
        # Kontur çizgisi çek (parlak neon kontur)
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay_bgr, contours, -1, (255, 255, 0), 2, cv2.LINE_AA)
        
        # İsteğe bağlı kutuları çiz
        for b in boxes_info:
            cv2.rectangle(overlay_bgr, (b["x1"], b["y1"]), (b["x2"], b["y2"]), (0, 255, 128), 2)
            cv2.putText(overlay_bgr, f"Lezyon %{b['conf']}", (b["x1"], max(b["y1"] - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 128), 1, cv2.LINE_AA)

    # 3. VLM MODELİ İLE RAPOR ÜRETİMİ (DENSENET-121 + TRANSFORMER CROSS-ATTENTION)
    vlm_t0 = time.time()
    cleaned_pil = Image.fromarray(cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2RGB))
    cnn_tensor = transform(cleaned_pil).unsqueeze(0).to(device)

    generated_tokens = []
    with torch.no_grad():
        # DenseNet-121 ile 49 mekansal bölge özniteliği çıkarılır (1, 49, 256)
        features = vlm_model.encoder(cnn_tensor)
        
        # Transformer Otoregresif Çözümleme (<SOS> ile başlar)
        uretilen_id_listesi = [SOZLUK["<SOS>"]]
        for _ in range(120): # maksimum 120 kelime
            tgt_tensor = torch.tensor([uretilen_id_listesi]).to(device)
            outputs = vlm_model.decoder(features, tgt_tensor)
            son_kelime_tahmini = outputs[0, -1, :]
            siradaki_id = son_kelime_tahmini.argmax().item()
            
            if siradaki_id == SOZLUK.get("<EOS>"):
                break
                
            uretilen_id_listesi.append(siradaki_id)
            
        for token in uretilen_id_listesi[1:]:
            word = SOZLUK_TERS.get(token, "")
            if word and word not in ["<SOS>", "<EOS>", "<PAD>", "<UNK>"]:
                generated_tokens.append(word)

    vlm_time_ms = int((time.time() - vlm_t0) * 1000)
    total_time_ms = int((time.time() - start_time) * 1000)

    # Raporu biçimlendir
    formatted_report = format_medical_report(generated_tokens)
    
    # 4. KLİNİK KARAR DESTEK (CDSS) VE TRİYAJ ANALİZİ
    cdss_analysis = analyze_clinical_triage(
        formatted_report["clean_text"],
        mask_ratio=mask_ratio,
        detection_count=len(boxes_info)
    )

    # 5. AKADEMİK NLP METRİKLERİ (BLEU & ROUGE)
    nlp_metrics = None
    if ground_truth_report:
        nlp_metrics = compute_nlp_metrics(ground_truth_report, formatted_report["clean_text"])

    # 6. GÖRSELLERİ BASE64 OLARAK HAZIRLA
    orig_b64 = cv2_to_base64(proc_img, "JPEG", quality=85)
    cleaned_b64 = cv2_to_base64(cleaned_bgr, "JPEG", quality=85)
    overlay_b64 = cv2_to_base64(overlay_bgr, "JPEG", quality=85)
    mask_b64 = cv2_to_base64(cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR), "PNG")

    return {
        "success": True,
        "sample_id": sample_id if is_sample else None,
        "images": {
            "original": orig_b64,
            "segmented_cleaned": cleaned_b64,
            "mask_overlay": overlay_b64,
            "mask_binary": mask_b64,
            "dimensions": {"width": w, "height": h}
        },
        "segmentation_metrics": {
            "has_mask": has_mask,
            "detection_count": len(boxes_info),
            "boxes": boxes_info,
            "avg_confidence": round(avg_conf, 1),
            "mask_area_ratio": mask_ratio,
            "yolo_time_ms": yolo_time_ms
        },
        "vlm_report": {
            "raw_report": formatted_report["raw_text"],
            "clean_report": formatted_report["clean_text"],
            "sections": formatted_report["sections"],
            "vlm_time_ms": vlm_time_ms,
            "word_count": len(generated_tokens)
        },
        "cdss": cdss_analysis,
        "ground_truth": {
            "has_ground_truth": bool(ground_truth_report),
            "report_text": ground_truth_report.strip() if ground_truth_report else None,
            "metrics": nlp_metrics
        },
        "timing": {
            "total_ms": total_time_ms,
            "yolo_ms": yolo_time_ms,
            "vlm_ms": vlm_time_ms
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("\n========================================================")
    print(" MediVision AI - Klinik Karar Destek Sistemi Sunucusu")
    print(" Yerel Adres: http://localhost:8000")
    print("========================================================\n")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
