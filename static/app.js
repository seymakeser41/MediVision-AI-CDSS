/**
 * MediVision AI - Frontend Uygulama Mantığı
 * Klinik Karar Destek Sistemi (CDSS) & Çoklu Model Şeffaflık Arayüzü
 */

document.addEventListener("DOMContentLoaded", () => {
    // ==========================================
    // 1. MODEL TANIMLARI VE BİLGİ KARTLARI KONFİGÜRASYONU
    // ==========================================
    const MODEL_CONFIGS = {
        v3: {
            id: "v3",
            name: "[V3] DenseNet + Transformer + YOLO (Tam Sistem / Önerilen)",
            badge: "Tam Sistem / Önerilen Mimari",
            badgeClass: "badge-v3",
            tag: "49 Mekansal Bölge & Çapraz Dikkat",
            title: "YOLO11-seg + DenseNet-121 + Transformer Decoder",
            desc: "YOLOv11 ile %94.6 Dice kesinlikli akciğer izolasyonu yapılır. DenseNet-121 omurgası 49 uzamsal bölge özniteliği çıkarır; 8 başlıklı Transformer Çapraz Dikkat (Cross-Attention) mekanizması ile en yüksek klinik doğruluk, BLEU-4 ve ROUGE-L skorlarını üretir.",
            pills: [
                { icon: "shield-check", text: "%94.6 Dice ROI" },
                { icon: "zap", text: "Cross-Attention" },
                { icon: "award", text: "En Yüksek BLEU-4 (%14.6)" }
            ]
        },
        v2: {
            id: "v2",
            name: "[V2] YOLO'suz DenseNet + Transformer",
            badge: "Ablasyon: YOLO Etkisi",
            badgeClass: "badge-v2",
            tag: "Ham Röntgen Girişi & Çapraz Dikkat",
            title: "Ham Röntgen + DenseNet-121 + Transformer Decoder",
            desc: "Segmentasyon olmadan doğrudan ham röntgen girer. Transformer dikkat mekanizması sayesinde LSTM'den daha üstündür; fakat YOLO desteği olmadan çevre dokulardaki parazitler nedeniyle V3'e göre %50 daha düşük BLEU-4 üretir (YOLO'nun net katkısını kanıtlar).",
            pills: [
                { icon: "alert-triangle", text: "YOLO Devre Dışı" },
                { icon: "zap", text: "Cross-Attention" },
                { icon: "trending-up", text: "%9.7 BLEU-4" }
            ]
        },
        v1: {
            id: "v1",
            name: "[V1] ResNet + LSTM + YOLO",
            badge: "Ablasyon: Transformer Etkisi",
            badgeClass: "badge-v1",
            tag: "Global Havuzlama & LSTM Sıralı Çözümleme",
            title: "YOLO11-seg + ResNet-50 + LSTM Decoder",
            desc: "YOLO maskelemesi sayesinde ham resme (V0) göre arka plan parazitlerini engeller. Ancak ResNet'in 2048 boyutlu tekil global vektörü mekansal detayları kaybeder ve LSTM uzun klinik raporlarda gradyan zayıflaması yaşar.",
            pills: [
                { icon: "shield-check", text: "YOLO Korumalı" },
                { icon: "clock", text: "Hızlı LSTM" },
                { icon: "bar-chart-2", text: "%6.8 BLEU-4" }
            ]
        },
        v0: {
            id: "v0",
            name: "[V0] YOLO'suz Uçtan Uca ResNet + LSTM",
            badge: "Temel Model (Baseline)",
            badgeClass: "badge-v0",
            tag: "Segmentasyonsuz Ham Röntgen Girişi",
            title: "Ham Röntgen + ResNet-50 + LSTM Decoder",
            desc: "Hiçbir anatomik bölütleme veya arka plan temizliği yapılmadan doğrudan ham röntgen girer. Kemik, yumuşak doku ve parazitler nedeniyle model dikkat kaybı yaşar; segmentasyonun klinik rapordaki kritik gerekliliğini açıkça kanıtlar.",
            pills: [
                { icon: "alert-triangle", text: "YOLO Devre Dışı" },
                { icon: "eye-off", text: "Gürültülü Ham Doku" },
                { icon: "bar-chart-2", text: "%2.9 BLEU-4 (Baseline)" }
            ]
        }
    };

    // ==========================================
    // 2. STATE VE DEĞİŞKENLER
    // ==========================================
    const state = {
        selectedModel: "v3", // v3 | v2 | v1 | v0
        currentMode: "split", // split | overlay | cleaned | raw
        activeCaseId: null,
        isAnalyzing: false,
        analysisData: null,
        sliderPosition: 50, // %
        maskOpacity: 0.85,
        showContour: true,
        theme: "dark",
        activeDatasetSplit: "train",
        datasetCache: { train: null, val: null, test: null },
        datasetFilterQuery: ""
    };

    // DOM Elementleri
    const elements = {
        // Model Seçim Menüsü ve Bilgi Kartı
        modelSelectDropdown: document.getElementById("modelSelectDropdown"),
        modelInfoCard: document.getElementById("modelInfoCard"),
        micBadge: document.getElementById("micBadge"),
        micTag: document.getElementById("micTag"),
        micTitle: document.getElementById("micTitle"),
        micDesc: document.getElementById("micDesc"),
        micPillsRow: document.getElementById("micPillsRow"),

        // Sekmeler
        tabFeatured: document.getElementById("tabFeatured"),
        tabUpload: document.getElementById("tabUpload"),
        tabDataset: document.getElementById("tabDataset"),
        featuredView: document.getElementById("featuredView"),
        uploadView: document.getElementById("uploadView"),
        datasetView: document.getElementById("datasetView"),
        featuredGrid: document.getElementById("featuredGrid"),
        datasetList: document.getElementById("datasetList"),
        datasetSearchInput: document.getElementById("datasetSearchInput"),
        btnRandomSample: document.getElementById("btnRandomSample"),

        // Dropzone & Dosya Yükleme
        dropzone: document.getElementById("dropzone"),
        fileInput: document.getElementById("fileInput"),
        btnBrowseFile: document.getElementById("btnBrowseFile"),

        // Görüntü Sahnesi & Modlar
        imageStage: document.getElementById("imageStage"),
        stagePlaceholder: document.getElementById("stagePlaceholder"),
        stageLoading: document.getElementById("stageLoading"),
        loadingStageTitle: document.getElementById("loadingStageTitle"),
        loadingStageDesc: document.getElementById("loadingStageDesc"),
        activeCaseTag: document.getElementById("activeCaseTag"),
        
        // Slider Elementleri
        splitSliderContainer: document.getElementById("splitSliderContainer"),
        splitImgBefore: document.getElementById("splitImgBefore"),
        splitImgAfter: document.getElementById("splitImgAfter"),
        splitAfterWrapper: document.getElementById("splitAfterWrapper"),
        sliderHandle: document.getElementById("sliderHandle"),
        
        // Tekli Görünüm
        singleImgContainer: document.getElementById("singleImgContainer"),
        singleStageImg: document.getElementById("singleStageImg"),

        // Statik YOLO Dice Skoru Etiketi
        yoloDiceBadgeCard: document.getElementById("yoloDiceBadgeCard"),

        // Mod Düğmeleri & Kontroller
        modeButtons: document.querySelectorAll(".mode-btn"),
        maskOpacitySlider: document.getElementById("maskOpacitySlider"),
        maskOpacityVal: document.getElementById("maskOpacityVal"),
        btnToggleContour: document.getElementById("btnToggleContour"),
        btnResetZoom: document.getElementById("btnResetZoom"),
        btnFullscreen: document.getElementById("btnFullscreen"),

        // Segmentasyon Metrikleri
        valDetectionCount: document.getElementById("valDetectionCount"),
        valYoloConf: document.getElementById("valYoloConf"),
        valAreaRatio: document.getElementById("valAreaRatio"),
        valYoloTime: document.getElementById("valYoloTime"),

        // CDSS Triyaj & Risk
        cdssTriageCard: document.getElementById("cdssTriageCard"),
        triageBadge: document.getElementById("triageBadge"),
        triageSummary: document.getElementById("triageSummary"),
        clinicalTagsContainer: document.getElementById("clinicalTagsContainer"),
        triageIconWrap: document.getElementById("triageIconWrap"),

        // Rapor Sekmeleri
        rtabButtons: document.querySelectorAll(".rtab-btn"),
        reportPanes: document.querySelectorAll(".report-tab-pane"),
        tabGroundTruthBtn: document.getElementById("tabGroundTruthBtn"),
        badgeGtAvailable: document.getElementById("badgeGtAvailable"),

        // VLM Rapor Alanları
        reportSecExam: document.getElementById("reportSecExam"),
        reportSecInd: document.getElementById("reportSecInd"),
        reportSecFind: document.getElementById("reportSecFind"),
        reportSecImp: document.getElementById("reportSecImp"),
        rawTokenBox: document.getElementById("rawTokenBox"),
        cdssAdviceText: document.getElementById("cdssAdviceText"),
        btnCopyReport: document.getElementById("btnCopyReport"),
        btnTransferToEditor: document.getElementById("btnTransferToEditor"),

        // Akademik Karşılaştırma & NLP
        valBleu1: document.getElementById("valBleu1"),
        valBleu4: document.getElementById("valBleu4"),
        valRouge1: document.getElementById("valRouge1"),
        valRougeL: document.getElementById("valRougeL"),
        gtReportText: document.getElementById("gtReportText"),
        gtPredText: document.getElementById("gtPredText"),

        // Doktor Çalışma Alanı
        doctorEditorText: document.getElementById("doctorEditorText"),
        icd10Chips: document.getElementById("icd10Chips"),
        inputDocName: document.getElementById("inputDocName"),
        inputHospital: document.getElementById("inputHospital"),
        chkDoctorApproved: document.getElementById("chkDoctorApproved"),
        btnPrintReport: document.getElementById("btnPrintReport"),

        // Navbar & Modallar
        statusLabel: document.getElementById("statusLabel"),
        btnThemeToggle: document.getElementById("btnThemeToggle"),
        themeIcon: document.getElementById("themeIcon"),
        btnQuickGuide: document.getElementById("btnQuickGuide"),
        infoModal: document.getElementById("infoModal"),
        btnCloseModal: document.getElementById("btnCloseModal"),

        // Model Kıyaslama & Metrik Modalı
        btnOpenBenchmarkModal: document.getElementById("btnOpenBenchmarkModal"),
        btnStripBenchmark: document.getElementById("btnStripBenchmark"),
        benchmarkModal: document.getElementById("benchmarkModal"),
        btnCloseBenchmarkModal: document.getElementById("btnCloseBenchmarkModal"),
        bTabButtons: document.querySelectorAll(".b-tab-btn"),
        bTabPanes: document.querySelectorAll(".b-tab-pane"),

        // Deney Günlüğü Elementleri
        experimentsTimeline: document.getElementById("experimentsTimeline"),
        expTotalCount: document.getElementById("expTotalCount"),
        btnOpenAddExpModal: document.getElementById("btnOpenAddExpModal"),
        addExperimentModal: document.getElementById("addExperimentModal"),
        btnCloseAddExpModal: document.getElementById("btnCloseAddExpModal"),
        btnCancelAddExp: document.getElementById("btnCancelAddExp"),
        addExperimentForm: document.getElementById("addExperimentForm"),

        // Veri Seti Kayıtları Modalı
        btnDatasetRecords: document.getElementById("btnDatasetRecords"),
        datasetModal: document.getElementById("datasetModal"),
        btnCloseDatasetModal: document.getElementById("btnCloseDatasetModal"),
        badgeTrainCount: document.getElementById("badgeTrainCount"),
        badgeValCount: document.getElementById("badgeValCount"),
        badgeTestCount: document.getElementById("badgeTestCount"),
        datasetFilterInput: document.getElementById("datasetFilterInput"),
        currentSplitLabel: document.getElementById("currentSplitLabel"),
        btnDownloadDatasetTxt: document.getElementById("btnDownloadDatasetTxt"),
        datasetScrollContainer: document.getElementById("datasetScrollContainer"),
        datasetLoadingState: document.getElementById("datasetLoadingState"),
        datasetFilesWrapper: document.getElementById("datasetFilesWrapper"),
        datasetTableBody: document.getElementById("datasetTableBody"),
        datasetFooterStatus: document.getElementById("datasetFooterStatus"),
        dsTabButtons: document.querySelectorAll(".ds-tab-btn"),

        // Printable Alanlar
        printableReportArea: document.getElementById("printableReportArea"),
        printReportNo: document.getElementById("printReportNo"),
        printDate: document.getElementById("printDate"),
        printCaseId: document.getElementById("printCaseId"),
        printPatientId: document.getElementById("printPatientId"),
        printTriageLevel: document.getElementById("printTriageLevel"),
        printImgOrig: document.getElementById("printImgOrig"),
        printImgSeg: document.getElementById("printImgSeg"),
        printReportText: document.getElementById("printReportText"),
        printSegConf: document.getElementById("printSegConf"),
        printAreaRatio: document.getElementById("printAreaRatio"),
        printIcd10: document.getElementById("printIcd10"),
        printDoctorName: document.getElementById("printDoctorName")
    };

    // ==========================================
    // 3. BAŞLANGIÇ YÜKLEMESİ
    // ==========================================
    initSystem();

    async function initSystem() {
        setupEventListeners();
        setupSplitSlider();
        updateModelInfoCard(state.selectedModel);
        await checkSystemStatus();
        await loadSampleCases();
        loadDatasetSummary(); // Arka planda sayıları çek
        loadExperimentsLog(); // Deney günlüğünü çek
    }

    // Dinamik Bilgi Kartını Güncelle
    function updateModelInfoCard(modelId) {
        const config = MODEL_CONFIGS[modelId] || MODEL_CONFIGS.v3;
        
        if (elements.micBadge) {
            elements.micBadge.className = `mic-badge ${config.badgeClass}`;
            const iconName = modelId === 'v3' ? 'sparkles' : (modelId === 'v2' ? 'cpu' : (modelId === 'v1' ? 'layers' : 'alert-circle'));
            elements.micBadge.innerHTML = `<i data-lucide="${iconName}"></i> ${config.badge}`;
        }
        if (elements.micTag) elements.micTag.textContent = config.tag;
        if (elements.micTitle) elements.micTitle.textContent = config.title;
        if (elements.micDesc) elements.micDesc.textContent = config.desc;

        if (elements.micPillsRow) {
            elements.micPillsRow.innerHTML = config.pills.map(p => 
                `<span class="mic-pill"><i data-lucide="${p.icon}"></i> ${p.text}</span>`
            ).join("");
        }

        if (window.lucide) lucide.createIcons();
    }

    // Sistem Durumu Kontrolü
    async function checkSystemStatus() {
        try {
            const res = await fetch("/api/status");
            const data = await res.json();
            if (data.cuda_available) {
                elements.statusLabel.textContent = `YOLO & VLM Hazır (${data.gpu_name} - CUDA Aktif)`;
            } else {
                elements.statusLabel.textContent = "YOLO & VLM Hazır (CPU Modu)";
            }
        } catch (err) {
            console.error("Sistem durumu alınamadı:", err);
            elements.statusLabel.textContent = "Sunucu Bağlantısı Bekleniyor...";
        }
    }

    // Örnek Vakaları Yükle
    async function loadSampleCases() {
        try {
            const res = await fetch("/api/samples");
            const data = await res.json();

            // Öne Çıkan Kartları Oluştur
            if (data.featured && elements.featuredGrid) {
                elements.featuredGrid.innerHTML = "";
                data.featured.forEach((item, idx) => {
                    const card = document.createElement("div");
                    card.className = "featured-item";
                    card.dataset.id = item.id;
                    card.innerHTML = `
                        <div class="fi-top">
                            <span class="fi-badge" style="background:${item.badge_color}22; color:${item.badge_color}; border:1px solid ${item.badge_color}44">${item.category}</span>
                        </div>
                        <div class="fi-title">${item.title}</div>
                        <div class="fi-finding">${item.finding}</div>
                    `;
                    card.addEventListener("click", () => selectSampleCase(item.id, card));
                    elements.featuredGrid.appendChild(card);
                });

                // Varsayılan olarak ilk vakayı hemen seç ve çalıştır!
                if (data.featured.length > 0) {
                    const firstCard = elements.featuredGrid.querySelector(".featured-item");
                    selectSampleCase(data.featured[0].id, firstCard);
                }
            }

            // Tüm 2354 Test Veri Listesini Doldur
            if (data.sample_ids && elements.datasetList) {
                renderDatasetList(data.sample_ids);
            }
        } catch (err) {
            console.error("Örnek vakalar yüklenemedi:", err);
        }
    }

    function renderDatasetList(list) {
        elements.datasetList.innerHTML = "";
        list.forEach(id => {
            const li = document.createElement("li");
            li.className = "dataset-item";
            li.dataset.id = id;
            li.innerHTML = `
                <span>${id.substring(0, 24)}...</span>
                <i data-lucide="play" style="width:12px;height:12px"></i>
            `;
            li.addEventListener("click", () => {
                selectSampleCase(id);
                document.querySelectorAll(".dataset-item").forEach(el => el.classList.remove("active"));
                li.classList.add("active");
            });
            elements.datasetList.appendChild(li);
        });
        if (window.lucide) lucide.createIcons();
    }

    // ==========================================
    // 4. ANALİZ VE ÇIKARIM ÇAĞRISI
    // ==========================================
    async function selectSampleCase(sampleId, cardEl = null) {
        if (state.isAnalyzing) return;
        state.activeCaseId = sampleId;

        if (cardEl) {
            document.querySelectorAll(".featured-item").forEach(el => el.classList.remove("active"));
            cardEl.classList.add("active");
        }

        elements.activeCaseTag.textContent = `Seçili Vaka: ${sampleId.substring(0, 16)}...`;

        const formData = new FormData();
        formData.append("sample_id", sampleId);
        formData.append("model_version", state.selectedModel);
        await runAnalysis(formData);
    }

    async function analyzeUploadedFile(file) {
        if (state.isAnalyzing) return;
        state.activeCaseId = file.name;
        elements.activeCaseTag.textContent = `Yüklenen Dosya: ${file.name}`;

        const formData = new FormData();
        formData.append("file", file);
        formData.append("model_version", state.selectedModel);
        await runAnalysis(formData);
    }

    async function runAnalysis(formData) {
        state.isAnalyzing = true;
        showLoading(true);

        try {
            elements.loadingStageTitle.textContent = "1/3 YOLO Lezyon/Akciğer Bölütlemesi Yapılıyor...";
            elements.loadingStageDesc.textContent = "YOLOv11-seg derin öğrenme modeli akciğer dokusunu tespit edip maskeliyor (%94.6 Dice).";

            const res = await fetch("/api/analyze", {
                method: "POST",
                body: formData
            });

            if (!res.ok) {
                const errData = await res.json().catch(() => ({}));
                throw new Error(errData.detail || "Analiz sırasında sunucu hatası oluştu.");
            }

            elements.loadingStageTitle.textContent = "2/3 [V3], [V2], [V1] ve [V0] Modelleri Çıkarım Yapıyor...";
            elements.loadingStageDesc.textContent = "YOLO'lu/YOLO'suz DenseNet+Transformer ve ResNet+LSTM (2x2 Ablasyon Matrisi) raporları üretiliyor.";

            const data = await res.json();
            state.analysisData = data;

            // Sonuçları arayüze yerleştir
            renderAnalysisResults(data);

        } catch (err) {
            console.error("Analiz Hatası:", err);
            alert(`Hata: ${err.message}`);
        } finally {
            state.isAnalyzing = false;
            showLoading(false);
        }
    }

    function showLoading(show) {
        elements.stageLoading.style.display = show ? "flex" : "none";
        elements.stagePlaceholder.style.display = "none";
    }

    // ==========================================
    // 5. SONUÇLARIN ARAYÜZE GİYDİRİLMESİ
    // ==========================================
    function renderAnalysisResults(data) {
        const { images, segmentation_metrics, timing } = data;

        // 1. Görselleri Sahneye Yerleştir
        elements.splitImgBefore.src = images.original;
        elements.splitImgAfter.src = images.mask_overlay;
        elements.singleStageImg.src = images.mask_overlay;

        // Sahneyi göster
        setViewerMode(state.currentMode);

        // 2. Segmentasyon Metrikleri
        elements.valDetectionCount.textContent = segmentation_metrics.has_mask ? `${segmentation_metrics.detection_count} Bölge` : "Tespit Yok";
        elements.valYoloConf.textContent = segmentation_metrics.avg_confidence > 0 ? `%${segmentation_metrics.avg_confidence}` : "-";
        elements.valAreaRatio.textContent = `%${segmentation_metrics.mask_area_ratio}`;
        elements.valYoloTime.textContent = `${timing.yolo_ms} ms`;

        // 3. Seçili Modele Ait Rapor ve Metrikleri Giydir
        renderActiveModel(state.selectedModel, data);

        // İkonları tazele
        if (window.lucide) lucide.createIcons();
    }

    // Seçilen Modele Göre Raporu, CDSS Triyajını ve Metrikleri Anlık Olarak Güncelle
    function renderActiveModel(modelId, data) {
        if (!data) return;

        // data.models içerisinde o model var mı kontrol et
        const modelData = (data.models && data.models[modelId]) ? data.models[modelId] : {
            vlm_report: data.vlm_report,
            cdss: data.cdss,
            metrics: data.ground_truth ? data.ground_truth.metrics : null,
            architecture: "YOLO + VLM Modeli"
        };

        const vlm = modelData.vlm_report;
        const cdss = modelData.cdss;
        const metrics = modelData.metrics;

        // 1. CDSS Triyaj & Risk Rozeti
        elements.triageBadge.textContent = cdss.urgency_label;
        elements.triageBadge.style.backgroundColor = cdss.urgency_color;
        elements.triageSummary.textContent = cdss.clinical_summary_tr || "Klinik değerlendirme tamamlandı.";
        
        elements.cdssTriageCard.style.borderColor = `${cdss.urgency_color}55`;
        elements.triageIconWrap.style.backgroundColor = `${cdss.urgency_color}22`;
        elements.triageIconWrap.style.color = cdss.urgency_color;

        // Triyaj kartındaki model kaynağı etiketi
        const sourceLabel = elements.cdssTriageCard.querySelector(".triage-model-source");
        if (sourceLabel) {
            sourceLabel.textContent = modelData.architecture || modelData.name;
        }

        // Klinik Etiketler
        elements.clinicalTagsContainer.innerHTML = "";
        cdss.tags.forEach(t => {
            const span = document.createElement("span");
            span.className = `c-tag ${t.type}`;
            span.textContent = t.name;
            elements.clinicalTagsContainer.appendChild(span);
        });

        // 2. VLM Rapor Bölümleri
        const secs = vlm.sections;
        elements.reportSecExam.textContent = secs.examination || "-";
        elements.reportSecInd.textContent = secs.indication || "-";
        elements.reportSecFind.textContent = secs.findings || "-";
        elements.reportSecImp.textContent = secs.impression || "-";
        elements.rawTokenBox.textContent = vlm.raw_report || "-";

        // CDSS Tavsiye
        if (cdss.recommendations && cdss.recommendations.length > 0) {
            elements.cdssAdviceText.innerHTML = cdss.recommendations.map(r => `• ${r}`).join("<br>");
        }

        // 3. Doktor Düzenleme Alanına Aktar
        elements.doctorEditorText.value = vlm.clean_report;

        // ICD-10 Kodları
        elements.icd10Chips.innerHTML = "";
        if (cdss.suggested_icd10 && cdss.suggested_icd10.length > 0) {
            cdss.suggested_icd10.forEach(codeObj => {
                const cSpan = document.createElement("span");
                cSpan.className = "icd-chip";
                cSpan.textContent = `${codeObj.code} - ${codeObj.desc}`;
                elements.icd10Chips.appendChild(cSpan);
            });
        }

        // 4. Ground Truth ve Akademik Metrikler (BLEU-1, BLEU-4, ROUGE-1, ROUGE-L)
        if (data.ground_truth && data.ground_truth.has_ground_truth) {
            elements.badgeGtAvailable.style.display = "inline-block";
            elements.gtReportText.textContent = data.ground_truth.report_text;
            elements.gtPredText.textContent = vlm.clean_report;

            if (metrics) {
                elements.valBleu1.textContent = `%${metrics.bleu1}`;
                elements.valBleu4.textContent = `%${metrics.bleu4}`;
                elements.valRouge1.textContent = `%${metrics.rouge1}`;
                elements.valRougeL.textContent = `%${metrics.rougeL}`;
            } else {
                elements.valBleu1.textContent = "-";
                elements.valBleu4.textContent = "-";
                elements.valRouge1.textContent = "-";
                elements.valRougeL.textContent = "-";
            }
        } else {
            elements.badgeGtAvailable.style.display = "none";
            elements.gtReportText.textContent = "Bu vaka için harici referans raporu yüklenmedi (Manuel yüklenen röntgen).";
            elements.gtPredText.textContent = vlm.clean_report;
            elements.valBleu1.textContent = "-";
            elements.valBleu4.textContent = "-";
            elements.valRouge1.textContent = "-";
            elements.valRougeL.textContent = "-";
        }

        // 5. Yazdırılabilir Raporu Güncelle
        updatePrintableReport(data, modelData);

        if (window.lucide) lucide.createIcons();
    }

    // Yazdırılabilir Rapor Verilerini Hazırla
    function updatePrintableReport(data, modelData = null) {
        const now = new Date();
        const dateStr = now.toLocaleDateString("tr-TR");
        const reportNo = `MV-${now.getFullYear()}-${Math.floor(1000 + Math.random() * 9000)}`;

        const activeModel = modelData || (data.models ? data.models[state.selectedModel] : null);
        const cdss = activeModel ? activeModel.cdss : data.cdss;

        elements.printReportNo.textContent = reportNo;
        elements.printDate.textContent = dateStr;
        elements.printCaseId.textContent = state.activeCaseId ? state.activeCaseId.substring(0, 20) : "VAKA-GENEL";
        elements.printPatientId.textContent = `PT-${Math.floor(100000 + Math.random() * 900000)}`;
        elements.printTriageLevel.textContent = cdss.urgency_label;
        elements.printImgOrig.src = data.images.original;
        elements.printImgSeg.src = data.images.mask_overlay;
        elements.printReportText.textContent = elements.doctorEditorText.value;
        elements.printSegConf.textContent = `%${data.segmentation_metrics.avg_confidence}`;
        elements.printAreaRatio.textContent = `%${data.segmentation_metrics.mask_area_ratio}`;
        
        const firstIcd = cdss.suggested_icd10 && cdss.suggested_icd10[0] ? cdss.suggested_icd10[0].code : "R91.8";
        elements.printIcd10.textContent = firstIcd;
        elements.printDoctorName.textContent = elements.inputDocName.value;
    }

    // ==========================================
    // 6. İNTERAKTİF BÖLÜNMÜŞ ÖNCESİ/SONRASI SLİDER'I
    // ==========================================
    function setupSplitSlider() {
        let isDragging = false;

        const updatePosition = (clientX) => {
            const rect = elements.splitSliderContainer.getBoundingClientRect();
            let x = clientX - rect.left;
            x = Math.max(0, Math.min(x, rect.width));
            const percentage = (x / rect.width) * 100;

            state.sliderPosition = percentage;
            elements.splitAfterWrapper.style.width = `${percentage}%`;
            elements.sliderHandle.style.left = `${percentage}%`;
        };

        elements.sliderHandle.addEventListener("mousedown", (e) => {
            isDragging = true;
            e.preventDefault();
        });

        window.addEventListener("mouseup", () => {
            isDragging = false;
        });

        window.addEventListener("mousemove", (e) => {
            if (!isDragging) return;
            updatePosition(e.clientX);
        });

        // Dokunmatik Ekran Desteği
        elements.sliderHandle.addEventListener("touchstart", (e) => {
            isDragging = true;
        }, { passive: true });

        window.addEventListener("touchend", () => {
            isDragging = false;
        });

        window.addEventListener("touchmove", (e) => {
            if (!isDragging || !e.touches[0]) return;
            updatePosition(e.touches[0].clientX);
        }, { passive: true });

        // Tıklayarak doğrudan pozisyona gitme
        elements.splitSliderContainer.addEventListener("click", (e) => {
            if (e.target === elements.sliderHandle || elements.sliderHandle.contains(e.target)) return;
            updatePosition(e.clientX);
        });
    }

    // Görünüm Modunu Değiştir (Split / Overlay / Cleaned / Raw)
    function setViewerMode(mode) {
        state.currentMode = mode;

        elements.modeButtons.forEach(btn => {
            btn.classList.toggle("active", btn.dataset.mode === mode);
        });

        if (!state.analysisData) return;

        const imgs = state.analysisData.images;

        if (mode === "split") {
            elements.splitSliderContainer.style.display = "block";
            elements.singleImgContainer.style.display = "none";
            elements.splitImgBefore.src = imgs.original;
            elements.splitImgAfter.src = imgs.mask_overlay;
            elements.splitAfterWrapper.style.width = `${state.sliderPosition}%`;
            elements.sliderHandle.style.left = `${state.sliderPosition}%`;
        } else {
            elements.splitSliderContainer.style.display = "none";
            elements.singleImgContainer.style.display = "flex";

            if (mode === "overlay") {
                elements.singleStageImg.src = imgs.mask_overlay;
                elements.singleStageImg.style.opacity = state.maskOpacity;
            } else if (mode === "cleaned") {
                elements.singleStageImg.src = imgs.segmented_cleaned;
                elements.singleStageImg.style.opacity = 1;
            } else if (mode === "raw") {
                elements.singleStageImg.src = imgs.original;
                elements.singleStageImg.style.opacity = 1;
            }
        }
    }

    // ==========================================
    // 7. EĞİTİM & TEST VERİ SETİ KAYITLARI MODALI
    // ==========================================
    async function loadDatasetSummary() {
        try {
            const res = await fetch("/api/dataset-records/summary");
            const data = await res.json();
            if (data.splits) {
                if (elements.badgeTrainCount && data.splits.train) {
                    elements.badgeTrainCount.textContent = Number(data.splits.train.count).toLocaleString("tr-TR");
                }
                if (elements.badgeValCount && data.splits.val) {
                    elements.badgeValCount.textContent = Number(data.splits.val.count).toLocaleString("tr-TR");
                }
                if (elements.badgeTestCount && data.splits.test) {
                    elements.badgeTestCount.textContent = Number(data.splits.test.count).toLocaleString("tr-TR");
                }
            }
        } catch (err) {
            console.error("Veri seti özeti alınamadı:", err);
        }
    }

    async function openDatasetModal() {
        elements.datasetModal.style.display = "flex";
        await loadDatasetSplit(state.activeDatasetSplit);
    }

    async function loadDatasetSplit(split, searchQuery = "") {
        state.activeDatasetSplit = split;

        // Sekme vurgusu
        elements.dsTabButtons.forEach(btn => {
            btn.classList.toggle("active", btn.dataset.split === split);
        });

        // Etiket ve indirme linki
        const splitNames = {
            train: "Eğitim Seti (train_images.txt - 18.826 Görüntü)",
            val: "Doğrulama Seti (val_images.txt - 2.353 Görüntü)",
            test: "Test Seti (test_images.txt - 2.354 Görüntü)"
        };
        elements.currentSplitLabel.textContent = splitNames[split] || split;
        elements.btnDownloadDatasetTxt.href = `/api/dataset-records/download/${split}`;

        // Yükleme durumu
        elements.datasetLoadingState.style.display = "flex";
        elements.datasetFilesWrapper.style.display = "none";

        try {
            // Önbellekte varsa ve arama yoksa oradan kullan
            if (!searchQuery && state.datasetCache[split]) {
                renderDatasetTable(state.datasetCache[split], split);
                return;
            }

            const url = `/api/dataset-records/${split}?limit=1200${searchQuery ? `&search=${encodeURIComponent(searchQuery)}` : ""}`;
            const res = await fetch(url);
            const result = await res.json();

            if (!searchQuery) {
                state.datasetCache[split] = result;
            }

            renderDatasetTable(result, split);

        } catch (err) {
            console.error("Veri seti spliti yüklenemedi:", err);
            elements.datasetLoadingState.innerHTML = `<span style="color:#ef4444">Hata: Dosyalar yüklenemedi. (${err.message})</span>`;
        }
    }

    function renderDatasetTable(data, split) {
        elements.datasetLoadingState.style.display = "none";
        elements.datasetFilesWrapper.style.display = "block";

        const items = data.items || [];
        elements.datasetTableBody.innerHTML = "";

        if (items.length === 0) {
            elements.datasetTableBody.innerHTML = `<tr><td colspan="3" style="text-align:center; padding:2rem; color:var(--text-muted)">Arama kriterine uygun resim kaydı bulunamadı.</td></tr>`;
            elements.datasetFooterStatus.textContent = `0 adet resim eşleşti (Toplam: ${data.total})`;
            return;
        }

        const fragment = document.createDocumentFragment();
        items.forEach((fileName, index) => {
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td class="row-num">${index + 1}</td>
                <td class="file-name">${fileName}</td>
                <td><span class="file-badge ${split}">.${fileName.split('.').pop().toUpperCase()} / ${split.toUpperCase()}</span></td>
            `;
            fragment.appendChild(tr);
        });

        elements.datasetTableBody.appendChild(fragment);
        elements.datasetFooterStatus.textContent = `Toplam ${Number(data.total).toLocaleString('tr-TR')} adet ${split.toUpperCase()} resmi kayıtlıdır (${items.length} tanesi listeleniyor).`;
    }

    // ==========================================
    // 7.5. AŞAMA AŞAMA DENEY GÜNLÜĞÜ (1, 2, 3, 4...)
    // ==========================================
    async function loadExperimentsLog() {
        if (!elements.experimentsTimeline) return;
        try {
            const res = await fetch("/api/experiments");
            const data = await res.json();
            const experiments = data.experiments || [];
            if (elements.expTotalCount) {
                elements.expTotalCount.textContent = experiments.length;
            }
            renderExperiments(experiments);
        } catch (err) {
            console.error("Deney günlüğü yüklenemedi:", err);
            elements.experimentsTimeline.innerHTML = `<div style="color:#ef4444; padding:1rem;">Deney günlüğü yüklenemedi: ${err.message}</div>`;
        }
    }

    function renderExperiments(experiments) {
        if (!elements.experimentsTimeline) return;
        elements.experimentsTimeline.innerHTML = "";

        experiments.forEach(exp => {
            const isSota = exp.id === "exp-4" || (exp.status && exp.status.includes("SOTA")) || (exp.status && exp.status.includes("Önerilen"));
            const card = document.createElement("div");
            card.className = `exp-card ${isSota ? "is-sota" : ""}`;
            card.id = `card-${exp.id}`;

            const badgeBg = exp.badge_color ? `${exp.badge_color}22` : "rgba(6, 182, 212, 0.15)";
            const badgeBorder = exp.badge_color ? `${exp.badge_color}55` : "rgba(6, 182, 212, 0.4)";
            const badgeText = exp.badge_color || "var(--cyan-400)";

            // Metric cells
            const m = exp.metrics || {};
            let metricsHtml = "";
            
            if (m.dice && m.dice !== "-") {
                metricsHtml += `<div class="exp-metric-cell ${isSota ? "highlight" : ""}"><span class="exp-m-label">YOLO Dice</span><span class="exp-m-val">${m.dice}</span></div>`;
            }
            if (m.map50 && m.map50 !== "-") {
                metricsHtml += `<div class="exp-metric-cell"><span class="exp-m-label">mAP50</span><span class="exp-m-val">${m.map50}</span></div>`;
            }
            if (m.bleu4 && m.bleu4 !== "-") {
                metricsHtml += `<div class="exp-metric-cell ${isSota ? "highlight" : ""}"><span class="exp-m-label">BLEU-4</span><span class="exp-m-val" style="color:${isSota ? '#10b981' : '#38bdf8'}">${m.bleu4}</span></div>`;
            }
            if (m.bleu1 && m.bleu1 !== "-") {
                metricsHtml += `<div class="exp-metric-cell"><span class="exp-m-label">BLEU-1</span><span class="exp-m-val">${m.bleu1}</span></div>`;
            }
            if (m.rougeL && m.rougeL !== "-") {
                metricsHtml += `<div class="exp-metric-cell"><span class="exp-m-label">ROUGE-L</span><span class="exp-m-val">${m.rougeL}</span></div>`;
            }
            if (m.top1_acc && m.top1_acc !== "-") {
                metricsHtml += `<div class="exp-metric-cell"><span class="exp-m-label">Top-1 Acc</span><span class="exp-m-val">${m.top1_acc}</span></div>`;
            }
            if (m.top5_acc && m.top5_acc !== "-") {
                metricsHtml += `<div class="exp-metric-cell"><span class="exp-m-label">Top-5 Acc</span><span class="exp-m-val">${m.top5_acc}</span></div>`;
            }
            if (m.val_loss && m.val_loss !== "-") {
                metricsHtml += `<div class="exp-metric-cell"><span class="exp-m-label">Val Loss</span><span class="exp-m-val">${m.val_loss}</span></div>`;
            }
            if (m.perplexity && m.perplexity !== "-") {
                metricsHtml += `<div class="exp-metric-cell"><span class="exp-m-label">Perplexity (PPL)</span><span class="exp-m-val">${m.perplexity}</span></div>`;
            }
            if (m.latency && m.latency !== "-") {
                metricsHtml += `<div class="exp-metric-cell"><span class="exp-m-label">Hız / Gecikme</span><span class="exp-m-val">${m.latency}</span></div>`;
            }

            // Components
            const c = exp.components || {};
            const compHtml = `
                <div class="exp-components-row">
                    <span class="exp-comp-chip"><i data-lucide="scan" style="width:12px;height:12px;"></i> Bölütleme: <strong>${c.segmentation || '-'}</strong></span>
                    <span class="exp-comp-chip"><i data-lucide="image" style="width:12px;height:12px;"></i> Omurga: <strong>${c.encoder || '-'}</strong></span>
                    <span class="exp-comp-chip"><i data-lucide="message-square" style="width:12px;height:12px;"></i> Dil Modeli: <strong>${c.decoder || '-'}</strong></span>
                    <span class="exp-comp-chip" style="margin-left:auto;"><i data-lucide="calendar" style="width:12px;height:12px;"></i> <strong>${exp.date || ''}</strong></span>
                </div>
            `;

            card.innerHTML = `
                <div class="exp-card-dot"></div>
                <div class="exp-header">
                    <div class="exp-header-left">
                        <span class="exp-trial-badge" style="background:${badgeBg}; border:1px solid ${badgeBorder}; color:${badgeText};">
                            <i data-lucide="award" style="width:12px;height:12px;"></i> DENEME ${exp.trial_num}
                        </span>
                        <h4 class="exp-title">${exp.title}</h4>
                        <span class="exp-phase-sub">${exp.phase} &bull; <em>${exp.architecture}</em></span>
                    </div>
                    <div class="exp-header-right">
                        <span class="exp-status-pill" style="background:${badgeBg}; color:${badgeText}; border:1px solid ${badgeBorder};">
                            <i data-lucide="${isSota ? 'check-circle-2' : 'info'}" style="width:12px;height:12px;"></i> ${exp.status}
                        </span>
                    </div>
                </div>

                ${compHtml}

                <div class="exp-metrics-grid">
                    ${metricsHtml}
                </div>

                <div class="exp-details-split">
                    <div class="exp-box exp-box-improved">
                        <div class="exp-box-title">
                            <i data-lucide="trending-up" style="width:14px;height:14px;"></i>
                            <span>Bu Deneme Neyi İyileştirdi? (Kazanım / Başarı)</span>
                        </div>
                        <p style="margin:0;">${exp.what_improved}</p>
                    </div>

                    <div class="exp-box exp-box-lesson ${isSota ? 'sota-final' : ''}">
                        <div class="exp-box-title">
                            <i data-lucide="${isSota ? 'trophy' : 'arrow-right-circle'}" style="width:14px;height:14px;"></i>
                            <span>${isSota ? 'Nihai Karar & Savunma Sonucu:' : 'Çıkarılan Ders & Neden Bir Sonraki Modele Geçildi?'}</span>
                        </div>
                        <p style="margin:0;">${exp.lesson_learned_next_step}</p>
                    </div>
                </div>
            `;

            elements.experimentsTimeline.appendChild(card);
        });

        // Add Experiment Placeholder Card at the end
        const placeholder = document.createElement("div");
        placeholder.className = "exp-add-placeholder";
        placeholder.innerHTML = `
            <i data-lucide="plus-circle"></i>
            <span>+ Yeni Eğitilen Modeli / Denemeyi Kaydet</span>
            <p>Swin Transformer, Med-Llama, ViT veya yeni denediğiniz mimarilerin test sonuçlarını günlüğe ekleyin</p>
        `;
        placeholder.addEventListener("click", openAddExpModal);
        elements.experimentsTimeline.appendChild(placeholder);

        if (window.lucide) lucide.createIcons();
    }

    function openAddExpModal() {
        if (elements.addExperimentModal) {
            elements.addExperimentModal.style.display = "flex";
            const form = document.getElementById("addExperimentForm");
            if (form) form.reset();
        }
    }

    function closeAddExpModal() {
        if (elements.addExperimentModal) {
            elements.addExperimentModal.style.display = "none";
        }
    }

    // ==========================================
    // 8. ETKİNLİK DİNLEYİCİLERİ (EVENT LISTENERS)
    // ==========================================
    function setupEventListeners() {
        // --- MODEL SEÇİM DROPDOWN DİNLENMESİ ---
        elements.modelSelectDropdown.addEventListener("change", (e) => {
            const modelId = e.target.value;
            state.selectedModel = modelId;
            
            // Bilgi kartını güncelle
            updateModelInfoCard(modelId);

            // Eğer daha önceden analiz yapılmışsa anlık olarak raporu ve metrikleri güncelle!
            if (state.analysisData) {
                renderActiveModel(modelId, state.analysisData);
            }
        });

        // --- VERİ SETİ KAYITLARI BUTONU VE MODALI ---
        elements.btnDatasetRecords.addEventListener("click", () => {
            openDatasetModal();
        });

        elements.btnCloseDatasetModal.addEventListener("click", () => {
            elements.datasetModal.style.display = "none";
        });

        elements.datasetModal.addEventListener("click", (e) => {
            if (e.target === elements.datasetModal) {
                elements.datasetModal.style.display = "none";
            }
        });

        // Split sekmeleri (Train / Val / Test)
        elements.dsTabButtons.forEach(btn => {
            btn.addEventListener("click", () => {
                const targetSplit = btn.dataset.split;
                elements.datasetFilterInput.value = "";
                loadDatasetSplit(targetSplit);
            });
        });

        // Veri seti içi filtreleme (debounce)
        let filterTimeout = null;
        elements.datasetFilterInput.addEventListener("input", (e) => {
            clearTimeout(filterTimeout);
            const q = e.target.value.trim();
            filterTimeout = setTimeout(() => {
                loadDatasetSplit(state.activeDatasetSplit, q);
            }, 300);
        });

        // Sol Panel Sekmeleri
        const tabs = [
            { btn: elements.tabFeatured, view: elements.featuredView },
            { btn: elements.tabUpload, view: elements.uploadView },
            { btn: elements.tabDataset, view: elements.datasetView }
        ];

        tabs.forEach(t => {
            t.btn.addEventListener("click", () => {
                tabs.forEach(o => {
                    o.btn.classList.remove("active");
                    o.view.classList.remove("active");
                });
                t.btn.classList.add("active");
                t.view.classList.add("active");
            });
        });

        // Dosya Yükleme Dropzone
        elements.btnBrowseFile.addEventListener("click", () => elements.fileInput.click());
        elements.dropzone.addEventListener("click", (e) => {
            if (e.target !== elements.btnBrowseFile) elements.fileInput.click();
        });

        elements.fileInput.addEventListener("change", (e) => {
            if (e.target.files && e.target.files[0]) {
                analyzeUploadedFile(e.target.files[0]);
            }
        });

        elements.dropzone.addEventListener("dragover", (e) => {
            e.preventDefault();
            elements.dropzone.classList.add("dragover");
        });

        elements.dropzone.addEventListener("dragleave", () => {
            elements.dropzone.classList.remove("dragover");
        });

        elements.dropzone.addEventListener("drop", (e) => {
            e.preventDefault();
            elements.dropzone.classList.remove("dragover");
            if (e.dataTransfer.files && e.dataTransfer.files[0]) {
                analyzeUploadedFile(e.dataTransfer.files[0]);
            }
        });

        // Test Veri Seti Arama Çubuğu
        elements.datasetSearchInput.addEventListener("input", (e) => {
            const query = e.target.value.toLowerCase().trim();
            const items = elements.datasetList.querySelectorAll(".dataset-item");
            items.forEach(item => {
                const id = item.dataset.id.toLowerCase();
                item.style.display = id.includes(query) ? "flex" : "none";
            });
        });

        // Rastgele Vaka Butonu
        elements.btnRandomSample.addEventListener("click", () => {
            const items = Array.from(elements.datasetList.querySelectorAll(".dataset-item"));
            if (items.length > 0) {
                const randomItem = items[Math.floor(Math.random() * items.length)];
                randomItem.click();
            }
        });

        // Görünüm Modu Butonları (Split / Overlay / Cleaned / Raw)
        elements.modeButtons.forEach(btn => {
            btn.addEventListener("click", () => setViewerMode(btn.dataset.mode));
        });

        // Opaklık Slider
        elements.maskOpacitySlider.addEventListener("input", (e) => {
            const val = e.target.value;
            elements.maskOpacityVal.textContent = `%${val}`;
            state.maskOpacity = val / 100;
            elements.splitImgAfter.style.opacity = state.maskOpacity;
            if (state.currentMode === "overlay") {
                elements.singleStageImg.style.opacity = state.maskOpacity;
            }
        });

        // Tam Ekran
        elements.btnFullscreen.addEventListener("click", () => {
            if (!document.fullscreenElement) {
                elements.imageStage.requestFullscreen().catch(() => {});
            } else {
                document.exitFullscreen().catch(() => {});
            }
        });

        // Sıfırla
        elements.btnResetZoom.addEventListener("click", () => {
            state.sliderPosition = 50;
            elements.splitAfterWrapper.style.width = "50%";
            elements.sliderHandle.style.left = "50%";
        });

        // Rapor Sekmeleri (VLM Raporu / Ground Truth / Doktor Onay)
        elements.rtabButtons.forEach(btn => {
            btn.addEventListener("click", () => {
                elements.rtabButtons.forEach(b => b.classList.remove("active"));
                elements.reportPanes.forEach(p => p.classList.remove("active"));

                btn.classList.add("active");
                const targetPane = document.getElementById(btn.dataset.tab);
                if (targetPane) targetPane.classList.add("active");
            });
        });

        // Raporu Kopyala
        elements.btnCopyReport.addEventListener("click", () => {
            if (!state.analysisData) return;
            const activeModel = state.analysisData.models ? state.analysisData.models[state.selectedModel] : null;
            const textToCopy = activeModel ? activeModel.vlm_report.clean_report : elements.doctorEditorText.value;
            navigator.clipboard.writeText(textToCopy);
            alert("Radyoloji raporu panoya kopyalandı!");
        });

        // Editöre Aktar
        elements.btnTransferToEditor.addEventListener("click", () => {
            const doctorTabBtn = document.querySelector('[data-tab="doctorEditTab"]');
            if (doctorTabBtn) doctorTabBtn.click();
        });

        // Hızlı Klinik Şablonlar
        document.querySelectorAll(".template-chip").forEach(chip => {
            chip.addEventListener("click", () => {
                const phrase = chip.dataset.phrase;
                elements.doctorEditorText.value += `\n- ${phrase}`;
            });
        });

        // Raporu Yazdır / PDF İndir
        elements.btnPrintReport.addEventListener("click", () => {
            if (state.analysisData) {
                updatePrintableReport(state.analysisData);
            }
            window.print();
        });

        // Tema Değiştirme
        elements.btnThemeToggle.addEventListener("click", () => {
            state.theme = state.theme === "dark" ? "light" : "dark";
            document.documentElement.setAttribute("data-theme", state.theme);
            elements.themeIcon.setAttribute("data-lucide", state.theme === "dark" ? "sun" : "moon");
            if (window.lucide) lucide.createIcons();
        });

        // Tez Bilgi Modalı
        elements.btnQuickGuide.addEventListener("click", () => {
            elements.infoModal.style.display = "flex";
        });

        elements.btnCloseModal.addEventListener("click", () => {
            elements.infoModal.style.display = "none";
        });

        elements.infoModal.addEventListener("click", (e) => {
            if (e.target === elements.infoModal) elements.infoModal.style.display = "none";
        });

        // --- MODEL KIYASLAMA & METRİK ANALİZ MODALI ---
        if (elements.btnOpenBenchmarkModal) {
            elements.btnOpenBenchmarkModal.addEventListener("click", () => {
                elements.benchmarkModal.style.display = "flex";
                loadExperimentsLog();
            });
        }

        if (elements.btnStripBenchmark) {
            elements.btnStripBenchmark.addEventListener("click", () => {
                elements.benchmarkModal.style.display = "flex";
                loadExperimentsLog();
            });
        }

        if (elements.btnCloseBenchmarkModal) {
            elements.btnCloseBenchmarkModal.addEventListener("click", () => {
                elements.benchmarkModal.style.display = "none";
            });
        }

        if (elements.benchmarkModal) {
            elements.benchmarkModal.addEventListener("click", (e) => {
                if (e.target === elements.benchmarkModal) {
                    elements.benchmarkModal.style.display = "none";
                }
            });
        }

        // --- YENİ DENEME / MODEL EKLEME MODALI DİNLENMESİ ---
        if (elements.btnOpenAddExpModal) {
            elements.btnOpenAddExpModal.addEventListener("click", openAddExpModal);
        }

        if (elements.btnCloseAddExpModal) {
            elements.btnCloseAddExpModal.addEventListener("click", closeAddExpModal);
        }

        if (elements.btnCancelAddExp) {
            elements.btnCancelAddExp.addEventListener("click", closeAddExpModal);
        }

        if (elements.addExperimentModal) {
            elements.addExperimentModal.addEventListener("click", (e) => {
                if (e.target === elements.addExperimentModal) {
                    closeAddExpModal();
                }
            });
        }

        if (elements.addExperimentForm) {
            elements.addExperimentForm.addEventListener("submit", async (e) => {
                e.preventDefault();
                const payload = {
                    title: document.getElementById("inpExpTitle").value.trim(),
                    phase: document.getElementById("inpExpPhase").value.trim(),
                    architecture: document.getElementById("inpExpArch").value.trim() || "Özel Hibrit Mimari",
                    components: {
                        segmentation: document.getElementById("inpExpSeg").value.trim() || "YOLOv11-seg",
                        encoder: document.getElementById("inpExpEnc").value.trim() || "DenseNet-121",
                        decoder: document.getElementById("inpExpDec").value.trim() || "Transformer"
                    },
                    metrics: {
                        dice: document.getElementById("inpExpDice").value.trim() || "%94.6",
                        bleu4: document.getElementById("inpExpBleu4").value.trim() || "%14.0",
                        rougeL: document.getElementById("inpExpRougeL").value.trim() || "%40.0",
                        top1_acc: document.getElementById("inpExpTop1").value.trim() || "%80.0",
                        val_loss: document.getElementById("inpExpValLoss").value.trim() || "1.80",
                        perplexity: document.getElementById("inpExpPpl").value.trim() || "6.2",
                        latency: document.getElementById("inpExpLatency").value.trim() || "50 ms"
                    },
                    status: "Tamamlandı (Yeni Deneme)",
                    badge_color: "#06b6d4",
                    what_improved: document.getElementById("inpExpWhatImproved").value.trim(),
                    lesson_learned_next_step: document.getElementById("inpExpLesson").value.trim()
                };

                try {
                    const btn = document.getElementById("btnSaveExperiment");
                    if (btn) btn.disabled = true;
                    const res = await fetch("/api/experiments", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify(payload)
                    });
                    const result = await res.json();
                    if (result.success) {
                        closeAddExpModal();
                        await loadExperimentsLog();
                        alert(`Harika! "${payload.title}" başarıyla deneme günlüğüne eklendi.`);
                    }
                } catch (err) {
                    alert("Kayıt sırasında hata oluştu: " + err.message);
                } finally {
                    const btn = document.getElementById("btnSaveExperiment");
                    if (btn) btn.disabled = false;
                }
            });
        }

        // Benchmark Sekme Geçişleri
        elements.bTabButtons.forEach(btn => {
            btn.addEventListener("click", () => {
                elements.bTabButtons.forEach(b => b.classList.remove("active"));
                elements.bTabPanes.forEach(p => p.classList.remove("active"));
                btn.classList.add("active");
                const targetPane = document.getElementById(btn.dataset.tab);
                if (targetPane) targetPane.classList.add("active");
                if (window.lucide) lucide.createIcons();
            });
        });

        // Benchmark Tablosundan Doğrudan Model Seçme Butonları
        document.querySelectorAll(".btn-switch-model").forEach(btn => {
            btn.addEventListener("click", () => {
                const modelId = btn.dataset.model;
                if (modelId && MODEL_CONFIGS[modelId]) {
                    state.selectedModel = modelId;
                    if (elements.modelSelectDropdown) {
                        elements.modelSelectDropdown.value = modelId;
                    }
                    updateModelInfoCard(modelId);
                    if (state.analysisData) {
                        renderActiveModel(modelId, state.analysisData);
                    }
                    elements.benchmarkModal.style.display = "none";
                }
            });
        });

        // Tooltip Açıklamaları (❓ Soru İşareti İkonları - Tıklama ve Dokunma Desteği)
        document.querySelectorAll(".tooltip-icon").forEach(icon => {
            icon.addEventListener("click", (e) => {
                e.stopPropagation();
                const content = icon.nextElementSibling;
                if (!content) return;
                const wasActive = content.classList.contains("is-active");
                document.querySelectorAll(".css-tooltip-content").forEach(c => c.classList.remove("is-active"));
                if (!wasActive) {
                    content.classList.add("is-active");
                }
            });
        });

        // Sayfa tıklamasında açık olan tooltip'leri kapat
        document.addEventListener("click", () => {
            document.querySelectorAll(".css-tooltip-content").forEach(c => c.classList.remove("is-active"));
        });
    }

});
