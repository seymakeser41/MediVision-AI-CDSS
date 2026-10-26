/**
 * MediVision AI - Frontend Uygulama Mantığı
 * Klinik Karar Destek Sistemi & İnteraktif Segmentasyon Arayüzü
 */

document.addEventListener("DOMContentLoaded", () => {
    // ==========================================
    // 1. STATE VE DEĞİŞKENLER
    // ==========================================
    const state = {
        currentMode: "split", // split | overlay | cleaned | raw
        activeCaseId: null,
        isAnalyzing: false,
        analysisData: null,
        sliderPosition: 50, // %
        maskOpacity: 0.85,
        showContour: true,
        theme: "dark"
    };

    // DOM Elementleri
    const elements = {
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
    // 2. BAŞLANGIÇ YÜKLEMESİ
    // ==========================================
    initSystem();

    async function initSystem() {
        setupEventListeners();
        setupSplitSlider();
        await checkSystemStatus();
        await loadSampleCases();
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
    // 3. ANALİZ VE ÇIKARIM ÇAĞRISI
    // ==========================================
    async function selectSampleCase(sampleId, cardEl = null) {
        if (state.isAnalyzing) return;
        state.activeCaseId = sampleId;

        // Kart vurgusu
        if (cardEl) {
            document.querySelectorAll(".featured-item").forEach(el => el.classList.remove("active"));
            cardEl.classList.add("active");
        }

        elements.activeCaseTag.textContent = `Seçili Vaka: ${sampleId.substring(0, 16)}...`;

        // Analizi başlat
        const formData = new FormData();
        formData.append("sample_id", sampleId);
        await runAnalysis(formData);
    }

    async function analyzeUploadedFile(file) {
        if (state.isAnalyzing) return;
        state.activeCaseId = file.name;
        elements.activeCaseTag.textContent = `Yüklenen Dosya: ${file.name}`;

        const formData = new FormData();
        formData.append("file", file);
        await runAnalysis(formData);
    }

    async function runAnalysis(formData) {
        state.isAnalyzing = true;
        showLoading(true);

        try {
            // Aşamalı bildirim efekti
            elements.loadingStageTitle.textContent = "1/3 YOLO Lezyon/Akciğer Bölütlemesi Yapılıyor...";
            elements.loadingStageDesc.textContent = "Derin öğrenme modeli doku maskesini hesaplıyor ve arka planı temizliyor.";

            const res = await fetch("/api/analyze", {
                method: "POST",
                body: formData
            });

            if (!res.ok) {
                const errData = await res.json().catch(() => ({}));
                throw new Error(errData.detail || "Analiz sırasında sunucu hatası oluştu.");
            }

            elements.loadingStageTitle.textContent = "2/3 DenseNet-121 + Transformer VLM Raporu Üretiliyor...";
            elements.loadingStageDesc.textContent = "Çapraz dikkat (Cross-Attention) ile 49 akciğer bölgesinden rapor sentezleniyor.";

            const data = await res.json();
            state.analysisData = data;

            // Sonuçları Arayüze Yerleştir
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
    // 4. SONUÇLARIN ARAYÜZE GİYDİRİLMESİ
    // ==========================================
    function renderAnalysisResults(data) {
        const { images, segmentation_metrics, vlm_report, cdss, ground_truth, timing } = data;

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

        // 3. CDSS Triyaj & Risk Rozeti
        elements.triageBadge.textContent = cdss.urgency_label;
        elements.triageBadge.style.backgroundColor = cdss.urgency_color;
        elements.triageSummary.textContent = cdss.clinical_summary_tr || "Klinik değerlendirme tamamlandı.";
        
        // Kart kenarlık ve arka plan rengi
        elements.cdssTriageCard.style.borderColor = `${cdss.urgency_color}55`;
        elements.triageIconWrap.style.backgroundColor = `${cdss.urgency_color}22`;
        elements.triageIconWrap.style.color = cdss.urgency_color;

        // Klinik Etiketler
        elements.clinicalTagsContainer.innerHTML = "";
        cdss.tags.forEach(t => {
            const span = document.createElement("span");
            span.className = `c-tag ${t.type}`;
            span.textContent = t.name;
            elements.clinicalTagsContainer.appendChild(span);
        });

        // 4. VLM Rapor Bölümleri
        const secs = vlm_report.sections;
        elements.reportSecExam.textContent = secs.examination || "-";
        elements.reportSecInd.textContent = secs.indication || "-";
        elements.reportSecFind.textContent = secs.findings || "-";
        elements.reportSecImp.textContent = secs.impression || "-";
        elements.rawTokenBox.textContent = vlm_report.raw_report || "-";

        // Tavsiye
        if (cdss.recommendations && cdss.recommendations.length > 0) {
            elements.cdssAdviceText.innerHTML = cdss.recommendations.map(r => `• ${r}`).join("<br>");
        }

        // 5. Doktor Düzenleme Alanına Ön Doldurma
        elements.doctorEditorText.value = vlm_report.clean_report;

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

        // 6. Ground Truth ve Akademik Metrikler
        if (ground_truth && ground_truth.has_ground_truth) {
            elements.badgeGtAvailable.style.display = "inline-block";
            elements.gtReportText.textContent = ground_truth.report_text;
            elements.gtPredText.textContent = vlm_report.clean_report;

            if (ground_truth.metrics) {
                elements.valBleu1.textContent = `%${ground_truth.metrics.bleu1}`;
                elements.valBleu4.textContent = `%${ground_truth.metrics.bleu4}`;
                elements.valRouge1.textContent = `%${ground_truth.metrics.rouge1}`;
                elements.valRougeL.textContent = `%${ground_truth.metrics.rougeL}`;
            }
        } else {
            elements.badgeGtAvailable.style.display = "none";
            elements.gtReportText.textContent = "Bu vaka için harici referans raporu yüklenmedi (Manuel yüklenen röntgen).";
            elements.gtPredText.textContent = vlm_report.clean_report;
            elements.valBleu1.textContent = "-";
            elements.valBleu4.textContent = "-";
            elements.valRouge1.textContent = "-";
            elements.valRougeL.textContent = "-";
        }

        // 7. Yazdırılabilir Formu Güncelle
        updatePrintableReport(data);

        // İkonları tazele
        if (window.lucide) lucide.createIcons();
    }

    // Yazdırılabilir Rapor Verilerini Hazırla
    function updatePrintableReport(data) {
        const now = new Date();
        const dateStr = now.toLocaleDateString("tr-TR");
        const reportNo = `MV-${now.getFullYear()}-${Math.floor(1000 + Math.random() * 9000)}`;

        elements.printReportNo.textContent = reportNo;
        elements.printDate.textContent = dateStr;
        elements.printCaseId.textContent = state.activeCaseId ? state.activeCaseId.substring(0, 20) : "VAKA-GENEL";
        elements.printPatientId.textContent = `PT-${Math.floor(100000 + Math.random() * 900000)}`;
        elements.printTriageLevel.textContent = data.cdss.urgency_label;
        elements.printImgOrig.src = data.images.original;
        elements.printImgSeg.src = data.images.mask_overlay;
        elements.printReportText.textContent = elements.doctorEditorText.value;
        elements.printSegConf.textContent = `%${data.segmentation_metrics.avg_confidence}`;
        elements.printAreaRatio.textContent = `%${data.segmentation_metrics.mask_area_ratio}`;
        
        const firstIcd = data.cdss.suggested_icd10 && data.cdss.suggested_icd10[0] ? data.cdss.suggested_icd10[0].code : "R91.8";
        elements.printIcd10.textContent = firstIcd;
        elements.printDoctorName.textContent = elements.inputDocName.value;
    }

    // ==========================================
    // 5. İNTERAKTİF BÖLÜNMÜŞ SLİDER
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
    }

    // Görünüm Modu Değiştirme
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
        } else {
            elements.splitSliderContainer.style.display = "none";
            elements.singleImgContainer.style.display = "flex";

            if (mode === "overlay") {
                elements.singleStageImg.src = imgs.mask_overlay;
            } else if (mode === "cleaned") {
                elements.singleStageImg.src = imgs.segmented_cleaned;
            } else if (mode === "raw") {
                elements.singleStageImg.src = imgs.original;
            }
        }
    }

    // ==========================================
    // 6. ETKİNLİK DİNLEYİCİLERİ (EVENT LISTENERS)
    // ==========================================
    function setupEventListeners() {
        // Sol Sekmeler (Örnekler / Yükle / Veri Seti)
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

        // Veri Seti Araması
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
            // Overlay görüntüsünün opaklığını CSS ile ayarla
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
            navigator.clipboard.writeText(state.analysisData.vlm_report.clean_report);
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

        // Model & Tez Bilgi Modalı
        elements.btnQuickGuide.addEventListener("click", () => {
            elements.infoModal.style.display = "flex";
        });

        elements.btnCloseModal.addEventListener("click", () => {
            elements.infoModal.style.display = "none";
        });

        elements.infoModal.addEventListener("click", (e) => {
            if (e.target === elements.infoModal) elements.infoModal.style.display = "none";
        });
    }

});
