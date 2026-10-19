/**
 * OcuScan AI — Production Diagnostic Workspace Client Logic
 * Handles file uploads, preset samples, Grad-CAM heatmap crossfades,
 * and 8-class probability distribution animations.
 */

document.addEventListener("DOMContentLoaded", () => {
    // --- DOM Elements ---
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const browseBtn = document.getElementById("browse-btn");
    const sampleButtons = document.querySelectorAll(".sample-pill-btn");

    const emptyState = document.getElementById("empty-state");
    const diagContent = document.getElementById("diagnostic-content");
    
    // Visualizer Elements
    const previewImage = document.getElementById("preview-image");
    const gradcamImage = document.getElementById("gradcam-image");
    const previewFilename = document.getElementById("preview-filename");
    const layerLabel = document.getElementById("layer-label");
    const toggleGradcamBtn = document.getElementById("toggle-gradcam-btn");
    const toggleBtnText = document.getElementById("toggle-btn-text");
    const scanLaser = document.getElementById("scan-laser");
    const scanningBadge = document.getElementById("scanning-badge");

    // Diagnostic Metrics Elements
    const primaryCard = document.getElementById("primary-card");
    const diagName = document.getElementById("diag-name");
    const diagConfVal = document.getElementById("diag-confidence-val");
    const diagUrgency = document.getElementById("diag-urgency-pill");
    const diagAction = document.getElementById("diag-action");
    const diagDesc = document.getElementById("diag-description");
    const breakdownList = document.getElementById("breakdown-list");

    // Action Controls
    const btnReset = document.getElementById("btn-reset");
    const btnPrint = document.getElementById("btn-print");

    // State Tracking
    let currentGradcamUrl = "";
    let isShowingGradcam = false;

    // Constants for Circular Gauges (radius=32 -> 2 * PI * 32 = 201.06)
    const GAUGE_CIRCUMFERENCE = 201.06;

    // Default 8 Disease Classes for Horizontal Gauge Bar
    const DEFAULT_CLASSES = [
        { class_id: 0, disease: "Normal", tag: "Healthy", color: "#10B981" },
        { class_id: 1, disease: "Diabetes", tag: "Retinopathy", color: "#EF4444" },
        { class_id: 2, disease: "Glaucoma", tag: "Optic CDR", color: "#8B5CF6" },
        { class_id: 3, disease: "Cataract", tag: "Lens Opacity", color: "#F59E0B" },
        { class_id: 4, disease: "AMD", tag: "Maculopathy", color: "#EC4899" },
        { class_id: 5, disease: "Hypertension", tag: "Vascular", color: "#F97316" },
        { class_id: 6, disease: "Myopia", tag: "High Myopia", color: "#06B6D4" },
        { class_id: 7, disease: "Others", tag: "Non-Specific", color: "#64748B" }
    ];

    // ==========================================
    // 1. File Upload Handlers (Drag & Drop + Browse)
    // ==========================================
    browseBtn.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        fileInput.click();
    });

    dropZone.addEventListener("click", (e) => {
        // Prevent double trigger if clicking directly on browse button
        if (e.target !== browseBtn) {
            fileInput.click();
        }
    });

    // Drag Events
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add('drag-active');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove('drag-active');
        });
    });

    dropZone.addEventListener("drop", (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files && files.length > 0) {
            handleUploadedFile(files[0]);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleUploadedFile(e.target.files[0]);
        }
    });

    // ==========================================
    // 2. Preset Sample Buttons
    // ==========================================
    sampleButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            const sampleName = btn.getAttribute("data-sample");
            
            // Mark active button
            sampleButtons.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");

            handleSamplePreset(sampleName);
        });
    });

    // ==========================================
    // 3. Grad-CAM Crossfade Toggle
    // ==========================================
    toggleGradcamBtn.addEventListener("click", () => {
        if (!currentGradcamUrl) return;

        isShowingGradcam = !isShowingGradcam;
        applyHeatmapLayerState();
    });

    function applyHeatmapLayerState() {
        if (isShowingGradcam && currentGradcamUrl) {
            gradcamImage.classList.add("visible");
            toggleGradcamBtn.classList.add("active");
            toggleBtnText.textContent = "Show Original Fundus";
            layerLabel.textContent = "Grad-CAM Saliency Heatmap";
            layerLabel.style.background = "var(--brand-gradient)";
        } else {
            gradcamImage.classList.remove("visible");
            toggleGradcamBtn.classList.remove("active");
            toggleBtnText.textContent = "Show Grad-CAM Heatmap";
            layerLabel.textContent = "Original Fundus";
            layerLabel.style.background = "#141416";
        }
    }

    // ==========================================
    // 4. Action Controls (Reset & Print)
    // ==========================================
    btnReset.addEventListener("click", resetDiagnosticState);
    btnPrint.addEventListener("click", () => window.print());

    // ==========================================
    // 5. Core Diagnostic Inference Pipelines
    // ==========================================

    async function handleUploadedFile(file) {
        if (!file.type.startsWith("image/")) {
            alert("Please upload a valid image file (PNG, JPG, JPEG).");
            return;
        }

        sampleButtons.forEach(b => b.classList.remove("active"));

        const imageUrl = URL.createObjectURL(file);
        prepareScanUI(imageUrl, file.name);

        const formData = new FormData();
        formData.append("file", file);

        try {
            const response = await fetch("/api/predict", {
                method: "POST",
                body: formData
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || "Diagnosis failed on server.");
            }

            const data = await response.json();
            renderDiagnosisResults(data);
        } catch (error) {
            console.error("Diagnosis Request Error:", error);
            alert(`Analysis Error: ${error.message}`);
            stopScanUI();
        }
    }

    async function handleSamplePreset(sampleName) {
        const sampleUrl = `/static/samples/${sampleName}.png`;
        prepareScanUI(sampleUrl, `Sample Case: ${sampleName.toUpperCase()}`);

        try {
            const response = await fetch("/api/predict-sample", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ sample_name: sampleName })
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || "Sample diagnosis failed.");
            }

            const data = await response.json();
            renderDiagnosisResults(data);
        } catch (error) {
            console.error("Sample Request Error:", error);
            alert(`Sample Analysis Error: ${error.message}`);
            stopScanUI();
        }
    }

    function prepareScanUI(imageSrc, filename) {
        // Switch view from empty state to workspace
        emptyState.style.display = "none";
        diagContent.style.display = "flex";

        // Set Images
        previewImage.src = imageSrc;
        previewFilename.textContent = filename;
        gradcamImage.src = "";
        gradcamImage.classList.remove("visible");
        currentGradcamUrl = "";
        isShowingGradcam = false;
        applyHeatmapLayerState();
        toggleGradcamBtn.style.opacity = "0.5";
        toggleGradcamBtn.style.pointerEvents = "none";

        // Activate Laser & Badge
        scanLaser.style.display = "block";
        scanningBadge.style.display = "flex";

        // Loading placeholders
        diagName.textContent = "Inferring ResNet50 convolutional features...";
        diagConfVal.textContent = "--";
        diagUrgency.textContent = "SCANNING";
        diagUrgency.style.background = "rgba(124, 58, 237, 0.1)";
        diagUrgency.style.color = "var(--brand-violet)";
        diagUrgency.style.borderColor = "rgba(124, 58, 237, 0.3)";

        btnReset.disabled = true;
        btnPrint.disabled = true;
    }

    function stopScanUI() {
        scanLaser.style.display = "none";
        scanningBadge.style.display = "none";
        btnReset.disabled = false;
        btnPrint.disabled = false;
    }

    function renderDiagnosisResults(data) {
        stopScanUI();

        // 1. Store Grad-CAM image
        if (data.gradcam_url) {
            currentGradcamUrl = data.gradcam_url;
            gradcamImage.src = data.gradcam_url;
            toggleGradcamBtn.style.opacity = "1";
            toggleGradcamBtn.style.pointerEvents = "auto";
        } else {
            toggleGradcamBtn.style.opacity = "0.5";
            toggleGradcamBtn.style.pointerEvents = "none";
        }

        // 2. Primary Diagnostic Details
        diagName.textContent = data.primary_diagnosis;
        diagConfVal.textContent = data.confidence_pct;
        diagAction.textContent = data.clinical_action || "Comprehensive ophthalmic examination advised.";
        diagDesc.textContent = data.description || "Feature extraction completed across 8 clinical categories.";

        // Urgency Pill Styling
        const accentColor = data.color || "#10B981";
        diagConfVal.style.color = accentColor;
        primaryCard.style.borderLeftColor = accentColor;

        diagUrgency.textContent = data.urgency || "ROUTINE";
        diagUrgency.style.background = hexToRgba(accentColor, 0.12);
        diagUrgency.style.color = accentColor;
        diagUrgency.style.borderColor = hexToRgba(accentColor, 0.35);

        // Co-Occurring Detections Box
        const coOccurBox = document.getElementById("co-occurring-box");
        const coOccurChips = document.getElementById("co-occurring-chips");
        if (coOccurBox && coOccurChips) {
            if (data.co_occurring && data.co_occurring.length > 0) {
                coOccurChips.innerHTML = "";
                data.co_occurring.forEach(item => {
                    const chip = document.createElement("div");
                    chip.className = "co-occurring-chip";
                    const itemColor = item.color || "#7C3AED";
                    const cutoff = item.threshold_pct || 20.0;
                    chip.style.borderColor = hexToRgba(itemColor, 0.35);
                    chip.style.background = hexToRgba(itemColor, 0.08);
                    chip.innerHTML = `
                        <span class="chip-dot" style="background: ${itemColor};"></span>
                        <span class="chip-name" style="color: ${itemColor};">${item.disease}</span>
                        <span class="chip-pct">${item.percentage}%</span>
                        <span class="chip-cutoff">(cutoff ${cutoff}%)</span>
                    `;
                    coOccurChips.appendChild(chip);
                });
                coOccurBox.style.display = "block";
            } else {
                coOccurBox.style.display = "none";
            }
        }

        // 3. Render Multi-Disease Probability Breakdown with Cutoff Thresholds & Detection Badges
        renderBreakdownBars(data.breakdown, data.primary_diagnosis);
    }

    const CLINICAL_THRESHOLDS_MAP = {
        "Age-related Macular Degeneration (AMD)": 10.0,
        "Cataract": 18.0,
        "Diabetes": 20.0,
        "Glaucoma": 12.0,
        "Hypertension": 50.0,
        "Myopia": 20.0,
        "Normal": 35.0,
        "Others": 10.0
    };

    function renderBreakdownBars(breakdown, primaryDiagnosis) {
        breakdownList.innerHTML = "";

        const isNormalPrimary = (primaryDiagnosis === "Normal");

        breakdown.forEach((item, index) => {
            const card = document.createElement("div");
            const isTop = index === 0;
            const barColor = item.color || "#7C3AED";

            // Safe cutoff lookup guaranteeing a real number
            const cutoffVal = (item.threshold_pct !== undefined && item.threshold_pct !== null)
                ? item.threshold_pct
                : (CLINICAL_THRESHOLDS_MAP[item.disease] || 20.0);

            // Mutual exclusivity enforcement: If Normal is diagnosed, no other disease can be detected!
            let isDetected = false;
            if (isNormalPrimary) {
                isDetected = (item.disease === "Normal");
            } else {
                if (item.disease === "Normal") {
                    isDetected = false; // Disease detected -> Normal is clear
                } else {
                    isDetected = (item.is_detected !== undefined && item.is_detected !== null)
                        ? item.is_detected
                        : (item.percentage >= cutoffVal);
                }
            }

            card.className = `circular-meter-card ${isDetected ? 'condition-detected' : ''} ${isTop ? 'top-match' : ''}`;
            card.id = `meter-card-${item.class_id}`;

            // Clean concise display name
            let displayName = item.disease;
            let wikiUrl = "https://en.wikipedia.org/wiki/Retinal_disease";
            if (displayName.includes("Age-related Macular Degeneration") || displayName.includes("AMD")) {
                displayName = "AMD";
                wikiUrl = "https://en.wikipedia.org/wiki/Macular_degeneration";
            } else if (displayName.includes("Diabetes")) {
                displayName = "Diabetes";
                wikiUrl = "https://en.wikipedia.org/wiki/Diabetic_retinopathy";
            } else if (displayName.includes("Glaucoma")) {
                displayName = "Glaucoma";
                wikiUrl = "https://en.wikipedia.org/wiki/Glaucoma";
            } else if (displayName.includes("Cataract")) {
                displayName = "Cataract";
                wikiUrl = "https://en.wikipedia.org/wiki/Cataract";
            } else if (displayName.includes("Hypertension")) {
                displayName = "Hypertension";
                wikiUrl = "https://en.wikipedia.org/wiki/Hypertensive_retinopathy";
            } else if (displayName.includes("Myopia")) {
                displayName = "High Myopia";
                wikiUrl = "https://en.wikipedia.org/wiki/Degenerative_myopia";
            } else if (displayName.includes("Normal")) {
                displayName = "Normal";
                wikiUrl = "https://en.wikipedia.org/wiki/Fundus_(eye)";
            } else if (displayName.includes("Others")) {
                displayName = "Other Lesions";
                wikiUrl = "https://en.wikipedia.org/wiki/Retinal_disease";
            }

            const statusBadge = isDetected
                ? `<span class="status-pill detected" style="color: ${barColor}; background: ${hexToRgba(barColor, 0.12)}; border-color: ${hexToRgba(barColor, 0.35)};"><span class="pill-dot"></span>DETECTED</span>`
                : `<span class="status-pill clear"><span class="pill-dot" style="background: #94A3B8;"></span>CLEAR</span>`;

            card.innerHTML = `
                ${isTop ? '<div class="top-match-badge">Primary Match</div>' : ''}
                <div class="card-status-header">
                    ${statusBadge}
                </div>
                <div class="gauge-wrapper">
                    <svg class="circular-gauge-svg" viewBox="0 0 80 80">
                        <circle class="gauge-bg" cx="40" cy="40" r="32" />
                        <circle class="gauge-meter" id="gauge-meter-${item.class_id}" cx="40" cy="40" r="32"
                            style="stroke: ${barColor}; stroke-dasharray: ${GAUGE_CIRCUMFERENCE}; stroke-dashoffset: ${GAUGE_CIRCUMFERENCE};" />
                    </svg>
                    <div class="gauge-value-center">
                        <span class="gauge-pct" id="gauge-pct-${item.class_id}" style="color: ${isDetected ? barColor : 'var(--text-primary)'}">
                            0.0%
                        </span>
                    </div>
                </div>
                <div class="cutoff-indicator-row">
                    <span class="cutoff-label">Cutoff: <strong>${cutoffVal}%</strong></span>
                </div>
                <div class="meter-info">
                    <a href="${wikiUrl}" target="_blank" rel="noopener noreferrer" class="meter-name" title="Read ${item.disease} on Wikipedia">${displayName} ↗</a>
                    <span class="meter-tag-pill" style="color: ${barColor}; background: ${hexToRgba(barColor, 0.08)}">${item.tag}</span>
                </div>
            `;

            breakdownList.appendChild(card);

            // Animate circular gauge meter & counter with staggered delay
            const targetOffset = GAUGE_CIRCUMFERENCE - (item.percentage / 100) * GAUGE_CIRCUMFERENCE;
            setTimeout(() => {
                const meterCircle = document.getElementById(`gauge-meter-${item.class_id}`);
                const pctLabel = document.getElementById(`gauge-pct-${item.class_id}`);
                if (meterCircle) {
                    meterCircle.style.strokeDashoffset = targetOffset.toFixed(2);
                }
                if (pctLabel) {
                    animateCounter(pctLabel, 0, item.percentage, 900);
                }
            }, 60 + index * 45);
        });
    }

    function animateCounter(element, start, end, duration) {
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            const currentVal = (progress * (end - start) + start).toFixed(1);
            element.textContent = `${currentVal}%`;
            if (progress < 1) {
                window.requestAnimationFrame(step);
            } else {
                element.textContent = `${end}%`;
            }
        };
        window.requestAnimationFrame(step);
    }

    function initializeDefaultGauges() {
        renderBreakdownBars(DEFAULT_CLASSES.map(cls => ({
            ...cls,
            percentage: 0,
            confidence: 0
        })));
    }

    function resetDiagnosticState() {
        emptyState.style.display = "flex";
        diagContent.style.display = "none";
        previewImage.src = "";
        gradcamImage.src = "";
        currentGradcamUrl = "";
        isShowingGradcam = false;
        applyHeatmapLayerState();
        previewFilename.textContent = "";
        fileInput.value = "";
        sampleButtons.forEach(b => b.classList.remove("active"));
        btnReset.disabled = true;
        btnPrint.disabled = true;
        initializeDefaultGauges();
    }

    function hexToRgba(hex, alpha = 1) {
        let clean = hex.replace('#', '');
        if (clean.length === 3) {
            clean = clean.split('').map(c => c + c).join('');
        }
        const num = parseInt(clean, 16);
        const r = (num >> 16) & 255;
        const g = (num >> 8) & 255;
        const b = num & 255;
        return `rgba(${r}, ${g}, ${b}, ${alpha})`;
    }

    // ==========================================
    // Hero Headline Typewriter Effect
    // ==========================================
    function initTypewriterEffect() {
        const textTarget = document.getElementById("typewriter-text");
        if (!textTarget) return;

        // Respect prefers-reduced-motion
        if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
            textTarget.textContent = "eight clinical insights.";
            return;
        }

        const phrases = [
            "eight clinical insights.",
            "instant disease triage.",
            "explainable Grad-CAM maps.",
            "sub-2-second screening."
        ];

        let phraseIndex = 0;
        let charIndex = phrases[0].length;
        let isDeleting = false;
        let typingSpeed = 75;

        function tick() {
            const currentPhrase = phrases[phraseIndex];

            if (isDeleting) {
                charIndex--;
                textTarget.textContent = currentPhrase.substring(0, charIndex);
                typingSpeed = 35; // Fast backspace
            } else {
                charIndex++;
                textTarget.textContent = currentPhrase.substring(0, charIndex);
                typingSpeed = 75; // Calm, deliberate keystroke
            }

            if (!isDeleting && charIndex === currentPhrase.length) {
                // Hold complete phrase for readability
                isDeleting = true;
                setTimeout(tick, 2800);
                return;
            } else if (isDeleting && charIndex === 0) {
                // Switch to next phrase
                isDeleting = false;
                phraseIndex = (phraseIndex + 1) % phrases.length;
                setTimeout(tick, 450);
                return;
            }

            setTimeout(tick, typingSpeed);
        }

        // Begin typewriter cycling after initial calm view (3 seconds)
        setTimeout(() => {
            isDeleting = true;
            tick();
        }, 3000);
    }

    // ==========================================
    // Smooth Scrolling & Navigation Controller
    // ==========================================
    function initSmoothScrolling() {
        const scrollTopBtn = document.getElementById("scroll-top-btn");
        const prefersReduced = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
        const scrollBehavior = prefersReduced ? "auto" : "smooth";

        // 1. Intercept internal anchor navigation links for seamless scrolling
        const internalAnchorLinks = document.querySelectorAll('a[href^="#"]');
        internalAnchorLinks.forEach(link => {
            link.addEventListener("click", (e) => {
                const href = link.getAttribute("href");
                if (!href || href === "#") {
                    e.preventDefault();
                    window.scrollTo({ top: 0, behavior: scrollBehavior });
                    return;
                }

                const targetEl = document.querySelector(href);
                if (targetEl) {
                    e.preventDefault();

                    // Smoothly scroll to target
                    targetEl.scrollIntoView({
                        behavior: scrollBehavior,
                        block: "start"
                    });

                    // Update URL hash cleanly without immediate jump
                    if (window.history && window.history.pushState) {
                        window.history.pushState(null, null, href);
                    }

                    // Special visual feedback: if jumping to the diagnostic workspace
                    if (href === "#workspace" && dropZone) {
                        setTimeout(() => {
                            dropZone.classList.remove("pulse-focus");
                            // Force reflow
                            void dropZone.offsetWidth;
                            dropZone.classList.add("pulse-focus");
                            setTimeout(() => {
                                dropZone.classList.remove("pulse-focus");
                            }, 1800);
                        }, 400);
                    }
                }
            });
        });

        // 2. Floating Back-to-Top Button Visibility Toggle
        if (scrollTopBtn) {
            let ticking = false;

            const handleScroll = () => {
                if (!ticking) {
                    window.requestAnimationFrame(() => {
                        const scrollY = window.pageYOffset || document.documentElement.scrollTop;
                        if (scrollY > 380) {
                            scrollTopBtn.classList.add("visible");
                        } else {
                            scrollTopBtn.classList.remove("visible");
                        }
                        ticking = false;
                    });
                    ticking = true;
                }
            };

            window.addEventListener("scroll", handleScroll, { passive: true });
            // Initial check
            handleScroll();

            // Click listener for smooth return to top
            scrollTopBtn.addEventListener("click", (e) => {
                e.preventDefault();
                window.scrollTo({
                    top: 0,
                    behavior: scrollBehavior
                });
                if (window.history && window.history.pushState) {
                    window.history.pushState(null, null, window.location.pathname);
                }
            });
        }
    }

    // Initialize components on page load
    initializeDefaultGauges();
    initTypewriterEffect();
    initSmoothScrolling();
});
