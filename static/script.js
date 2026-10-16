// National Flood Incident Command System (ICS) Frontend Controller

let selectedImageFile = null;

const PRESET_DATA = {
    1: {
        text: "sunny asphalt street, dry pavement, clear storm drain, normal vehicular traffic, pedestrians walking, blue sky, unobstructed sidewalk, zero water accumulation, dry surface",
        desc: "Routine conditions: sunny afternoon on dry municipal pavement with unobstructed drainage and normal traffic."
    },
    3: {
        text: "ankle-deep puddle, clogged sewer grate with wet leaves, car tire throwing water plume, standing water near curb, nuisance ponding, passable road",
        desc: "Minor nuisance ponding: shallow ankle-deep water around blocked curbside catch basins; curbs remain visible."
    },
    5: {
        text: "knee-deep murky floodwater, stalled compact sedan with hazard lights flashing, sandbag barriers stacked at shop doors, water reaching car bumpers, wheel-height inundation",
        desc: "Major flood warning: knee-deep brown torrent touching car bumpers and wheel rims; stalled sedans and shop sandbagging."
    },
    7: {
        text: "completely submerged passenger cars with only roofs poking out, first story of single-family homes completely underwater, motorized rescue boat navigating street, residents on second-floor balconies",
        desc: "State of emergency: first floor structures inundated to ceiling; cars fully submerged to roofs; rescue watercraft deployed."
    },
    8: {
        text: "residents stranded on rooftop peak, waving orange distress flag and bedsheet, violent brown torrent swirling below, single-story houses submerged to roof gutter lines, catastrophic flash flood",
        desc: "Disaster declaration: single-story homes drowned to roofs; families trapped on roof peaks awaiting tactical airborne helicopter extraction."
    },
    10: {
        text: "dead body floating face-down in stagnant brown floodwater, deceased human victim in waterlogged clothes, emergency coroner team approaching in inflatable dinghy, yellow body recovery bag, mass casualty flood disaster",
        desc: "Mass casualty disaster: confirmed deceased human victims floating in floodwaters; coroner recovery units deployed."
    }
};

const SEVERITY_COLORS = {
    1: "#10b981",
    2: "#22c55e",
    3: "#eab308",
    4: "#f59e0b",
    5: "#f97316",
    6: "#ef4444",
    7: "#dc2626",
    8: "#b91c1c",
    9: "#991b1b",
    10: "#581c87"
};

// Drag and drop setup
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");

["dragenter", "dragover"].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.add("dragover");
    });
});

["dragleave", "drop"].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.remove("dragover");
    });
});

dropzone.addEventListener("drop", (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files.length > 0) {
        processSelectedFile(files[0]);
    }
});

function handleFileSelect(event) {
    if (event.target.files.length > 0) {
        processSelectedFile(event.target.files[0]);
    }
}

function processSelectedFile(file) {
    if (!file.type.startsWith("image/")) {
        alert("Please select a valid image file (JPEG, PNG, HEIC, WebP).");
        return;
    }
    
    selectedImageFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        document.getElementById("imagePreview").src = e.target.result;
        document.getElementById("dropzonePrompt").style.display = "none";
        document.getElementById("imagePreviewContainer").style.display = "block";
        document.getElementById("analyzePhotoBtn").disabled = false;
    };
    reader.readAsDataURL(file);
}

function removeImage(event) {
    if (event) event.stopPropagation();
    selectedImageFile = null;
    fileInput.value = "";
    document.getElementById("imagePreview").src = "";
    document.getElementById("imagePreviewContainer").style.display = "none";
    document.getElementById("dropzonePrompt").style.display = "flex";
    document.getElementById("analyzePhotoBtn").disabled = true;
}

function switchTab(tab) {
    if (tab === "photo") {
        document.getElementById("tabPhoto").classList.add("active");
        document.getElementById("tabText").classList.remove("active");
        document.getElementById("tabPhotoBtn").classList.add("active");
        document.getElementById("tabTextBtn").classList.remove("active");
    } else {
        document.getElementById("tabPhoto").classList.remove("active");
        document.getElementById("tabText").classList.add("active");
        document.getElementById("tabPhotoBtn").classList.remove("active");
        document.getElementById("tabTextBtn").classList.add("active");
    }
}

function loadPreset(severityLevel) {
    const preset = PRESET_DATA[severityLevel];
    if (!preset) return;
    
    switchTab("text");
    document.getElementById("textInput").value = preset.text;
    analyzeText();
}

async function analyzePhoto() {
    if (!selectedImageFile) return;

    const btn = document.getElementById("analyzePhotoBtn");
    const textSpan = document.getElementById("btnPhotoText");
    const spinner = document.getElementById("btnPhotoSpinner");

    btn.disabled = true;
    textSpan.textContent = "Executing Multi-Stage Disaster Grounding & Consensus...";
    spinner.style.display = "inline-block";

    const formData = new FormData();
    formData.append("image", selectedImageFile);

    try {
        const response = await fetch("/api/analyze-image", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!response.ok || data.error) {
            throw new Error(data.error || "Failed to analyze incident image");
        }

        renderGovernmentReport(data, "photo");
    } catch (err) {
        alert("Incident Assessment Error: " + err.message);
    } finally {
        btn.disabled = false;
        textSpan.textContent = "⚡ Execute Incident Triage & Verification";
        spinner.style.display = "none";
    }
}

async function analyzeText() {
    const textVal = document.getElementById("textInput").value.trim();
    if (!textVal) {
        alert("Please enter field officer observations or keywords.");
        return;
    }

    const btn = document.getElementById("analyzeTextBtn");
    const textSpan = document.getElementById("btnTextText");
    const spinner = document.getElementById("btnTextSpinner");

    btn.disabled = true;
    textSpan.textContent = "Evaluating Field Report with ML Consensus...";
    spinner.style.display = "inline-block";

    try {
        const response = await fetch("/api/analyze-text", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: textVal })
        });

        const data = await response.json();

        if (!response.ok || data.error) {
            throw new Error(data.error || "Failed to evaluate field report");
        }

        renderGovernmentReport(data, "text");
    } catch (err) {
        alert("Prediction Error: " + err.message);
    } finally {
        btn.disabled = false;
        textSpan.textContent = "⚡ Evaluate Field Report";
        spinner.style.display = "none";
    }
}

function renderGovernmentReport(data, source) {
    document.getElementById("resultsPlaceholder").style.display = "none";
    document.getElementById("resultsContent").style.display = "block";
    document.getElementById("assessmentBadge").style.display = "inline-block";

    const rep = data.incident_report;
    const vision = data.vision_grounding || {};
    const score = rep.severity_score;
    const color = rep.badge_color || SEVERITY_COLORS[score] || "#3b82f6";

    // Score Circle
    const scoreCircle = document.getElementById("scoreCircle");
    scoreCircle.style.borderColor = color;
    scoreCircle.style.boxShadow = `0 0 25px ${color}66`;
    document.getElementById("scoreVal").textContent = score;
    document.getElementById("scoreVal").style.color = color;

    // Badges & Title
    document.getElementById("threatBadge").textContent = `Threat Level: ${rep.threat_level}`;
    document.getElementById("femaCategoryBadge").textContent = rep.fema_incident_category;
    document.getElementById("severityTitle").textContent = `Level ${score}: ${rep.severity_level}`;

    // Confidence
    const confBadge = document.getElementById("confidenceBadge");
    confBadge.textContent = `Confidence: ${rep.confidence_level}`;
    const pct = Math.round(rep.confidence_score * 100);
    document.getElementById("confidencePct").textContent = `${pct}% (${rep.confidence_score})`;
    document.getElementById("confidenceRationale").textContent = rep.confidence_rationale;

    // Key Metrics Grid
    document.getElementById("waterDepthVal").textContent = rep.water_depth_estimate;
    document.getElementById("sceneSettingVal").textContent = vision.scene_setting || "Field Incident Location";

    // ICS Directives
    document.getElementById("icsPhaseVal").textContent = rep.ics_operational_phase;
    document.getElementById("icsAgencyVal").textContent = rep.ics_lead_agency;

    // AI Vision Grounding Description
    document.getElementById("visionDescText").textContent = vision.objective_description || "Field evidence analyzed.";

    // Submerged Landmarks Tags
    const landmarksContainer = document.getElementById("landmarksTags");
    landmarksContainer.innerHTML = "";
    const landmarks = rep.submerged_landmarks_detected || [];
    if (landmarks.length === 0 || (landmarks.length === 1 && landmarks[0] === "none")) {
        landmarksContainer.innerHTML = `<span class="tag-pill" style="color:#94a3b8">None (dry infrastructure)</span>`;
    } else {
        landmarks.forEach(lm => {
            const span = document.createElement("span");
            span.className = "landmark-pill";
            span.textContent = lm.replace(/_/g, " ");
            landmarksContainer.appendChild(span);
        });
    }

    // Keywords Tags
    const kwContainer = document.getElementById("keywordsTags");
    kwContainer.innerHTML = "";
    const kws = vision.exact_keywords || [];
    kws.forEach(kw => {
        const span = document.createElement("span");
        span.className = "tag-pill";
        span.textContent = kw;
        kwContainer.appendChild(span);
    });

    // Action Alert
    document.getElementById("actionText").textContent = rep.recommended_action;

    // ML Model Evaluation Audit Box
    const consensusDiv = document.getElementById("consensusContent");
    const cd = rep.consensus_details || {};
    consensusDiv.innerHTML = `
        <div class="consensus-grid">
            <div class="consensus-cell">
                <span>ML Model Prediction</span>
                <strong>Level ${cd.ml_predicted_score || score}</strong>
            </div>
            <div class="consensus-cell">
                <span>Continuous Value</span>
                <strong>${rep.continuous_score || score} / 10</strong>
            </div>
        </div>
        <div class="audit-note">
            <strong>Engine:</strong> Trained Scikit-Learn Model (TF-IDF + CalibratedClassifierCV + Ridge)<br>
            <strong>Role:</strong> Vision model extracts objective physical description &rarr; Trained ML model computes severity score & ICS protocol.<br>
            ${(cd.audit_notes || []).join("<br>")}
        </div>
    `;

    // Probability Bars
    const probBars = document.getElementById("probBars");
    probBars.innerHTML = "";
    const dist = data.probability_distribution || {};

    for (let i = 1; i <= 10; i++) {
        const key = `Level_${i}`;
        const p = dist[key] || 0;
        const pctVal = Math.round(p * 100);
        const barColor = SEVERITY_COLORS[i];

        const row = document.createElement("div");
        row.className = "prob-row";
        row.innerHTML = `
            <span class="prob-label">Lvl ${i}</span>
            <div class="prob-track">
                <div class="prob-fill" style="width: ${pctVal}%; background: ${barColor};"></div>
            </div>
            <span class="prob-val">${pctVal}%</span>
        `;
        probBars.appendChild(row);
    }

    // JSON Details Viewer
    document.getElementById("jsonViewer").querySelector("code").textContent = JSON.stringify(data, null, 2);
}
