document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const dropArea = document.getElementById('drop-area');
    const fileElem = document.getElementById('file-elem');
    const previewSection = document.getElementById('preview-section');
    const uploadForm = document.getElementById('upload-form');
    const imagePreview = document.getElementById('image-preview');
    const analyzeBtn = document.getElementById('analyze-btn');
    const changeFileBtn = document.getElementById('change-file-btn');
    const loading = document.getElementById('loading');
    const resultsCard = document.getElementById('results-card');
    const errorMessage = document.getElementById('error-message');
    const errorText = document.getElementById('error-text');

    // UI Result Elements
    const statusBadge = document.getElementById('status-badge');
    const primaryDiagnosisText = document.getElementById('primary-diagnosis-text');
    const radiologicalFindings = document.getElementById('radiological-findings');
    const confidenceValue = document.getElementById('confidence-value');
    const probabilityList = document.getElementById('probability-list');
    const stageValue = document.getElementById('stage-value');
    const timeValue = document.getElementById('time-value');
    const imageViewport = document.querySelector('.image-viewport');

    let selectedFile = null;

    // Drag and drop event handlers
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => dropArea.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => dropArea.classList.remove('dragover'), false);
    });

    dropArea.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files && files.length > 0) {
            handleFiles(files[0]);
        }
    }

    fileElem.addEventListener('change', function() {
        if (this.files && this.files.length > 0) {
            handleFiles(this.files[0]);
        }
    });

    function handleFiles(file) {
        if (!file.type.match('image.*')) {
            showError("Please select a valid image file (PNG, JPG, JPEG).");
            return;
        }

        if (file.size > 10 * 1024 * 1024) {
            showError("File size exceeds 10MB limit.");
            return;
        }

        selectedFile = file;
        hideError();

        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onloadend = function() {
            imagePreview.src = reader.result;
            uploadForm.classList.add('hidden');
            previewSection.classList.remove('hidden');
        };
    }

    // Reset preview
    changeFileBtn.addEventListener('click', () => {
        selectedFile = null;
        fileElem.value = '';
        imagePreview.src = '';
        previewSection.classList.add('hidden');
        uploadForm.classList.remove('hidden');
        hideError();
    });

    // Run AI Multi-Disease Analysis
    analyzeBtn.addEventListener('click', async () => {
        if (!selectedFile) {
            showError("No image selected for AI scanning.");
            return;
        }

        // Show loading state
        previewSection.classList.add('hidden');
        loading.classList.remove('hidden');
        if (imageViewport) imageViewport.classList.add('scanning');
        hideError();

        const formData = new FormData();
        formData.append('image', selectedFile);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            loading.classList.add('hidden');
            previewSection.classList.remove('hidden');
            if (imageViewport) imageViewport.classList.remove('scanning');

            if (data.error) {
                showError(data.error);
                return;
            }

            renderDiagnosticResults(data);

        } catch (err) {
            loading.classList.add('hidden');
            previewSection.classList.remove('hidden');
            if (imageViewport) imageViewport.classList.remove('scanning');
            showError("Connection failed. Could not communicate with AI analysis server.");
            console.error(err);
        }
    });

    function renderDiagnosticResults(data) {
        resultsCard.classList.remove('hidden');
        resultsCard.scrollIntoView({ behavior: 'smooth', block: 'start' });

        // Primary Diagnosis & Findings
        primaryDiagnosisText.textContent = data.primary_diagnosis;
        radiologicalFindings.textContent = data.radiological_findings || "";
        confidenceValue.textContent = `${data.confidence_level} (${data.accuracy}%)`;
        
        if (stageValue) stageValue.textContent = data.stage_of_disease || "N/A";
        if (timeValue) timeValue.textContent = `${data.execution_time_seconds}s`;

        // Status Badge Styling
        if (data.is_disease_detected) {
            statusBadge.className = "result-badge detected";
            statusBadge.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i> ${data.primary_diagnosis}`;
        } else {
            statusBadge.className = "result-badge healthy";
            statusBadge.innerHTML = `<i class="fa-solid fa-circle-check"></i> Normal (Healthy)`;
        }

        // Render Multi-Disease Probability Distribution Bars
        probabilityList.innerHTML = '';
        const supported = data.supported_diseases || [];

        supported.forEach(disease => {
            const prob = disease.probability || 0;
            const isTop = (disease.title === data.primary_diagnosis);

            const item = document.createElement('div');
            item.className = 'prob-item';

            let barColor = 'linear-gradient(90deg, #2563eb, #38bdf8)';
            if (isTop) {
                barColor = data.is_disease_detected 
                    ? 'linear-gradient(90deg, #dc2626, #f87171)' 
                    : 'linear-gradient(90deg, #059669, #34d399)';
            }

            item.innerHTML = `
                <div class="prob-header">
                    <span class="prob-name">
                        ${isTop ? '<strong>' + disease.title + '</strong>' : disease.title}
                        <small style="color: #64748b; margin-left: 6px;">(Acc: ${disease.accuracy})</small>
                    </span>
                    <span class="prob-val" style="color: ${isTop ? (data.is_disease_detected ? '#dc2626' : '#059669') : '#334155'};">${prob}%</span>
                </div>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill" style="width: ${prob}%; background: ${barColor};"></div>
                </div>
            `;
            probabilityList.appendChild(item);
        });
    }

    function showError(msg) {
        errorText.textContent = msg;
        errorMessage.classList.remove('hidden');
    }

    function hideError() {
        errorMessage.classList.add('hidden');
    }
});
