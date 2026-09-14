/**
 * SynapseRAG Knowledge Base and Ingestion Task Controller
 * Handles document upload, progress tracking, document library listing, previewing, and deletion.
 */

let activePreviewDoc = null;

document.addEventListener("DOMContentLoaded", () => {
    initIngestion();
    initKnowledgeBaseLibrary();
    initPreviewModal();
});

function initIngestion() {
    const dropzone = document.getElementById("upload-dropzone");
    const fileInput = document.getElementById("file-input-ingest");

    if (!dropzone || !fileInput) return;

    // Drag & Drop events
    ["dragenter", "dragover"].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropzone.classList.add("dragover");
        });
    });

    ["dragleave", "drop"].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropzone.classList.remove("dragover");
        });
    });

    dropzone.addEventListener("drop", (e) => {
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });
}

async function handleFileUpload(file) {
    const jobsList = document.getElementById("jobs-list");

    // Remove placeholder if present
    const placeholder = jobsList.querySelector(".job-card-placeholder");
    if (placeholder) {
        placeholder.remove();
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch("/api/v1/ingest/upload", {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`Upload failed with status ${response.status}`);
        }

        const data = await response.json();
        createJobProgressCard(data.job_id, data.filename);
        if (window.showToast) window.showToast(`Ingesting "${data.filename}" into Knowledge Base...`, "info");

        // Start Polling Job Status
        pollJobStatus(data.job_id);

    } catch (err) {
        console.error("Ingestion failed:", err);
        if (window.showToast) window.showToast(`Upload failed: ${err.message}`, "info");
    }
}

function createJobProgressCard(jobId, filename) {
    const jobsList = document.getElementById("jobs-list");
    const card = document.createElement("div");
    card.className = "job-card";
    card.id = `job-${jobId}`;

    card.innerHTML = `
        <div class="job-header">
            <div>
                <strong>${escapeHtml(filename)}</strong>
                <span style="font-size: 11px; color: #78716C; margin-left: 8px;">ID: ${jobId.substring(0, 8)}...</span>
            </div>
            <span class="job-status-pill processing" id="badge-${jobId}">PROCESSING</span>
        </div>
        <div class="job-meta-row">
            <span class="job-stage-text" id="stage-${jobId}">Stage: Initializing ingestion pipeline...</span>
            <span class="job-pct-text" id="pct-${jobId}">10%</span>
        </div>
        <div class="progress-bar-container">
            <div class="progress-bar-fill" id="progress-${jobId}" style="width: 10%;"></div>
        </div>
        <div class="job-result-bar" id="result-${jobId}" style="display: none;"></div>
    `;

    jobsList.prepend(card);
}

async function pollJobStatus(jobId) {
    const interval = setInterval(async () => {
        try {
            const res = await fetch(`/api/v1/jobs/${jobId}`);
            if (res.status === 404) {
                clearInterval(interval);
                const badge = document.getElementById(`badge-${jobId}`);
                const stage = document.getElementById(`stage-${jobId}`);
                if (badge) {
                    badge.className = "job-status-pill failed";
                    badge.textContent = "SESSION RESTARTED";
                }
                if (stage) stage.textContent = "Server restarted with active task worker. Please upload your document again to process.";
                return;
            }
            if (!res.ok) return;

            const job = await res.json();
            const badge = document.getElementById(`badge-${jobId}`);
            const stage = document.getElementById(`stage-${jobId}`);
            const pct = document.getElementById(`pct-${jobId}`);
            const progress = document.getElementById(`progress-${jobId}`);
            const resultBar = document.getElementById(`result-${jobId}`);

            const percentage = job.progress_percentage || (job.status === "completed" ? 100 : 25);

            if (stage) stage.textContent = `Stage: ${job.current_stage || "Processing"}`;
            if (pct) pct.textContent = `${percentage}%`;
            if (progress) progress.style.width = `${percentage}%`;

            if (job.status === "completed") {
                clearInterval(interval);
                if (badge) {
                    badge.className = "job-status-pill completed";
                    badge.textContent = "COMPLETED ✓";
                }
                if (progress) progress.style.width = "100%";
                if (pct) pct.textContent = "100%";
                if (stage) stage.textContent = job.current_stage || "Finished: Nodes, Chunks & Vectors indexed.";

                // Show result statistics
                if (resultBar && job.result) {
                    resultBar.style.display = "flex";
                    resultBar.innerHTML = `
                        <span class="job-metric-chip">Chunks: <strong>${job.result.chunks_created}</strong></span>
                        <span class="job-metric-chip">Entities: <strong>${job.result.entities_extracted}</strong></span>
                        <span class="job-metric-chip">Relations: <strong>${job.result.edges_created}</strong></span>
                        <span class="job-metric-chip">Elapsed: <strong>${job.result.execution_time_seconds}s</strong></span>
                        <button class="btn-job-action" onclick="askDocument('${escapeForAttr(job.filename)}')">Ask Questions in Chat 💬</button>
                    `;
                }

                if (window.showToast) window.showToast(`Document "${job.filename}" indexed into Knowledge Base!`, "success");

                // Refresh Knowledge Base Library & Graph
                loadKnowledgeBaseDocuments();
                if (typeof loadGraphData === "function") {
                    loadGraphData();
                }

            } else if (job.status === "failed") {
                clearInterval(interval);
                if (badge) {
                    badge.className = "job-status-pill failed";
                    badge.textContent = "FAILED ✗";
                }
                if (stage) stage.textContent = `Error: ${job.result?.error_message || "Ingestion interrupted"}`;
            }

        } catch (err) {
            console.warn("Polling error:", err);
        }
    }, 1500);
}

// ==============================================================================
// KNOWLEDGE BASE DOCUMENTS LIBRARY
// ==============================================================================

function initKnowledgeBaseLibrary() {
    const btnRefresh = document.getElementById("btn-refresh-kb");
    if (btnRefresh) {
        btnRefresh.addEventListener("click", () => {
            loadKnowledgeBaseDocuments();
            if (window.showToast) window.showToast("Refreshed Knowledge Base documents", "info");
        });
    }

    loadKnowledgeBaseDocuments();
}

let autoRetryTimer = null;

async function loadKnowledgeBaseDocuments() {
    const listEl = document.getElementById("kb-documents-list");
    const countBadge = document.getElementById("kb-doc-count");
    if (!listEl) return;

    // Helper to render documents cards
    function renderDocCards(docs, isCached = false) {
        if (!docs || docs.length === 0) {
            listEl.innerHTML = `
                <div class="kb-card-placeholder">
                    No documents currently in the Knowledge Base. Upload PDF, schematics, or image files above to build your multimodal index.
                </div>
            `;
            return;
        }

        const cacheNotice = isCached ? `
            <div style="grid-column: 1 / -1; background: #FEF3C7; border: 1px solid #F59E0B; border-radius: 8px; padding: 8px 14px; font-size: 12px; color: #92400E; display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px;">
                <span>⚠️ Backend server offline. Showing <strong>${docs.length}</strong> cached documents. Attempting to reconnect...</span>
                <button type="button" onclick="loadKnowledgeBaseDocuments()" style="background:#D97706;color:#fff;border:none;border-radius:4px;padding:3px 8px;font-size:11px;cursor:pointer;">Retry</button>
            </div>
        ` : '';

        listEl.innerHTML = cacheNotice + docs.map(doc => {
            const isPdf = doc.extension === ".pdf";
            const isImg = [".png", ".jpg", ".jpeg", ".webp"].includes(doc.extension);
            const icon = isPdf ? "📄" : (isImg ? "🖼️" : "📑");
            const dateStr = doc.modified_at ? new Date(doc.modified_at * 1000).toLocaleDateString([], { month: 'short', day: 'numeric', year: 'numeric' }) : "Recently Added";

            return `
                <div class="kb-doc-card" id="doc-card-${escapeForAttr(doc.filename)}">
                    <div class="kb-card-top">
                        <div class="kb-doc-icon">${icon}</div>
                        <div class="kb-doc-details">
                            <div class="kb-doc-filename" title="${escapeHtml(doc.filename)}">${escapeHtml(doc.filename)}</div>
                            <div class="kb-doc-meta">
                                <span>${doc.size_human || "N/A"}</span>
                                <span>•</span>
                                <span>${dateStr}</span>
                                <span>•</span>
                                <span class="kb-doc-status-badge">${doc.status || "Indexed"}</span>
                            </div>
                        </div>
                    </div>
                    <div class="kb-card-actions">
                        <button type="button" class="btn-kb-action" onclick="viewDocument('${escapeForAttr(doc.filename)}', '${doc.size_human || ""}', '${doc.mime_type || ""}')" title="View document in viewer">
                            <span>👁️ View</span>
                        </button>
                        <button type="button" class="btn-kb-action primary" onclick="askDocument('${escapeForAttr(doc.filename)}')" title="Ask questions about this document">
                            <span>💬 Ask Questions</span>
                        </button>
                        <button type="button" class="btn-kb-action delete" onclick="deleteDocument('${escapeForAttr(doc.filename)}')" title="Delete from Knowledge Base">
                            <span>🗑️</span>
                        </button>
                    </div>
                </div>
            `;
        }).join("");
    }

    try {
        const response = await fetch("/api/v1/documents");
        if (!response.ok) throw new Error(`HTTP ${response.status}`);

        const docs = await response.json();
        
        // Cache valid documents list
        try {
            localStorage.setItem("synapse_cached_docs", JSON.stringify(docs));
        } catch (e) {
            console.warn("Storage quota exceeded or storage unavailable");
        }

        if (countBadge) {
            countBadge.textContent = `${docs.length} ${docs.length === 1 ? 'Document' : 'Documents'}`;
            countBadge.style.color = "";
        }

        if (autoRetryTimer) {
            clearTimeout(autoRetryTimer);
            autoRetryTimer = null;
        }

        renderDocCards(docs, false);

    } catch (err) {
        console.error("Failed to load Knowledge Base documents:", err);

        // Check if cached documents exist in localStorage
        let cachedDocs = null;
        try {
            const raw = localStorage.getItem("synapse_cached_docs");
            if (raw) cachedDocs = JSON.parse(raw);
        } catch (e) {}

        if (cachedDocs && Array.isArray(cachedDocs) && cachedDocs.length > 0) {
            if (countBadge) {
                countBadge.textContent = `${cachedDocs.length} Docs (Cached)`;
                countBadge.style.color = "#D97706";
            }
            renderDocCards(cachedDocs, true);
        } else {
            if (countBadge) {
                countBadge.textContent = "Offline";
                countBadge.style.color = "#DC2626";
            }
            listEl.innerHTML = `
                <div class="kb-card-placeholder" style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 32px 16px; text-align: center;">
                    <div style="font-size: 32px; margin-bottom: 8px;">🔌</div>
                    <div style="font-weight: 600; font-size: 14px; color: #1C1917; margin-bottom: 4px;">
                        Backend Engine Disconnected
                    </div>
                    <div style="font-size: 12px; color: #78716C; max-width: 420px; line-height: 1.5; margin-bottom: 14px;">
                        Unable to reach the Multimodal API on port 8000 (<code>${escapeHtml(err.message)}</code>). The server may be restarting or offline.
                    </div>
                    <button type="button" class="btn-primary" onclick="loadKnowledgeBaseDocuments()" style="padding: 6px 16px; font-size: 12px; border-radius: 6px; cursor: pointer;">
                        🔄 Retry Connection
                    </button>
                </div>
            `;
        }

        // Schedule auto-retry probe after 4 seconds
        if (!autoRetryTimer) {
            autoRetryTimer = setTimeout(() => {
                autoRetryTimer = null;
                loadKnowledgeBaseDocuments();
            }, 4000);
        }
    }
}

// View Document in Preview Modal
window.viewDocument = function(filename, sizeHuman = "", mimeType = "") {
    activePreviewDoc = { filename, sizeHuman, mimeType };

    const modal = document.getElementById("doc-preview-modal");
    const modalTitle = document.getElementById("preview-modal-title");
    const modalMeta = document.getElementById("preview-modal-meta");
    const modalIcon = document.getElementById("preview-modal-icon");
    const modalIframe = document.getElementById("doc-preview-iframe");
    const btnExternal = document.getElementById("btn-open-external");

    if (!modal) return;

    const viewUrl = `/api/v1/documents/${encodeURIComponent(filename)}/view`;

    if (modalTitle) modalTitle.textContent = filename;
    if (modalMeta) modalMeta.textContent = `${sizeHuman || "Uploaded Document"} • ${mimeType || "media/stream"}`;
    if (modalIcon) {
        const isPdf = filename.toLowerCase().endsWith(".pdf");
        const isImg = [".png", ".jpg", ".jpeg"].some(ext => filename.toLowerCase().endsWith(ext));
        modalIcon.textContent = isPdf ? "📄" : (isImg ? "🖼️" : "📑");
    }

    if (modalIframe) modalIframe.src = viewUrl;
    if (btnExternal) btnExternal.href = viewUrl;

    modal.style.display = "flex";
};

// Delete Document from Knowledge Base
window.deleteDocument = async function(filename) {
    const confirmDelete = confirm(`Are you sure you want to delete "${filename}" from the Knowledge Base?\n\nThis will remove the file from disk, vector store embeddings, and associated knowledge graph citations.`);
    if (!confirmDelete) return;

    try {
        const res = await fetch(`/api/v1/documents/${encodeURIComponent(filename)}`, {
            method: "DELETE"
        });

        if (!res.ok) {
            throw new Error(`HTTP ${res.status}: Failed to delete document`);
        }

        const data = await res.json();
        if (window.showToast) window.showToast(`Deleted "${filename}" from Knowledge Base`, "success");

        // Close preview modal if viewing deleted doc
        if (activePreviewDoc && activePreviewDoc.filename === filename) {
            closePreviewModal();
        }

        // Refresh Knowledge Base Library
        loadKnowledgeBaseDocuments();

        // If active attached document in chat is this one, clear it
        if (window.clearAttachedDocumentIfMatches) {
            window.clearAttachedDocumentIfMatches(filename);
        }

    } catch (err) {
        console.error("Delete failed:", err);
        if (window.showToast) window.showToast(`Delete failed: ${err.message}`, "info");
    }
};

// Ask questions about document: Switches to Chat and sets up question prompt
window.askDocument = function(filename) {
    // 1. Switch to Chat tab
    const chatNavBtn = document.getElementById("btn-nav-chat");
    if (chatNavBtn) chatNavBtn.click();

    // 2. Set attached document banner
    const banner = document.getElementById("attached-doc-banner");
    const bannerName = document.getElementById("attached-doc-name");
    const bannerIcon = document.getElementById("attached-doc-icon");

    if (banner && bannerName) {
        bannerName.textContent = filename;
        const isPdf = filename.toLowerCase().endsWith(".pdf");
        if (bannerIcon) bannerIcon.textContent = isPdf ? "📄" : "🖼️";
        banner.style.display = "flex";
    }

    // 3. Populate chat input with targeted questions
    const chatInput = document.getElementById("chat-input");
    if (chatInput) {
        chatInput.value = `Analyze "${filename}" and summarize key insights, operational risks, and critical entity relationships.`;
        chatInput.focus();
    }

    // Close preview modal if open
    closePreviewModal();

    if (window.showToast) window.showToast(`Document "${filename}" attached for Q&A in Chat`, "info");
};

function initPreviewModal() {
    const modal = document.getElementById("doc-preview-modal");
    const btnClose = document.getElementById("btn-close-doc-modal");
    const btnAskDoc = document.getElementById("btn-modal-ask-doc");
    const btnDeleteDoc = document.getElementById("btn-modal-delete-doc");

    if (btnClose) {
        btnClose.addEventListener("click", closePreviewModal);
    }

    if (modal) {
        modal.addEventListener("click", (e) => {
            if (e.target === modal) closePreviewModal();
        });
    }

    if (btnAskDoc) {
        btnAskDoc.addEventListener("click", () => {
            if (activePreviewDoc) {
                askDocument(activePreviewDoc.filename);
            }
        });
    }

    if (btnDeleteDoc) {
        btnDeleteDoc.addEventListener("click", () => {
            if (activePreviewDoc) {
                deleteDocument(activePreviewDoc.filename);
            }
        });
    }
}

function closePreviewModal() {
    const modal = document.getElementById("doc-preview-modal");
    const modalIframe = document.getElementById("doc-preview-iframe");
    if (modal) modal.style.display = "none";
    if (modalIframe) modalIframe.src = "";
    activePreviewDoc = null;
}

function escapeHtml(str) {
    if (!str) return "";
    return str
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

function escapeForAttr(str) {
    if (!str) return "";
    return str.replace(/'/g, "\\'").replace(/"/g, '&quot;');
}

