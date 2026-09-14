/**
 * SynapseRAG App State, Navigation, and Interactive Controls Manager
 */

document.addEventListener("DOMContentLoaded", () => {
    initNavigation();
    initGlobalShortcuts();
    initLayoutToggle();
    initDropdowns();
    initHeaderActionButtons();
    initHealthPolling();
});

function initNavigation() {
    const navItems = document.querySelectorAll(".nav-item");
    const tabPanes = document.querySelectorAll(".tab-pane");

    navItems.forEach(item => {
        item.addEventListener("click", () => {
            const targetTabId = item.getAttribute("data-tab");

            navItems.forEach(n => n.classList.remove("active"));
            tabPanes.forEach(p => p.classList.remove("active"));

            item.classList.add("active");
            const targetPane = document.getElementById(targetTabId);
            if (targetPane) {
                targetPane.classList.add("active");
            }

            // Trigger robust re-render if switching to graph view
            if (targetTabId === "tab-graph") {
                if (window.renderGraphCanvas) {
                    setTimeout(() => window.renderGraphCanvas(), 100);
                } else if (window.graphViewer) {
                    window.graphViewer.setSize("100%", "100%");
                    window.graphViewer.redraw();
                    window.graphViewer.fit();
                }
            }

            // Refresh Knowledge Base documents if switching to ingest tab
            if (targetTabId === "tab-ingest") {
                if (typeof loadKnowledgeBaseDocuments === "function") {
                    loadKnowledgeBaseDocuments();
                }
            }
        });
    });
}

function initLayoutToggle() {
    const btnToggle = document.getElementById("btn-toggle-layout");
    const panel = document.getElementById("provenance-panel");
    if (btnToggle && panel) {
        btnToggle.addEventListener("click", () => {
            panel.classList.toggle("provenance-hidden");
            const isHidden = panel.classList.contains("provenance-hidden");
            btnToggle.title = isHidden ? "Show Provenance Sidebar" : "Hide Provenance Sidebar (Maximize Canvas)";
            showToast(isHidden ? "Provenance sidebar hidden (Maximized View)" : "Provenance sidebar restored", "info");

            // Redraw & fit graph if canvas is visible
            if (window.renderGraphCanvas) {
                setTimeout(() => window.renderGraphCanvas(), 150);
            }
        });
    }
}

function initDropdowns() {
    const modelChip = document.getElementById("chip-model-selector");
    const modelMenu = document.getElementById("model-dropdown-menu");
    const modelText = document.getElementById("selected-model-text");

    const ragChip = document.getElementById("chip-rag-selector");
    const ragMenu = document.getElementById("rag-dropdown-menu");
    const ragText = document.getElementById("selected-rag-text");

    function closeAllDropdowns() {
        if (modelMenu) modelMenu.classList.remove("show");
        if (modelChip) modelChip.classList.remove("open");
        if (ragMenu) ragMenu.classList.remove("show");
        if (ragChip) ragChip.classList.remove("open");
    }

    if (modelChip && modelMenu) {
        modelChip.addEventListener("click", (e) => {
            e.stopPropagation();
            const isOpen = modelMenu.classList.contains("show");
            closeAllDropdowns();
            if (!isOpen) {
                modelMenu.classList.add("show");
                modelChip.classList.add("open");
            }
        });

        modelMenu.querySelectorAll(".dropdown-item").forEach(item => {
            item.addEventListener("click", (e) => {
                e.stopPropagation();
                modelMenu.querySelectorAll(".dropdown-item").forEach(i => i.classList.remove("active"));
                item.classList.add("active");
                const val = item.getAttribute("data-model");
                if (modelText) modelText.textContent = val;
                closeAllDropdowns();
                showToast(`Switched reasoning engine to ${val}`, "info");
            });
        });
    }

    if (ragChip && ragMenu) {
        ragChip.addEventListener("click", (e) => {
            e.stopPropagation();
            const isOpen = ragMenu.classList.contains("show");
            closeAllDropdowns();
            if (!isOpen) {
                ragMenu.classList.add("show");
                ragChip.classList.add("open");
            }
        });

        ragMenu.querySelectorAll(".dropdown-item").forEach(item => {
            item.addEventListener("click", (e) => {
                e.stopPropagation();
                ragMenu.querySelectorAll(".dropdown-item").forEach(i => i.classList.remove("active"));
                item.classList.add("active");
                const val = item.getAttribute("data-rag");
                if (ragText) ragText.textContent = val;
                closeAllDropdowns();
                showToast(`Retriever configured: ${val}`, "info");
            });
        });
    }

    document.addEventListener("click", () => {
        closeAllDropdowns();
    });
}

function initHeaderActionButtons() {
    const btnNotifications = document.getElementById("btn-notifications");
    if (btnNotifications) {
        btnNotifications.addEventListener("click", () => {
            showToast("🔔 3 Ingestion pipelines completed • Neo4j graph synced • Qdrant index active", "info");
        });
    }

    const btnConfig = document.getElementById("btn-config");
    if (btnConfig) {
        btnConfig.addEventListener("click", () => {
            showToast("⚙️ Core Config: ColPali 768-dim, Neo4j Bolt 5.20, Hybrid RRF k=60, Temp 0.2", "info");
        });
    }

    const btnProfile = document.getElementById("btn-profile-gear");
    if (btnProfile) {
        btnProfile.addEventListener("click", () => {
            showToast("👤 Dr. Elena Rostova • Principal AI Architect (Role: Enterprise SuperAdmin)", "info");
        });
    }

    const globalSearch = document.getElementById("global-search-input");
    if (globalSearch) {
        globalSearch.addEventListener("keydown", (e) => {
            if (e.key === "Enter") {
                const val = globalSearch.value.trim();
                if (val) {
                    window.executePrompt(val);
                    globalSearch.value = "";
                }
            }
        });
    }
}

function initGlobalShortcuts() {
    // ⌘K or Ctrl+K for Global Search focus
    document.addEventListener("keydown", (e) => {
        if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
            e.preventDefault();
            const searchInput = document.getElementById("global-search-input");
            if (searchInput) {
                searchInput.focus();
                searchInput.select();
            }
        }
    });

    const tenancyBtn = document.getElementById("btn-tenancy");
    if (tenancyBtn) {
        tenancyBtn.addEventListener("click", () => {
            showToast("🏢 Enterprise Tenancy: [AlphaCorp Global], Tenant [TEN-0824-SYN], Pods [8/8 Dedicated]", "info");
        });
    }
}

let isBackendLive = null;

async function initHealthPolling() {
    const pill = document.querySelector(".cluster-status-pill");
    const dot = pill ? pill.querySelector(".status-dot") : null;
    const text = pill ? pill.querySelector(".cluster-text") : null;

    async function checkHealth() {
        try {
            const response = await fetch("/api/v1/health", { cache: "no-store" });
            if (response.ok) {
                if (isBackendLive !== true) {
                    isBackendLive = true;
                    if (dot) {
                        dot.className = "status-dot green";
                        dot.style.background = "";
                        dot.style.boxShadow = "";
                    }
                    if (text) text.innerHTML = `US-East-1 Cluster • <strong>Connected</strong>`;
                    
                    // Auto-refresh document list if it was offline
                    if (typeof loadKnowledgeBaseDocuments === "function") {
                        loadKnowledgeBaseDocuments();
                    }
                }
            } else {
                throw new Error(`HTTP ${response.status}`);
            }
        } catch (err) {
            if (isBackendLive !== false) {
                isBackendLive = false;
                if (dot) {
                    dot.className = "status-dot red";
                    dot.style.background = "#EF4444";
                    dot.style.boxShadow = "0 0 8px rgba(239, 68, 68, 0.7)";
                }
                if (text) text.innerHTML = `Backend Core • <strong style="color: #EF4444;">Offline (Port 8000)</strong>`;
            }
        }
    }

    checkHealth();
    setInterval(checkHealth, 4000);
}

// Global Toast System
window.showToast = function(message, type = "info") {
    const container = document.getElementById("toast-container");
    if (!container) return;

    const toast = document.createElement("div");
    toast.className = `toast toast-${type}`;
    const icon = type === "success" ? "✓" : "ℹ️";
    toast.innerHTML = `<span>${icon}</span><span>${message}</span>`;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = "0";
        setTimeout(() => toast.remove(), 260);
    }, 3200);
};

// Global prompt executor called by prompt cards and search
window.executePrompt = function(queryText) {
    const chatTabBtn = document.getElementById("btn-nav-chat");
    if (chatTabBtn && !chatTabBtn.classList.contains("active")) {
        chatTabBtn.click();
    }

    const input = document.getElementById("chat-input");
    const form = document.getElementById("chat-form");
    if (input && form) {
        input.value = queryText;
        input.focus();
        form.dispatchEvent(new Event("submit", { cancelable: true, bubbles: true }));
    }
};

window.setQuery = window.executePrompt;
