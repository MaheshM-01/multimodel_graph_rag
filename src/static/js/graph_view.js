/**
 * Vis-Network Force-Directed Knowledge Graph Visualizer
 * Fully Query-Driven: dynamically loads and highlights subgraphs tailored to user queries.
 */

let networkInstance = null;
let graphDataLoaded = false;
let currentActiveQuery = null;

document.addEventListener("DOMContentLoaded", () => {
    initGraphView();
});

function initGraphView() {
    const container = document.getElementById("graph-canvas");
    const btnReload = document.getElementById("btn-reload-graph");
    const btnFit = document.getElementById("btn-fit-graph");
    const btnSearch = document.getElementById("btn-search-subgraph");
    const queryInput = document.getElementById("graph-query-input");

    if (!container) return;

    // Reset Graph button
    if (btnReload) {
        btnReload.addEventListener("click", () => {
            if (queryInput) queryInput.value = "";
            currentActiveQuery = null;
            loadGraphData(true, null);
            if (window.showToast) window.showToast("Reset graph to root overview", "info");
        });
    }

    // Fit View button
    if (btnFit) {
        btnFit.addEventListener("click", () => {
            if (networkInstance) {
                networkInstance.fit({
                    animation: { duration: 400, easingFunction: "easeInOutQuad" }
                });
            }
        });
    }

    // Graph Query Search Button
    if (btnSearch && queryInput) {
        btnSearch.addEventListener("click", () => {
            const q = queryInput.value.trim();
            if (q) {
                loadGraphData(true, q);
            }
        });

        queryInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter") {
                e.preventDefault();
                const q = queryInput.value.trim();
                if (q) {
                    loadGraphData(true, q);
                }
            }
        });
    }

    // Top Global Header Search Integration
    const globalSearchInput = document.getElementById("global-search-input");
    if (globalSearchInput) {
        globalSearchInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter") {
                const q = globalSearchInput.value.trim();
                if (q) {
                    // Switch to Graph tab
                    const navGraph = document.getElementById("btn-nav-graph");
                    if (navGraph) navGraph.click();

                    if (queryInput) queryInput.value = q;
                    loadGraphData(true, q);
                    if (window.showToast) window.showToast(`Exploring graph for: "${q}"`, "info");
                }
            }
        });
    }

    const btnQueryNode = document.getElementById("btn-inspect-query-node");
    if (btnQueryNode) {
        btnQueryNode.addEventListener("click", () => {
            const target = currentInspectedNode.label || currentInspectedNode.id || "Deep Learning";
            if (window.executePrompt) {
                window.executePrompt(`Traverse knowledge graph relationships and principles for entity: ${target}`);
            }
        });
    }

    const btnFocusCanvas = document.getElementById("btn-inspect-focus-graph");
    if (btnFocusCanvas) {
        btnFocusCanvas.addEventListener("click", () => {
            const navGraph = document.getElementById("btn-nav-graph");
            if (navGraph) navGraph.click();

            if (networkInstance && currentInspectedNode) {
                networkInstance.focus(currentInspectedNode.id, {
                    scale: 1.4,
                    animation: { duration: 500, easingFunction: "easeInOutQuad" }
                });
                if (window.showToast) window.showToast(`Focused on node: ${currentInspectedNode.label || currentInspectedNode.id}`, "info");
            }
        });
    }
}

let currentInspectedNode = {
    id: "1",
    label: "Andrew Ng",
    group: "Entity"
};

async function loadGraphData(forceFit = false, query = null) {
    const container = document.getElementById("graph-canvas");
    const statNodes = document.getElementById("graph-stat-nodes");
    const statEdges = document.getElementById("graph-stat-edges");
    const activeQueryTag = document.getElementById("active-subgraph-tag");
    const queryInput = document.getElementById("graph-query-input");

    if (!container) return;

    currentActiveQuery = query || null;

    if (activeQueryTag) {
        if (query) {
            activeQueryTag.style.display = "inline-flex";
            activeQueryTag.innerHTML = `<span>🎯 Subgraph: <strong>${escapeHtml(query)}</strong></span> <button type="button" onclick="window.resetGraphQuery()" style="background:none;border:none;color:#C2410C;cursor:pointer;font-weight:bold;margin-left:4px;" title="Clear filter">✕</button>`;
            if (queryInput && queryInput.value !== query) {
                queryInput.value = query;
            }
        } else {
            activeQueryTag.style.display = "none";
        }
    }

    try {
        let url = "/api/v1/graph/overview?limit=60";
        if (query) {
            url += `&query=${encodeURIComponent(query)}`;
        }

        const response = await fetch(url);
        const data = await response.json();

        if (statNodes) statNodes.textContent = `Nodes: ${data.total_nodes}`;
        if (statEdges) statEdges.textContent = `Edges: ${data.total_edges}`;

        // Map groups to curated visual themes with high contrast
        const groupOptions = {
            Entity: {
                color: { background: "#D9531E", border: "#C84815", highlight: { background: "#EA580C", border: "#C2410C" } },
                shape: "dot",
                size: 24,
                font: { color: "#FFFFFF", face: "Inter", size: 12, strokeWidth: 3, strokeColor: "#0B0F19" },
            },
            Image: {
                color: { background: "#B45309", border: "#92400E", highlight: { background: "#D97706", border: "#B45309" } },
                shape: "diamond",
                size: 26,
                font: { color: "#FFFFFF", face: "Inter", size: 12, strokeWidth: 3, strokeColor: "#0B0F19" },
            },
            Chunk: {
                color: { background: "#2563EB", border: "#1D4ED8", highlight: { background: "#3B82F6", border: "#1E40AF" } },
                shape: "box",
                margin: 8,
                font: { color: "#FFFFFF", face: "Inter", size: 11 },
            },
            Community: {
                color: { background: "#16A34A", border: "#15803D", highlight: { background: "#22C55E", border: "#16A34A" } },
                shape: "hexagon",
                size: 28,
                font: { color: "#FFFFFF", face: "Inter", size: 12, strokeWidth: 3, strokeColor: "#0B0F19" },
            },
        };

        const nodes = new vis.DataSet(data.nodes);
        const edges = new vis.DataSet(data.edges);

        const networkData = { nodes, edges };
        const options = {
            groups: groupOptions,
            nodes: {
                borderWidth: 2,
                shadow: { enabled: true, color: "rgba(0,0,0,0.5)", size: 6, x: 2, y: 2 },
            },
            edges: {
                color: { color: "rgba(255,255,255,0.45)", highlight: "#D9531E", hover: "#D9531E" },
                font: { color: "#E2E8F0", size: 11, face: "JetBrains Mono", align: "middle", strokeWidth: 3, strokeColor: "#0B0F19" },
                width: 2,
                smooth: { type: "continuous" },
                arrows: { to: { enabled: true, scaleFactor: 0.8 } },
            },
            physics: {
                enabled: true,
                stabilization: {
                    enabled: true,
                    iterations: 150,
                    updateInterval: 25,
                },
                barnesHut: {
                    gravitationalConstant: -2500,
                    centralGravity: 0.3,
                    springLength: 140,
                    springConstant: 0.05,
                    damping: 0.09,
                    avoidOverlap: 0.2,
                },
            },
            interaction: {
                hover: true,
                tooltipDelay: 100,
                navigationButtons: true,
                keyboard: false,
            },
        };

        // If existing network instance, destroy before recreating
        if (networkInstance) {
            networkInstance.destroy();
            networkInstance = null;
        }

        networkInstance = new vis.Network(container, networkData, options);
        window.graphViewer = networkInstance;
        graphDataLoaded = true;

        networkInstance.once("stabilizationIterationsDone", () => {
            if (networkInstance) {
                networkInstance.fit();
            }
        });

        // Click handler to update live inspector in right sidebar
        networkInstance.on("click", (params) => {
            if (params.nodes.length > 0) {
                const nodeId = params.nodes[0];
                const nodeData = nodes.get(nodeId);

                // Extract connected relations
                const rels = [];
                edges.forEach(edge => {
                    if (edge.from === nodeId) {
                        const target = nodes.get(edge.to);
                        rels.push(`[:${edge.label || "RELATES"}] ➔ ${target ? target.label : edge.to}`);
                    } else if (edge.to === nodeId) {
                        const source = nodes.get(edge.from);
                        rels.push(`${source ? source.label : edge.from} ➔ [:${edge.label || "RELATES"}]`);
                    }
                });

                window.updateLiveInspector(nodeData, rels);
            }
        });

        // Automatically focus inspector on first node of the query subgraph
        if (data.nodes && data.nodes.length > 0) {
            const firstNode = data.nodes[0];
            const rels = [];
            data.edges.forEach(edge => {
                if (edge.from === firstNode.id) {
                    const target = data.nodes.find(n => n.id === edge.to);
                    rels.push(`[:${edge.label || "RELATES"}] ➔ ${target ? target.label : edge.to}`);
                }
            });
            window.updateLiveInspector(firstNode, rels);
        }

        if (forceFit) {
            setTimeout(() => {
                if (networkInstance) networkInstance.fit();
            }, 300);
        }

    } catch (err) {
        console.error("Failed to load Knowledge Graph data:", err);
    }
}

window.resetGraphQuery = function() {
    const queryInput = document.getElementById("graph-query-input");
    if (queryInput) queryInput.value = "";
    loadGraphData(true, null);
};

window.setGraphQuery = function(query) {
    const queryInput = document.getElementById("graph-query-input");
    if (queryInput) queryInput.value = query;
    loadGraphData(true, query);
};

window.loadGraphData = loadGraphData;

window.updateLiveInspector = function(node, relations = null) {
    if (!node) return;
    currentInspectedNode = node;

    const badgeId = document.getElementById("inspector-node-badge");
    const typePill = document.getElementById("inspector-type-pill");
    const nameEl = document.getElementById("inspector-node-name");
    const categoryEl = document.getElementById("inspector-attr-category");
    const riskEl = document.getElementById("inspector-attr-risk");
    const modalityEl = document.getElementById("inspector-attr-modality");
    const centralityEl = document.getElementById("inspector-attr-centrality");
    const relTagsContainer = document.getElementById("inspector-relation-tags");

    if (badgeId) badgeId.textContent = `ID: ${node.id}`;
    if (nameEl) nameEl.textContent = node.label || node.id;
    if (typePill) {
        typePill.textContent = node.group || "Entity";
        typePill.className = `inspector-badge-type pill-${(node.group || "entity").toLowerCase()}`;
    }

    if (categoryEl) {
        if (node.group === "Image") categoryEl.textContent = "Visual Schematic / Asset";
        else if (node.group === "Chunk") categoryEl.textContent = "Text / Unstructured Excerpt";
        else if (node.group === "Community") categoryEl.textContent = "Leiden Graph Cluster";
        else categoryEl.textContent = "Concept / Mathematical Entity";
    }

    if (riskEl) {
        riskEl.textContent = "Nominal / Monitored";
        riskEl.className = "attr-val text-green strong";
    }

    if (modalityEl) {
        if (node.group === "Image") modalityEl.textContent = "Vision (Page Diagram)";
        else if (node.group === "Chunk") modalityEl.textContent = "Dense Semantic + Lexical";
        else modalityEl.textContent = "Graph Structure + Multimodal Grounding";
    }

    if (centralityEl) {
        centralityEl.textContent = node.group === "Community" ? "0.95 (Cluster Modularity)" : "0.87 (High Centrality)";
    }

    if (relTagsContainer) {
        if (relations && relations.length > 0) {
            relTagsContainer.innerHTML = relations.slice(0, 4).map(r => `<span class="rel-tag">${escapeHtml(r)}</span>`).join("");
        } else {
            relTagsContainer.innerHTML = `
                <span class="rel-tag">[:CONNECTED_TO] ➔ Synapse Knowledge Graph</span>
                <span class="rel-tag">[:GROUNDED_IN] ➔ Multimodal Evidence</span>
            `;
        }
    }
};

function escapeHtml(str) {
    if (!str) return "";
    return str
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

// Global render/refresh method triggered on tab switch
window.renderGraphCanvas = function() {
    const container = document.getElementById("graph-canvas");
    if (!container) return;

    if (!networkInstance || !graphDataLoaded) {
        loadGraphData(true, currentActiveQuery);
    } else {
        networkInstance.setSize("100%", "100%");
        networkInstance.redraw();
        setTimeout(() => {
            if (networkInstance) {
                networkInstance.fit({
                    animation: { duration: 400, easingFunction: "easeInOutQuad" }
                });
            }
        }, 100);
    }
};
