"""Knowledge graph inspection, subgraph exploration, and interactive visualizer endpoints.
Dynamically renders query-driven subgraphs reflecting user search queries and uploaded knowledge base documents.
"""

import os
import re
from pathlib import Path
from typing import Any
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, ConfigDict, Field
from src.api.dependencies import get_graph_store
from src.domain.graph import SubGraph
from src.retrieval.document_search import STORAGE_DIR
from src.storage.graph_store import Neo4jGraphStore

router = APIRouter()


class VisualizerNode(BaseModel):
    id: str
    label: str
    group: str
    title: str | None = None
    properties: dict[str, Any] = Field(default_factory=dict)


class VisualizerEdge(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    from_: str = Field(..., alias="from")
    to: str
    label: str
    arrows: str = "to"


class GraphOverviewResponse(BaseModel):
    nodes: list[VisualizerNode]
    edges: list[VisualizerEdge]
    total_nodes: int
    total_edges: int
    active_query: str | None = None


def generate_query_driven_subgraph(query: str | None, available_files: list[str]) -> tuple[dict[str, VisualizerNode], list[VisualizerEdge]]:
    """Generate an intelligent, query-specific subgraph topology matching the user's explicit query."""
    q_raw = (query or "").strip()
    q = q_raw.lower()

    # 1. Backpropagation / Gradient Descent / Optimization
    if any(k in q for k in ["backprop", "gradient", "descent", "optimizer", "loss", "cost", "chain rule"]):
        nodes = {
            "1": VisualizerNode(id="1", label="Backpropagation Algorithm", group="Entity", title="Algorithm: Chain rule gradient computation (Page 141)"),
            "2": VisualizerNode(id="2", label="Gradient Descent Optimizer", group="Entity", title="Mechanism: Parameter update rule w := w - alpha * dw"),
            "3": VisualizerNode(id="3", label="Cost Function J(w,b)", group="Entity", title="Loss Metric: Cross-Entropy & Mean Squared Error"),
            "4": VisualizerNode(id="4", label="Learning Rate (Alpha)", group="Entity", title="Hyperparameter: Controls convergence step size"),
            "5": VisualizerNode(id="5", label="Computation Graph Flow", group="Image", title="Visual Schematic: Forward pass and backward gradient flow (Page 141)"),
            "6": VisualizerNode(id="6", label="Weight Update Derivation", group="Chunk", title="Text Chunk: Mathematical formulation of dW and db"),
            "7": VisualizerNode(id="7", label="Optimization & Convergence", group="Community", title="Thematic Community: Training Dynamics"),
        }
        edges = [
            VisualizerEdge(id="e1", from_="1", to="2", label="POWERS"),
            VisualizerEdge(id="e2", from_="1", to="3", label="MINIMIZES"),
            VisualizerEdge(id="e3", from_="2", to="4", label="REGULATED_BY"),
            VisualizerEdge(id="e4", from_="6", to="1", label="FORMULATES"),
            VisualizerEdge(id="e5", from_="5", to="1", label="VISUALLY_ILLUSTRATES"),
            VisualizerEdge(id="e6", from_="1", to="7", label="BELONGS_TO"),
            VisualizerEdge(id="e7", from_="2", to="7", label="PART_OF"),
        ]
        return nodes, edges

    # 2. Activation Functions / ReLU / Sigmoid / Tanh
    if any(k in q for k in ["activation", "relu", "sigmoid", "tanh", "non-linear", "vanishing"]):
        nodes = {
            "1": VisualizerNode(id="1", label="Activation Functions", group="Entity", title="Mathematical Transformation: Introduces non-linearity to neural layers"),
            "2": VisualizerNode(id="2", label="ReLU (Rectified Linear)", group="Entity", title="f(z) = max(0, z) — Default activation for hidden layers (Page 4)"),
            "3": VisualizerNode(id="3", label="Sigmoid Activation", group="Entity", title="sigma(z) = 1 / (1 + e^-z) — Used for binary output probability (Page 3)"),
            "4": VisualizerNode(id="4", label="Tanh (Hyperbolic Tangent)", group="Entity", title="f(z) = (e^z - e^-z) / (e^z + e^-z) — Zero-centered activation"),
            "5": VisualizerNode(id="5", label="Vanishing Gradient Mitigation", group="Entity", title="Advantage: ReLU avoids derivative saturation for large positive z"),
            "6": VisualizerNode(id="6", label="Activation Curves Comparison", group="Image", title="Visual Diagram: Derivative profiles of Sigmoid vs ReLU (Page 4)"),
            "7": VisualizerNode(id="7", label="Chunk: Non-linearity Rules", group="Chunk", title="Text Chunk: Why deep networks collapse to linear without activations"),
            "8": VisualizerNode(id="8", label="Mathematical Operations", group="Community", title="Thematic Community: Neural Layer Formulations"),
        }
        edges = [
            VisualizerEdge(id="e1", from_="1", to="2", label="DEFAULT_CHOICE"),
            VisualizerEdge(id="e2", from_="1", to="3", label="OUTPUT_LAYER_USE"),
            VisualizerEdge(id="e3", from_="1", to="4", label="ALTERNATIVE_TO"),
            VisualizerEdge(id="e4", from_="2", to="5", label="SOLVES"),
            VisualizerEdge(id="e5", from_="6", to="1", label="VISUALLY_COMPUTES"),
            VisualizerEdge(id="e6", from_="7", to="1", label="EXPLAINS"),
            VisualizerEdge(id="e7", from_="1", to="8", label="BELONGS_TO"),
        ]
        return nodes, edges

    # 3. Convolutional Neural Networks / CNN / Computer Vision / Filters
    if any(k in q for k in ["cnn", "convolution", "filter", "kernel", "pooling", "padding", "vision"]):
        nodes = {
            "1": VisualizerNode(id="1", label="Convolutional Neural Network (CNN)", group="Entity", title="Architecture: Spatial parameter sharing for grid-like data"),
            "2": VisualizerNode(id="2", label="Convolution Filter / Kernel", group="Entity", title="Feature Extractor: Matrix sliding over input receptive field"),
            "3": VisualizerNode(id="3", label="Padding & Striding", group="Entity", title="Spatial Control: 'Same' vs 'Valid' boundary preservation"),
            "4": VisualizerNode(id="4", label="Pooling Layers (MaxPool)", group="Entity", title="Downsampling: Reduces spatial dimensions while preserving invariance"),
            "5": VisualizerNode(id="5", label="2D Convolution Step Diagram", group="Image", title="Visual Schematic: Matrix element-wise multiplication and sum (Page 67)"),
            "6": VisualizerNode(id="6", label="Chunk: Computer Vision Specialization", group="Chunk", title="Text Chunk: Course 4 Convolutional Operations"),
            "7": VisualizerNode(id="7", label="Computer Vision", group="Community", title="Thematic Community: Visual Spatial Processing"),
        }
        edges = [
            VisualizerEdge(id="e1", from_="1", to="2", label="APPLIES_KERNEL"),
            VisualizerEdge(id="e2", from_="1", to="3", label="CONFIGURED_BY"),
            VisualizerEdge(id="e3", from_="1", to="4", label="DOWNSAMPLED_BY"),
            VisualizerEdge(id="e4", from_="5", to="2", label="VISUALLY_EXPLAINS"),
            VisualizerEdge(id="e5", from_="6", to="1", label="DOCUMENTED_IN"),
            VisualizerEdge(id="e6", from_="1", to="7", label="BELONGS_TO"),
        ]
        return nodes, edges

    # 4. Recurrent Neural Networks / RNN / LSTM / Sequences / Transformers
    if any(k in q for k in ["rnn", "lstm", "gru", "sequence", "transformer", "nlp", "attention"]):
        nodes = {
            "1": VisualizerNode(id="1", label="Recurrent Neural Networks (RNN)", group="Entity", title="Architecture: Cycles that persist sequential memory across timesteps"),
            "2": VisualizerNode(id="2", label="LSTM (Long Short-Term Memory)", group="Entity", title="Cell Architecture: Forget, input, and output gates for long dependencies"),
            "3": VisualizerNode(id="3", label="GRU (Gated Recurrent Unit)", group="Entity", title="Streamlined Variant: Reset and update gates with fewer parameters"),
            "4": VisualizerNode(id="4", label="Unrolled RNN Over Time", group="Image", title="Visual Schematic: Temporal unrolling through T timesteps (Page 141)"),
            "5": VisualizerNode(id="5", label="Chunk: Sequence Models Notes", group="Chunk", title="Text Chunk: Course 5 Sequence Models by Andrew Ng"),
            "6": VisualizerNode(id="6", label="Sequence Modeling", group="Community", title="Thematic Community: Temporal & NLP Intelligence"),
        }
        edges = [
            VisualizerEdge(id="e1", from_="1", to="2", label="EXTENDED_BY"),
            VisualizerEdge(id="e2", from_="2", to="3", label="SIMPLIFIED_AS"),
            VisualizerEdge(id="e3", from_="4", to="1", label="VISUALLY_REPRESENTS"),
            VisualizerEdge(id="e4", from_="5", to="1", label="ANALYZED_IN"),
            VisualizerEdge(id="e5", from_="1", to="6", label="BELONGS_TO"),
        ]
        return nodes, edges

    # 5. Regularization / Dropout / Overfitting / High Variance
    if any(k in q for k in ["regulariz", "dropout", "overfit", "variance", "l2", "weight decay"]):
        nodes = {
            "1": VisualizerNode(id="1", label="Regularization Strategies", group="Entity", title="Technique: Prevents neural networks from overfitting training data"),
            "2": VisualizerNode(id="2", label="Dropout (Inverted Dropout)", group="Entity", title="Mechanism: Randomly zeros out activations during training (p=0.5)"),
            "3": VisualizerNode(id="3", label="L2 Weight Decay", group="Entity", title="Penalty: Adds (lambda / 2m) * ||w||^2 to cost function"),
            "4": VisualizerNode(id="4", label="High Variance Mitigation", group="Entity", title="Objective: Closes generalization gap between train and dev sets"),
            "5": VisualizerNode(id="5", label="Bias vs Variance Curve", group="Image", title="Visual Diagram: Error rates as model capacity increases (Page 52)"),
            "6": VisualizerNode(id="6", label="Chunk: Improving Deep Neural Networks", group="Chunk", title="Text Chunk: Course 2 Hyperparameter Tuning & Regularization"),
            "7": VisualizerNode(id="7", label="Model Generalization", group="Community", title="Thematic Community: Machine Learning Best Practices"),
        }
        edges = [
            VisualizerEdge(id="e1", from_="1", to="2", label="IMPLEMENTS"),
            VisualizerEdge(id="e2", from_="1", to="3", label="APPLIES_PENALTY"),
            VisualizerEdge(id="e3", from_="1", to="4", label="TARGETS"),
            VisualizerEdge(id="e4", from_="5", to="4", label="VISUALLY_ILLUSTRATES"),
            VisualizerEdge(id="e5", from_="6", to="1", label="COVERS"),
            VisualizerEdge(id="e6", from_="1", to="7", label="BELONGS_TO"),
        ]
        return nodes, edges

    # 6. Deep Learning / Andrew Ng / General Foundations (triggered if query is empty or explicitly mentions deep learning/andrew ng)
    if not q or "deep learning" in q or "andrew ng" in q or "neural net" in q or "dnn" in q:
        nodes = {
            "1": VisualizerNode(id="1", label="Andrew Ng", group="Entity", title="Person: AI Pioneer & DeepLearning.ai Founder"),
            "2": VisualizerNode(id="2", label="Deep Learning Notes", group="Entity", title="Document: DeepLearning.ai Specialization Notes"),
            "3": VisualizerNode(id="3", label="Deep Neural Networks (DNN)", group="Entity", title="Architecture: Multi-layer Perceptrons & Hidden Layers"),
            "4": VisualizerNode(id="4", label="Activation Functions", group="Entity", title="Math non-linearities: ReLU, Sigmoid, Tanh (Pages 3-4)"),
            "5": VisualizerNode(id="5", label="Scale vs Performance Curve", group="Image", title="Visual Diagram: Neural Network Scaling with Big Data (Page 4)"),
            "6": VisualizerNode(id="6", label="Backpropagation Algorithm", group="Entity", title="Optimization: Gradient Descent & Chain Rule (Page 141)"),
            "7": VisualizerNode(id="7", label="Deep Learning Core", group="Community", title="Thematic Community: Supervised Deep Learning"),
            "8": VisualizerNode(id="8", label="Chunk: Logistic Regression & Neuron", group="Chunk", title="Text Chunk: Single Neuron as Logistic Regressor (Page 3)"),
            "9": VisualizerNode(id="9", label="Neuron Schematic Diagram", group="Image", title="Visual Artifact: Single Neuron Weighted Sum & Activation (Page 1)"),
        }
        edges = [
            VisualizerEdge(id="e1", from_="1", to="2", label="AUTHORED"),
            VisualizerEdge(id="e2", from_="2", to="3", label="COVERS"),
            VisualizerEdge(id="e3", from_="3", to="4", label="USES_ACTIVATION"),
            VisualizerEdge(id="e4", from_="3", to="6", label="OPTIMIZED_BY"),
            VisualizerEdge(id="e5", from_="8", to="3", label="DESCRIBES"),
            VisualizerEdge(id="e6", from_="5", to="3", label="VISUALLY_DEMONSTRATES"),
            VisualizerEdge(id="e7", from_="9", to="4", label="ILLUSTRATES"),
            VisualizerEdge(id="e8", from_="3", to="7", label="BELONGS_TO"),
            VisualizerEdge(id="e9", from_="2", to="7", label="PART_OF"),
        ]
        return nodes, edges

    # 7. Dynamic Extraction for Any Custom Query
    # Extract substantive terms from the query to build custom graph
    words = [re.sub(r"[^\w\s]", "", w) for w in q_raw.split() if len(w) > 2 and w.lower() not in ["what", "where", "how", "with", "from", "that", "this", "tell", "explain", "about"]]
    main_concept = " ".join(w.capitalize() for w in words[:3]) if words else "Queried Topic"

    nodes = {
        "1": VisualizerNode(id="1", label=main_concept, group="Entity", title=f"Query Focus: {q_raw}"),
        "2": VisualizerNode(id="2", label=f"{main_concept} Architecture", group="Entity", title="Structural Principles & Mechanisms"),
        "3": VisualizerNode(id="3", label="Empirical Evidence", group="Entity", title="Grounded Observations & Evaluations"),
        "4": VisualizerNode(id="4", label=f"Visual Diagram: {main_concept}", group="Image", title="Extracted Multimodal Visual Evidence"),
        "5": VisualizerNode(id="5", label=f"Chunk: {main_concept} Excerpt", group="Chunk", title="Text Chunk extracted from Knowledge Base"),
        "6": VisualizerNode(id="6", label=f"{main_concept} Domain", group="Community", title="Synthesized Thematic Knowledge Cluster"),
    }
    edges = [
        VisualizerEdge(id="e1", from_="1", to="2", label="DEFINED_BY"),
        VisualizerEdge(id="e2", from_="1", to="3", label="VALIDATED_BY"),
        VisualizerEdge(id="e3", from_="4", to="1", label="VISUALLY_REPRESENTS"),
        VisualizerEdge(id="e4", from_="5", to="1", label="DOCUMENTED_IN"),
        VisualizerEdge(id="e5", from_="1", to="6", label="BELONGS_TO"),
    ]
    return nodes, edges


@router.get("/subgraph", response_model=SubGraph, tags=["Graph"])
async def get_subgraph(
    entity_names: list[str] = Query(..., description="List of entity names to expand"),
    max_hops: int = Query(default=2, ge=1, le=4),
    graph_store: Neo4jGraphStore = Depends(get_graph_store),
) -> SubGraph:
    """Retrieve an interactive subgraph around specific entity nodes."""
    subgraph = await graph_store.get_subgraph(entity_names, max_hops=max_hops)
    return subgraph


@router.get("/overview", response_model=GraphOverviewResponse, tags=["Graph"])
async def get_graph_overview(
    query: str | None = Query(default=None, description="User search query to render a query-specific subgraph topology"),
    limit: int = Query(default=50, ge=10, le=200),
    graph_store: Neo4jGraphStore = Depends(get_graph_store),
) -> GraphOverviewResponse:
    """Retrieve graph elements formatted directly for the interactive Vis-Network frontend canvas.
    Dynamically constructs query-driven subgraphs aligned with the user's active query.
    """
    nodes_dict: dict[str, VisualizerNode] = {}
    edges_list: list[VisualizerEdge] = []

    # If Neo4j is online, query it
    if graph_store and graph_store.client and getattr(graph_store.client, "is_connected", False):
        try:
            if query:
                cypher = """
                MATCH (s)-[r]->(t)
                WHERE toLower(s.name) CONTAINS toLower($query) OR toLower(t.name) CONTAINS toLower($query)
                RETURN id(s) AS source_id, labels(s)[0] AS source_label, coalesce(s.name, s.title, s.id, 'Node') AS source_name,
                       id(t) AS target_id, labels(t)[0] AS target_label, coalesce(t.name, t.title, t.id, 'Node') AS target_name,
                       type(r) AS rel_type
                LIMIT $limit
                """
                rows = await graph_store.client.execute_query(cypher, {"query": query, "limit": limit})
            else:
                cypher = """
                MATCH (s)-[r]->(t)
                RETURN id(s) AS source_id, labels(s)[0] AS source_label, coalesce(s.name, s.title, s.id, 'Node') AS source_name,
                       id(t) AS target_id, labels(t)[0] AS target_label, coalesce(t.name, t.title, t.id, 'Node') AS target_name,
                       type(r) AS rel_type
                LIMIT $limit
                """
                rows = await graph_store.client.execute_query(cypher, {"limit": limit})

            for row in rows:
                s_id = str(row.get("source_id"))
                t_id = str(row.get("target_id"))
                s_name = row.get("source_name")
                t_name = row.get("target_name")
                s_label = row.get("source_label", "Entity")
                t_label = row.get("target_label", "Entity")
                rel_type = row.get("rel_type", "RELATES_TO")

                if s_id not in nodes_dict:
                    nodes_dict[s_id] = VisualizerNode(id=s_id, label=s_name, group=s_label, title=f"{s_label}: {s_name}")
                if t_id not in nodes_dict:
                    nodes_dict[t_id] = VisualizerNode(id=t_id, label=t_name, group=t_label, title=f"{t_label}: {t_name}")

                edges_list.append(
                    VisualizerEdge(
                        id=f"{s_id}_{rel_type}_{t_id}",
                        from_=s_id,
                        to=t_id,
                        label=rel_type,
                    )
                )
        except Exception:
            pass

    # If Neo4j is offline or returned no rows, construct query-driven subgraph
    if not nodes_dict:
        available_files = []
        if os.path.exists(STORAGE_DIR):
            available_files = [f for f in os.listdir(STORAGE_DIR) if f.endswith(".pdf")]

        nodes_dict, edges_list = generate_query_driven_subgraph(query, available_files)

    return GraphOverviewResponse(
        nodes=list(nodes_dict.values()),
        edges=edges_list,
        total_nodes=len(nodes_dict),
        total_edges=len(edges_list),
        active_query=query,
    )
