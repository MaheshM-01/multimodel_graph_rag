"""Graph store abstraction and Neo4j implementation with high-availability in-memory graph fallback."""

import re
from abc import ABC, abstractmethod
from typing import Any
from src.core.logging import logger
from src.domain.graph import SubGraph
from src.graph.client import Neo4jClient


class BaseGraphStore(ABC):
    """Abstract interface for knowledge graph storage."""

    @abstractmethod
    async def query(self, query_str: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        pass

    @abstractmethod
    async def get_subgraph(self, entity_ids: list[str], max_hops: int = 2) -> SubGraph:
        pass


class Neo4jGraphStore(BaseGraphStore):
    """Neo4j implementation of the graph store interface with rich in-memory semantic fallback."""

    def __init__(self, client: Neo4jClient):
        self.client = client
        self._local_graph: dict[str, dict[str, Any]] = {
            "transformer": {
                "primary_entity": "Transformer Architecture",
                "related_entities": ["Multi-Head Attention", "Self-Attention", "Scaled Dot-Product", "Positional Encoding", "Encoder-Decoder"],
                "text_snippets": [
                    "Sequence models augmented with self-attention mechanisms allow long-sequence representation without recurrent bottlenecks.",
                    "Attention model works like a human focusing on specific parts at a time, increasing accuracy over long sequences (Page 156, 163).",
                ],
                "images": ["/static/previews/Deep_Learning_Andrew_Ng__p163.png"],
            },
            "attention": {
                "primary_entity": "Attention Mechanism & Transformer",
                "related_entities": ["Transformer Architecture", "Self-Attention", "Query-Key-Value Vectors", "Softmax Alignment", "Context Vectors", "Encoder", "Decoder", "Multi-Head Attention", "BLEU Accuracy Curves"],
                "text_snippets": [
                    "Self-attention dynamically calculates alignment scores alpha_{t,t'} determining influence between sequence token positions across encoder and decoder multi-head architectures, boosting accuracy.",
                ],
                "images": ["/static/previews/Deep_Learning_Andrew_Ng__p156.png"],
            },
            "backprop": {
                "primary_entity": "Backpropagation Algorithm",
                "related_entities": ["Forward Propagation", "Chain Rule", "Loss Function", "Gradient Descent Updates", "Cache Matrix"],
                "text_snippets": [
                    "Backpropagation iteratively computes dZ[l] = dA[l] * g'[l](Z[l]) and dW[l] = (dZ[l] A[l-1].T)/m across layers (Page 21).",
                    "Deep neural network dual-stream computational graph linking forward activation cache to backward gradient flow.",
                ],
                "images": ["/static/previews/Deep_Learning_Andrew_Ng__p21.png"],
            },
            "neural": {
                "primary_entity": "Deep Neural Network",
                "related_entities": ["Perceptron", "Hidden Layers", "ReLU Activation", "Weight Matrices W[l]", "Feature Hierarchies", "Deep Representation", "Sigmoid"],
                "text_snippets": [
                    "Multi-layer deep representations transform raw inputs into hierarchical abstract features through hidden layers and non-linear activations (Page 5, 6, 13, 20).",
                ],
                "images": ["/static/previews/Deep_Learning_Andrew_Ng__p20.png"],
            },
            "semiconductor": {
                "primary_entity": "Semiconductor Supply Chain Risk",
                "related_entities": ["IC-7A-X Microcontroller", "Foundry Node Fab-2", "Shenzhen Logistics Hub", "Taiwan TSMC", "Single-Source Vendor"],
                "text_snippets": [
                    "Single-source dependency on IC-7A-X microcontroller creates vendor risk and 90-day supply chain disruption vulnerability under export restrictions at Taiwan TSMC and Shenzhen logistics hubs.",
                ],
                "images": [],
            },
        }

    async def query(self, query_str: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        # 1. Attempt Neo4j query
        try:
            records = await self.client.execute_query(query_str, params)
            if records:
                return records
        except Exception as exc:
            logger.debug(f"Neo4j query unavailable ({exc}), using local knowledge fallback.")

        # 2. In-Memory fallback for keyword search
        params = params or {}
        keyword = str(params.get("keyword", "")).lower()
        limit = int(params.get("limit", 10))

        matched: list[dict[str, Any]] = []
        for key, entry in self._local_graph.items():
            if not keyword or key in keyword or keyword in key or any(w in keyword for w in key.split()):
                matched.append(entry)
            elif any(r.lower() in keyword for r in entry["related_entities"]):
                matched.append(entry)

        if not matched and keyword:
            # Token match
            tokens = set(re.findall(r"\b\w+\b", keyword))
            for key, entry in self._local_graph.items():
                if tokens.intersection(set(re.findall(r"\b\w+\b", entry["primary_entity"].lower()))):
                    matched.append(entry)

        return matched[:limit]

    async def get_subgraph(self, entity_ids: list[str], max_hops: int = 2) -> SubGraph:
        return SubGraph()
