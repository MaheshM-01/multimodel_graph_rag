"""Smart Query Router classifying intents and dynamically activating optimal retrieval channels."""

import re
from src.core.logging import logger
from src.domain.retrieval import QueryIntent, QueryRequest, RouterDecision


class QueryRouter:
    """Classifies user query intent and selects the optimal subset of the 4 retrieval channels.

    Prevents cold, unnecessary traversals and dynamically balances latency vs recall.
    """

    def __init__(self):
        logger.info("Initialized Smart QueryRouter")

    async def route(self, request: QueryRequest) -> RouterDecision:
        query = (request.query_text or "").strip()
        query_lower = query.lower()

        # 1. Visual / Chart intent detection
        if any(w in query_lower for w in ["chart", "diagram", "figure", "graph of", "visualize", "plot", "infographic", "show me image"]):
            return RouterDecision(
                intent=QueryIntent.VISUAL_COMPARATIVE,
                active_channels=["visual", "sparse", "graph"],
                channel_weights={"visual": 0.50, "sparse": 0.25, "graph": 0.15, "dense": 0.10},
                extracted_entities=self._extract_entities(query),
                reasoning="Query requests visual artifacts (charts/diagrams). Prioritizing ColPali/SigLIP visual search.",
            )

        # 2. Global / Thematic Community intent detection
        if any(w in query_lower for w in ["summarize", "main themes", "overview", "overall", "comprehensive report", "general summary"]):
            return RouterDecision(
                intent=QueryIntent.GLOBAL_COMMUNITY,
                active_channels=["graph", "dense"],
                channel_weights={"graph": 0.60, "dense": 0.40, "sparse": 0.0, "visual": 0.0},
                extracted_entities=self._extract_entities(query),
                reasoning="Broad, thematic holistic query. Prioritizing Global community summaries.",
            )

        # 3. Supply Chain / Enterprise Entity Knowledge Graph intent detection
        if any(w in query_lower for w in ["supply chain", "semiconductor", "single-source", "single source", "vendor", "ic-7a-x", "shenzhen", "tsmc", "foundry", "board member", "ownership", "ceo"]):
            return RouterDecision(
                intent=QueryIntent.FACTUAL_ENTITY,
                active_channels=["graph", "sparse", "dense"],
                channel_weights={"graph": 0.65, "sparse": 0.20, "dense": 0.15, "visual": 0.0},
                extracted_entities=self._extract_entities(query),
                reasoning="Enterprise or Supply Chain entity query. Prioritizing Knowledge Graph traversal.",
            )

        # 4. Analytical / Aggregation intent detection
        if any(w in query_lower for w in ["how many", "count of", "list all", "total number", "average of", "more than", "greater than"]):
            return RouterDecision(
                intent=QueryIntent.ANALYTICAL_CYPHER,
                active_channels=["graph", "sparse"],
                channel_weights={"graph": 0.65, "sparse": 0.25, "dense": 0.10, "visual": 0.0},
                extracted_entities=self._extract_entities(query),
                suggested_cypher=self._suggest_cypher(query),
                reasoning="Quantitative or aggregation query. Activating Text2Cypher and Sparse exact matching.",
            )

        # 5. Technical Conceptual Intent (Deep Learning, Transformers, Math, Foundations)
        entities = self._extract_entities(query)
        if any(w in query_lower for w in ["transformer", "attention", "back propagation", "backprop", "backward", "neural network", "deep neural", "perceptron"]):
            return RouterDecision(
                intent=QueryIntent.FACTUAL_ENTITY,
                active_channels=["sparse", "dense", "visual", "graph"],
                channel_weights={"sparse": 0.45, "dense": 0.35, "visual": 0.10, "graph": 0.10},
                extracted_entities=entities,
                reasoning="Core technical ML concept query. Activating high-precision BM25 and dense semantic search.",
            )

        # 6. General Quad-Hybrid (Fallback)
        return RouterDecision(
            intent=QueryIntent.GENERAL_HYBRID,
            active_channels=["dense", "sparse", "visual", "graph"],
            channel_weights={"dense": 0.35, "graph": 0.35, "sparse": 0.15, "visual": 0.15},
            extracted_entities=entities,
            reasoning="Broad ambiguous query. Activating all 4 channels (Quad-Hybrid) for maximum recall.",
        )

    def _extract_entities(self, query: str) -> list[str]:
        """Named-entity and domain keyword extractor."""
        quoted = re.findall(r'"([^"]*)"', query)
        capitalized = re.findall(r"\b[A-Z][a-zA-Z0-9_]*(?:\s+[A-Z0-9][a-zA-Z0-9_]*)*\b", query)

        stopwords = {"What", "How", "Who", "Where", "When", "Why", "Show", "List", "Tell", "Explain", "Summarize", "Can", "Could"}
        filtered_caps = [c for c in capitalized if c not in stopwords]

        # Domain entities
        domain_matches = []
        q_lower = query.lower()
        domain_patterns = [
            ("semiconductor", "Semiconductor"),
            ("supply chain", "Supply Chain"),
            ("ic-7a-x", "IC-7A-X"),
            ("transformer", "Transformer"),
            ("attention", "Attention Mechanism"),
            ("back propagation", "Backpropagation"),
            ("backprop", "Backpropagation"),
            ("neural network", "Neural Network"),
        ]
        for pat, ent in domain_patterns:
            if pat in q_lower:
                domain_matches.append(ent)

        candidates = list(set(quoted + filtered_caps + domain_matches))
        return candidates

    def _suggest_cypher(self, query: str) -> str | None:
        """Template heuristic for common analytical queries."""
        if "how many" in query.lower():
            return "MATCH (n) RETURN labels(n)[0] AS type, count(n) AS total"
        return None
