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
                channel_weights={"visual": 0.45, "graph": 0.30, "sparse": 0.15, "dense": 0.10},
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
                reasoning="Broad, thematic holistic query. Prioritizing Global Leiden/Louvain community summaries.",
            )

        # 3. Analytical / Aggregation intent detection
        if any(w in query_lower for w in ["how many", "count of", "list all", "total number", "average of", "more than", "greater than"]):
            return RouterDecision(
                intent=QueryIntent.ANALYTICAL_CYPHER,
                active_channels=["graph", "sparse"],
                channel_weights={"graph": 0.65, "sparse": 0.25, "dense": 0.10, "visual": 0.0},
                extracted_entities=self._extract_entities(query),
                suggested_cypher=self._suggest_cypher(query),
                reasoning="Quantitative or aggregation query. Activating Text2Cypher and Sparse exact matching.",
            )

        # 4. Factual / Entity-centric intent (Default high-precision mode)
        entities = self._extract_entities(query)
        if entities:
            return RouterDecision(
                intent=QueryIntent.FACTUAL_ENTITY,
                active_channels=["dense", "sparse", "graph"],
                channel_weights={"dense": 0.40, "graph": 0.40, "sparse": 0.20, "visual": 0.0},
                extracted_entities=entities,
                reasoning=f"Entity-centric factual question targeting entities: {entities}. Activating Local K-hop graph traversal.",
            )

        # 5. General Quad-Hybrid (Fallback)
        return RouterDecision(
            intent=QueryIntent.GENERAL_HYBRID,
            active_channels=["dense", "sparse", "visual", "graph"],
            channel_weights={"dense": 0.35, "graph": 0.35, "sparse": 0.15, "visual": 0.15},
            extracted_entities=entities,
            reasoning="Broad ambiguous query. Activating all 4 channels (Quad-Hybrid) for maximum recall.",
        )

    def _extract_entities(self, query: str) -> list[str]:
        """Heuristic named-entity extractor identifying capitalized terms and quoted phrases."""
        # Extract quoted phrases
        quoted = re.findall(r'"([^"]*)"', query)
        # Extract sequences of capitalized words/letters (e.g. 'Sundar Pichai', 'Cloud Services', 'Model X')
        capitalized = re.findall(r"\b[A-Z][a-zA-Z0-9_]*(?:\s+[A-Z0-9][a-zA-Z0-9_]*)*\b", query)

        # Exclude common sentence starters
        stopwords = {"What", "How", "Who", "Where", "When", "Why", "Show", "List", "Tell", "Explain", "Summarize", "Can", "Could"}
        filtered_caps = [c for c in capitalized if c not in stopwords]

        candidates = list(set(quoted + filtered_caps))
        return candidates

    def _suggest_cypher(self, query: str) -> str | None:
        """Template heuristic for common analytical queries."""
        if "how many" in query.lower():
            return "MATCH (n) RETURN labels(n)[0] AS type, count(n) AS total"
        return None
