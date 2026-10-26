"""Smart Query Router classifying intents, selecting domain indices (MoDI), and activating retrieval channels."""

import re
from typing import Optional
from src.core.logging import logger
from src.domain.retrieval import DomainType, QueryIntent, QueryRequest, RouterDecision
from src.retrieval.domain_router import DomainClusterRouter


class QueryRouter:
    """Classifies user query intent, routes to specialized domain clusters (MoDI),

    and selects the optimal subset of retrieval channels (MoR).
    Prevents noise, cross-domain false positives, and cold unnecessary traversals.
    """

    def __init__(self, domain_router: Optional[DomainClusterRouter] = None):
        self.domain_router = domain_router or DomainClusterRouter()
        logger.info("Initialized Smart QueryRouter with Mixture of Domain Indices (MoDI) support.")

    async def route(self, request: QueryRequest) -> RouterDecision:
        query = (request.query_text or "").strip()
        query_lower = query.lower()

        # 1. Evaluate Domain Embedding Router (MoDI)
        domain_routing = self.domain_router.route_query(
            query=query,
            manual_override=request.target_domain,
        )
        primary_domain = domain_routing.primary_domain

        # 2. Visual / Chart intent detection
        if any(w in query_lower for w in ["chart", "diagram", "figure", "graph of", "visualize", "plot", "infographic", "show me image"]):
            return RouterDecision(
                intent=QueryIntent.VISUAL_COMPARATIVE,
                active_channels=["visual", "sparse", "graph"],
                channel_weights={"visual": 0.50, "sparse": 0.25, "graph": 0.15, "dense": 0.10},
                extracted_entities=self._extract_entities(query),
                domain_routing=domain_routing,
                reasoning=(
                    f"Visual artifact request in domain [{primary_domain.value}]. "
                    f"Pruned {domain_routing.search_space_pruned_pct:.1f}% noise. Prioritizing ColPali/SigLIP visual search."
                ),
            )

        # 3. Global / Thematic Community intent detection
        if any(w in query_lower for w in ["summarize", "main themes", "overview", "overall", "comprehensive report", "general summary"]):
            return RouterDecision(
                intent=QueryIntent.GLOBAL_COMMUNITY,
                active_channels=["graph", "dense"],
                channel_weights={"graph": 0.60, "dense": 0.40, "sparse": 0.0, "visual": 0.0},
                extracted_entities=self._extract_entities(query),
                domain_routing=domain_routing,
                reasoning=(
                    f"Broad thematic query in domain [{primary_domain.value}]. "
                    f"Pruned {domain_routing.search_space_pruned_pct:.1f}% noise. Prioritizing Global community reports."
                ),
            )

        # 4. Supply Chain / Corporate Entity Knowledge Graph intent detection
        if any(w in query_lower for w in ["supply chain", "semiconductor", "single-source", "single source", "vendor", "ic-7a-x", "shenzhen", "tsmc", "foundry", "board member", "ownership", "ceo"]) or primary_domain == DomainType.CORPORATE_GOVERNANCE:
            return RouterDecision(
                intent=QueryIntent.FACTUAL_ENTITY,
                active_channels=["graph", "sparse", "dense"],
                channel_weights={"graph": 0.65, "sparse": 0.20, "dense": 0.15, "visual": 0.0},
                extracted_entities=self._extract_entities(query),
                domain_routing=domain_routing,
                reasoning=(
                    f"Corporate Governance / Supply Chain traversal in domain [{primary_domain.value}]. "
                    f"Pruned {domain_routing.search_space_pruned_pct:.1f}% noise. Prioritizing Neo4j Knowledge Graph."
                ),
            )

        # 5. Analytical / Aggregation intent detection
        if any(w in query_lower for w in ["how many", "count of", "list all", "total number", "average of", "more than", "greater than"]):
            return RouterDecision(
                intent=QueryIntent.ANALYTICAL_CYPHER,
                active_channels=["graph", "sparse"],
                channel_weights={"graph": 0.65, "sparse": 0.25, "dense": 0.10, "visual": 0.0},
                extracted_entities=self._extract_entities(query),
                suggested_cypher=self._suggest_cypher(query),
                domain_routing=domain_routing,
                reasoning=(
                    f"Quantitative / Aggregation query in domain [{primary_domain.value}]. "
                    f"Pruned {domain_routing.search_space_pruned_pct:.1f}% noise. Activating Text2Cypher."
                ),
            )

        # 6. Technical Conceptual Intent (Deep Learning, Transformers, Math, Engineering)
        entities = self._extract_entities(query)
        if any(w in query_lower for w in ["transformer", "attention", "back propagation", "backprop", "backward", "neural network", "deep neural", "perceptron", "classification", "regression"]) or primary_domain == DomainType.ENGINEERING_TECHNICAL:
            return RouterDecision(
                intent=QueryIntent.FACTUAL_ENTITY,
                active_channels=["sparse", "dense", "visual", "graph"],
                channel_weights={"sparse": 0.45, "dense": 0.35, "visual": 0.10, "graph": 0.10},
                extracted_entities=entities,
                domain_routing=domain_routing,
                reasoning=(
                    f"Technical engineering concept query in domain [{primary_domain.value}]. "
                    f"Pruned {domain_routing.search_space_pruned_pct:.1f}% noise. Activating exact BM25 and dense semantic search."
                ),
            )

        # 7. General Quad-Hybrid (Fallback)
        return RouterDecision(
            intent=QueryIntent.GENERAL_HYBRID,
            active_channels=["dense", "sparse", "visual", "graph"],
            channel_weights={"dense": 0.35, "graph": 0.35, "sparse": 0.15, "visual": 0.15},
            extracted_entities=entities,
            domain_routing=domain_routing,
            reasoning=(
                f"Multi-faceted hybrid query in domain [{primary_domain.value}]. "
                f"Pruned {domain_routing.search_space_pruned_pct:.1f}% noise. Activating Quad-Hybrid channels."
            ),
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
            ("binary classification", "Binary Classification"),
            ("sentiment classification", "Sentiment Classification"),
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
