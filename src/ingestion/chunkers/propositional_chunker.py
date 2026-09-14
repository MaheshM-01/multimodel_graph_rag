"""Propositional Chunker (Dense X / Atomic Fact Chunking for zero-hallucination GraphRAG triple extraction)."""

import re
from src.core.logging import logger
from src.domain.multimodal import Document, PropositionChunk, TextChunk


class PropositionalChunker:
    """Splits complex sentences into atomic factual propositions (Dense X).

    Each proposition contains a single assertion that maps directly to:
    (Subject) -[PREDICATE]-> (Object) in the Knowledge Graph.
    """

    def __init__(self, use_llm: bool = False):
        self.use_llm = use_llm

    async def extract_propositions(self, document: Document, chunk: TextChunk) -> list[PropositionChunk]:
        """Deconstructs text chunk into atomic proposition statements."""
        logger.debug(f"Extracting atomic propositions from chunk {chunk.id}")

        sentences = re.split(r"(?<=[.!?])\s+", chunk.content)
        propositions: list[PropositionChunk] = []

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue

            # Check for relative clauses: e.g. "Subject, who did X, did Y"
            rel_match = re.match(r"^([^,]+),\s*who\s+([^,]+),\s*(.+)$", sent, re.IGNORECASE)
            if rel_match:
                subject = rel_match.group(1).strip()
                rel_action = rel_match.group(2).strip()
                main_action = rel_match.group(3).strip().rstrip(".")

                prop1 = f"{subject} {rel_action}."
                prop2 = f"{subject} {main_action}."

                for p_str in [prop1, prop2]:
                    propositions.append(
                        PropositionChunk(
                            document_id=document.id,
                            chunk_index=len(propositions),
                            source_chunk_id=chunk.id,
                            proposition_statement=p_str,
                            subject=subject,
                        )
                    )
                continue

            # Fallback: split on conjunctions and clauses
            clauses = re.split(r"(?:,\s*(?:and|after|before|while)\s+|\s+(?:and|after|before)\s+)", sent)
            for clause in clauses:
                clean_clause = clause.strip().rstrip(".")
                if len(clean_clause.split()) >= 3:
                    propositions.append(
                        PropositionChunk(
                            document_id=document.id,
                            chunk_index=len(propositions),
                            source_chunk_id=chunk.id,
                            proposition_statement=clean_clause + ".",
                        )
                    )

            if not propositions:
                propositions.append(
                    PropositionChunk(
                        document_id=document.id,
                        chunk_index=0,
                        source_chunk_id=chunk.id,
                        proposition_statement=sent,
                    )
                )

        return propositions
