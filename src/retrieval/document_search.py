"""Knowledge Base Document Search and Ingestion Engine.
Extracts, indexes, and retrieves text, pages, and visual diagram evidence from media store documents.
"""

import math
import mimetypes
import re
from pathlib import Path
from collections import defaultdict
from src.core.constants import ModalityType
from src.core.logging import logger
from src.domain.retrieval import SearchResult

STORAGE_DIR = Path("./data/media_store")


class DocumentSearchEngine:
    """In-memory BM25 and multi-modal page search over local Knowledge Base documents."""

    def __init__(self, storage_dir: Path | None = None):
        self.storage_dir = storage_dir or STORAGE_DIR
        self._page_cache: dict[str, list[dict]] = {}
        self._doc_mtimes: dict[str, float] = {}

    def _load_document_pages(self, file_path: Path) -> list[dict]:
        """Extract pages, text snippets, and visual image availability from a document."""
        filename = file_path.name
        mtime = file_path.stat().st_mtime

        # Return cached pages if file has not changed
        if filename in self._page_cache and self._doc_mtimes.get(filename) == mtime:
            return self._page_cache[filename]

        pages: list[dict] = []
        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            try:
                import fitz
                pdf_doc = fitz.open(str(file_path))
                for page_idx in range(len(pdf_doc)):
                    page = pdf_doc[page_idx]
                    text = page.get_text().strip()
                    imgs = page.get_images()
                    has_images = len(imgs) > 0

                    if text or has_images:
                        # Normalize whitespace and strip social/disclaimer boilerplate
                        clean_text = re.sub(r"[ \t]+", " ", text)
                        clean_text = re.sub(r"\n{3,}", "\n\n", clean_text)
                        clean_text = re.sub(r"Follow Arpit Singh on LinkedIn[^\n]*\n?", "", clean_text, flags=re.IGNORECASE)
                        clean_text = re.sub(r"This PDF contains Mahmoud Badry[^\n]*\n?", "", clean_text, flags=re.IGNORECASE)
                        clean_text = clean_text.strip()
                        
                        pages.append({
                            "document_id": filename,
                            "page_number": page_idx + 1,
                            "text": clean_text or f"[Visual Diagram Page {page_idx + 1}]",
                            "has_images": has_images,
                            "image_count": len(imgs),
                            "preview_url": f"/api/v1/documents/{filename}/pages/{page_idx + 1}/preview",
                        })
                pdf_doc.close()
                logger.info(f"Parsed {len(pages)} pages from PDF: {filename}")
            except Exception as e:
                logger.warning(f"Error parsing PDF {filename} with fitz: {e}")

        elif suffix in [".png", ".jpg", ".jpeg", ".webp"]:
            pages.append({
                "document_id": filename,
                "page_number": 1,
                "text": f"Visual asset: {filename}. Multimodal image and diagram context.",
                "has_images": True,
                "image_count": 1,
                "preview_url": f"/api/v1/documents/{filename}/pages/1/preview",
            })

        elif suffix in [".txt", ".md", ".json"]:
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                # Split into page-like sections of ~1500 chars
                lines = content.split("\n")
                curr_chunk = []
                curr_len = 0
                page_idx = 1
                for line in lines:
                    curr_chunk.append(line)
                    curr_len += len(line)
                    if curr_len > 1200:
                        pages.append({
                            "document_id": filename,
                            "page_number": page_idx,
                            "text": "\n".join(curr_chunk),
                            "has_images": False,
                            "image_count": 0,
                            "preview_url": f"/api/v1/documents/{filename}/view",
                        })
                        curr_chunk = []
                        curr_len = 0
                        page_idx += 1
                if curr_chunk:
                    pages.append({
                        "document_id": filename,
                        "page_number": page_idx,
                        "text": "\n".join(curr_chunk),
                        "has_images": False,
                        "image_count": 0,
                        "preview_url": f"/api/v1/documents/{filename}/view",
                    })
            except Exception as e:
                logger.warning(f"Error reading text file {filename}: {e}")

        self._page_cache[filename] = pages
        self._doc_mtimes[filename] = mtime
        return pages

    def _ensure_documents_indexed(self, target_doc: str | None = None) -> list[dict]:
        """Scan media store directory and index relevant documents."""
        all_pages: list[dict] = []
        if not self.storage_dir.exists():
            return all_pages

        if target_doc:
            target_path = self.storage_dir / target_doc
            if target_path.exists() and target_path.is_file():
                return self._load_document_pages(target_path)

        for file_path in sorted(self.storage_dir.glob("*")):
            if file_path.is_file() and not file_path.name.startswith("."):
                all_pages.extend(self._load_document_pages(file_path))

        return all_pages

    def _compute_idf(self, pages: list[dict]) -> tuple[dict[str, float], float]:
        """Compute Inverse Document Frequency for vocabulary across all indexed pages."""
        total_docs = len(pages)
        if total_docs == 0:
            return {}, 1.0

        doc_freq = defaultdict(int)
        total_len = 0
        for page in pages:
            tokens = set(re.findall(r"\b\w+\b", page["text"].lower()))
            total_len += len(page["text"].split())
            for token in tokens:
                doc_freq[token] += 1

        avg_dl = max(1.0, total_len / total_docs)
        idf = {}
        for token, count in doc_freq.items():
            # Standard BM25 IDF formula with smoothing
            idf[token] = math.log((total_docs - count + 0.5) / (count + 0.5) + 1.0)

        return idf, avg_dl

    async def search(
        self,
        query: str,
        document_name: str | None = None,
        top_k: int = 6,
    ) -> list[SearchResult]:
        """Execute high-precision BM25 and semantic-expansion search over document pages."""
        pages = self._ensure_documents_indexed(target_doc=document_name)
        if not pages:
            return []

        clean_query = query.lower()
        query_words = re.findall(r"\b\w+\b", clean_query)
        stop_words = {
            "machi", "ta", "the", "a", "an", "and", "or", "in", "on", "at", "of", "to",
            "is", "are", "by", "for", "with", "what", "how", "why", "does", "do", "can",
            "you", "tell", "me", "give", "explain", "describe", "about", "show", "details",
            "mean", "meant", "meaning", "define", "definition"
        }
        base_keywords = [w for w in query_words if w not in stop_words and len(w) > 1]
        wants_visuals = any(w in clean_query for w in ["visual", "visuals", "diagram", "diagrams", "image", "chart", "picture", "figure", "draw", "schematic"])

        if not base_keywords:
            base_keywords = [w for w in query_words if len(w) > 1]

        # Domain query expansions for high-precision retrieval
        expansion_map = {
            "transformer": ["attention", "self-attention", "sequence", "encoder", "decoder", "weights"],
            "transformers": ["attention", "self-attention", "sequence", "encoder", "decoder", "weights"],
            "attention": ["attention model", "attention weights", "context", "sequence-to-sequence", "encoder"],
            "cnn": ["convolutional", "convolution", "pooling", "stride", "filter", "kernel"],
            "rnn": ["recurrent", "lstm", "gru", "sequence", "hidden state"],
            "backprop": ["backpropagation", "gradient", "derivatives", "chain rule", "backward"],
            "backpropagation": ["gradient descent", "derivatives", "chain rule", "backward propagation", "dz", "dw"],
            "propagation": ["backward", "backpropagation", "backward propagation", "gradient", "derivatives", "chain rule", "dz", "dw"],
            "back": ["backward", "backpropagation", "backward propagation", "gradient"],
            "neural": ["artificial neural network", "deep neural network", "hidden layer", "perceptron", "neurons", "layers"],
            "network": ["neural network", "deep neural network", "hidden layer", "architecture"],
            "networks": ["neural networks", "deep neural networks", "hidden layers", "architecture"],
            "regularization": ["dropout", "l2", "weight decay", "overfitting"],
            "adam": ["rmsprop", "momentum", "optimizer", "exponentially weighted"],
            "loss": ["cost function", "cross-entropy", "log loss"],
            "activation": ["relu", "sigmoid", "tanh", "softmax", "leaky relu"],
        }

        expanded_terms: list[str] = []
        for kw in base_keywords:
            if kw in expansion_map:
                expanded_terms.extend(expansion_map[kw])

        # Precompute corpus-level IDF
        idf_table, avg_dl = self._compute_idf(pages)

        # Build bigrams and key phrases
        phrases = []
        for i in range(len(query_words) - 1):
            pair = f"{query_words[i]} {query_words[i+1]}"
            if not any(sw in pair for sw in ["machi", "ta", "the", "a"]):
                phrases.append(pair)

        # Add expanded domain phrases
        if any(w in clean_query for w in ["transformer", "attention"]):
            phrases.extend(["attention model", "attention mechanism", "sequence models"])
        if ("back" in clean_query and "propagat" in clean_query) or "backprop" in clean_query:
            phrases.extend(["back propagation", "backward propagation", "forward and backward propagation", "backward function", "vectorized backpropagation"])
        if "neural" in clean_query or "network" in clean_query:
            phrases.extend(["neural network", "neural networks", "deep neural network", "deep neural networks", "hidden layers", "what is a neural network", "neural networks overview", "supervised learning with neural networks"])

        k1 = 1.5
        b = 0.75
        scored_pages = []

        for page in pages:
            text_lower = page["text"].lower()
            tokens = re.findall(r"\b\w+\b", text_lower)
            doc_len = max(1, len(tokens))
            term_counts = defaultdict(int)
            for t in tokens:
                term_counts[t] += 1

            score = 0.0

            # 1. BM25 scoring for core query keywords (with 2.5x base weighting)
            for kw in base_keywords:
                count = term_counts.get(kw, 0)
                if count > 0:
                    kw_idf = idf_table.get(kw, 1.0)
                    tf = (count * (k1 + 1.0)) / (count + k1 * (1.0 - b + b * (doc_len / avg_dl)))
                    score += kw_idf * tf * 2.5

            # 2. BM25 scoring for domain expanded synonyms (0.8x weighting)
            for exp in expanded_terms:
                exp_words = exp.split()
                if len(exp_words) == 1:
                    count = term_counts.get(exp, 0)
                    if count > 0:
                        exp_idf = idf_table.get(exp, 0.8)
                        tf = (count * (k1 + 1.0)) / (count + k1 * (1.0 - b + b * (doc_len / avg_dl)))
                        score += exp_idf * tf * 0.8
                else:
                    if exp in text_lower:
                        score += 4.0

            # 3. Exact phrase match boost
            for phrase in phrases:
                if phrase in text_lower:
                    score += 6.0

            # 4. Target document preference
            if document_name and page["document_id"] == document_name:
                score += 5.0

            # 5. Visual bonus if user specifically requested diagrams/visuals
            if wants_visuals and page["has_images"]:
                score += 4.0

            # 6. Table of contents and title/cover penalty (content pages should outrank table of contents)
            is_toc = any(toc in text_lower for toc in ["table of contents", "notes on deeplearning.ai", "personal notes and summaries", "course summary"])
            if is_toc and page.get("page_number", 0) <= 2:
                score -= 25.0

            if score > 0:
                scored_pages.append((score, page))

        # Sort strictly by score descending
        scored_pages.sort(key=lambda x: x[0], reverse=True)

        results: list[SearchResult] = []
        seen_pages: set[tuple[str, int]] = set()

        for score, page in scored_pages:
            doc_id = page["document_id"]
            page_num = page["page_number"]
            if (doc_id, page_num) in seen_pages:
                continue
            seen_pages.add((doc_id, page_num))

            # If user wanted visuals or page has images, mark as IMAGE modality to trigger multimodal rendering
            is_visual = page["has_images"] and (wants_visuals or page.get("image_count", 0) > 0)
            modality = ModalityType.IMAGE if is_visual else ModalityType.TEXT

            results.append(
                SearchResult(
                    id=f"{doc_id}_p{page_num}",
                    modality=modality,
                    score=round(float(score), 4),
                    content=page["text"],
                    image_url=page["preview_url"] if is_visual else None,
                    source_type="document_page",
                    data_points={"page": page_num, "image_count": page.get("image_count", 0)},
                    metadata={
                        "document_id": doc_id,
                        "page_number": page_num,
                        "has_visuals": page["has_images"],
                        "preview_url": page["preview_url"],
                    },
                )
            )

            if len(results) >= top_k:
                break

        return results


_doc_search_engine: DocumentSearchEngine | None = None


def get_document_search_engine() -> DocumentSearchEngine:
    global _doc_search_engine
    if _doc_search_engine is None:
        _doc_search_engine = DocumentSearchEngine()
    return _doc_search_engine
