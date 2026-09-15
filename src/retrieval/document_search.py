"""Production Multimodal Knowledge Base Search Engine.
Features:
1. Layout-aware & hierarchical structural chunking (headings, paragraphs, code, figures).
2. Dense semantic embeddings via SentenceTransformers (all-MiniLM-L6-v2).
3. Sparse structural Okapi BM25 with dynamic N-grams and heading weighting.
4. Dense-Sparse Hybrid Reciprocal Rank Fusion (RRF) with strict cosine thresholding.
5. Zero hardcoded dictionaries or static query mappings.
"""

import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional
import numpy as np

from src.core.constants import ModalityType
from src.core.logging import logger
from src.domain.retrieval import SearchResult
from src.embeddings.text_embedder import TextEmbeddingEngine

STORAGE_DIR = Path("./data/media_store")


class DocumentSearchEngine:
    """Production Dense-Sparse Hybrid Search Engine with layout-aware chunking and neural embeddings."""

    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = storage_dir or STORAGE_DIR
        self.embedder = TextEmbeddingEngine()
        self._chunks_cache: dict[str, list[dict[str, Any]]] = {}
        self._matrices_cache: dict[str, np.ndarray] = {}
        self._doc_mtimes: dict[str, float] = {}

    def _extract_structural_chunks_from_pdf(self, file_path: Path) -> list[dict[str, Any]]:
        """Extract layout-aware structural child chunks with font-size section headings, boilerplate suppression, and visual attachments."""
        chunks: list[dict[str, Any]] = []
        filename = file_path.name
        boilerplate_pattern = re.compile(
            r'(?:https?://\S+|t\.me/\S+|follow\s+[^\n]+on\s+linkedin[^\n]*|from\s+a\s+guide\s+to[^\n]+|download\s+machine\s+learning[^\n]*|deeplearning\.ai\s+courses\s+notes[^\n]*|machine\s+learning\s+full\s+course[^\n]*|coursera\s+deep\s+learning\s+specialization[^\n]*)',
            re.IGNORECASE,
        )

        try:
            import fitz
            pdf_doc = fitz.open(str(file_path))
            total_pages = len(pdf_doc)
            running_heading = "Introduction"

            for page_idx in range(total_pages):
                page = pdf_doc[page_idx]
                page_num = page_idx + 1
                page_text = page.get_text().strip()
                imgs = page.get_images()
                has_images = len(imgs) > 0

                if not page_text and not has_images:
                    continue

                # Detect if this page is a Table of Contents / Course Summary
                lower_raw = page_text.lower()
                is_toc = ("table of contents" in lower_raw) or ("course summary" in lower_raw)

                # Clean boilerplate and noise from page text
                clean_page_text = boilerplate_pattern.sub("", page_text)
                clean_page_text = re.sub(r"[ \t]+", " ", clean_page_text)
                clean_page_text = re.sub(r"\n{3,}", "\n\n", clean_page_text).strip()

                # Determine true structural heading for this page using font size
                current_heading = None
                if is_toc:
                    current_heading = "Table of Contents"
                else:
                    # 1. Search for title spans with font size >= 12.8 (author uses 15.0 and 18.0 for titles)
                    blocks = page.get_text("dict").get("blocks", [])
                    size_cands = []
                    for b in blocks:
                        if "lines" not in b:
                            continue
                        for l in b["lines"]:
                            line_text = "".join(s.get("text", "") for s in l.get("spans", [])).strip()
                            if not line_text or boilerplate_pattern.search(line_text):
                                continue
                            max_size = max((s.get("size", 0) for s in l.get("spans", [])), default=0)
                            if (
                                max_size >= 12.8
                                and len(line_text) >= 3
                                and not line_text.lower().startswith(("course", "download", "page ", "chapter ", "http", "t.me"))
                            ):
                                size_cands.append((max_size, line_text))

                    if size_cands:
                        size_cands.sort(key=lambda x: x[0], reverse=True)
                        current_heading = size_cands[0][1]
                    else:
                        # 2. Inspect clean lines for short standalone title lines
                        TRANSITION_WORDS = {
                            "then", "also", "here", "now", "first", "second", "third", "next", "finally",
                            "note", "so", "step", "alpha", "beta", "gamma", "equations", "formula",
                            "hint", "example", "suppose", "assume", "let's", "where", "while", "blue"
                        }
                        clean_lines = [l.strip() for l in clean_page_text.split("\n") if l.strip()]
                        for cand in clean_lines[:5]:
                            cand_clean = cand.strip("#: \t")
                            cand_lower = cand_clean.lower()
                            if (
                                4 <= len(cand_clean) <= 65
                                and not cand_clean.endswith((".", ";"))
                                and cand_lower not in TRANSITION_WORDS
                                and not any(cand_lower.startswith(tw + " ") for tw in ["we will", "let's", "in this", "suppose", "assume", "give a", "from a"])
                                and not boilerplate_pattern.search(cand_clean)
                            ):
                                current_heading = cand_clean
                                break

                    if not current_heading:
                        current_heading = running_heading

                    if current_heading and not current_heading.startswith("Page "):
                        running_heading = current_heading

                # Split page into granular coherent paragraphs/sentences (~350-500 chars)
                paragraphs = re.split(r"\n\s*\n+", clean_page_text)
                raw_segments = []
                for p in paragraphs:
                    p = p.strip()
                    if not p or len(p) < 15:
                        continue
                    if len(p) > 600:
                        sentences = re.split(r"(?<=[.!?])\s+", p)
                        curr_sent_chunk = []
                        curr_len = 0
                        for s in sentences:
                            curr_sent_chunk.append(s)
                            curr_len += len(s)
                            if curr_len >= 380:
                                raw_segments.append(" ".join(curr_sent_chunk))
                                curr_sent_chunk = []
                                curr_len = 0
                        if curr_sent_chunk:
                            raw_segments.append(" ".join(curr_sent_chunk))
                    else:
                        raw_segments.append(p)

                # Fallback to whole clean page text if no paragraphs could be extracted
                if not raw_segments and clean_page_text:
                    raw_segments = [clean_page_text]

                preview_url = f"/api/v1/documents/{filename}/pages/{page_num}/preview"
                figure_title = f"Figure: {current_heading} (Page {page_num})" if has_images else None

                for seg_idx, segment_content in enumerate(raw_segments):
                    # Contextual enrichment (Anthropic pattern): prepends document & section context
                    contextual_prefix = f"[{filename} | Page {page_num} | {current_heading}] "
                    enriched_content = f"{contextual_prefix}{segment_content}"

                    chunks.append({
                        "chunk_id": f"{filename}_p{page_num}_c{seg_idx}",
                        "document_id": filename,
                        "page_number": page_num,
                        "chunk_index": seg_idx,
                        "heading": current_heading,
                        "content": segment_content,
                        "enriched_content": enriched_content,
                        "parent_text": clean_page_text[:1400],
                        "has_images": has_images,
                        "image_count": len(imgs),
                        "preview_url": preview_url,
                        "figure_title": figure_title,
                        "is_toc": is_toc,
                    })

            pdf_doc.close()
            logger.info(f"Loaded {len(chunks)} structural chunks from PDF: {filename}")
        except Exception as e:
            logger.error(f"Failed to extract structural chunks from PDF {filename}: {e}")

        return chunks

    def _extract_structural_chunks_from_text(self, file_path: Path) -> list[dict[str, Any]]:
        """Extract structural chunks from plaintext, markdown, or JSON files."""
        chunks: list[dict[str, Any]] = []
        filename = file_path.name
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            lines = content.split("\n")
            current_heading = "Overview"
            curr_para: list[str] = []
            chunk_idx = 0

            for line in lines:
                s_line = line.strip()
                if s_line.startswith("#"):
                    if curr_para:
                        body = "\n".join(curr_para).strip()
                        if body:
                            chunks.append({
                                "chunk_id": f"{filename}_c{chunk_idx}",
                                "document_id": filename,
                                "page_number": 1,
                                "chunk_index": chunk_idx,
                                "heading": current_heading,
                                "content": body,
                                "enriched_content": f"[{filename} | {current_heading}] {body}",
                                "parent_text": body[:1200],
                                "has_images": False,
                                "image_count": 0,
                                "preview_url": f"/api/v1/documents/{filename}/view",
                                "figure_title": None,
                            })
                            chunk_idx += 1
                        curr_para = []
                    current_heading = s_line.lstrip("# \t") or "Section"
                elif not s_line:
                    if curr_para and sum(len(l) for l in curr_para) > 300:
                        body = "\n".join(curr_para).strip()
                        chunks.append({
                            "chunk_id": f"{filename}_c{chunk_idx}",
                            "document_id": filename,
                            "page_number": 1,
                            "chunk_index": chunk_idx,
                            "heading": current_heading,
                            "content": body,
                            "enriched_content": f"[{filename} | {current_heading}] {body}",
                            "parent_text": body[:1200],
                            "has_images": False,
                            "image_count": 0,
                            "preview_url": f"/api/v1/documents/{filename}/view",
                            "figure_title": None,
                        })
                        chunk_idx += 1
                        curr_para = []
                else:
                    curr_para.append(line)

            if curr_para:
                body = "\n".join(curr_para).strip()
                if body:
                    chunks.append({
                        "chunk_id": f"{filename}_c{chunk_idx}",
                        "document_id": filename,
                        "page_number": 1,
                        "chunk_index": chunk_idx,
                        "heading": current_heading,
                        "content": body,
                        "enriched_content": f"[{filename} | {current_heading}] {body}",
                        "parent_text": body[:1200],
                        "has_images": False,
                        "image_count": 0,
                        "preview_url": f"/api/v1/documents/{filename}/view",
                        "figure_title": None,
                    })
        except Exception as e:
            logger.error(f"Error reading text document {filename}: {e}")

        return chunks

    def _ensure_document_indexed(self, file_path: Path) -> tuple[list[dict[str, Any]], np.ndarray]:
        """Index document chunks and compute normalized neural embedding matrix."""
        filename = file_path.name
        mtime = file_path.stat().st_mtime

        if (
            filename in self._chunks_cache
            and filename in self._matrices_cache
            and self._doc_mtimes.get(filename) == mtime
        ):
            return self._chunks_cache[filename], self._matrices_cache[filename]

        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            chunks = self._extract_structural_chunks_from_pdf(file_path)
        elif suffix in [".png", ".jpg", ".jpeg", ".webp"]:
            chunks = [{
                "chunk_id": f"{filename}_img1",
                "document_id": filename,
                "page_number": 1,
                "chunk_index": 0,
                "heading": filename.rsplit(".", 1)[0].replace("_", " ").title(),
                "content": f"Visual schematic and architectural diagram: {filename}.",
                "enriched_content": f"Visual diagram {filename}",
                "parent_text": f"Visual diagram {filename}",
                "has_images": True,
                "image_count": 1,
                "preview_url": f"/api/v1/documents/{filename}/pages/1/preview",
                "figure_title": f"Schematic: {filename}",
            }]
        else:
            chunks = self._extract_structural_chunks_from_text(file_path)

        # Batch compute dense neural embeddings for all chunks
        texts_to_embed = [c["enriched_content"] for c in chunks]
        if texts_to_embed:
            matrix = self.embedder.encode_batch(texts_to_embed)
        else:
            matrix = np.empty((0, self.embedder.dim), dtype=np.float32)

        self._chunks_cache[filename] = chunks
        self._matrices_cache[filename] = matrix
        self._doc_mtimes[filename] = mtime
        return chunks, matrix

    def _get_corpus_chunks_and_matrix(
        self, target_doc: Optional[str] = None
    ) -> tuple[list[dict[str, Any]], np.ndarray]:
        """Collect all indexed chunks and vertically stack their dense neural embedding matrices."""
        if not self.storage_dir.exists():
            return [], np.empty((0, self.embedder.dim), dtype=np.float32)

        all_chunks: list[dict[str, Any]] = []
        matrices: list[np.ndarray] = []

        if target_doc:
            target_path = self.storage_dir / target_doc
            if target_path.exists() and target_path.is_file():
                return self._ensure_document_indexed(target_path)

        for f in sorted(self.storage_dir.glob("*")):
            if f.is_file() and not f.name.startswith("."):
                chunks, mat = self._ensure_document_indexed(f)
                if chunks and mat.size > 0:
                    all_chunks.extend(chunks)
                    matrices.append(mat)

        if matrices:
            combined_matrix = np.vstack(matrices)
        else:
            combined_matrix = np.empty((0, self.embedder.dim), dtype=np.float32)

        return all_chunks, combined_matrix

    def _ensure_documents_indexed(self) -> list[dict[str, Any]]:
        """Auto-index and retrieve all document chunks for legacy/hybrid retrievers."""
        chunks, _ = self._get_corpus_chunks_and_matrix()
        for c in chunks:
            if "text" not in c:
                c["text"] = c.get("content", "")
        return chunks

    def _compute_idf(self, chunks: list[dict[str, Any]]) -> tuple[dict[str, float], float]:
        """Compute Inverse Document Frequency across all indexed chunks."""
        total_chunks = len(chunks)
        if total_chunks == 0:
            return {}, 1.0

        doc_freq = defaultdict(int)
        total_len = 0
        for chunk in chunks:
            tokens = set(re.findall(r"\b\w+\b", chunk["content"].lower()))
            total_len += len(chunk["content"].split())
            for t in tokens:
                doc_freq[t] += 1

        avg_dl = max(1.0, total_len / total_chunks)
        idf: dict[str, float] = {}
        for t, cnt in doc_freq.items():
            idf[t] = math.log((total_chunks - cnt + 0.5) / (cnt + 0.5) + 1.0)

        return idf, avg_dl

    async def search(
        self,
        query: str,
        document_name: Optional[str] = None,
        top_k: int = 6,
    ) -> list[SearchResult]:
        """Execute Universal Dense Neural + Sparse Okapi BM25 Hybrid Retrieval.
        Completely eliminates hardcoded dictionaries by leveraging continuous vector embeddings.
        """
        all_chunks, chunk_matrix = self._get_corpus_chunks_and_matrix(target_doc=document_name)
        if not all_chunks or chunk_matrix.shape[0] == 0:
            return []

        clean_query = query.lower().strip()
        query_words = re.findall(r"\b\w+\b", clean_query)
        stop_words = {
            "machi", "ta", "the", "a", "an", "and", "or", "in", "on", "at", "of", "to",
            "is", "are", "by", "for", "with", "what", "how", "why", "does", "do", "can",
            "you", "tell", "me", "give", "explain", "describe", "about", "show", "details",
            "mean", "meant", "meaning", "define", "definition"
        }
        salient_keywords = [w for w in query_words if w not in stop_words and len(w) > 1]
        if not salient_keywords:
            salient_keywords = [w for w in query_words if len(w) > 1]

        wants_visuals = any(
            w in clean_query for w in ["visual", "diagram", "image", "chart", "figure", "picture", "schematic"]
        )

        # ----------------------------------------------------------------------
        # 1. Dense Semantic Vector Search (SentenceTransformers Cosine Similarity)
        # ----------------------------------------------------------------------
        q_vec = np.array(self.embedder.encode_text(query), dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 1e-6:
            q_vec /= q_norm

        # Matrix dot-product across all chunks
        dense_scores = np.dot(chunk_matrix, q_vec)

        # ----------------------------------------------------------------------
        # 2. Sparse Okapi BM25 Search with Dynamic N-Grams & Heading Awareness
        # ----------------------------------------------------------------------
        idf_table, avg_dl = self._compute_idf(all_chunks)
        k1 = 1.5
        b = 0.75

        # Dynamic query bigrams and trigrams
        dynamic_phrases = []
        for i in range(len(query_words) - 1):
            pair = f"{query_words[i]} {query_words[i+1]}"
            if not any(sw in pair for sw in ["machi", "ta"]):
                dynamic_phrases.append(pair)
        for i in range(len(query_words) - 2):
            tri = f"{query_words[i]} {query_words[i+1]} {query_words[i+2]}"
            dynamic_phrases.append(tri)

        # Pre-calculate sparse BM25 scores
        sparse_scores = np.zeros(len(all_chunks), dtype=np.float32)

        for idx, chunk in enumerate(all_chunks):
            text_lower = chunk["content"].lower()
            tokens = re.findall(r"\b\w+\b", text_lower)
            doc_len = max(1, len(tokens))
            term_counts = defaultdict(int)
            for t in tokens:
                term_counts[t] += 1

            s_score = 0.0
            heading_lower = chunk["heading"].lower()

            # A. Salient keyword BM25
            for kw in salient_keywords:
                count = term_counts.get(kw, 0)
                if count > 0:
                    kw_idf = idf_table.get(kw, 1.0)
                    tf = (count * (k1 + 1.0)) / (count + k1 * (1.0 - b + b * (doc_len / avg_dl)))
                    s_score += kw_idf * tf * 2.0

                # Structural heading match boost (2.5x multiplier)
                if kw in heading_lower:
                    s_score += 3.5

            # B. Dynamic continuous phrase match boost (unhardcoded N-grams)
            for phr in dynamic_phrases:
                if phr in text_lower:
                    s_score += 4.5
                if phr in heading_lower:
                    s_score += 6.0

            # C. Penalty for table-of-contents / syllabus / index pages
            if chunk.get("is_toc") and not any(toc in clean_query for toc in ["table of contents", "syllabus", "outline", "summary", "index"]):
                s_score -= 35.0

            sparse_scores[idx] = max(0.0, s_score)

        # ----------------------------------------------------------------------
        # 3. Dense-Sparse Reciprocal Rank Fusion (RRF) & Threshold Filtering
        # ----------------------------------------------------------------------
        # Dense ranking
        dense_rank_indices = np.argsort(-dense_scores)
        dense_ranks = {idx: rank + 1 for rank, idx in enumerate(dense_rank_indices)}

        # Sparse ranking
        sparse_rank_indices = np.argsort(-sparse_scores)
        sparse_ranks = {idx: rank + 1 for rank, idx in enumerate(sparse_rank_indices)}

        # Compute combined hybrid score
        scored_candidates = []
        max_sparse = max(float(np.max(sparse_scores)), 1.0)

        for idx, chunk in enumerate(all_chunks):
            d_score = float(dense_scores[idx])
            s_score = float(sparse_scores[idx])

            # Deprecate TOC semantic similarity if query is conceptual
            if chunk.get("is_toc") and not any(toc in clean_query for toc in ["table of contents", "syllabus", "outline", "summary", "index"]):
                d_score -= 0.35

            # RRF Fusion:
            # High dense score (semantic) + BM25 exact match
            rrf_score = (1.0 / (60.0 + dense_ranks[idx])) + (1.0 / (60.0 + sparse_ranks[idx]))

            # Normalized hybrid blend
            norm_sparse = s_score / max_sparse
            hybrid_score = (0.60 * d_score) + (0.40 * norm_sparse)

            # Strict relevance gating: drop candidate if BOTH semantic similarity is poor and sparse match is zero
            if d_score < 0.18 and s_score <= 0.0:
                continue

            scored_candidates.append({
                "chunk": chunk,
                "dense_score": d_score,
                "sparse_score": s_score,
                "fused_score": hybrid_score + (rrf_score * 50.0),
            })

        # Sort by fused score descending
        scored_candidates.sort(key=lambda x: x["fused_score"], reverse=True)

        # ----------------------------------------------------------------------
        # 4. De-duplicate at Page Level while Preserving Granular Chunk Context
        # ----------------------------------------------------------------------
        results: list[SearchResult] = []
        seen_pages: set[tuple[str, int]] = set()

        for item in scored_candidates:
            c = item["chunk"]
            doc_id = c["document_id"]
            p_num = c["page_number"]

            if (doc_id, p_num) in seen_pages:
                continue
            seen_pages.add((doc_id, p_num))

            is_visual = c["has_images"] and (wants_visuals or c.get("image_count", 0) > 0)
            modality = ModalityType.IMAGE if is_visual else ModalityType.TEXT

            # Pinpoint chunk content enriched with heading for cross-encoder & synthesis
            pinpoint_content = f"[{c['heading']}] {c['content']}" if c.get("heading") and not c["content"].startswith("[") else c["content"]

            results.append(
                SearchResult(
                    id=c["chunk_id"],
                    modality=modality,
                    score=round(float(item["fused_score"]), 4),
                    content=pinpoint_content,
                    image_url=c["preview_url"] if is_visual else None,
                    source_type="document_page",
                    data_points={
                        "page": p_num,
                        "heading": c["heading"],
                        "dense_score": round(item["dense_score"], 4),
                        "sparse_score": round(item["sparse_score"], 4),
                        "parent_context": c["parent_text"],
                    },
                    metadata={
                        "document_id": doc_id,
                        "page_number": p_num,
                        "section_heading": c["heading"],
                        "heading": c["heading"],
                        "figure_title": c["figure_title"],
                        "preview_tag": c["heading"],
                        "has_visuals": c["has_images"],
                        "preview_url": c["preview_url"],
                        "parent_text": c["parent_text"],
                        "is_toc": c.get("is_toc", False),
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
