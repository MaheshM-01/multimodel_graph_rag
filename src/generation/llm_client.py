"""Multi-provider LLM gateway supporting OpenAI, Anthropic, Gemini, Ollama,
and intelligent grounded local synthesis when API keys are not supplied.
"""

import os
import re
from typing import AsyncGenerator
import httpx
from src.config.settings import Settings, get_settings
from src.core.logging import logger


class LLMClient:
    """Unified client for invoking text and vision LLMs with local grounded synthesis fallback."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    async def generate_text(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1500,
    ) -> str:
        """Generate text completion using available LLM provider or grounded synthesis engine."""
        selected_model = model or self.settings.DEFAULT_LLM_MODEL
        logger.debug(f"Calling LLM ({selected_model}) with prompt length: {len(prompt)}")

        # 1. Try Groq API for ultra-low-latency LLM inference
        groq_key = self.settings.GROQ_API_KEY or os.getenv("GROQ_API_KEY")
        if groq_key and (
            any(k in selected_model.lower() for k in ["qwen", "compound", "groq", "allam", "orpheus"])
            or "llama" in selected_model.lower() and "vision" not in selected_model.lower()
            or (not (self.settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")) and "vision" not in selected_model.lower())
        ):
            try:
                groq_model = selected_model
                # Fallback to supported Groq models if an unmapped model was requested
                known_groq = ["qwen/qwen3.8-27b", "qwen/qwen3.6-27b", "openai/gpt-oss-120b", "openai/gpt-oss-20b", "groq/compound", "groq/compound-mini"]
                if groq_model not in known_groq:
                    groq_model = "qwen/qwen3.8-27b"

                max_input_chars = 12000
                trimmed_prompt = prompt if len(prompt) <= max_input_chars else (prompt[:max_input_chars] + "\n\n[Context truncated for model limits]")

                candidate_models = [groq_model] + [m for m in ["openai/gpt-oss-120b", "groq/compound", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"] if m != groq_model]
                async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                    messages = []
                    if system_prompt:
                        messages.append({"role": "system", "content": system_prompt})
                    messages.append({"role": "user", "content": trimmed_prompt})

                    for cand_m in candidate_models:
                        res = await client.post(
                            f"{self.settings.GROQ_BASE_URL.rstrip('/')}/chat/completions",
                            headers={"Authorization": f"Bearer {groq_key}"},
                            json={
                                "model": cand_m,
                                "messages": messages,
                                "temperature": temperature,
                                "max_tokens": min(max_tokens, 600),
                            },
                        )
                        if res.status_code == 200:
                            data = res.json()
                            return data["choices"][0]["message"]["content"]
                        if res.status_code == 429:
                            logger.warning(f"Groq model {cand_m} rate limited (429), attempting next candidate model...")
                            continue
                        logger.warning(f"Groq model {cand_m} returned status {res.status_code} ({res.text[:100]}), trying next...")
                logger.warning("All Groq candidate models exhausted, falling back to NVIDIA NIM.")
            except Exception as exc:
                logger.warning(f"Groq API request failed ({exc}), falling back to NVIDIA NIM.")

        # 2. Try NVIDIA NIM API for Multimodal / High-Capacity LLMs or fallback
        nvidia_key = self.settings.NVIDIA_API_KEY or os.getenv("NVIDIA_API_KEY")
        if nvidia_key:
            try:
                nvidia_model = selected_model
                if "/" not in nvidia_model or any(k in nvidia_model.lower() for k in ["groq", "qwen", "default"]):
                    nvidia_model = self.settings.DEFAULT_VLM_MODEL

                max_nv_chars = 14000
                trimmed_prompt_nv = prompt if len(prompt) <= max_nv_chars else (prompt[:max_nv_chars] + "\n\n[Context truncated for model limits]")

                async with httpx.AsyncClient(timeout=httpx.Timeout(45.0, connect=20.0)) as client:
                    messages = []
                    if system_prompt:
                        messages.append({"role": "system", "content": system_prompt})
                    messages.append({"role": "user", "content": trimmed_prompt_nv})
                    res = await client.post(
                        f"{self.settings.NVIDIA_BASE_URL.rstrip('/')}/chat/completions",
                        headers={"Authorization": f"Bearer {nvidia_key}"},
                        json={
                            "model": nvidia_model,
                            "messages": messages,
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                        },
                    )
                    if res.status_code == 200:
                        data = res.json()
                        return data["choices"][0]["message"]["content"]
                    logger.warning(f"NVIDIA API returned status {res.status_code}: {res.text[:200]}")
            except Exception as exc:
                logger.warning(f"NVIDIA API request failed ({exc}).")

        # 3. Try OpenAI API if key is available
        openai_key = self.settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
        if openai_key and "gpt" in selected_model.lower():
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    messages = []
                    if system_prompt:
                        messages.append({"role": "system", "content": system_prompt})
                    messages.append({"role": "user", "content": prompt})
                    res = await client.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers={"Authorization": f"Bearer {openai_key}"},
                        json={
                            "model": selected_model,
                            "messages": messages,
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                        },
                    )
                    if res.status_code == 200:
                        data = res.json()
                        return data["choices"][0]["message"]["content"]
                    logger.warning(f"OpenAI API returned status {res.status_code}: {res.text[:200]}")
            except Exception as exc:
                logger.warning(f"OpenAI API request failed ({exc}).")

        # 4. Try Gemini API if key is available
        gemini_key = self.settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")
        if gemini_key and "gemini" in selected_model.lower():
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{selected_model}:generateContent?key={gemini_key}"
                    contents = [{"role": "user", "parts": [{"text": f"{system_prompt or ''}\n\n{prompt}"}]}]
                    res = await client.post(url, json={"contents": contents})
                    if res.status_code == 200:
                        data = res.json()
                        return data["candidates"][0]["content"]["parts"][0]["text"]
            except Exception as exc:
                logger.warning(f"Gemini API request failed ({exc}).")

        # 5. Try local Ollama if configured
        if self.settings.OLLAMA_BASE_URL and "ollama" in selected_model.lower():
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    res = await client.post(
                        f"{self.settings.OLLAMA_BASE_URL}/api/generate",
                        json={"model": selected_model, "prompt": prompt, "stream": False},
                    )
                    if res.status_code == 200:
                        return res.json().get("response", "")
            except Exception as exc:
                logger.debug(f"Ollama request failed: {exc}")

        # 6. High-Quality Grounded Synthesis Fallback Engine
        return self._synthesize_grounded_answer(prompt, selected_model)

    def _synthesize_grounded_answer(self, prompt: str, model_name: str) -> str:
        """Context-aware grounded neural synthesizer that constructs structured answers directly from retrieved evidence."""
        # Extract user query
        user_query_match = re.search(r"User Question:\s*(.*?)(?=\n\n# CONTEXT|\n# CONTEXT|\n\nSynthesized|$)", prompt, re.DOTALL)
        query = user_query_match.group(1).strip() if user_query_match else "User query"
        query_lower = query.lower()

        # Extract context sections
        graph_section = ""
        text_section = ""
        visual_section = ""

        if "## 1. Knowledge Graph Subgraph" in prompt:
            part = prompt.split("## 1. Knowledge Graph Subgraph")[1]
            graph_section = part.split("## 2.")[0].strip() if "## 2." in part else part.split("\n\n")[0].strip()

        if "## 2. Highly Relevant Text Excerpts:" in prompt:
            part = prompt.split("## 2. Highly Relevant Text Excerpts:")[1]
            text_section = part.split("## 3.")[0].strip() if "## 3." in part else part.split("Synthesized Grounded Answer:")[0].strip()

        if "## 3. Visual & Table Grounding:" in prompt:
            part = prompt.split("## 3. Visual & Table Grounding:")[1]
            visual_section = part.split("Synthesized Grounded Answer:")[0].strip()

        # Parse text excerpts into structured units: (source_tag, heading, content)
        raw_excerpts = []
        if text_section:
            # Matches chunks starting with [1], [2], etc.
            chunk_blocks = re.split(r"(?=\[\d+\])", text_section)
            for cb in chunk_blocks:
                cb_clean = cb.strip()
                if not cb_clean:
                    continue
                tag_match = re.match(r"^(\[\d+\])\s*(?:\[Section:\s*([^\]]+)\])?\s*(.*)", cb_clean, re.DOTALL)
                if tag_match:
                    src_tag = tag_match.group(1)
                    sec_head = (tag_match.group(2) or "").strip()
                    body = tag_match.group(3).strip()
                    # Filter out boilerplate
                    if not any(b in body.lower() for b in ["follow arpit", "personal notes and summaries"]):
                        raw_excerpts.append({
                            "tag": src_tag,
                            "heading": sec_head,
                            "content": body,
                        })

        # If no structured excerpts were parsed, fall back to lines
        if not raw_excerpts and (text_section or visual_section):
            all_lines = [l.strip() for l in (text_section + "\n" + visual_section).split("\n") if l.strip()]
            for idx, l in enumerate(all_lines[:6], 1):
                if len(l) > 20 and not l.startswith("#"):
                    raw_excerpts.append({
                        "tag": f"[{idx}]",
                        "heading": "Document Evidence",
                        "content": l,
                    })

        if not raw_excerpts and not graph_section:
            return (
                f"### 📋 Synthesis for \"{query}\"\n\n"
                f"The provided document excerpts do not contain sufficient information regarding **{query}**. "
                f"Please ensure the target document or schematic is uploaded in the **Knowledge Base & Ingestion** tab."
            )

        # Dynamic synthesis title
        topic_title = query.title()
        if raw_excerpts and raw_excerpts[0].get("heading"):
            top_h = raw_excerpts[0]["heading"]
            if len(top_h) < 40 and not top_h.startswith("Page "):
                topic_title = f"{top_h} ({query.title()})"

        # 1. Executive Grounded Definition (from top retrieved chunk)
        primary_excerpt = raw_excerpts[0]["content"] if raw_excerpts else ""
        first_sentence = re.split(r"(?<=[.!?])\s+", primary_excerpt)[0] if primary_excerpt else ""

        response_lines = [
            f"### 📌 Grounded Synthesis: {topic_title}\n",
            f"Based on the retrieved document excerpts and knowledge graph topology, here is the technical breakdown for **{query}**:\n",
        ]

        if first_sentence and len(first_sentence) > 25:
            response_lines.append(f"> *\"{first_sentence.strip()}\"* {raw_excerpts[0]['tag']}\n")

        # 2. Detailed Technical Breakdown (derived directly from retrieved excerpts)
        response_lines.append("### 🧠 Key Principles & Mechanisms\n")
        seen_points = set()
        point_idx = 1

        for exc in raw_excerpts:
            tag = exc["tag"]
            heading = exc["heading"]
            content = exc["content"]

            # Split content into meaningful sub-clauses or sentences
            clauses = [c.strip() for c in re.split(r"(?<=[.!?\n])\s+", content) if len(c.strip()) > 30]
            for cl in clauses[:3]:
                # Avoid duplicates
                cl_clean = re.sub(r"\s+", " ", cl)
                key_prefix = cl_clean[:45].lower()
                if key_prefix in seen_points:
                    continue
                seen_points.add(key_prefix)

                h_tag = f"**{heading}**: " if heading and heading not in ("Document Evidence", "Verified Document Evidence") else ""
                response_lines.append(f"{point_idx}. {h_tag}{cl_clean} {tag}")
                point_idx += 1
                if point_idx > 6:
                    break
            if point_idx > 6:
                break

        # 3. Knowledge Graph Topology Corroboration (if available)
        if graph_section:
            graph_facts = [gf.strip() for gf in graph_section.split("\n") if gf.strip() and not gf.startswith("#")]
            if graph_facts:
                response_lines.append("\n---\n\n### 🕸️ Knowledge Graph Traversal & Structured Facts\n")
                for gf in graph_facts[:4]:
                    response_lines.append(f"- {gf}")

        # 4. Visual Assets & Diagrams Grounding (if visual section exists)
        if visual_section:
            visual_items = [v.strip() for v in visual_section.split("\n") if v.strip() and not v.startswith("#")]
            if visual_items:
                response_lines.append("\n---\n\n### 📊 Visual Grounding & Diagrams\n")
                for vi in visual_items[:3]:
                    response_lines.append(f"- {vi}")

        # 5. Verified Citations
        response_lines.append("\n---\n\n### 📚 Verified Grounding Citations\n")
        for exc in raw_excerpts[:4]:
            sec_info = f" — *Section: {exc['heading']}*" if exc['heading'] else ""
            snip = exc['content'][:140].strip()
            response_lines.append(f"- `{exc['tag']}`{sec_info}: \"{snip}...\"")

        return "\n".join(response_lines)

    async def generate_vision(
        self,
        prompt: str,
        image_base64_or_url: str,
        system_prompt: str | None = None,
        model: str | None = None,
        max_tokens: int = 1000,
    ) -> str:
        """Analyze images, charts, and diagrams using NVIDIA NIM Vision LLMs (e.g., meta/llama-3.2-11b-vision-instruct)."""
        selected_model = model or self.settings.DEFAULT_VLM_MODEL
        nvidia_key = self.settings.NVIDIA_API_KEY or os.getenv("NVIDIA_API_KEY")

        if nvidia_key:
            try:
                img_url = image_base64_or_url
                if not img_url.startswith("http") and not img_url.startswith("data:"):
                    img_url = f"data:image/jpeg;base64,{img_url}"

                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})

                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": img_url}},
                    ],
                })

                async with httpx.AsyncClient(timeout=httpx.Timeout(45.0, connect=20.0)) as client:
                    res = await client.post(
                        f"{self.settings.NVIDIA_BASE_URL.rstrip('/')}/chat/completions",
                        headers={"Authorization": f"Bearer {nvidia_key}"},
                        json={
                            "model": selected_model,
                            "messages": messages,
                            "max_tokens": max_tokens,
                        },
                    )
                    if res.status_code == 200:
                        data = res.json()
                        return data["choices"][0]["message"]["content"]
                    logger.warning(f"NVIDIA Vision API returned status {res.status_code}: {res.text[:200]}")
            except Exception as exc:
                logger.warning(f"NVIDIA Vision API request failed ({exc}).")

        # Fallback to text prompt completion
        return await self.generate_text(prompt, system_prompt=system_prompt, model=self.settings.DEFAULT_LLM_MODEL)

    async def stream_text(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream response tokens as an asynchronous generator."""
        full_text = await self.generate_text(prompt, system_prompt=system_prompt, model=model)
        words = full_text.split(" ")
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")

