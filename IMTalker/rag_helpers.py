"""rag_helpers.py — RAG retrieval, context compression, web search, and the
isolated upstream-`moshi` STT loader, ported from personaplex-with-rag-new's
`moshi_local/moshi/server.py` for use by the live IMTalker avatar pipeline
(`liveTry.py` / `liveTryHeliumFrontendDequeStaticPoseFP32FM_ws_binary_AHAudioPace.py`).

Everything here is intentionally free of any dependency on the PersonaPlex
`moshi` fork, `lm_gen`, or CUDA-graph state — these are pure retrieval/
compression/HTTP helpers plus one namespaced import shim, safe to call from a
background thread. The only GPU-touching class is `ContextCompressor`, which
owns its own small model and never touches the main PersonaPlex LM.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

_SYMBOL_RE = re.compile(r"[*_#`~]+")


# ── Text-injection tag helpers (verbatim from server.py) ────────────────────

def wrap_with_system_tags(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
        return cleaned
    return f"<system> {cleaned} <system>"


def wrap_with_ref_tags(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("<ref>") and cleaned.endswith("<ref>"):
        return cleaned
    return f"<ref> {cleaned} <ref>"


def wrap_with_lookup_tags() -> str:
    return "<lookup> Please wait a minute."


# ── RAG index loading + retrieval ────────────────────────────────────────────

def load_rag_index(index_dir: str):
    """Returns (chunks, embeddings[float32 N,D], embedding_model_name)."""
    manifest_path = os.path.join(index_dir, "manifest.json")
    chunks_path = os.path.join(index_dir, "chunks.npz")
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest.get("mode") == "text", (
        f"Index at {index_dir} has mode='{manifest.get('mode')}'. Rebuild with --mode text."
    )
    arrays = np.load(chunks_path)
    embeddings = arrays["text_embeddings"].astype(np.float32)
    chunks = manifest["chunks"]
    embedding_model_name = manifest.get("embedding_model", "all-MiniLM-L6-v2")
    return chunks, embeddings, embedding_model_name


def build_faiss_index(embeddings: np.ndarray):
    import faiss

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    return index


def retrieve_chunks_fast(
    transcript: str, chunks: list, faiss_index, embedding_model, top_k: int, min_score: float = 0.45
) -> list[dict]:
    """Fast retrieval using FAISS."""
    if not transcript.strip():
        return []
    query_vec = embedding_model.encode(transcript, normalize_embeddings=True).astype(np.float32)
    query_vec = query_vec.reshape(1, -1)
    scores, indices = faiss_index.search(query_vec, top_k)
    results = []
    for idx, score in zip(indices[0], scores[0]):
        if score < min_score:
            continue
        chunk = dict(chunks[idx])
        chunk["similarity_score"] = float(score)
        results.append(chunk)
    return results


def retrieve_chunks_hierarchical_fast(
    transcript: str, chunks: list, faiss_index, embeddings_t: torch.Tensor,
    embedding_model, top_k: int, min_score: float = 0.40,
    sibling_min_score: Optional[float] = None,
) -> list[dict]:
    """Hierarchical retrieval: FAISS base hits, then sibling expansion via GPU dot product."""
    base_hits = retrieve_chunks_fast(transcript, chunks, faiss_index, embedding_model, top_k, min_score)
    if not base_hits:
        return base_hits

    sibling_min_score = min_score if sibling_min_score is None else sibling_min_score
    query_vec = embedding_model.encode(transcript, normalize_embeddings=True).astype(np.float32)
    query_vec_t = torch.from_numpy(query_vec).to(embeddings_t.device)

    by_source: dict[str, list] = {}
    for i, c in enumerate(chunks):
        by_source.setdefault(c["source"], []).append((i, c))

    expanded = []
    seen_ids = set()
    for hit in base_hits:
        if hit["id"] not in seen_ids:
            expanded.append(hit)
            seen_ids.add(hit["id"])

        src = hit["source"]
        siblings = by_source[src]
        pos = next(j for j, (_, c) in enumerate(siblings) if c["id"] == hit["id"])
        sib_pos = pos + 1
        if sib_pos >= len(siblings):
            continue
        sib_idx, sib_chunk = siblings[sib_pos]
        if sib_chunk["id"] in seen_ids:
            continue

        sib_emb = embeddings_t[sib_idx]
        sib_score = float((sib_emb @ query_vec_t).cpu().item())
        if sib_score < sibling_min_score:
            continue

        merged = dict(sib_chunk)
        merged["similarity_score"] = sib_score
        expanded.append(merged)
        seen_ids.add(sib_chunk["id"])

    expanded.sort(key=lambda c: c["similarity_score"], reverse=True)
    return expanded


def summarize_context(
    transcript: str, retrieved_chunks: list[dict], embedding_model,
    max_sentences: int = 3, max_chars: int = 400,
) -> str:
    """Extractive fallback summarizer, used only if ContextCompressor isn't loaded/fails."""
    full_text = " ".join(c["text"] for c in retrieved_chunks)
    sentences = re.split(r"(?<=[.!?])\s+", full_text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 15]
    # Web-scraped passages frequently end mid-fragment -- the text after the
    # last '.'/'!'/'?' in the source is often nav/alt-text junk with no
    # sentence punctuation of its own (confirmed via conversation_logs_5:
    # a gold-price query pulled in a trailing fragment reading "Image 13:
    # Dollar IconCalculate Gold ValueImage 14: Bell", which scored high on
    # keyword overlap and was read into the assistant's <ref> context
    # verbatim). Prefer properly terminated sentences; only fall back to the
    # unfiltered list if that would leave nothing to summarize from.
    terminated = [s for s in sentences if s.endswith((".", "!", "?"))]
    if terminated:
        sentences = terminated
    if not sentences:
        return full_text[:max_chars]

    query_vec = embedding_model.encode(transcript, normalize_embeddings=True).astype(np.float32)
    sent_vecs = embedding_model.encode(sentences, normalize_embeddings=True).astype(np.float32)
    scores = sent_vecs @ query_vec

    top_indices = sorted(np.argsort(scores)[::-1][:max_sentences])
    summary = " ".join(sentences[i] for i in top_indices)
    return summary[:max_chars]


# ── STT transcript decoding ──────────────────────────────────────────────────

def decode_stt_tokens(text_tokens: list[torch.Tensor], tokenizer, padding_token_id: int) -> str:
    if not text_tokens:
        return ""
    all_tokens = torch.cat(text_tokens, dim=-1)
    all_tokens = all_tokens.cpu().view(-1)
    valid = all_tokens[all_tokens > padding_token_id]
    if valid.numel() == 0:
        return ""
    return tokenizer.decode(valid.tolist())


# ── Web search (Tavily / Serper / Bing) ──────────────────────────────────────

async def web_search_query(
    query: str,
    api_key: Optional[str],
    provider: str,
    max_results: int,
    timeout: float,
) -> list[dict]:
    """Async web search, normalized into the same chunk-dict shape as local RAG hits."""
    if not api_key:
        return []

    import aiohttp

    try:
        timeout_cfg = aiohttp.ClientTimeout(total=timeout)
        async with aiohttp.ClientSession(timeout=timeout_cfg) as session:
            if provider == "tavily":
                async with session.post(
                    "https://api.tavily.com/search",
                    json={"api_key": api_key, "query": query, "max_results": max_results, "search_depth": "basic"},
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                raw_results = data.get("results", [])[:max_results]
                return [
                    {"id": f"web-{i}", "source": r.get("url", "web"), "text": r.get("content", "")[:1000],
                     "similarity_score": float(r.get("score", 0.5))}
                    for i, r in enumerate(raw_results)
                ]
            elif provider == "serper":
                async with session.post(
                    "https://google.serper.dev/search",
                    headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
                    json={"q": query, "num": max_results},
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                raw_results = data.get("organic", [])[:max_results]
                return [
                    {"id": f"web-{i}", "source": r.get("link", "web"), "text": r.get("snippet", "")[:1000],
                     "similarity_score": max(0.5, 0.9 - 0.05 * i)}
                    for i, r in enumerate(raw_results)
                ]
            elif provider == "bing":
                async with session.get(
                    "https://api.bing.microsoft.com/v7.0/search",
                    headers={"Ocp-Apim-Subscription-Key": api_key},
                    params={"q": query, "count": max_results},
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                raw_results = data.get("webPages", {}).get("value", [])[:max_results]
                return [
                    {"id": f"web-{i}", "source": r.get("url", "web"), "text": r.get("snippet", "")[:1000],
                     "similarity_score": max(0.5, 0.9 - 0.05 * i)}
                    for i, r in enumerate(raw_results)
                ]
            else:
                print(f"[rag_helpers] unknown web search provider {provider!r}", flush=True)
                return []
    except asyncio.TimeoutError:
        print(f"[rag_helpers] web search timed out after {timeout}s: {query!r}", flush=True)
        return []
    except Exception as e:
        print(f"[rag_helpers] web search failed: {e!r}", flush=True)
        return []


def web_search_query_sync(query: str, api_key: Optional[str], provider: str, max_results: int, timeout: float) -> list[dict]:
    """Sync wrapper for use from a plain (non-asyncio) background thread."""
    return asyncio.run(web_search_query(query, api_key, provider, max_results, timeout))


# ── Context compressor: small LLM, query + top-k hits -> 1-2 sentence grounding ──

class ContextCompressor:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str = "cuda",
        max_new_tokens: int = 40,
        quantize_4bit: bool = True,
        max_passages: int = 2,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

        print(f"[rag_helpers][compressor] loading {model_name} on {device} (4bit={quantize_4bit}) ...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.getenv("HF_TOKEN"))
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        common_kwargs = dict(token=os.getenv("HF_TOKEN"), attn_implementation="sdpa")
        if quantize_4bit and device != "cpu":
            from transformers import BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, quantization_config=bnb_config, device_map=device, **common_kwargs,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device != "cpu" else torch.float32,
                device_map=device, **common_kwargs,
            )
        self.model.eval()
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.max_passages = max_passages
        self.eos_ids = [self.tokenizer.eos_token_id]

        class _SentenceEndCriteria(StoppingCriteria):
            def __init__(self, tokenizer, prompt_len, min_new=6):
                self.tokenizer = tokenizer
                self.prompt_len = prompt_len
                self.min_new = min_new

            def __call__(self, input_ids, scores, **kwargs):
                new_len = input_ids.shape[1] - self.prompt_len
                if new_len < self.min_new:
                    return False
                tail = self.tokenizer.decode(input_ids[0, -3:])
                return tail.rstrip().endswith((".", "!", "?", "\n"))

        self._StoppingCriteriaList = StoppingCriteriaList
        self._SentenceEndCriteria = _SentenceEndCriteria
        n_params = sum(p.numel() for p in self.model.parameters()) / 1e9
        print(f"[rag_helpers][compressor] ready — {n_params:.2f}B params", flush=True)

    def compress(self, question: str, chunks: list[dict]) -> str:
        # NOTE: this used to also accept a `history` parameter (recent
        # conversation turns), but it was never referenced anywhere in the
        # prompt below -- confirmed by forensic review of conversation_logs_1
        # (Issue #5). Removed rather than wired up, since using it would be a
        # behavior change (untested prompt content) beyond fixing the
        # confirmed problem (a parameter that implied functionality it didn't
        # have). If conversation-aware compression is wanted later, that's a
        # separate, deliberately-designed feature, not a bugfix.
        if not chunks:
            return ""
        passages = "\n".join(
            f"[{i + 1}] {c['text'][:180]}" for i, c in enumerate(chunks[: min(self.max_passages, 2)])
        )
        user_content = (
            "Answer in 1 short spoken sentence, plain text, no markdown, no lead-in. "
            "If the passages don't answer the question, reply NO_CONTEXT.\n"
            f"Q: {question}\nPassages:\n{passages}\nA:"
        )
        messages = [{"role": "user", "content": user_content}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        prompt_len = inputs["input_ids"].shape[1]

        stopping = self._StoppingCriteriaList([self._SentenceEndCriteria(self.tokenizer, prompt_len)])
        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.eos_ids,
                stopping_criteria=stopping,
            )
        new_tokens = out[0, prompt_len:]
        result = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        result = _SYMBOL_RE.sub("", result).strip()
        result = re.sub(r"^[\-•\d\.\)]+\s*", "", result)
        if "NO_CONTEXT" in result or not result:
            # Logged so a fallback-to-extractive-summary event (visible in
            # the conversation log as compressor fallback=True) can be traced
            # back to why the LLM answer was discarded -- previously this
            # rejection left no trace anywhere (confirmed gap hit while
            # diagnosing conversation_logs_5's turn 4).
            print(f"[rag_helpers][compressor] rejected (NO_CONTEXT/empty): {result!r}", flush=True)
            return ""

        passage_words = set(w.lower() for c in chunks[: self.max_passages] for w in c["text"].split())
        answer_words = set(w.lower().strip(".,!?") for w in result.split())
        overlap = len(answer_words & passage_words) / max(1, len(answer_words))
        if overlap < 0.15:
            print(
                f"[rag_helpers][compressor] rejected (low overlap={overlap:.2f}): {result!r}",
                flush=True,
            )
            return ""
        return result


# ── Isolated upstream-`moshi` loader (for the STT/VAD submodel only) ────────
#
# Both the PersonaPlex fork this project already runs (`liveTry.py`'s
# `import moshi`, resolved via `_ensure_moshi_importable`) and the real
# upstream Kyutai `moshi` PyPI package (needed here for
# `moshi.models.loaders.CheckpointInfo`, which the fork's own loaders.py does
# not define) both package themselves under the same top-level import name
# `moshi`. They cannot both occupy `sys.modules["moshi"]`. This loads the
# upstream package from an isolated install directory under a private alias
# (`moshi_stt`) via importlib, so it never touches `sys.modules["moshi"]` and
# never fights with the PersonaPlex fork for the name.

def load_upstream_moshi_stt(stt_pkg_dir: str):
    """Load the genuine upstream Kyutai `moshi` PyPI package (installed via
    `pip install --no-deps --target <stt_pkg_dir> moshi`) under the private
    module name `moshi_stt`. Returns the loaded module, or raises on failure
    -- callers should wrap this in try/except and disable STT/RAG on failure,
    never let it block avatar startup.
    """
    if "moshi_stt" in sys.modules:
        return sys.modules["moshi_stt"]

    init_path = Path(stt_pkg_dir) / "moshi" / "__init__.py"
    if not init_path.is_file():
        raise FileNotFoundError(
            f"upstream moshi package not found at {init_path} -- run "
            f"`pip install --no-deps --target {stt_pkg_dir} moshi` first"
        )
    spec = importlib.util.spec_from_file_location(
        "moshi_stt", init_path, submodule_search_locations=[str(init_path.parent)]
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"could not build an import spec for {init_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["moshi_stt"] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop("moshi_stt", None)
        raise
    return module
