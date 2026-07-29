"""conversation_logger.py — structured conversation logging for the live
PersonaPlex + IMTalker + RAG pipeline.

Three outputs, all append-only (never truncated/overwritten while the
process is alive; each server run gets its own timestamped session_id, so
older sessions' files are never touched either):

  1. Console (stdout, already captured into live_server.log by the launch
     notebook) -- short one-line-per-event summaries.
  2. `conversation_<session>.log` / `.jsonl` -- the same short events, as
     text and as machine-parseable JSON lines, for scripts/dashboards.
  3. `detailed_<session>.log` -- a plain-English, step-by-step narrative of
     every request: what the user said, what the assistant decided to do,
     every document/web search attempted (with scores and running counts),
     why and how a summary was created, exactly what was fed to the model
     and how, what it answered, and the "thinking sound" start/stop/loop
     detail. Written so a non-technical reader can follow the whole story
     of a single request top to bottom without needing to know what RAG,
     FAISS, or a token is.

Thread-safety: called from both the GPU thread and the background
retrieval thread (see MoshiOnlyEngineWithHidden._rag_* in the AHAudioPace
script). `logging.Logger`/`Handler` are internally thread-safe; the JSONL
and detailed-narrative write paths take their own lock. The turn-machine
these are called from only ever has one request in flight at a time, so the
detailed log's events arrive in true chronological order -- no buffering or
reordering needed to keep the narrative readable.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any


class ConversationLogger:
    def __init__(self, log_dir: str = "", session_id: str = ""):
        self.session_id = session_id or time.strftime("%Y%m%d_%H%M%S")
        self.log_dir = str(log_dir or "")
        self._jsonl_path: str | None = None
        self._detail_path: str | None = None
        self._lock = threading.Lock()

        # Running totals for this server process (i.e. across the whole
        # session, not per-request) -- "how many times document search is
        # performed" / "how many times online search is performed".
        self.doc_search_count = 0
        self.web_search_count = 0

        self.logger = logging.getLogger(f"conversation.{id(self)}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        if not self.logger.handlers:
            console = logging.StreamHandler(sys.stdout)
            console.setFormatter(logging.Formatter("[%(asctime)s] [CONV] %(message)s", datefmt="%H:%M:%S"))
            self.logger.addHandler(console)

            if self.log_dir:
                os.makedirs(self.log_dir, exist_ok=True)
                text_path = os.path.join(self.log_dir, f"conversation_{self.session_id}.log")
                file_handler = logging.FileHandler(text_path, encoding="utf-8")
                file_handler.setFormatter(
                    logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
                )
                self.logger.addHandler(file_handler)
                self._jsonl_path = os.path.join(self.log_dir, f"conversation_{self.session_id}.jsonl")

                self._detail_path = os.path.join(self.log_dir, f"detailed_{self.session_id}.log")
                with open(self._detail_path, "a", encoding="utf-8") as f:
                    f.write(
                        "=" * 80 + "\n"
                        f"Detailed conversation log -- session {self.session_id}\n"
                        f"Started {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                        "This file records the full story of every request in plain language:\n"
                        "what was said, what was searched, what was found, what was given to\n"
                        "the assistant, and what it answered. Nothing here is ever overwritten --\n"
                        "each new request is appended below the last.\n"
                        + "=" * 80 + "\n\n"
                    )

                print(
                    f"[conversation_logger] logging conversation events to {text_path}, "
                    f"{self._jsonl_path}, and {self._detail_path}",
                    flush=True,
                )

    # -- low-level ---------------------------------------------------------

    def _write_jsonl(self, record: dict[str, Any]) -> None:
        if not self._jsonl_path:
            return
        record = dict(record)
        record.setdefault("ts", time.time())
        record.setdefault("session_id", self.session_id)
        line = json.dumps(record, default=str, ensure_ascii=False)
        try:
            with self._lock, open(self._jsonl_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as e:
            # Logging must never crash the live pipeline.
            print(f"[conversation_logger] failed to write jsonl event: {e!r}", flush=True)

    def event(self, kind: str, summary: str = "", **fields: Any) -> None:
        line = f"[{kind}] {summary}" if summary else f"[{kind}]"
        extra = " ".join(f"{k}={v!r}" for k, v in fields.items() if v is not None)
        if extra:
            line = f"{line} {extra}"
        self.logger.info(line)
        self._write_jsonl({"kind": kind, "summary": summary, **fields})

    def _write_detail(self, turn_id: Any, heading: str, body: list[str]) -> None:
        """Append one readable, timestamped section to the detailed
        narrative log. Never raises -- logging must never break the pipeline."""
        if not self._detail_path:
            return
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        turn_label = f"Turn #{turn_id}" if turn_id is not None else "—"
        lines = [f"[{ts}] {turn_label} — {heading}"]
        for b in body:
            for sub in str(b).splitlines() or [""]:
                lines.append(f"    {sub}")
        lines.append("")
        text = "\n".join(lines) + "\n"
        try:
            with self._lock, open(self._detail_path, "a", encoding="utf-8") as f:
                f.write(text)
        except Exception as e:
            print(f"[conversation_logger] failed to write detailed log: {e!r}", flush=True)

    # -- convenience wrappers matching the actual pipeline events (short,
    # machine-friendly console/JSONL events) --------------------------------

    def user_transcript(self, transcript: str, turn_epoch: int) -> None:
        self.event("user_transcript", transcript, turn_epoch=turn_epoch)

    def rag_gate(self, transcript: str, top_score: float, lexical_hit: bool, triggered: bool) -> None:
        self.event(
            "rag_gate",
            f"triggered={triggered}",
            transcript=transcript, top_score=round(top_score, 4), lexical_hit=lexical_hit,
        )

    def retrieval(self, transcript: str, source: str, hits: list[dict], elapsed_s: float) -> None:
        preview = [
            {"source": h.get("source"), "score": round(float(h.get("similarity_score", 0.0)), 4),
             "text": str(h.get("text", ""))[:200]}
            for h in hits
        ]
        self.event(
            "retrieval", f"source={source} n_hits={len(hits)} elapsed={elapsed_s:.3f}s",
            transcript=transcript, hits=preview,
        )

    def web_search(self, query: str, provider: str, n_results: int, elapsed_s: float, triggered_reason: str = "") -> None:
        self.event(
            "web_search", f"provider={provider} n_results={n_results} elapsed={elapsed_s:.3f}s",
            query=query, triggered_reason=triggered_reason,
        )

    def compressor_call(self, question: str, passages: list[str], result: str, elapsed_s: float, used_fallback: bool) -> None:
        self.event(
            "compressor",
            f"elapsed={elapsed_s:.3f}s fallback={used_fallback}",
            question=question, passages=[p[:200] for p in passages], result=result,
        )

    def ref_injected(self, ref_text: str, n_tokens: int, elapsed_s: float, kind: str = "ref") -> None:
        self.event(
            f"{kind}_injected", f"n_tokens={n_tokens} elapsed={elapsed_s:.3f}s",
            text=ref_text,
        )

    def assistant_response(self, transcript: str, response_text: str) -> None:
        self.event("assistant_response", response_text, user_transcript=transcript)

    def component_status(self, **fields: Any) -> None:
        self.event("component_status", **fields)

    def error(self, where: str, exc: BaseException, traceback_text: str = "") -> None:
        self.event("error", f"{where}: {exc!r}", where=where, traceback=traceback_text[-4000:])

    # -- detailed, plain-English narrative (the new "any user can read it"
    # log) ------------------------------------------------------------------

    def narrate_user_message(self, turn_id: Any, transcript: str) -> None:
        self._write_detail(turn_id, "User spoke", [f'The user said: "{transcript}"'])

    def narrate_decision(
        self, turn_id: Any, top_score: float, min_score: float, lexical_hit: bool, triggered: bool
    ) -> None:
        lines = [
            "The assistant did a quick check of its documents to see if this question "
            "needs a real search.",
            f"Best quick match score: {top_score:.3f} (a score above ~{min_score:.3f}, or a "
            f"matching keyword, is enough to trigger a full search).",
        ]
        if lexical_hit:
            lines.append("A matching keyword was also found between the question and a document.")
        if triggered:
            lines.append(
                "Decision: this looks like it needs a document search, so the assistant is "
                "searching before answering."
            )
        else:
            lines.append(
                "Decision: nothing looked relevant enough, so the assistant will answer "
                "normally from what it already knows, without searching."
            )
        self._write_detail(turn_id, "Deciding how to respond", lines)

    def narrate_doc_search_start(self, turn_id: Any, query: str, n_chunks: int, sources: list[str]) -> None:
        self.doc_search_count += 1
        src_desc = ", ".join(sources[:5]) + (", ..." if len(sources) > 5 else "") if sources else "unknown"
        self._write_detail(
            turn_id, f"Document search #{self.doc_search_count} this session",
            [
                f"Searching the assistant's local knowledge base: {n_chunks} indexed passage(s) "
                f"from {len(sources)} file(s) ({src_desc}).",
                f'Search query (from what the user said): "{query}"',
            ],
        )

    def narrate_doc_search_results(self, turn_id: Any, hits: list[dict]) -> None:
        if not hits:
            self._write_detail(
                turn_id, "Document search results",
                ["No matching passages were found in the local documents."],
            )
            return
        lines = [f"Found {len(hits)} matching passage(s), best match first:"]
        for i, h in enumerate(hits, 1):
            score = float(h.get("similarity_score", 0.0))
            src = h.get("source", "unknown")
            text = str(h.get("text", ""))[:220]
            lines.append(f'{i}. match quality {score:.3f} (higher is better), from "{src}":')
            lines.append(f'   "{text}"')
        self._write_detail(turn_id, "Document search results", lines)

    def narrate_web_search_start(self, turn_id: Any, query: str, provider: str, reason: str) -> None:
        self.web_search_count += 1
        self._write_detail(
            turn_id, f"Online search #{self.web_search_count} this session",
            [
                f"Why: {reason}",
                f'Search query: "{query}"',
                f"Search provider: {provider}",
            ],
        )

    def narrate_web_search_results(self, turn_id: Any, all_hits: list[dict], selected_hits: list[dict], max_results: int) -> None:
        if not all_hits:
            self._write_detail(turn_id, "Online search results", ["No results were returned by the web search."])
            return
        lines = [f"Found {len(all_hits)} result(s) from the web:"]
        for i, h in enumerate(all_hits, 1):
            score = float(h.get("similarity_score", 0.0))
            src = h.get("source", "unknown")
            text = str(h.get("text", ""))[:220]
            lines.append(f'{i}. relevance {score:.3f}, from "{src}": "{text}"')
        lines.append(
            f"Preprocessing: results were sorted by relevance (highest first) and the top "
            f"{min(len(selected_hits), max_results)} of {len(all_hits)} were kept for use."
        )
        self._write_detail(turn_id, "Online search results", lines)

    def narrate_summary(self, turn_id: Any, source: str, n_passages: int, summary_text: str, used_fallback: bool) -> None:
        method = (
            "a short extractive summary (the most relevant sentences picked out directly)"
            if used_fallback
            else "a small AI model that reads the passages and writes one short spoken sentence"
        )
        lines = [
            f"Why: to turn the {n_passages} retrieved passage(s) from {source} into one short, "
            f"natural-sounding piece of information the assistant can work into its reply, "
            f"instead of reading raw document text aloud.",
            f"How: {method}.",
            f'What it contains: "{summary_text}"' if summary_text else "Nothing usable was produced.",
        ]
        self._write_detail(turn_id, "Summary created", lines)

    def narrate_no_information(self, turn_id: Any) -> None:
        self._write_detail(
            turn_id, "No information found",
            [
                "Neither the documents nor (if enabled) the web search turned up anything "
                "relevant, so the assistant was told to answer from its own general knowledge "
                "instead of pretending to have a specific source."
            ],
        )

    def narrate_injection(
        self, turn_id: Any, injected_text: str, n_tokens_final: int, n_tokens_before_trim: int, max_tokens: int, kind: str
    ) -> None:
        if n_tokens_before_trim > max_tokens:
            preprocessing = (
                f"The information was shortened from {n_tokens_before_trim} to {max_tokens} tokens "
                f"(roughly word-pieces) so it stays short enough for the assistant to read quickly."
            )
        else:
            preprocessing = f"No shortening was needed ({n_tokens_final} tokens, under the {max_tokens}-token limit)."
        label = "grounding information" if kind == "ref" else "filler notice"
        lines = [
            f'What is given to the assistant: "{injected_text}"',
            "How it is given: it is fed to the assistant word-by-word as a hidden note "
            "inserted directly into the live, ongoing conversation -- the assistant's memory "
            "of everything said so far is NOT cleared or restarted, this is simply added on top of it.",
            f"Preprocessing: {preprocessing}",
            f"This {label} is exactly what the assistant receives as input before its next reply.",
        ]
        self._write_detail(turn_id, "Information given to the assistant", lines)

    def narrate_response(self, turn_id: Any, transcript: str, response_text: str) -> None:
        if response_text:
            self._write_detail(
                turn_id, "Assistant replied",
                [f'In reply to: "{transcript}"', f'The assistant said: "{response_text}"'],
            )
        else:
            self._write_detail(
                turn_id, "Assistant replied",
                [f'In reply to: "{transcript}"', "(no speech was captured for this turn)"],
            )

    def narrate_thinking_start(self, turn_id: Any) -> None:
        self._write_detail(
            turn_id, "Thinking sound started",
            ["Started playing so the user hears something while the assistant searches, instead of silence."],
        )

    def narrate_thinking_stop(self, turn_id: Any, reason: str, duration_s: float, play_count: int, clip_duration_s: float) -> None:
        if play_count <= 1:
            loop_desc = f"played once, did not need to loop (the clip is {clip_duration_s:.1f}s long)."
        else:
            loop_desc = f"played {play_count} times in a row (looped), the clip is {clip_duration_s:.1f}s long."
        self._write_detail(
            turn_id, "Thinking sound stopped",
            [
                f"Why it stopped: {reason}.",
                f"How long it played: {duration_s:.1f} seconds.",
                f"How many times it played: {loop_desc}",
            ],
        )
