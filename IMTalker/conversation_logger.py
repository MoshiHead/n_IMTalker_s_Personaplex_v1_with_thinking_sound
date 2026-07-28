"""conversation_logger.py — structured conversation logging for the live
PersonaPlex + IMTalker + RAG pipeline.

Every event is written two ways:
  1. A human-readable line to stdout (already captured into live_server.log
     by the launch notebook) AND to a dedicated per-session .log file, so RAG
     activity is easy to find without wading through the full moshi-step
     spam.
  2. A machine-parseable line to a per-session .jsonl file, so a whole
     session's RAG/STT/web-search/compressor activity can be re-loaded and
     inspected (e.g. `[json.loads(l) for l in open(path)]`) without regexing
     the text log.

Thread-safety: this is called from both the GPU thread and the background
retrieval thread (see MoshiOnlyEngineWithHidden._rag_* in the AHAudioPace
script). `logging.Logger`/`Handler` are internally thread-safe; the JSONL
write path takes its own lock.
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
        self._lock = threading.Lock()

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
                print(f"[conversation_logger] logging conversation events to {text_path} and {self._jsonl_path}", flush=True)

    # -- low-level -------------------------------------------------------

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

    # -- convenience wrappers matching the actual pipeline events --------

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
