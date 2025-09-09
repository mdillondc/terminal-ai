"""
RAG-Log Memory Service

Indexes and retrieves conversation turns from the current conversation history
(JSON-backed, in-memory) to provide reliable long-horizon recall for local
Ollama chats. It deterministically fetches earlier turns and packs them into a
compact system message for injection before model calls.

See timeline.md for details.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

import tiktoken
from datetime import datetime

from settings_manager import SettingsManager
from rag_embedding_service import EmbeddingService
from llm_client_manager import LLMClientManager
from print_helper import print_md


class TimelineLogMemory:
    """
    Conversation memory via RAG over the current conversation history.

    Responsibilities:
    - Build an in-memory embedding index of user+assistant turns
    - Retrieve earlier turns relevant to the current user intent
    - Temporal modes: earliest-global, earliest-topic, latest-topic (deterministic by turn order)
    - General mode: semantic top‑K packing under a token budget
    """

    RECALL_PREFIX = "Info:Timeline-Recall:"
    SUPPORTED_ROLES = {"user", "assistant", "system"}

    # Intent modes
    MODE_GENERAL = "general"
    MODE_EARLIEST_GLOBAL = "earliest_global"
    MODE_EARLIEST_TOPIC = "earliest_topic"
    MODE_LATEST_TOPIC = "latest_topic"

    def __init__(self, openai_client) -> None:
        self.settings = SettingsManager.getInstance()
        self.embedding_service = EmbeddingService(openai_client)
        self.llm_client_manager = LLMClientManager(openai_client)
        self.encoding = tiktoken.get_encoding("cl100k_base")

        # In-memory index of turns:
        # [{"turn_index": int, "role": str, "content": str, "message_index": int, "embedding": List[float]}]
        self._index: List[Dict[str, Any]] = []

    # ---------------------------
    # Public API
    # ---------------------------

    def get_recall_system_message(
        self, conversation_history: List[Dict[str, Any]]
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Build (or rebuild) the in-memory index and return a system message based on intent:
        - earliest_global: exact first user message (deterministic, no embeddings)
        - earliest_topic: semantic pool, then min(turn_index)
        - latest_topic: semantic pool, then max(turn_index)
        - general: semantic top‑K packing within token budget

        Returns:
            (system_message, metadata)
        """
        # Phase 2 settings
        top_k: int = int(self.settings.setting_get("timeline_top_k"))
        exclude_recent: int = int(self.settings.setting_get("timeline_exclude_recent"))
        max_tokens: int = int(self.settings.setting_get("timeline_max_tokens"))
        quote_mode: str = str(self.settings.setting_get("timeline_quote_mode") or "verbatim")
        temporal_pool_size: int = int(self.settings.setting_get("timeline_temporal_pool_size"))
        intent_use_llm: bool = bool(self.settings.setting_get("timeline_intent_use_llm"))
        intent_user_window: int = int(self.settings.setting_get("timeline_intent_user_window"))
        intent_assistant_window: int = int(self.settings.setting_get("timeline_intent_assistant_window"))
        intent_temperature: float = float(self.settings.setting_get("timeline_intent_temperature"))
        intent_max_tokens: int = int(self.settings.setting_get("timeline_intent_max_tokens"))

        # Rebuild index every call (accuracy first)
        provider = self.embedding_service._get_provider()
        print_md(f"Timeline: rebuilding index for {provider} embeddings...")
        self._build_index(conversation_history)

        # Last user prompt
        last_user = self._find_last_user_message(conversation_history)
        if last_user is None:
            raise Exception("No user message found for RAG-Log recall query")
        current_prompt = last_user.get("content") or ""
        if not isinstance(current_prompt, str) or not current_prompt.strip():
            raise Exception("Last user message is empty or invalid for RAG-Log recall")

        # Deterministic: dates we spoke (list all distinct dates from timestamps)
        if self._is_dates_we_spoke_request(current_prompt):
            sys_msg = self._build_dates_we_spoke(conversation_history)
            return sys_msg, self._meta([], self._count_tokens(sys_msg), len(self._index), top_k, exclude_recent, max_tokens)

        # Detect temporal intent (pre-check earliest_global, then LLM with recent window context). If LLM fails, default to general.
        if self._matches_earliest_global(current_prompt):
            mode, topic_query = self.MODE_EARLIEST_GLOBAL, ""
        else:
            mode, topic_query = self._detect_intent_llm_safe(
                conversation_history,
                current_prompt,
                user_window=intent_user_window,
                assistant_window=intent_assistant_window,
                temperature=intent_temperature,
                max_tokens=intent_max_tokens,
                enabled=intent_use_llm,
            )
        # Transparency: always show intent routing
        notice = f"Timeline intent: mode={mode}"
        if topic_query:
            notice += f", topic='{topic_query}'"
        print_md(notice)

        # Route based on intent
        if mode == self.MODE_EARLIEST_GLOBAL:
            # Deterministic first user message (no embeddings)
            first_user = self._find_nth_user_message(conversation_history, 1)
            if not first_user:
                # If there is no 1st user message (edge case), return empty
                return "", self._meta([], 0, 0, top_k, exclude_recent, max_tokens)

            # Single exact recall
            sys_msg = (
                f"{self.RECALL_PREFIX} Exact recall (first user message)\n"
                f"    Turn #1 (user): {self._single_line_snippet(self._normalize_text(first_user.get('content', '')))}"
            )
            token_est = self._count_tokens(sys_msg)
            return sys_msg, self._meta([1], token_est, len(self._index), top_k, 0, max_tokens)

        # For topic/general modes, we need embeddings
        # Build query for embeddings (topic_query preferred for topic modes)
        query_for_embed = topic_query.strip() if topic_query and mode in (self.MODE_EARLIEST_TOPIC, self.MODE_LATEST_TOPIC) else current_prompt

        try:
            query_embedding = self.embedding_service.generate_embedding(query_for_embed)
        except Exception as e:
            # Topic/general: embedding failure is a hard error per policy
            raise Exception(f"RAG-Log: embedding failed for recall query: {e}")

        # Choose candidate set
        if mode in (self.MODE_EARLIEST_TOPIC, self.MODE_LATEST_TOPIC):
            # Consider full timeline (do not exclude recent for temporal modes)
            candidates = self._index[:]
        else:
            # General recall: exclude recent window
            last_turn_index = self._last_turn_index()
            min_allowed_turn_index = max(0, last_turn_index - exclude_recent)
            candidates = [it for it in self._index if it["turn_index"] <= min_allowed_turn_index]

        if not candidates:
            # Nothing to recall; return empty
            return "", self._meta([], 0, 0, top_k, exclude_recent, max_tokens)

        # Score by cosine similarity
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for item in candidates:
            try:
                score = self.embedding_service.cosine_similarity(query_embedding, item["embedding"])
            except Exception:
                continue
            scored.append((score, item))

        if not scored:
            # If temporal modes produce no scores, fall back to general semantic recall for this turn
            if mode in (self.MODE_EARLIEST_TOPIC, self.MODE_LATEST_TOPIC):
                mode = self.MODE_GENERAL
            else:
                return "", self._meta([], 0, 0, top_k, exclude_recent, max_tokens)

        # ----- Temporal modes: earliest/latest among a semantic pool -----
        if mode in (self.MODE_EARLIEST_TOPIC, self.MODE_LATEST_TOPIC):
            scored.sort(key=lambda x: x[0], reverse=True)
            pool = [it for _, it in scored[:max(1, temporal_pool_size)]]

            if not pool:
                # Fallback to general recall if no pool
                mode = self.MODE_GENERAL
            else:
                chosen = min(pool, key=lambda it: it["turn_index"]) if mode == self.MODE_EARLIEST_TOPIC else max(pool, key=lambda it: it["turn_index"])
                # Build single-snippet message
                header = (
                    f"{self.RECALL_PREFIX} "
                    f"{'Earliest' if mode == self.MODE_EARLIEST_TOPIC else 'Latest'} mention"
                    + (f' (topic)' if topic_query else '')
                )
                is_timeline = self._is_timeline_request(current_prompt)
                ts = chosen.get("timestamp")
                ts_label = f"{self._format_ts(ts)}, " if (is_timeline and ts) else ""
                snippet_line = f"    Turn #{chosen['turn_index']} ({ts_label}{chosen['role']}): {self._single_line_snippet(chosen['content'])}"
                sys_msg = header + "\n" + snippet_line
                if is_timeline:
                    # Derived timeline from timestamps (start/current/days)
                    derived = self._build_timestamp_derived(conversation_history)
                    if derived:
                        sys_msg = derived + "\n" + sys_msg
                    # Add anti-parroting directive tailored for timestamps
                    timeline_directive = ("Instruction: Use the timestamps to compute dates and day counts "
                                          "(start, current, days_elapsed, days_inclusive). Do not repeat raw snippets; "
                                          "synthesize a clear summary.")
                    sys_msg = sys_msg + "\n" + timeline_directive
                else:
                    # General anti-parroting directive
                    directive = ("Instruction: Use the snippets above only as context. Do not quote or repeat them. "
                                 "Synthesize a concise answer to the user's latest prompt.")
                    sys_msg = sys_msg + "\n" + directive
                token_est = self._count_tokens(sys_msg)
                return sys_msg, self._meta([chosen["turn_index"]], token_est, len(candidates), top_k, 0, max_tokens)

        # ----- General mode: semantic top‑K packing within token budget -----
        scored.sort(key=lambda x: (round(x[0], 6), 1 if x[1]["role"] == "assistant" else 2), reverse=True)
        selected = [item for _, item in scored[:top_k]]

        if quote_mode != "verbatim":
            quote_mode = "verbatim"

        is_timeline = self._is_timeline_request(current_prompt)
        if is_timeline:
            # Sort chronologically when answering timeline questions
            selected.sort(key=lambda it: ((it.get("timestamp") or 0), it["turn_index"]))
        system_message, included_turn_indices, injected_tokens = self._pack_verbatim(selected, max_tokens, include_timestamps=is_timeline)
        if is_timeline:
            derived = self._build_timestamp_derived(conversation_history)
            if derived:
                system_message = derived + "\n" + system_message
            timeline_directive = ("Instruction: Use the timestamps to compute dates and day counts "
                                  "(start, current, days_elapsed, days_inclusive). Do not repeat raw snippets; "
                                  "synthesize a clear summary.")
            system_message = system_message + "\n" + timeline_directive
        else:
            # General anti-parroting directive
            directive = ("Instruction: Use the snippets above only as context. Do not quote or repeat them. "
                         "Synthesize a concise answer to the user's latest prompt.")
            system_message = system_message + "\n" + directive
        injected_tokens = self._count_tokens(system_message)
        return system_message, self._meta(included_turn_indices, injected_tokens, len(candidates), top_k, exclude_recent, max_tokens)

    # ---------------------------
    # Intent detection (LLM-first with last-K window)
    # ---------------------------

    def _detect_intent_llm_safe(
        self,
        conversation_history: List[Dict[str, Any]],
        current_prompt: str,
        user_window: int,
        assistant_window: int,
        temperature: float,
        max_tokens: int,
        enabled: bool,
    ) -> Tuple[str, str]:
        """
        Returns (mode, topic_query). If LLM disabled or classification fails, returns (general, "").
        """
        if not enabled:
            # Heuristic fallback: minimal cues, otherwise general
            text = current_prompt.lower()
            if any(k in text for k in ["first thing i told", "at the beginning", "from the beginning", "initially"]):
                return self.MODE_EARLIEST_GLOBAL, ""
            if any(k in text for k in ["earliest", "beginning about", "initial mention"]):
                return self.MODE_EARLIEST_TOPIC, current_prompt
            if any(k in text for k in ["latest", "most recent", "recently", "current"]):
                return self.MODE_LATEST_TOPIC, current_prompt
            return self.MODE_GENERAL, ""

        try:
            # Assemble last-K context window (interleaved recent user/assistant turns)
            recent_context = self._build_recent_context(conversation_history, user_window, assistant_window)

            system = (
                "You classify the user's intent for conversation recall.\n"
                "Return a strict JSON object with keys: mode, topic_query.\n"
                "Modes: earliest_global | earliest_topic | latest_topic | general.\n"
                "- earliest_global: user asks for 'first thing I told you' (global first user turn)\n"
                "- earliest_topic: earliest mention of a topic in this conversation\n"
                "- latest_topic: most recent mention/update about a topic\n"
                "- general: no temporal preference; just general recall\n"
                "topic_query should be a short string capturing the topic terms, or empty if not needed."
            )
            user = (
                "Classify the intent for the CURRENT user prompt using the brief recent context.\n\n"
                "RECENT CONTEXT:\n"
                f"{recent_context}\n\n"
                "CURRENT PROMPT:\n"
                f"{current_prompt}\n\n"
                "Return JSON only, e.g.: {\"mode\":\"earliest_topic\",\"topic_query\":\"alcohol withdrawal\"}"
            )
            model = self.settings.setting_get("model")
            resp = self.llm_client_manager.create_chat_completion(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = (resp.choices[0].message.content or "").strip()
            data = json.loads(content)
            mode = str(data.get("mode", "")).strip() or self.MODE_GENERAL
            topic_query = str(data.get("topic_query", "") or "").strip()
            if mode not in {self.MODE_GENERAL, self.MODE_EARLIEST_GLOBAL, self.MODE_EARLIEST_TOPIC, self.MODE_LATEST_TOPIC}:
                mode = self.MODE_GENERAL
            return mode, topic_query
        except Exception:
            # If classification fails, fall back to general
            return self.MODE_GENERAL, ""

    def _build_recent_context(self, conversation_history: List[Dict[str, Any]], uw: int, aw: int) -> str:
        """
        Build a compact recent context string of the last uw user turns and aw assistant turns,
        interleaved in actual order, most recent last.
        """
        # Collect last N user/assistant messages preserving order
        user_msgs = []
        assistant_msgs = []
        for msg in reversed(conversation_history):
            role = msg.get("role")
            if role == "user" and len(user_msgs) < uw:
                user_msgs.append(f"User: {self._normalize_text(msg.get('content', ''))}")
            elif role == "assistant" and len(assistant_msgs) < aw:
                assistant_msgs.append(f"Assistant: {self._normalize_text(msg.get('content', ''))}")
            if len(user_msgs) >= uw and len(assistant_msgs) >= aw:
                break

        # The above lists are from newest to oldest; combine in chronological order
        combined = []
        # Gather timestamps by walking forward from the start, but we only have slices; just reverse each and append
        for s in reversed(assistant_msgs):
            combined.append(s)
        for s in reversed(user_msgs):
            combined.append(s)
        return "\n".join(combined) if combined else "No recent context."

    # ---------------------------
    # Indexing
    # ---------------------------

    def _build_index(self, conversation_history: List[Dict[str, Any]]) -> None:
        """
        Build the embedding index from scratch using the provided conversation history.
        Index user, assistant, and system messages for completeness.
        Exclude prior recall injections (messages starting with Info:RAG-Log-Recall:).
        """
        self._index = []

        # Walk history, track a monotonic "turn_index" for user/assistant roles
        turn_index = 0
        to_embed: List[str] = []
        items: List[Dict[str, Any]] = []

        for i, msg in enumerate(conversation_history):
            role = str(msg.get("role", "")).lower()
            content = msg.get("content", "")

            if role not in self.SUPPORTED_ROLES:
                continue

            # Exclude any recall-of-recall content
            if isinstance(content, str) and content.startswith(self.RECALL_PREFIX):
                continue

            # Assign turn index: increment for user/assistant messages (these define conversation turns)
            if role in {"user", "assistant"}:
                turn_index += 1
            assigned_turn_index = turn_index

            # Normalize content (avoid None, ensure string)
            norm_content = self._normalize_text(content)
            if not norm_content:
                norm_content = ""

            items.append({
                "turn_index": assigned_turn_index,
                "role": role,
                "content": norm_content,
                "message_index": i,
                "timestamp": int(msg.get("timestamp")) if msg.get("timestamp") is not None else None,
            })
            to_embed.append(norm_content)

        if not items:
            return

        # Batch embeddings
        try:
            embeddings = self.embedding_service.generate_embeddings_batch(to_embed)
        except Exception as e:
            raise Exception(f"RAG-Log: embedding batch failed: {e}")



        # Attach embeddings
        for idx, emb in enumerate(embeddings):
            items[idx]["embedding"] = emb

        self._index = items

    def _last_turn_index(self) -> int:
        """Return the highest turn_index in the current index (0 if empty)."""
        if not self._index:
            return 0
        return max(item["turn_index"] for item in self._index)

    # ---------------------------
    # Packing
    # ---------------------------

    def _pack_verbatim(
        self, selected_items: List[Dict[str, Any]], max_tokens: int, include_timestamps: bool = False
    ) -> Tuple[str, List[int], int]:
        """
        Pack selected items into a single system message with verbatim quotes,
        honoring the token budget (cl100k). Returns (message, included_indices, token_estimate).
        """
        if not selected_items:
            return "", [], 0

        header_lines = [
            f"{self.RECALL_PREFIX} Conversation memory (from local history)",
            "    Injected snippets (verbatim):",
        ]
        header = "\n".join(header_lines)

        included_indices: List[int] = []
        current_text = header
        current_tokens = self._count_tokens(current_text)

        for item in selected_items:
            turn_idx = item["turn_index"]
            role = item["role"]
            content = self._single_line_snippet(item["content"])

            label = f"({role})"
            ts_val = item.get("timestamp")
            if include_timestamps and ts_val:
                ts_str = self._format_ts(ts_val)
                if ts_str:
                    label = f"({ts_str}, {role})"
            snippet_line = f"        Turn #{turn_idx} {label}: {content}"
            tentative = current_text + "\n" + snippet_line
            tentative_tokens = self._count_tokens(tentative)

            if tentative_tokens <= max_tokens:
                current_text = tentative
                current_tokens = tentative_tokens
                included_indices.append(turn_idx)
            else:
                if not included_indices:
                    truncated = self._truncate_to_tokens(snippet_line, max_tokens - current_tokens)
                    current_text = current_text + "\n" + truncated
                    current_tokens = self._count_tokens(current_text)
                    included_indices.append(turn_idx)
                break

        return current_text, included_indices, current_tokens

    # ---------------------------
    # Utilities
    # ---------------------------

    def _find_last_user_message(self, conversation_history: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Return the last message with role == 'user'."""
        for msg in reversed(conversation_history):
            if str(msg.get("role", "")).lower() == "user":
                return msg
        return None

    def _format_ts(self, ts: int) -> str:
        """Format Unix epoch seconds to local time 'YYYY-MM-DD HH:mm'."""
        try:
            return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
        except Exception:
            return ""

    def _build_timestamp_derived(self, conversation_history: List[Dict[str, Any]]) -> str:
        """
        Build a small derived timeline line from per-message timestamps.
        Uses earliest and latest message timestamps to compute spans.
        """
        ts_list = []
        for msg in conversation_history:
            ts_val = msg.get("timestamp")
            if isinstance(ts_val, int):
                ts_list.append(ts_val)

        if not ts_list:
            return ""

        start_ts = min(ts_list)
        current_ts = max(ts_list)
        # Compute whole-day span (UTC epoch based)
        days_elapsed = max(0, int((current_ts - start_ts) // 86400))
        days_inclusive = days_elapsed + 1

        start_str = datetime.fromtimestamp(start_ts).strftime("%Y-%m-%d")
        current_str = datetime.fromtimestamp(current_ts).strftime("%Y-%m-%d")

        lines = [
            f"{self.RECALL_PREFIX} Derived timeline",
            f"    Derived: start={start_str}; current={current_str}; days_elapsed={days_elapsed}; days_inclusive={days_inclusive}",
        ]
        return "\n".join(lines)

    def _find_nth_user_message(self, conversation_history: List[Dict[str, Any]], n: int) -> Optional[Dict[str, Any]]:
        """Return the nth user message (1-based), or None if not found."""
        count = 0
        for msg in conversation_history:
            if str(msg.get("role", "")).lower() == "user":
                count += 1
                if count == n:
                    return msg
        return None

    def _matches_earliest_global(self, prompt: str) -> bool:
        """Heuristic pre-check for earliest_global intent (fast path)."""
        p = prompt.lower()
        patterns = [
            r"\bfirst thing i (asked|told|said)\b",
            r"\bwhat did i (ask|tell|say) first\b",
            r"\bat the beginning\b",
            r"\bfrom the beginning\b",
        ]
        return any(re.search(rx, p) for rx in patterns)

    def _is_dates_we_spoke_request(self, prompt: str) -> bool:
        """
        Detect requests that ask for an exhaustive list of dates we spoke.
        This triggers a deterministic path that lists all distinct dates from user+assistant timestamps.
        """
        p = prompt.lower()
        patterns = [
            r"\btell me (each|all) date(s)? we (have )?spoken\b",
            r"\blist (each|all) date(s)? we (have )?spoken\b",
            r"\bwhich date(s)? did we speak\b",
            r"\bshow (each|all) date(s) we spoke\b",
            r"\bdates we spoke\b",
        ]
        return any(re.search(rx, p) for rx in patterns)

    def _build_dates_we_spoke(self, conversation_history: List[Dict[str, Any]]) -> str:
        """
        Build a deterministic list of distinct local dates (YYYY-MM-DD) on which user or assistant spoke.
        Excludes system messages. Sorted ascending. Uses per-message Unix epoch timestamps.
        """
        dates_set = set()
        for msg in conversation_history:
            role = str(msg.get("role", "")).lower()
            if role not in {"user", "assistant"}:
                continue
            ts = msg.get("timestamp")
            if not isinstance(ts, int):
                continue
            try:
                d = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
                dates_set.add(d)
            except Exception:
                continue

        dates_sorted = sorted(dates_set)
        header = f"{self.RECALL_PREFIX} Dates we spoke (deterministic)"
        if not dates_sorted:
            return header + "\n    (no dated user/assistant messages found)"
        lines = [header]
        for d in dates_sorted:
            lines.append(f"    - {d}")
        # Anti-parroting directive tailored for date listing (concise output)
        lines.append("Instruction: Provide the dates above as the final answer, formatted as a simple list. Do not add commentary.")
        return "\n".join(lines)

    def _is_timeline_request(self, prompt: str) -> bool:
        """Detect if the user is asking for dates/timeline/timespan/current status."""
        p = prompt.lower()
        keywords = [
            "timeline", "date", "dates", "when did", "how many days",
            "timespan", "time span", "current date", "today", "days_elapsed",
        ]
        return any(k in p for k in keywords)

    # TimelineManager-based anchors removed (timestamp-first approach)

    @staticmethod
    def _normalize_text(content: Any) -> str:
        """Normalize content to a plain string; collapse whitespace."""
        if content is None:
            return ""
        if isinstance(content, str):
            text = content
        else:
            text = str(content)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _single_line_snippet(text: str, max_chars: int = 1800) -> str:
        """Single-line snippet; collapse whitespace and char-truncate (packing is token-based)."""
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > max_chars:
            text = text[: max_chars - 3] + "..."
        return text

    def _count_tokens(self, text: str) -> int:
        """Count tokens using cl100k encoding."""
        if not text:
            return 0
        return len(self.encoding.encode(text))

    def _truncate_to_tokens(self, text: str, token_budget: int) -> str:
        """Truncate text to fit within a token budget (best-effort, binary search on chars)."""
        if token_budget <= 0:
            return ""
        if self._count_tokens(text) <= token_budget:
            return text

        lo, hi = 0, len(text)
        best = ""
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = text[:mid]
            tokens = self._count_tokens(candidate)
            if tokens <= token_budget:
                best = candidate
                lo = mid + 1
            else:
                hi = mid - 1

        return best.rstrip() + ("..." if best and not best.endswith("...") else "")

    def _meta(
        self,
        included_turn_indices: List[int],
        injected_token_estimate: int,
        candidate_count: int,
        top_k: int,
        exclude_recent: int,
        max_tokens: int,
    ) -> Dict[str, Any]:
        """Helper to build metadata dict."""
        return {
            "included_turn_indices": included_turn_indices,
            "injected_token_estimate": injected_token_estimate,
            "candidate_count": candidate_count,
            "top_k": top_k,
            "exclude_recent": exclude_recent,
            "max_tokens": max_tokens,
        }