#!/usr/bin/env python3
"""
Search-Log Engine (v1)

Whoosh-based, per-search, in-memory indexing of the current conversation log (user+assistant role only to prevent noise).
- OR-based keyword matching with phrase-boosting for multi-word phrases
- Sentence-aware snippets with robust fallback (line-based or bounded context)
- Optional neighboring messages (by turn) to provide minimal extra context
- Temporal grouping by date for time-oriented queries
- Returns a formatted "Search-Log Summary:" block ready to append via conversation_manager
- Using Whoosh instead of the AI allows searching across massive context windows without the massive additional context the AI would add to the already existing context window.

Notes:
- Source of truth is the on-disk JSON file (read elsewhere). Pass the already-parsed indexable messages here.
- Indexable messages must be a list of dicts with keys:
  {
    "content": str,
    "timestamp": int|float|None,  # unix ts
    "role": "user"|"assistant",
    "turn_index": int,            # original position in full conversation array
    "message_id": int|None
  }
"""

from __future__ import annotations

import datetime
import re
from typing import Any, Dict, List, Optional, Tuple

from print_helper import print_md
from settings_manager import SettingsManager

# Whoosh
from whoosh import query as Q
from whoosh.analysis import StandardAnalyzer
from whoosh.fields import Schema, TEXT, NUMERIC, ID
from whoosh.filedb.filestore import RamStorage
from whoosh.highlight import SentenceFragmenter, NullFormatter, ContextFragmenter
from whoosh.index import create_in


IndexableMessage = Dict[str, Any]


class SearchLogEngine:
    def __init__(self, settings_manager: Optional[SettingsManager] = None) -> None:
        self._settings = settings_manager or SettingsManager.getInstance()
        # Analyzer: language-agnostic (no stoplist)
        self._analyzer = StandardAnalyzer(stoplist=None)

    # Public API ----------------------------------------------------------------

    def search(
        self,
        messages: List[IndexableMessage],
        keywords: List[str],
        phrases: List[str],
        is_temporal: bool,
    ) -> Dict[str, Any]:
        """
        Build an in-memory index over messages and run a query with phrase-boosting.
        Produce sentence-aware snippets with fallback and return a ready-to-append
        summary string (prefixed with 'Search-Log Summary:') alongside structured hits.

        Returns:
            {
              "hits": [
                  {
                    "timestamp": int|float|None,
                    "date": "YYYY-MM-DD"|None,
                    "time": "HH:MM"|None,
                    "message_id": int|None,
                    "role": "user"|"assistant",
                    "turn_index": int,
                    "snippets": [str, ...]
                  },
                  ...
              ],
              "is_temporal": bool,
              "stats": {"indexed": int, "matches": int},
              "formatted_summary": Optional[str]  # None if no hits
            }
        """
        transparency = bool(getattr(self._settings, "search_log_transparency", True))
        max_results = int(getattr(self._settings, "search_log_max_results", 400))
        ctx_neighbors = int(getattr(self._settings, "search_log_context_messages", 0))
        snippet_count = int(getattr(self._settings, "search_log_snippets_per_hit", 1))
        # Transparency will be printed after search execution (includes match counts)

        # Build index in RAM
        ix, total_indexed = self._build_index(messages)

        # No query terms? Quick exit
        if not keywords and not phrases:
            if transparency:
                print_md("Search-Log: no query terms — skipping search")
            return {
                "hits": [],
                "is_temporal": is_temporal,
                "stats": {"indexed": total_indexed, "matches": 0},
                "formatted_summary": None,
            }

        # Build query
        q = self._build_query(keywords, phrases)

        # Execute
        with ix.searcher() as searcher:
            results = searcher.search(q, limit=max_results)
            matches_count = len(results)
            if transparency:
                info = "Search-Log (engine):\n"
                info += f"    Keywords: {', '.join(keywords) if keywords else '(none)'}\n"
                info += f"    Phrases: {', '.join(phrases) if phrases else '(none)'}\n"
                info += f"    Temporal: {'yes' if is_temporal else 'no'}\n"
                info += f"    Matches: {matches_count} (indexed: {total_indexed})"
                print_md(info)

            if matches_count == 0:
                # Caller will handle "no results" policy (print only, do not append)
                return {
                    "hits": [],
                    "is_temporal": is_temporal,
                    "stats": {"indexed": total_indexed, "matches": 0},
                    "formatted_summary": None,
                }

            # Prepare highlighters
            sent_fragmenter = SentenceFragmenter()  # sentence-aware
            null_formatter = NullFormatter()

            # Pre-map turn_index -> position in indexable list for neighbor lookup
            turn_to_pos = self._build_turn_position_map(messages)

            # Collect hits with snippets
            hits: List[Dict[str, Any]] = []
            for hit in results:
                # Extract stored fields
                content = hit.get("content", "")
                ts = hit.get("timestamp")
                role = hit.get("role")
                turn_index = hit.get("turn_index", 0)
                message_id = hit.get("message_id")

                # Generate 1–2 sentence-aware fragments
                snippets = self._make_snippets(
                    hit,
                    content,
                    sent_fragmenter,
                    null_formatter,
                    keywords,
                    phrases,
                    max_fragments=snippet_count
                )

                # Optionally add ±N neighboring messages (user/assistant only), dedup by content
                if ctx_neighbors > 0:
                    neighbor_snips = self._neighbor_context(messages, turn_to_pos, turn_index, ctx_neighbors)
                    # Merge while avoiding duplicates
                    for s in neighbor_snips:
                        if s not in snippets:
                            snippets.append(s)

                date_str, time_str = self._format_ts(ts)

                hits.append({
                    "timestamp": ts,
                    "date": date_str,
                    "time": time_str,
                    "message_id": message_id,
                    "role": role,
                    "turn_index": turn_index,
                    "snippets": snippets[:snippet_count] if len(snippets) > snippet_count else snippets,
                })

        # Order/grouping
        if is_temporal:
            grouped = self._group_temporal(hits)
            formatted = self._format_temporal_summary(grouped, keywords, phrases)
        else:
            # Keep relevance order (as returned), format flat
            formatted = self._format_relevance_summary(hits, keywords, phrases)

        return {
            "hits": hits,
            "is_temporal": is_temporal,
            "stats": {"indexed": total_indexed, "matches": len(hits)},
            "formatted_summary": formatted,
        }

    # Indexing -----------------------------------------------------------------

    def _build_index(self, messages: List[IndexableMessage]):
        schema = Schema(
            content=TEXT(stored=True, analyzer=self._analyzer),
            timestamp=NUMERIC(stored=True, sortable=True),
            role=ID(stored=True),
            turn_index=NUMERIC(stored=True, sortable=True),
            message_id=NUMERIC(stored=True),
        )
        storage = RamStorage()
        ix = storage.create_index(schema)
        writer = ix.writer()
        indexed = 0
        for msg in messages:
            try:
                writer.add_document(
                    content=str(msg.get("content", "")),
                    timestamp=msg.get("timestamp"),
                    role=msg.get("role"),
                    turn_index=msg.get("turn_index", 0),
                    message_id=msg.get("message_id"),
                )
                indexed += 1
            except Exception:
                # Skip problematic documents
                continue
        writer.commit()
        return ix, indexed

    # Query building -----------------------------------------------------------

    def _build_query(self, keywords: List[str], phrases: List[str]) -> Q.Query:
        """
        Build a Whoosh query:
        - OR of all keyword Terms
        - Plus OR of Phrase queries for multi-word phrases
        - Phrase queries get a higher boost to prefer exact matches
        """
        terms: List[Q.Query] = []
        for kw in (keywords or []):
            kw = kw.strip()
            if not kw:
                continue
            terms.append(Q.Term("content", kw))

        # Phrase boost
        phrase_nodes: List[Q.Query] = []
        for ph in (phrases or []):
            # Split into tokens; skip if fewer than 2 words
            toks = [t for t in re.split(r"\s+", ph.strip()) if t]
            if len(toks) < 2:
                continue
            node = Q.Phrase("content", toks)
            node.boost = 2.0  # prefer exact multi-word matches
            phrase_nodes.append(node)

        # If only phrases exist, search by them; otherwise combine
        if terms and phrase_nodes:
            return Q.Or([Q.Or(terms), Q.Or(phrase_nodes)])
        elif phrase_nodes:
            return Q.Or(phrase_nodes)
        elif terms:
            return Q.Or(terms)
        else:
            # Should not happen because caller guards, but return a match-nothing query
            return Q.NullQuery()

    # Snippet generation -------------------------------------------------------

    def _make_snippets(
        self,
        hit,
        content: str,
        sent_fragmenter: SentenceFragmenter,
        formatter: NullFormatter,
        keywords: List[str],
        phrases: List[str],
        max_fragments: int = 2
    ) -> List[str]:
        """
        Try sentence-aware fragments first. If Whoosh returns empty (e.g., no sentence
        boundaries or odd content), fall back to line-based or bounded context.
        """
        snippets: List[str] = []

        try:
            frag_text = hit.highlights(
                "content",
                text=content,
                top=max_fragments,
                fragmenter=sent_fragmenter,
                formatter=formatter
            )
            frag_text = frag_text.strip()
            if frag_text:
                # Whoosh may concatenate multiple fragments with " ..." separators.
                # Split conservatively on newline boundaries first; if only one line, use ellipses as secondary split.
                parts = [p.strip() for p in frag_text.split("\n") if p.strip()]
                if len(parts) < 2 and "..." in frag_text:
                    parts = [p.strip() for p in frag_text.split("...") if p.strip()]
                # Keep up to max_fragments trimmed
                for p in parts:
                    if p and p not in snippets:
                        snippets.append(p)
                        if len(snippets) >= max_fragments:
                            break
        except Exception:
            # Ignore and fall back
            pass

        if snippets:
            return snippets

        # Fallback 1: line-based context (works better for code/log-like content)
        lines = content.splitlines()
        match_line_idx = self._find_first_match_line(lines, keywords, phrases)
        if match_line_idx is not None:
            start = max(0, match_line_idx - 1)
            end = min(len(lines), match_line_idx + 2)  # include ±1
            window = "\n".join(l.strip() for l in lines[start:end] if l.strip())
            if window:
                snippets.append(window)

        if snippets:
            return snippets[:max_fragments]

        # Fallback 2: bounded char window (last resort)
        window = self._bounded_char_window(content, keywords, phrases, radius=240)
        if window:
            snippets.append(window)

        return snippets[:max_fragments]

    def _find_first_match_line(self, lines: List[str], keywords: List[str], phrases: List[str]) -> Optional[int]:
        # Prefer phrase matches, then keywords
        lower_lines = [ln.lower() for ln in lines]
        # Phrases
        for ph in phrases or []:
            ph_l = ph.lower()
            for i, ln in enumerate(lower_lines):
                if ph_l in ln:
                    return i
        # Keywords
        for kw in keywords or []:
            kw_l = kw.lower()
            for i, ln in enumerate(lower_lines):
                if kw_l in ln:
                    return i
        return None

    def _bounded_char_window(self, text: str, keywords: List[str], phrases: List[str], radius: int = 240) -> Optional[str]:
        hay = text
        needle_pos = -1
        # Prefer phrases, then keywords
        for ph in phrases or []:
            pos = hay.lower().find(ph.lower())
            if pos != -1:
                needle_pos = pos
                break
        if needle_pos == -1:
            for kw in keywords or []:
                pos = hay.lower().find(kw.lower())
                if pos != -1:
                    needle_pos = pos
                    break
        if needle_pos == -1:
            return None

        start = max(0, needle_pos - radius)
        end = min(len(hay), needle_pos + radius)
        snippet = hay[start:end].strip()

        # Trim to word boundaries and add ellipses if clipped
        snippet = self._trim_to_word_boundaries(snippet)
        if start > 0:
            snippet = "... " + snippet
        if end < len(hay):
            snippet = snippet + " ..."
        return snippet

    def _trim_to_word_boundaries(self, text: str) -> str:
        # Trim leading/trailing partial words
        text = text.strip()
        text = re.sub(r"^\S{1,10}\b", lambda m: m.group(0) if " " not in text[:m.end()] else text[:m.end()], text)
        text = re.sub(r"\b\S{1,10}$", lambda m: m.group(0) if " " not in text[m.start():] else text[m.start():], text)
        return text

    # Neighboring messages -----------------------------------------------------

    def _build_turn_position_map(self, messages: List[IndexableMessage]) -> Dict[int, int]:
        # Build mapping from turn_index -> position in indexable list (sorted by turn_index)
        sorted_pairs = sorted(((m.get("turn_index", 0), i) for i, m in enumerate(messages)), key=lambda t: t[0])
        return {turn: pos for pos, (turn, i) in enumerate(sorted_pairs)}

    def _neighbor_context(
        self,
        messages: List[IndexableMessage],
        turn_to_pos: Dict[int, int],
        turn_index: int,
        ctx_neighbors: int
    ) -> List[str]:
        # Return content from ±N neighboring indexable messages (user/assistant only)
        snippets: List[str] = []
        # Build an ordered array by turn_index so neighbors are consistent
        ordered = sorted(messages, key=lambda m: m.get("turn_index", 0))
        # Find position of current turn_index
        pos = None
        for i, m in enumerate(ordered):
            if m.get("turn_index", 0) == turn_index:
                pos = i
                break
        if pos is None:
            return snippets
        start = max(0, pos - ctx_neighbors)
        end = min(len(ordered), pos + ctx_neighbors + 1)
        for i in range(start, end):
            if i == pos:
                continue
            content = str(ordered[i].get("content", "")).strip()
            if content:
                # Compress newlines for compactness
                compact = " ".join([ln.strip() for ln in content.splitlines() if ln.strip()])
                if compact and compact not in snippets:
                    snippets.append(compact)
        return snippets

    # Grouping/formatting ------------------------------------------------------

    def _group_temporal(self, hits: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        # Group by date (YYYY-MM-DD, None last), sort by time within date
        buckets: Dict[str, List[Dict[str, Any]]] = {}
        for h in hits:
            key = h.get("date") or "(no-date)"
            buckets.setdefault(key, []).append(h)

        # Sort keys with real dates ascending, then "(no-date)"
        def date_key(k: str):
            if k == "(no-date)":
                return (1, k)
            return (0, k)

        ordered: Dict[str, List[Dict[str, Any]]] = {}
        for d in sorted(buckets.keys(), key=date_key):
            items = buckets[d]
            # Within a date, order by time ascending where available
            def time_key(h: Dict[str, Any]):
                t = h.get("time")
                return ("~" if t is None else t)  # "~" sorts after numeric-like strings
            items_sorted = sorted(items, key=time_key)
            ordered[d] = items_sorted
        return ordered

    def _format_temporal_summary(self, grouped: Dict[str, List[Dict[str, Any]]], keywords: List[str], phrases: List[str]) -> str:
        lines: List[str] = []
        lines.append("Search-Log Summary:")
        if keywords or phrases:
            kline = []
            if keywords:
                kline.append("keywords: " + ", ".join(keywords))
            if phrases:
                kline.append("phrases: " + ", ".join(phrases))
            lines.append("  " + " | ".join(kline))
        lines.append("  Temporal: yes")

        for date_str, items in grouped.items():
            lines.append(f"  {date_str}")
            for h in items:
                t = h.get("time") or ""
                mid = h.get("message_id")
                role = h.get("role")
                # Each hit can have up to 2 snippets; print each as a bullet
                snips = h.get("snippets", []) or []
                if not snips:
                    # Fallback to a compact content if somehow empty
                    snips = ["(no snippet)"]
                for s in snips:
                    prefix = f"    - {t} (msg {mid}, {role}): " if t else f"    - (msg {mid}, {role}): "
                    lines.append(prefix + s)

        return "\n".join(lines)

    def _format_relevance_summary(self, hits: List[Dict[str, Any]], keywords: List[str], phrases: List[str]) -> str:
        lines: List[str] = []
        lines.append("Search-Log Summary:")
        if keywords or phrases:
            kline = []
            if keywords:
                kline.append("keywords: " + ", ".join(keywords))
            if phrases:
                kline.append("phrases: " + ", ".join(phrases))
            lines.append("  " + " | ".join(kline))
        lines.append("  Temporal: no")

        for h in hits:
            d = h.get("date") or ""
            t = h.get("time") or ""
            mid = h.get("message_id")
            role = h.get("role")
            snips = h.get("snippets", []) or []
            if not snips:
                snips = ["(no snippet)"]
            for s in snips:
                dt = f"{d} {t}".strip()
                if dt:
                    prefix = f"  - {dt} (msg {mid}, {role}): "
                else:
                    prefix = f"  - (msg {mid}, {role}): "
                lines.append(prefix + s)

        return "\n".join(lines)

    # Utilities ----------------------------------------------------------------

    def _format_ts(self, ts: Any) -> Tuple[Optional[str], Optional[str]]:
        if ts is None:
            return None, None
        try:
            # Assume unix timestamp (seconds)
            dt = datetime.datetime.fromtimestamp(float(ts))
            return dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M")
        except Exception:
            return None, None
