"""
KeywordExtractor for Search‑Log (v1)

Responsibilities:
- Generate stable keyword and phrase candidates from a user question using the active LLM
- Classify temporal intent (is the query time-oriented?)
- Provide robust fallbacks when LLM is unavailable or returns malformed output

Behavior:
- No exposed temperature/max-token knobs (kept intentionally simple for v1)
- Uses minimal recent context (optional) to improve keyword selection
- Prioritizes quality: returns both keywords and multi-word phrases
- Transparent printing (optional) via settings.search_log_transparency

Outputs (dict):
{
    "keywords": List[str],     # single tokens/terms
    "phrases": List[str],      # multi-word phrases to boost as exact matches
    "is_temporal": bool,       # true if query implies timeline/date ordering
    "raw": str                 # raw LLM JSON (if available) for debugging
}
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Any

from print_helper import print_md
from settings_manager import SettingsManager







class KeywordExtractor:
    """
    Extracts search keywords/phrases and temporal intent for Search‑Log.

    Usage:
        extractor = KeywordExtractor(llm_client_manager, SettingsManager.getInstance())
        result = extractor.extract(user_question, recent_messages=[...])
        # result: { "keywords": [...], "phrases": [...], "is_temporal": bool, "raw": "..." }
    """

    def __init__(self, llm_client_manager, settings_manager: SettingsManager) -> None:
        self._llm = llm_client_manager
        self._settings = settings_manager

    def extract(
        self,
        user_question: str,
        recent_messages: Optional[List[Dict[str, Any]]] = None,
        max_keywords: int = 32,
        max_phrases: int = 16,
    ) -> Dict[str, Any]:
        """
        Extract keywords/phrases and temporal intent.

        Args:
            user_question: The user's query text
            recent_messages: Optional list of recent conversation messages (role/content) to provide minimal context
            max_keywords: Soft cap on returned single-token keywords
            max_phrases: Soft cap on returned multi-word phrases

        Returns:
            Dict with fields: keywords, phrases, is_temporal, raw
        """
        recent_messages = recent_messages or []
        transparency = bool(getattr(self._settings, "search_log_transparency", True))

        # Try LLM extraction first
        try:
            llm_out = self._extract_via_llm(user_question, recent_messages, max_keywords, max_phrases)
            if transparency:
                self._print_transparency(llm_out, header="Search-Log (extractor: LLM)")
            return llm_out
        except Exception as e:
            if transparency:
                print_md(f"Search-Log (extractor): LLM extraction failed; using fallback\n    {e}")

        # Fallback: rule-based extraction
        fallback_out = self._extract_via_rules(user_question, recent_messages, max_keywords, max_phrases)
        if transparency:
            self._print_transparency(fallback_out, header="Search-Log (extractor: fallback)")
        return fallback_out

    # -----------------------
    # LLM-based implementation
    # -----------------------
    def _extract_via_llm(
        self,
        user_question: str,
        recent_messages: List[Dict[str, Any]],
        max_keywords: int,
        max_phrases: int,
    ) -> Dict[str, Any]:
        model = self._settings.setting_get("model")

        # Condense minimal context: last few user/assistant turns (shortened)
        condensed_context = self._build_condensed_context(recent_messages, max_chars=1000)

        system_prompt = (
            "You generate search terms for a local full-text search over a single conversation log.\n"
            "- Output must be STRICT JSON with keys: keywords (list of strings), phrases (list of strings), is_temporal (boolean).\n"
            "- keywords: single-word or short tokens; do not include stopwords or punctuation.\n"
            "- phrases: multi-word phrases to match exactly; prefer critical noun phrases.\n"
            "- is_temporal: true if the question implies a time-based grouping (e.g., asking 'when', dates, or chronological sequence).\n"
            "- Do NOT include explanations. Only return the JSON object.\n"
        )

        user_block = "User question:\n" + user_question.strip()
        if condensed_context:
            user_block += "\n\nRecent context (truncated):\n" + condensed_context

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_block},
        ]

        resp = self._llm.create_chat_completion(model=model, messages=messages, temperature=0.0)
        raw_text = self._extract_content_text(resp)

        data = self._parse_json_strict(raw_text)

        # Basic validation and caps
        keywords = self._normalize_terms(data.get("keywords", []))[:max_keywords]
        phrases = self._normalize_phrases(data.get("phrases", []))[:max_phrases]
        is_temporal = bool(data.get("is_temporal", False))

        # If LLM returned nothing meaningful, raise to trigger fallback
        if not keywords and not phrases:
            raise ValueError("Empty LLM keyword/phrase output")

        return {
            "keywords": keywords,
            "phrases": phrases,
            "is_temporal": is_temporal,
            "raw": raw_text or "",
        }

    # -----------------------
    # Fallback implementation
    # -----------------------
    def _extract_via_rules(
        self,
        user_question: str,
        recent_messages: List[Dict[str, Any]],
        max_keywords: int,
        max_phrases: int,
    ) -> Dict[str, Any]:
        text = user_question.lower()

        # Heuristic temporal detection
        is_temporal = self._heuristic_is_temporal(text)

        # Extract phrases: naive quoted phrases + basic n-gram detection on key segments
        phrases: List[str] = []
        phrases.extend(self._extract_quoted_phrases(text))
        # Minimal approach: pull simple bi-grams around key terms (very conservative)
        phrases = self._dedup_preserve_order([p for p in phrases if p.strip()])[:max_phrases]

        # Extract keywords: split on non-letters/digits, filter stopwords and short tokens
        tokens = [t for t in re.split(r"[^\w\-]+", text) if t]
        keywords = []
        for tok in tokens:
            if len(tok) < 2 and not tok.isdigit():
                continue
            keywords.append(tok)

        keywords = self._dedup_preserve_order(keywords)[:max_keywords]

        return {
            "keywords": keywords,
            "phrases": phrases,
            "is_temporal": is_temporal,
            "raw": "",
        }

    # -----------------------
    # Helpers
    # -----------------------
    def _build_condensed_context(self, messages: List[Dict[str, Any]], max_chars: int = 1000) -> str:
        # Include only last few user/assistant messages
        filtered = [m for m in messages if m.get("role") in ("user", "assistant")]
        # Take the last 4 messages (2 user + 2 assistant typically)
        tail = filtered[-4:]
        # Concatenate role-tagged short lines
        parts: List[str] = []
        for m in tail:
            role = m.get("role", "user")
            content = str(m.get("content", ""))[:256]
            parts.append(f"{role}: {content}")
        ctx = "\n".join(parts)
        return ctx[:max_chars]

    def _extract_content_text(self, resp: Any) -> str:
        """
        Extract text content from an OpenAI-compatible Chat Completions response.
        """
        try:
            # OpenAI-compatible schema (used for OpenAI, Ollama, and Google adapters here)
            return resp.choices[0].message.content or ""
        except Exception:
            pass
        # Fallback best-effort
        try:
            return str(resp)
        except Exception:
            return ""

    def _parse_json_strict(self, raw_text: str) -> Dict[str, Any]:
        """
        Parse a JSON object string strictly. Attempt light repair if needed.
        """
        raw_text = raw_text.strip()
        if not raw_text:
            raise ValueError("Empty LLM output")

        # If the model wrapped JSON in markdown fences, strip them
        fenced = re.match(r"^```(?:json)?\s*(\{.*\})\s*```$", raw_text, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            raw_text = fenced.group(1)

        # Try direct json.loads
        try:
            return json.loads(raw_text)
        except Exception:
            pass

        # Try to extract the first {...} block and parse
        try:
            brace_start = raw_text.find("{")
            brace_end = raw_text.rfind("}")
            if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
                candidate = raw_text[brace_start:brace_end + 1]
                return json.loads(candidate)
        except Exception:
            pass

        # Try json_repair if available
        try:
            import json_repair
            return json_repair.loads(raw_text)
        except Exception:
            pass

        raise ValueError("Failed to parse JSON from LLM output")

    def _normalize_terms(self, items: Any) -> List[str]:
        out: List[str] = []
        if not isinstance(items, list):
            return out
        for it in items:
            if not isinstance(it, str):
                continue
            it = it.strip()
            if not it:
                continue
            # Single-token keywords only
            if " " in it:
                continue
            out.append(it)
        return self._dedup_preserve_order(out)

    def _normalize_phrases(self, items: Any) -> List[str]:
        out: List[str] = []
        if not isinstance(items, list):
            return out
        for it in items:
            if not isinstance(it, str):
                continue
            it = " ".join(it.strip().split())
            if not it:
                continue
            # Multi-word phrases only
            if " " not in it:
                continue
            out.append(it)
        return self._dedup_preserve_order(out)

    def _heuristic_is_temporal(self, text: str) -> bool:
        # Language-agnostic temporal detection for fallback:
        # Look for numeric date/time patterns only.
        date_patterns = [
            r"\b\d{4}[-/.]\d{1,2}[-/.]\d{1,2}\b",  # YYYY-MM-DD / YYYY/MM/DD / YYYY.MM.DD
            r"\b\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\b",  # DD-MM-YYYY or DD/MM/YY etc.
            r"\b\d{1,2}:\d{2}\b",                   # HH:MM
        ]
        for pat in date_patterns:
            if re.search(pat, text, flags=re.IGNORECASE):
                return True
        return False

    def _extract_quoted_phrases(self, text: str) -> List[str]:
        # Pull phrases in double or single quotes
        phrases = re.findall(r"\"([^\"]{2,})\"", text)
        phrases += re.findall(r"'([^']{2,})'", text)
        # Normalize whitespace
        phrases = [" ".join(p.split()) for p in phrases]
        # Keep multi-word only
        return [p for p in phrases if " " in p]

    def _dedup_preserve_order(self, items: List[str]) -> List[str]:
        seen = set()
        out: List[str] = []
        for it in items:
            if it not in seen:
                seen.add(it)
                out.append(it)
        return out

    def _print_transparency(self, result: Dict[str, Any], header: str) -> None:
        try:
            keywords = result.get("keywords", [])
            phrases = result.get("phrases", [])
            is_temporal = result.get("is_temporal", False)
            text = f"{header}:\n"
            text += f"    Keywords: {', '.join(keywords) if keywords else '(none)'}\n"
            text += f"    Phrases: {', '.join(phrases) if phrases else '(none)'}\n"
            text += f"    Temporal: {'yes' if is_temporal else 'no'}"
            print_md(text)
        except Exception:
            # Never let transparency fail the flow
            pass