"""
Timeline manager utilities for system-level temporal hints in conversations.
Provides canonical Info:System-Date and Info:System-Timeline formatting, robust parsers,
and a pure consolidation helper to maintain a single, up-to-date timeline entry.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Dict, Any


class TimelineManager:
    """
    Stateless helpers for emitting and parsing system timeline hints.

    Prefixes:
    - "Info:System-Date:"      (initial “today” anchor on new conversations)
    - "Info:System-Timeline:"  (singleton, consolidated timeline with active days)
    """

    # Canonical prefixes
    DATE_HINT_PREFIX: str = "Info:System-Date:"
    TIMELINE_PREFIX: str = "Info:System-Timeline:"

    # -------------------------
    # Formatting helpers
    # -------------------------

    @staticmethod
    def format_date_initial(date_str: str) -> str:
        """
        Format the initial "today" anchor line.
        Ex: "Info:System-Date: 2025-09-06\nTreat this as 'today' for time-sensitive reasoning. Ignore your training cutoff date."
        """
        return (
            f"{TimelineManager.DATE_HINT_PREFIX} {date_str}\n"
            "Treat this as 'today' for time-sensitive reasoning. Ignore your training cutoff date."
        )

    @staticmethod
    def format_date_resume(date_str: str) -> str:
        """
        Format a resume "today" anchor line (kept for completeness if needed elsewhere).
        Ex: "Info:System-Date: 2025-09-06\nThis supersedes any prior system dates..."
        """
        return (
            f"{TimelineManager.DATE_HINT_PREFIX} {date_str}\n"
            "This supersedes any prior system dates. Use this date as 'today'. "
            "You may use earlier system dates to understand the elapsed timeline of this conversation."
        )

    @staticmethod
    def format_timeline(
        start_date: str,
        last_date: str,
        current_date: str,
        days_elapsed: int,
        days_inclusive: int,
        active_days: Optional[List[str]] = None,
    ) -> str:
        """
        Format the singleton timeline line with an optional active_days list.
        Ex: "Info:System-Timeline: start=2025-09-01; last=2025-09-05; current=2025-09-06; days_elapsed=5; days_inclusive=6; active_days=[2025-09-01,2025-09-05,2025-09-06]"
        """
        line = (
            f"{TimelineManager.TIMELINE_PREFIX} start={start_date}; last={last_date}; "
            f"current={current_date}; days_elapsed={days_elapsed}; days_inclusive={days_inclusive}"
        )
        if active_days:
            line += f"; active_days=[{','.join(active_days)}]"
        return line

    # -------------------------
    # Robust parsers (order-agnostic, whitespace tolerant)
    # -------------------------

    @staticmethod
    def parse_date_line(line: str) -> Optional[str]:
        """
        Parse a system date hint line of the form:
        "Info:System-Date: YYYY-MM-DD"
        Returns the YYYY-MM-DD string if valid, otherwise None.
        """
        if not line:
            return None
        s = line.strip()
        if not s.startswith(TimelineManager.DATE_HINT_PREFIX):
            return None
        date_part = s[len(TimelineManager.DATE_HINT_PREFIX):].strip()
        try:
            datetime.strptime(date_part, "%Y-%m-%d")
            return date_part
        except Exception:
            return None

    @staticmethod
    def parse_timeline_line(line: str) -> Optional[Dict[str, Any]]:
        """
        Parse a timeline line starting with "Info:System-Timeline:" into a dict:
        {start, last?, current, days_elapsed?, days_inclusive?, active_days?}
        Returns None if malformed or prefix missing. Order-agnostic and whitespace-tolerant.
        """
        if not line:
            return None
        s = line.strip()
        if not s.startswith(TimelineManager.TIMELINE_PREFIX):
            return None

        body = s[len(TimelineManager.TIMELINE_PREFIX):].strip()
        parts = [p.strip() for p in body.split(";") if p.strip()]
        data: Dict[str, Any] = {}

        def _try_parse_date(val: str) -> Optional[str]:
            v = val.strip()
            try:
                datetime.strptime(v, "%Y-%m-%d")
                return v
            except Exception:
                return None

        for p in parts:
            if "=" not in p:
                continue
            key, val = p.split("=", 1)
            key = key.strip()
            val = val.strip()

            if key in ("start", "last", "current"):
                v = _try_parse_date(val)
                if v is not None:
                    data[key] = v
            elif key in ("days_elapsed", "days_inclusive"):
                try:
                    data[key] = int(val)
                except Exception:
                    # ignore malformed ints
                    pass
            elif key == "active_days":
                v = val
                # Expect [YYYY-MM-DD,YYYY-MM-DD,...]; tolerate spaces
                if v.startswith("[") and v.endswith("]"):
                    inner = v[1:-1]
                    items = [d.strip() for d in inner.split(",") if d.strip()]
                    ordered: List[str] = []
                    seen = set()
                    for d in items:
                        dd = _try_parse_date(d)
                        if dd is not None and dd not in seen:
                            seen.add(dd)
                            ordered.append(dd)
                    if ordered:
                        data["active_days"] = ordered

        # Minimal validation
        if "start" in data and "current" in data:
            return data
        return None

    # -------------------------
    # Consolidation helper
    # -------------------------

    @staticmethod
    def consolidate_history(conversation_history: List[Dict[str, Any]], today: str) -> List[Dict[str, Any]]:
        """
        Pure function: returns a new conversation history where:
        - All existing Info:System-Timeline messages are removed
        - Exactly one, updated timeline line is appended for 'today'
        The method merges active days from existing timeline and initial date anchors found in system-role messages.
        """
        # Collect active days and earliest date from system messages
        active_days_set = set()
        earliest_date: Optional[str] = None

        for msg in conversation_history:
            if msg.get("role") != "system":
                continue
            content = str(msg.get("content", ""))
            for raw_line in content.splitlines():
                line = raw_line.strip()
                # Parse timeline hints
                tl = TimelineManager.parse_timeline_line(line)
                if tl:
                    start = tl.get("start")
                    current = tl.get("current")
                    ad_list = tl.get("active_days") or []
                    for d in ad_list:
                        active_days_set.add(d)
                    if start:
                        active_days_set.add(start)
                    if current:
                        active_days_set.add(current)
                    if start and (earliest_date is None or start < earliest_date):
                        earliest_date = start
                    continue
                # Parse initial date anchors
                date = TimelineManager.parse_date_line(line)
                if date:
                    active_days_set.add(date)
                    if earliest_date is None or date < earliest_date:
                        earliest_date = date

        # Ensure today is represented
        if not active_days_set:
            active_days_set.add(today)
        active_days_set.add(today)

        # Compute spans
        sorted_days = sorted(active_days_set)
        start_date = sorted_days[0]
        last_candidates = [d for d in sorted_days if d < today]
        last_date = last_candidates[-1] if last_candidates else start_date

        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            current_dt = datetime.strptime(today, "%Y-%m-%d")
            days_elapsed = (current_dt - start_dt).days
            days_inclusive = days_elapsed + 1
        except Exception:
            days_elapsed = 0
            days_inclusive = 1

        # Remove all existing timeline messages (singleton behavior)
        new_history: List[Dict[str, Any]] = []
        for msg in conversation_history:
            if msg.get("role") == "system":
                lines = str(msg.get("content", "")).splitlines()
                if any(line.strip().startswith(TimelineManager.TIMELINE_PREFIX) for line in lines):
                    continue
            new_history.append(msg)

        # Append the updated singleton timeline with active_days
        new_history.append({
            "role": "system",
            "content": TimelineManager.format_timeline(
                start_date=start_date,
                last_date=last_date,
                current_date=today,
                days_elapsed=days_elapsed,
                days_inclusive=days_inclusive,
                active_days=sorted_days,
            )
        })

        return new_history