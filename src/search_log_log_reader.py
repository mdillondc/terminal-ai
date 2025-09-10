#!/usr/bin/env python3
"""
Search-Log LogReader (v1)

Reads the current conversation JSON log from disk (source of truth) and returns
indexable messages for the Search‑Log pipeline. Indexable messages include only
roles: user and assistant. System messages are excluded in v1.

Responsibilities:
- Resolve the active log file path
- Parse the JSON array of messages (fresh read per search)
- Produce a normalized list of indexable message dicts:
  {
    "content": str,
    "timestamp": int | float | None,
    "role": "user" | "assistant",
    "turn_index": int,         # original position in the full conversation array
    "message_id": int | None
  }
- Optional transparency: print counts when enabled

Notes:
- We never write to the log file here; appending of search results occurs via
  the conversation manager (standard path), not direct file I/O.
- If no active log file is available (incognito or before first AI response),
  methods return empty results and optionally print a helpful message.

Usage:
    from settings_manager import SettingsManager
    from search_log_log_reader import SearchLogLogReader

    sm = SettingsManager.getInstance()
    reader = SearchLogLogReader(sm)
    msgs = reader.get_indexable_messages()
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple, Iterable

from settings_manager import SettingsManager
from print_helper import print_md


IndexableMessage = Dict[str, Any]


class SearchLogLogReader:
    def __init__(self, settings_manager: Optional[SettingsManager] = None) -> None:
        self._settings = settings_manager or SettingsManager.getInstance()

    # Public API --------------------------------------------------------------

    def get_indexable_messages(self) -> List[IndexableMessage]:
        """
        Read the active JSON log from disk and return indexable messages
        (roles: user and assistant only) with normalized fields.

        Returns:
            List of dicts with keys:
            - content (str)
            - timestamp (int|float|None)
            - role ("user"|"assistant")
            - turn_index (int)  original array position
            - message_id (int|None)
        """
        path = self.get_active_log_path()
        if not path:
            self._maybe_print_no_log()
            return []

        all_messages = self._read_messages_from_disk(path)
        if not all_messages:
            return []

        indexable, total = self._filter_and_normalize(all_messages)
        self._maybe_print_counts(total_messages=total, indexable_messages=len(indexable), path=path)
        return indexable

    def get_active_log_path(self) -> Optional[str]:
        """
        Resolve the active JSON log file path.

        Strategy:
        1) Prefer settings.log_file_location if present and exists.
           - If missing ".json" but "{path}.json" exists, use the latter.
        2) Fallback: construct from instructions + log_file_name
           working_dir/logs/{instructions_base}/{log_file_name_wo_ext}.json
        3) If not found, return None.
        """
        # Incognito: do not expect a log file
        try:
            if bool(self._settings.setting_get("incognito")):
                return None
        except Exception:
            pass

        # 1) Direct setting
        try:
            candidate = self._settings.setting_get("log_file_location")
        except Exception:
            candidate = None

        # Normalize candidate (handle missing extension)
        path = self._resolve_candidate_path(candidate)
        if path:
            return path

        # 2) Fallback by instructions + log_file_name
        try:
            working_dir = self._settings.setting_get("working_dir")
            instructions = self._settings.setting_get("instructions")  # e.g., "samantha.md"
            log_file_name = self._settings.setting_get("log_file_name")  # may end with .md
            instructions_base = instructions.rsplit(".", 1)[0] if instructions and "." in instructions else (instructions or "default")
            base_name = log_file_name[:-3] if (log_file_name and log_file_name.endswith(".md")) else (log_file_name or "")
            if not base_name:
                return None
            fallback = os.path.join(working_dir, "logs", instructions_base, base_name + ".json")
            if os.path.exists(fallback):
                return fallback
        except Exception:
            pass

        return None

    # Internals --------------------------------------------------------------

    def _resolve_candidate_path(self, candidate: Optional[str]) -> Optional[str]:
        if not candidate:
            return None
        # If candidate exists as-is, prefer it
        if os.path.exists(candidate):
            return candidate
        # If it lacks .json but "{candidate}.json" exists, use that
        if not candidate.endswith(".json"):
            alt = candidate + ".json"
            if os.path.exists(alt):
                return alt
        return None

    def _read_messages_from_disk(self, path: str) -> List[Dict[str, Any]]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print_md(f"Search-Log (log-reader): failed to read log file\n    {e}")
            return []

        if not isinstance(data, list):
            print_md("Search-Log (log-reader): log file is not a JSON array")
            return []

        # Normalize and attach original position as turn_index if missing
        normalized: List[Dict[str, Any]] = []
        for i, msg in enumerate(data):
            if not isinstance(msg, dict):
                continue
            # Only add turn_index if missing to preserve existing field if present
            if "turn_index" not in msg:
                msg = dict(msg)  # shallow copy to avoid mutating original structure
                msg["turn_index"] = i
            normalized.append(msg)
        return normalized

    def _filter_and_normalize(self, all_messages: List[Dict[str, Any]]) -> Tuple[List[IndexableMessage], int]:
        indexable: List[IndexableMessage] = []
        total = len(all_messages)

        for msg in all_messages:
            role = msg.get("role")
            if role not in ("user", "assistant"):
                continue

            content = msg.get("content", "")
            # Normalize to string
            try:
                content = str(content)
            except Exception:
                content = ""

            indexable.append({
                "content": content,
                "timestamp": msg.get("timestamp"),
                "role": role,
                "turn_index": msg.get("turn_index", 0),
                "message_id": msg.get("message_id"),
            })

        return indexable, total

    def _maybe_print_no_log(self) -> None:
        try:
            if bool(self._settings.setting_get("search_log_transparency")):
                print_md("Search-Log (log-reader): no active log file to search (incognito or no AI response yet)")
        except Exception:
            pass

    def _maybe_print_counts(self, total_messages: int, indexable_messages: int, path: str) -> None:
        try:
            if bool(self._settings.setting_get("search_log_transparency")):
                text = f"Search-Log (log-reader): loaded log\n"
                text += f"    Path: {path}\n"
                text += f"    Messages: {total_messages} total, {indexable_messages} indexable (user+assistant)"
                print_md(text)
        except Exception:
            # Never fail on transparency
            pass