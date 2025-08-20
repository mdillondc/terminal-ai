"""
RAG Query Analyzer Module - Simplified

Simple LLM-based query analysis that only determines if user wants recent/fresh information.
"""

import json
from typing import Dict, Any
from llm_client_manager import LLMClientManager
from settings_manager import SettingsManager
from print_helper import print_md

class RAGQueryAnalyzer:
    """
    Simple query analyzer that determines if user wants recent/fresh information.
    """

    def __init__(self, llm_client_manager: LLMClientManager):
        self.llm_client_manager = llm_client_manager
        self.settings_manager = SettingsManager.getInstance()
        self.analysis_temperature = 0.1

    def is_recent_query(self, query: str) -> bool:
        """
        Determine if a query wants recent/fresh information using LLM analysis

        Args:
            query: The search query

        Returns:
            True if query wants recent/fresh information, False otherwise
        """
        if not self.llm_client_manager:
            # Fallback to simple keyword detection
            return self._fallback_recent_detection(query)

        transparency_enabled = self.settings_manager.setting_get("rag_enable_search_transparency")

        try:
            prompt = f"""Does this query want recent, latest, current, or fresh information?

Query: "{query}"

Consider these examples:
- "latest email" → YES (wants most recent)
- "newest document" → YES (wants most recent)
- "recent changes" → YES (wants recent updates)
- "current status" → YES (wants current state)
- "what happened lately" → YES (wants recent events)
- "what is photosynthesis" → NO (general knowledge)
- "how to cook pasta" → NO (general instructions)
- "definition of democracy" → NO (general information)
- "Charlie's medical history" → NO (wants complete history)

Respond with only: YES or NO"""

            response = self.llm_client_manager.create_chat_completion(
                model=self.settings_manager.setting_get("model"),
                messages=[
                    {"role": "system", "content": "You are an expert at determining if queries want recent information. Respond only with YES or NO."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.analysis_temperature,
                max_tokens=10
            )

            content = response.choices[0].message.content
            if content is None:
                # Content is None, fall back to keyword detection
                return self._fallback_recent_detection(query)

            result = content.strip().upper() == "YES"

            # Display transparency information if enabled
            if transparency_enabled:
                search_type = "recent results" if result else "all results"
                print_md(f"Searching {search_type}...")

            return result

        except Exception as e:
            if transparency_enabled:
                print_md(f"Error in recent detection: {e}, using fallback")
            # Fallback to simple keyword detection
            return self._fallback_recent_detection(query)

    def _fallback_recent_detection(self, query: str) -> bool:
        """
        Simple keyword-based fallback for recent detection

        Args:
            query: The search query

        Returns:
            True if query appears to want recent information
        """
        query_lower = query.lower()
        recent_keywords = [
            'latest', 'newest', 'recent', 'last', 'current', 'now',
            'today', 'yesterday', 'this week', 'this month', 'lately'
        ]

        return any(keyword in query_lower for keyword in recent_keywords)