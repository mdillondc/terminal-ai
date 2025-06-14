"""
RAG Query Analyzer Module

LLM-based intelligent query analysis for RAG systems.
"""

import json
import time
from typing import Dict, Any, List
from openai import OpenAI
from settings_manager import SettingsManager
from llm_client_manager import LLMClientManager
from print_helper import print_info

class RAGQueryAnalyzer:
    """
    LLM-based query analyzer that intelligently determines query intent,
    temporal relevance, and optimal search strategies.
    """

    def __init__(self, llm_client_manager: LLMClientManager):
        self.llm_client_manager = llm_client_manager
        self.settings_manager = SettingsManager.getInstance()
        self._load_settings()

    def _load_settings(self):
        """Load analyzer settings from settings manager"""
        # Analysis settings
        self.enabled = self.settings_manager.setting_get("rag_enable_llm_query_analysis") if hasattr(self.settings_manager, 'rag_enable_llm_query_analysis') else True
        self.analysis_temperature = getattr(self.settings_manager, 'rag_query_analysis_temperature', 0.1)
        self.cache_duration_seconds = getattr(self.settings_manager, 'rag_query_analysis_cache_duration', 300)  # 5 minutes

        # Thresholds
        self.temporal_threshold = getattr(self.settings_manager, 'rag_temporal_detection_threshold', 0.5)
        self.confidence_threshold = getattr(self.settings_manager, 'rag_analysis_confidence_threshold', 0.7)

        # Cache for analysis results
        self._analysis_cache: Dict[str, Dict[str, Any]] = {}

    def _get_analysis_prompt(self, query: str, domain: str = "general") -> str:
        """
        Generate the analysis prompt for the LLM

        Args:
            query: The user's query to analyze
            domain: The domain context (e.g., "medical", "general", "technical")

        Returns:
            Formatted prompt for LLM analysis
        """
        return f"""Analyze this search query and provide structured information about its intent and characteristics.

Query: "{query}"
Domain: {domain}

Please analyze the query and respond with a JSON object containing:

1. "intent_type": The primary intent (choose from: "temporal", "factual", "comparative", "procedural", "analytical", "navigational")

2. "temporal_strength": A score from 0.0 to 1.0 indicating how much this query is asking for recent/latest information
   - 1.0: Clearly asking for most recent/latest (e.g., "latest visit", "most recent results")
   - 0.7-0.9: Strong temporal component (e.g., "recent changes", "current status")
   - 0.3-0.6: Some temporal aspect (e.g., "when did", "how long ago")
   - 0.0-0.2: No temporal focus (e.g., "what is", "how to")

3. "key_entities": List of important nouns/concepts to search for (max 5)

4. "temporal_indicators": List of words/phrases that suggest temporal intent

5. "search_strategy": Recommended approach ("semantic_only", "temporal_boost", "hybrid", "chronological")

6. "confidence": Your confidence in this analysis (0.0-1.0)

7. "reasoning": Brief explanation of your analysis

Examples:

Query: "What is the capital of France?"
{{"intent_type": "factual", "temporal_strength": 0.0, "key_entities": ["capital", "France"], "temporal_indicators": [], "search_strategy": "semantic_only", "confidence": 0.95, "reasoning": "Simple factual query with no temporal component"}}

Query: "When was our last visit to the vet?"
{{"intent_type": "temporal", "temporal_strength": 0.9, "key_entities": ["visit", "vet"], "temporal_indicators": ["last", "when"], "search_strategy": "temporal_boost", "confidence": 0.9, "reasoning": "Clear temporal query asking for most recent occurrence"}}

Query: "What are the recent changes in the treatment protocol?"
{{"intent_type": "temporal", "temporal_strength": 0.8, "key_entities": ["changes", "treatment", "protocol"], "temporal_indicators": ["recent"], "search_strategy": "hybrid", "confidence": 0.85, "reasoning": "Asking for recent updates, strong temporal component"}}

Now analyze the given query and respond with ONLY the JSON object:"""

    def _is_cache_valid(self, cache_entry: Dict[str, Any]) -> bool:
        """Check if a cached analysis is still valid"""
        if 'timestamp' not in cache_entry:
            return False

        age_seconds = time.time() - cache_entry['timestamp']
        return age_seconds < self.cache_duration_seconds

    def analyze_query(self, query: str, domain: str = "general", use_cache: bool = True) -> Dict[str, Any]:
        """
        Analyze a query using LLM to determine intent and characteristics

        Args:
            query: The search query to analyze
            domain: Domain context for better analysis
            use_cache: Whether to use cached results if available

        Returns:
            Dictionary with analysis results
        """
        if not self.enabled:
            # Fallback to simple analysis if LLM analysis is disabled
            return self._simple_fallback_analysis(query)

        # Check cache first
        cache_key = f"{query.lower().strip()}|{domain}"
        if use_cache and cache_key in self._analysis_cache:
            cached_result = self._analysis_cache[cache_key]
            if self._is_cache_valid(cached_result):
                return cached_result['analysis']

        try:
            print_info("Analyzing RAG query for intent (temporal, factual, analytical, etc)...")
            # Get analysis from LLM
            prompt = self._get_analysis_prompt(query, domain)

            response = self.llm_client_manager.create_chat_completion(
                model=self.settings_manager.setting_get("model"),
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing search queries. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.analysis_temperature,
                max_tokens=500
            )

            # Parse the JSON response
            content = response.choices[0].message.content
            if content is None:
                raise ValueError("Empty response from LLM")
            analysis_text = content.strip()
            analysis = json.loads(analysis_text)

            # Validate and clean the analysis
            analysis = self._validate_analysis(analysis)

            # Cache the result
            if use_cache:
                self._analysis_cache[cache_key] = {
                    'analysis': analysis,
                    'timestamp': time.time()
                }

            return analysis

        except Exception as e:
            print_info(f"Error in rag_query_analyzer: {e}")
            # Fallback to simple analysis on error
            return self._simple_fallback_analysis(query)

    def _validate_analysis(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and clean the LLM analysis results

        Args:
            analysis: Raw analysis from LLM

        Returns:
            Validated and cleaned analysis
        """
        # Ensure required fields exist with defaults
        defaults = {
            'intent_type': 'factual',
            'temporal_strength': 0.0,
            'key_entities': [],
            'temporal_indicators': [],
            'search_strategy': 'semantic_only',
            'confidence': 0.5,
            'reasoning': 'Analysis completed'
        }

        for key, default_value in defaults.items():
            if key not in analysis:
                analysis[key] = default_value

        # Clamp numeric values
        analysis['temporal_strength'] = max(0.0, min(1.0, float(analysis.get('temporal_strength', 0.0))))
        analysis['confidence'] = max(0.0, min(1.0, float(analysis.get('confidence', 0.5))))

        # Validate intent type
        valid_intents = ['temporal', 'factual', 'comparative', 'procedural', 'analytical', 'navigational']
        if analysis.get('intent_type') not in valid_intents:
            analysis['intent_type'] = 'factual'

        # Validate search strategy
        valid_strategies = ['semantic_only', 'temporal_boost', 'hybrid', 'chronological']
        if analysis.get('search_strategy') not in valid_strategies:
            analysis['search_strategy'] = 'semantic_only'

        # Ensure lists are actually lists
        for list_field in ['key_entities', 'temporal_indicators']:
            if not isinstance(analysis.get(list_field), list):
                analysis[list_field] = []

        return analysis

    def _simple_fallback_analysis(self, query: str) -> Dict[str, Any]:
        """
        Simple keyword-based fallback analysis when LLM is not available

        Args:
            query: The search query

        Returns:
            Basic analysis based on simple rules
        """
        query_lower = query.lower()

        # Simple temporal keyword detection
        temporal_keywords = [
            'latest', 'recent', 'newest', 'last', 'most recent', 'current',
            'when', 'today', 'yesterday', 'this week', 'this month',
            'now', 'just', 'lately'
        ]

        temporal_matches = sum(1 for keyword in temporal_keywords if keyword in query_lower)
        total_words = len(query.split())
        temporal_strength = min(1.0, temporal_matches / max(total_words, 1) * 2)  # Amplify for simple detection

        # Determine intent and strategy
        if temporal_strength > 0.3:
            intent_type = 'temporal'
            search_strategy = 'temporal_boost' if temporal_strength > 0.6 else 'hybrid'
        else:
            intent_type = 'factual'
            search_strategy = 'semantic_only'

        return {
            'intent_type': intent_type,
            'temporal_strength': temporal_strength,
            'key_entities': query.split()[:5],  # Simple: just use query words
            'temporal_indicators': [kw for kw in temporal_keywords if kw in query_lower],
            'search_strategy': search_strategy,
            'confidence': 0.6,  # Lower confidence for fallback
            'reasoning': 'Fallback analysis using simple keyword matching'
        }

    def is_temporal_query(self, query: str, domain: str = "general") -> bool:
        """
        Determine if a query has temporal intent

        Args:
            query: The search query
            domain: Domain context

        Returns:
            True if the query appears to be asking for temporal information
        """
        analysis = self.analyze_query(query, domain)
        return analysis.get('temporal_strength', 0.0) >= self.temporal_threshold

    def get_search_weights(self, query: str, domain: str = "general") -> Dict[str, float]:
        """
        Get recommended search weights based on query analysis

        Args:
            query: The search query
            domain: Domain context

        Returns:
            Dictionary with recommended weights for different search components
        """
        analysis = self.analyze_query(query, domain)
        temporal_strength = analysis.get('temporal_strength', 0.0)
        search_strategy = analysis.get('search_strategy', 'semantic_only')

        if search_strategy == 'semantic_only':
            return {'semantic': 1.0, 'temporal': 0.0, 'keyword': 0.0}
        elif search_strategy == 'temporal_boost':
            return {
                'semantic': 0.4,
                'temporal': 0.5,
                'keyword': 0.1
            }
        elif search_strategy == 'chronological':
            return {
                'semantic': 0.2,
                'temporal': 0.7,
                'keyword': 0.1
            }
        else:  # hybrid
            # Dynamic weighting based on temporal strength
            semantic_weight = max(0.3, 1.0 - temporal_strength)
            temporal_weight = min(0.6, temporal_strength)
            keyword_weight = 1.0 - semantic_weight - temporal_weight

            return {
                'semantic': semantic_weight,
                'temporal': temporal_weight,
                'keyword': max(0.1, keyword_weight)
            }

    def clear_cache(self):
        """Clear the analysis cache"""
        self._analysis_cache.clear()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the analysis cache"""
        total_entries = len(self._analysis_cache)
        valid_entries = sum(1 for entry in self._analysis_cache.values() if self._is_cache_valid(entry))

        return {
            'total_entries': total_entries,
            'valid_entries': valid_entries,
            'hit_rate': valid_entries / max(total_entries, 1)
        }

    def test_analysis(self, test_queries: List[str], domain: str = "general") -> Dict[str, Dict[str, Any]]:
        """
        Test the analyzer on a list of queries for debugging/tuning

        Args:
            test_queries: List of queries to analyze
            domain: Domain context

        Returns:
            Dictionary mapping queries to their analysis results
        """
        results = {}
        for query in test_queries:
            results[query] = self.analyze_query(query, domain, use_cache=False)
        return results