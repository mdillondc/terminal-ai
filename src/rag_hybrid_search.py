import re
from datetime import datetime
from typing import List, Dict, Any
from collections import Counter
import math

from settings_manager import SettingsManager
from rag_query_analyzer import RAGQueryAnalyzer
from llm_client_manager import LLMClientManager


class HybridSearchService:
    """
    Hybrid search service that combines semantic similarity, temporal relevance, and keyword matching
    for improved RAG retrieval, especially for temporal queries.
    """

    def __init__(self, llm_client_manager=None):
        self.llm_client_manager = llm_client_manager
        self.settings_manager = SettingsManager.getInstance()
        self.date_patterns = [
            r'\b(\d{4})-(\d{2})-(\d{2})\b',  # YYYY-MM-DD
            r'\b(\d{2})/(\d{2})/(\d{4})\b',  # MM/DD/YYYY
            r'\b(\d{2})\.(\d{2})\.(\d{4})\b',  # DD.MM.YYYY
            r'\b(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{4})\b',  # DD Mon YYYY
        ]
        self._load_settings()

        # Initialize LLM query analyzer if LLM client manager is available
        if self.llm_client_manager and self.enable_llm_analysis:
            self.query_analyzer = RAGQueryAnalyzer(self.llm_client_manager)
        else:
            self.query_analyzer = None

    def _load_settings(self):
        """Load hybrid search settings from settings manager"""
        self.enabled = self.settings_manager.setting_get("rag_enable_hybrid_search")
        self.enable_llm_analysis = self.settings_manager.setting_get("rag_enable_llm_query_analysis")

        # Default weights (used when LLM analysis is not available)
        self.default_semantic_weight = self.settings_manager.setting_get("rag_semantic_weight")
        self.default_temporal_weight = self.settings_manager.setting_get("rag_temporal_weight")
        self.default_keyword_weight = self.settings_manager.setting_get("rag_keyword_weight")

        self.temporal_boost_months = self.settings_manager.setting_get("rag_temporal_boost_months")
        self.temporal_detection_threshold = self.settings_manager.setting_get("rag_temporal_detection_threshold")

    def is_temporal_query(self, query: str) -> bool:
        """
        Detect if a query has temporal intent using LLM analysis or fallback

        Args:
            query: The search query

        Returns:
            True if query appears to be asking for recent/latest information
        """
        if not self.enabled:
            return False

        # Use LLM analysis if available
        if self.query_analyzer:
            return self.query_analyzer.is_temporal_query(query)

        # Fallback to simple keyword matching
        return self._fallback_temporal_detection(query)

    def _fallback_temporal_detection(self, query: str) -> bool:
        """
        Fallback temporal detection using simple keyword matching
        """
        temporal_keywords = [
            'latest', 'recent', 'newest', 'last', 'most recent', 'current',
            'when', 'today', 'yesterday', 'this week', 'this month'
        ]

        query_lower = query.lower()
        temporal_matches = sum(1 for keyword in temporal_keywords if keyword in query_lower)
        total_words = len(query_lower.split())

        confidence = temporal_matches / max(total_words, 1)
        return confidence >= self.temporal_detection_threshold

    def extract_dates_from_text(self, text: str) -> List[datetime]:
        """
        Extract dates from text content

        Args:
            text: Text content to extract dates from

        Returns:
            List of datetime objects found in the text
        """
        dates = []

        for pattern in self.date_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    date = None
                    if pattern == self.date_patterns[0]:  # YYYY-MM-DD
                        year, month, day = match.groups()
                        date = datetime(int(year), int(month), int(day))
                    elif pattern == self.date_patterns[1]:  # MM/DD/YYYY
                        month, day, year = match.groups()
                        date = datetime(int(year), int(month), int(day))
                    elif pattern == self.date_patterns[2]:  # DD.MM.YYYY
                        day, month, year = match.groups()
                        date = datetime(int(year), int(month), int(day))
                    elif pattern == self.date_patterns[3]:  # DD Mon YYYY
                        day, month_name, year = match.groups()
                        month_map = {
                            'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                            'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
                        }
                        month = month_map.get(month_name, 1)
                        date = datetime(int(year), month, int(day))

                    if date is not None:
                        dates.append(date)

                except (ValueError, KeyError):
                    continue

        return dates

    def calculate_temporal_score(self, chunk_dates: List[datetime], current_time: datetime) -> float:
        """
        Calculate temporal relevance score based on recency

        Args:
            chunk_dates: List of dates found in the chunk
            current_time: Current datetime for comparison

        Returns:
            Temporal score (0.0 to 1.0, higher = more recent)
        """
        if not chunk_dates:
            return 0.0

        # Use the most recent date in the chunk
        most_recent = max(chunk_dates)

        # Calculate months difference
        months_diff = (current_time.year - most_recent.year) * 12 + (current_time.month - most_recent.month)

        # Apply temporal boost for recent content
        if months_diff <= self.temporal_boost_months:
            # Exponential decay: more recent = higher score
            temporal_score = math.exp(-months_diff / self.temporal_boost_months)
        else:
            # Older content gets minimal temporal score
            temporal_score = 0.1 * math.exp(-(months_diff - self.temporal_boost_months) / 12)

        return min(temporal_score, 1.0)

    def calculate_bm25_score(self, query: str, document: str, corpus: List[str]) -> float:
        """
        Calculate BM25 score for keyword matching

        Args:
            query: Search query
            document: Document text
            corpus: List of all documents for IDF calculation

        Returns:
            BM25 score
        """
        # BM25 parameters
        k1 = 1.2
        b = 0.75

        # Tokenize
        query_terms = query.lower().split()
        doc_terms = document.lower().split()
        doc_length = len(doc_terms)

        # Calculate average document length
        avg_doc_length = sum(len(doc.split()) for doc in corpus) / len(corpus) if corpus else 1

        # Term frequency in document
        doc_term_freq = Counter(doc_terms)

        # Document frequency for IDF calculation
        doc_freq = {}
        for term in set(query_terms):
            doc_freq[term] = sum(1 for doc in corpus if term in doc.lower())

        score = 0.0
        for term in query_terms:
            if term in doc_term_freq:
                tf = doc_term_freq[term]
                df = doc_freq.get(term, 0)
                idf = math.log((len(corpus) - df + 0.5) / (df + 0.5)) if df > 0 else 0

                # BM25 formula
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * (doc_length / avg_doc_length))
                score += idf * (numerator / denominator)

        return score

    def enhance_chunks_with_metadata(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enhance chunks with temporal and keyword metadata

        Args:
            chunks: List of document chunks

        Returns:
            Enhanced chunks with temporal metadata
        """
        enhanced_chunks = []

        for chunk in chunks:
            enhanced_chunk = chunk.copy()

            # Extract dates from chunk content
            content = chunk.get('content', '')
            dates = self.extract_dates_from_text(content)
            enhanced_chunk['extracted_dates'] = dates

            # Store most recent date for easy access
            if dates:
                enhanced_chunk['most_recent_date'] = max(dates)
            else:
                enhanced_chunk['most_recent_date'] = None

            enhanced_chunks.append(enhanced_chunk)

        return enhanced_chunks

    def hybrid_search(self, query: str, chunks: List[Dict[str, Any]],
                     semantic_scores: List[float], top_k: int) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining semantic, temporal, and keyword scoring

        Args:
            query: Search query
            chunks: List of document chunks with embeddings
            semantic_scores: Pre-calculated semantic similarity scores
            top_k: Number of top results to return

        Returns:
            Re-ranked list of chunks with hybrid scores
        """
        if not self.enabled:
            # Fall back to pure semantic search
            results = []
            for i, chunk in enumerate(chunks):
                result = chunk.copy()
                result['similarity_score'] = semantic_scores[i]
                result['hybrid_score'] = semantic_scores[i]
                results.append(result)

            results.sort(key=lambda x: x['hybrid_score'], reverse=True)
            return results[:top_k]

        # Get intelligent weights from LLM analysis or use defaults
        if self.query_analyzer:
            weights = self.query_analyzer.get_search_weights(query)
            semantic_weight = weights.get('semantic', self.default_semantic_weight)
            temporal_weight = weights.get('temporal', self.default_temporal_weight)
            keyword_weight = weights.get('keyword', self.default_keyword_weight)
            analysis = self.query_analyzer.analyze_query(query)
            is_temporal = analysis.get('temporal_strength', 0.0) > 0.3
        else:
            # Fallback to default weights and simple detection
            is_temporal = self.is_temporal_query(query)
            if is_temporal:
                semantic_weight = self.default_semantic_weight
                temporal_weight = self.default_temporal_weight
                keyword_weight = self.default_keyword_weight
            else:
                semantic_weight = 0.8
                temporal_weight = 0.1
                keyword_weight = 0.1

        # Enhance chunks with temporal metadata if not already done
        if not chunks or 'extracted_dates' not in chunks[0]:
            chunks = self.enhance_chunks_with_metadata(chunks)

        # Prepare corpus for BM25
        corpus = [chunk.get('content', '') for chunk in chunks]
        current_time = datetime.now()

        results = []

        for i, chunk in enumerate(chunks):
            result = chunk.copy()

            # Semantic score (already calculated)
            semantic_score = semantic_scores[i] if i < len(semantic_scores) else 0.0

            # Temporal score
            chunk_dates = chunk.get('extracted_dates', [])
            temporal_score = self.calculate_temporal_score(chunk_dates, current_time)

            # Keyword score (BM25)
            content = chunk.get('content', '')
            keyword_score = self.calculate_bm25_score(query, content, corpus)

            # Normalize keyword score (simple min-max normalization)
            if keyword_score > 0:
                keyword_score = min(keyword_score / 10.0, 1.0)  # Rough normalization

            # Calculate hybrid score using intelligent or default weights
            # Ensure weights are never None
            semantic_weight = semantic_weight or 0.6
            temporal_weight = temporal_weight or 0.3
            keyword_weight = keyword_weight or 0.1

            hybrid_score = (
                semantic_weight * semantic_score +
                temporal_weight * temporal_score +
                keyword_weight * keyword_score
            )

            # Store individual scores for debugging
            result['similarity_score'] = semantic_score  # Keep original name for compatibility
            result['hybrid_score'] = hybrid_score
            result['temporal_score'] = temporal_score
            result['keyword_score'] = keyword_score
            result['is_temporal_query'] = is_temporal
            result['weights_used'] = {
                'semantic': semantic_weight,
                'temporal': temporal_weight,
                'keyword': keyword_weight
            }

            results.append(result)

        # Sort by hybrid score
        results.sort(key=lambda x: x['hybrid_score'], reverse=True)

        return results[:top_k]

    def get_search_explanation(self, query: str, top_result: Dict[str, Any]) -> str:
        """
        Generate explanation of how the search worked

        Args:
            query: Original query
            top_result: Top search result with scores

        Returns:
            Human-readable explanation of the search process
        """
        if not self.enabled:
            return f"Semantic similarity: {top_result.get('similarity_score', 0):.2f}"

        explanation_parts = []

        # Show analysis method
        if self.query_analyzer:
            explanation_parts.append("🤖 LLM analyzed")
        else:
            explanation_parts.append("📋 Rule-based")

        if top_result.get('is_temporal_query', False):
            explanation_parts.append("🕐 Temporal query")

        # Show scores
        explanation_parts.append(f"Semantic: {top_result.get('similarity_score', 0):.2f}")
        explanation_parts.append(f"Temporal: {top_result.get('temporal_score', 0):.2f}")
        explanation_parts.append(f"Keyword: {top_result.get('keyword_score', 0):.2f}")
        explanation_parts.append(f"Hybrid: {top_result.get('hybrid_score', 0):.2f}")

        # Show weights used
        weights = top_result.get('weights_used', {})
        if weights:
            weight_str = f"W({weights.get('semantic', 0):.1f}/{weights.get('temporal', 0):.1f}/{weights.get('keyword', 0):.1f})"
            explanation_parts.append(weight_str)

        if top_result.get('most_recent_date'):
            date_str = top_result['most_recent_date'].strftime('%Y-%m-%d')
            explanation_parts.append(f"Date: {date_str}")

        return " | ".join(explanation_parts)

    def debug_configuration(self) -> Dict[str, Any]:
        """
        Get debug information about hybrid search configuration

        Returns:
            Dictionary with configuration and status information
        """
        return {
            "enabled": self.enabled,
            "llm_analysis_enabled": self.enable_llm_analysis,
            "has_query_analyzer": self.query_analyzer is not None,
            "default_semantic_weight": self.default_semantic_weight,
            "default_temporal_weight": self.default_temporal_weight,
            "default_keyword_weight": self.default_keyword_weight,
            "temporal_boost_months": self.temporal_boost_months,
            "temporal_detection_threshold": self.temporal_detection_threshold,
            "date_patterns_count": len(self.date_patterns),
            "settings_loaded": hasattr(self, 'enabled')
        }

    def test_temporal_detection(self, test_queries: List[str]) -> Dict[str, Any]:
        """
        Test temporal detection on a list of queries

        Args:
            test_queries: List of queries to test

        Returns:
            Dictionary with test results
        """
        results = {}
        for query in test_queries:
            is_temporal = self.is_temporal_query(query)

            if self.query_analyzer:
                # Use LLM analysis
                analysis = self.query_analyzer.analyze_query(query)
                results[query] = {
                    "is_temporal": is_temporal,
                    "temporal_strength": analysis.get('temporal_strength', 0.0),
                    "analysis_method": "llm",
                    "word_count": len(query.split()),
                    "search_strategy": analysis.get('search_strategy', 'unknown')
                }
            else:
                # Use fallback method
                temporal_keywords = [
                    'latest', 'recent', 'newest', 'last', 'most recent', 'current',
                    'when', 'today', 'yesterday', 'this week', 'this month'
                ]
                results[query] = {
                    "is_temporal": is_temporal,
                    "keywords_found": [kw for kw in temporal_keywords if kw in query.lower()],
                    "analysis_method": "fallback",
                    "word_count": len(query.split())
                }
        return results