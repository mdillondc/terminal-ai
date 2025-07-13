"""
RAG Hybrid Search Module - Simplified

Simple semantic search with optional recent boost and result diversity.
"""

import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from settings_manager import SettingsManager
from print_helper import print_md


class HybridSearchService:
    """
    Simplified search service that performs semantic search with optional recent boost
    and result diversity.
    """

    def __init__(self, llm_client_manager=None):
        self.settings_manager = SettingsManager.getInstance()
        self.llm_client_manager = llm_client_manager
        self._load_settings()

        # Initialize query analyzer if available
        if self.llm_client_manager:
            from rag_query_analyzer import RAGQueryAnalyzer
            self.query_analyzer = RAGQueryAnalyzer(self.llm_client_manager)
        else:
            self.query_analyzer = None

        # Date extraction patterns for temporal metadata
        self.date_patterns = [
            r'\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b',  # YYYY-MM-DD or YYYY/MM/DD
            r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{4})\b',  # MM-DD-YYYY or MM/DD/YYYY
            r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{2})\b',  # MM-DD-YY or MM/DD/YY
        ]

    def _load_settings(self):
        """Load search settings from settings manager"""
        self.enabled = self.settings_manager.setting_get("rag_enable_hybrid_search")
        self.recent_boost_months = self.settings_manager.setting_get("rag_temporal_boost_months")

    def extract_dates_from_text(self, text: str) -> List[datetime]:
        """
        Extract dates from text content using regex patterns

        Args:
            text: Text content to search for dates

        Returns:
            List of datetime objects found in the text
        """
        dates = []
        transparency_enabled = self.settings_manager.setting_get("rag_enable_search_transparency")

        for pattern in self.date_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    # Try different date formats
                    for date_format in ['%Y-%m-%d', '%Y/%m/%d', '%m-%d-%Y', '%m/%d/%Y', '%m-%d-%y', '%m/%d/%y']:
                        try:
                            date_obj = datetime.strptime(match, date_format)
                            # Handle 2-digit years
                            if date_obj.year < 1950:
                                date_obj = date_obj.replace(year=date_obj.year + 2000)
                            # Only accept reasonable years (1900-2100)
                            if 1900 <= date_obj.year <= 2100:
                                dates.append(date_obj)
                            break
                        except ValueError:
                            continue
                except ValueError:
                    continue

        return dates

    def calculate_recent_score(self, chunk: Dict[str, Any]) -> float:
        """
        Calculate recent score for a chunk based on dates found in content

        Args:
            chunk: Document chunk with content and metadata

        Returns:
            Score from 0.0 to 1.0 based on recency (1.0 = very recent)
        """
        if 'most_recent_date' not in chunk:
            return 0.0

        most_recent_date = chunk['most_recent_date']
        if not most_recent_date:
            return 0.0

        # Calculate months since the most recent date
        now = datetime.now()
        months_ago = (now.year - most_recent_date.year) * 12 + (now.month - most_recent_date.month)

        if months_ago <= 0:
            score = 1.0  # Current month
        elif months_ago <= self.recent_boost_months:
            # Linear decay over the boost period
            score = max(0.0, 1.0 - (months_ago / self.recent_boost_months))
        else:
            score = 0.0

        return score

    def enhance_chunks_with_metadata(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enhance chunks with temporal metadata

        Args:
            chunks: List of document chunks

        Returns:
            Enhanced chunks with temporal metadata
        """
        enhanced_chunks = []

        for chunk in chunks:
            enhanced_chunk = chunk.copy()

            # Extract dates from content if not already done
            if 'extracted_dates' not in enhanced_chunk:
                content = enhanced_chunk.get('content', '')
                extracted_dates = self.extract_dates_from_text(content)
                enhanced_chunk['extracted_dates'] = extracted_dates
            else:
                extracted_dates = enhanced_chunk['extracted_dates']

            # Always ensure most_recent_date is set if we have dates
            if extracted_dates:
                enhanced_chunk['most_recent_date'] = max(extracted_dates)
            else:
                enhanced_chunk['most_recent_date'] = None

            enhanced_chunks.append(enhanced_chunk)

        return enhanced_chunks

    def search(self, query: str, chunks: List[Dict[str, Any]],
               semantic_scores: List[float], top_k: int) -> List[Dict[str, Any]]:
        """
        Perform simplified search: semantic with optional recent boost

        Args:
            query: Search query
            chunks: List of document chunks with embeddings
            semantic_scores: Pre-calculated semantic similarity scores
            top_k: Number of top results to return

        Returns:
            List of search results with scores
        """
        if not self.enabled:
            # Fall back to pure semantic search
            results = []
            for i, chunk in enumerate(chunks):
                result = chunk.copy()
                result['similarity_score'] = semantic_scores[i]
                result['final_score'] = semantic_scores[i]
                results.append(result)

            results.sort(key=lambda x: x['final_score'], reverse=True)
            return self._apply_result_diversity(results, top_k)

        # Determine if this is a recent query
        wants_recent = False
        transparency_enabled = self.settings_manager.setting_get("rag_enable_search_transparency")
        if self.query_analyzer:
            wants_recent = self.query_analyzer.is_recent_query(query)

        # Enhance chunks with temporal metadata if not already done
        if not chunks or 'extracted_dates' not in chunks[0]:
            chunks = self.enhance_chunks_with_metadata(chunks)

        # Calculate final scores
        results = []
        for i, chunk in enumerate(chunks):
            result = chunk.copy()
            semantic_score = semantic_scores[i]

            # Start with semantic score
            final_score = semantic_score

            # Apply recent boost if this is a recent query
            if wants_recent:
                recent_score = self.calculate_recent_score(chunk)
                # Strong boost: multiply semantic score by (1 + recent_score * 2.0)
                # This gives up to 200% boost for very recent content
                final_score = semantic_score * (1.0 + recent_score * 2.0)
            else:
                final_score = semantic_score

            result['similarity_score'] = semantic_score
            result['final_score'] = final_score
            result['wants_recent'] = wants_recent

            if wants_recent:
                result['recent_score'] = self.calculate_recent_score(chunk)

            results.append(result)

        # Debug: Show all vet.md chunks before sorting
        if transparency_enabled and wants_recent:
            vet_chunks = [r for r in results if 'vet.md' in r.get('filename', '')]
            if vet_chunks:
                debug = "All vet.md chunks:\n"
                for i, chunk in enumerate(vet_chunks, 1):
                    date = chunk.get('most_recent_date')
                    date_str = date.strftime('%Y-%m-%d') if date else 'no date'
                    semantic = chunk.get('similarity_score', 0)
                    final = chunk.get('final_score', 0)
                    recent = chunk.get('recent_score', 0) if wants_recent else 0
                    content_preview = chunk.get('content', '')[:50].replace('\n', ' ')
                    debug += f"    {i}. Date: {date_str}, Semantic: {semantic:.3f}, Recent: {recent:.2f}, Final: {final:.3f}\n"
                    debug += f"       Content: {content_preview}...\n"
                print_md(debug.rstrip())

        # Sort by final score
        results.sort(key=lambda x: x['final_score'], reverse=True)

        # Show summary if transparency enabled
        if transparency_enabled and wants_recent:
            top_3 = results[:3]
            summary = "Top results with recent boost:\n"
            for i, result in enumerate(top_3, 1):
                filename = result.get('filename', 'unknown').split('/')[-1]
                date = result.get('most_recent_date')
                date_str = date.strftime('%Y-%m-%d') if date else 'no date'
                semantic = result.get('similarity_score', 0)
                final = result.get('final_score', 0)
                summary += f"    {i}. {filename} ({date_str}) - semantic: {semantic:.3f}, final: {final:.3f}\n"
            print_md(summary.rstrip())

        # Apply result diversity
        return self._apply_result_diversity(results, top_k, wants_recent)

    def _apply_result_diversity(self, results: List[Dict[str, Any]], top_k: int, wants_recent: bool = False) -> List[Dict[str, Any]]:
        """
        Apply result diversity to prevent over-representation from single sources

        Args:
            results: Sorted list of search results
            top_k: Maximum number of results to return
            wants_recent: Whether this is a recent query (prioritize by date within sources)

        Returns:
            Diversified list of results
        """
        if not results:
            return results

        if not self.settings_manager.setting_get("rag_enable_result_diversity"):
            return results[:top_k]

        max_chunks_per_source = self.settings_manager.setting_get("rag_max_chunks_per_source")
        transparency_enabled = self.settings_manager.setting_get("rag_enable_search_transparency")

        if wants_recent:
            # For recent queries: group by source and prioritize by date within each source
            source_groups = {}
            for result in results:
                source_filename = result.get('filename', 'unknown')
                if source_filename not in source_groups:
                    source_groups[source_filename] = []
                source_groups[source_filename].append(result)

            # Sort each source group by most recent date first, then by final score
            for source_filename in source_groups:
                source_groups[source_filename].sort(key=lambda x: (
                    x.get('most_recent_date') or datetime(1900, 1, 1),  # Primary: most recent date
                    x.get('final_score', 0)  # Secondary: final score
                ), reverse=True)

            # Now apply diversity using the reordered groups
            diversified_results = []
            source_counts = {}

            # Interleave results from all sources, taking best from each
            max_rounds = max_chunks_per_source
            for round_num in range(max_rounds):
                for source_filename, source_results in source_groups.items():
                    if round_num < len(source_results) and len(diversified_results) < top_k:
                        diversified_results.append(source_results[round_num])
                        source_counts[source_filename] = source_counts.get(source_filename, 0) + 1
        else:
            # Original logic for non-recent queries
            source_counts = {}
            diversified_results = []

            for result in results:
                source_filename = result.get('filename', 'unknown')

                # Count how many chunks we've already taken from this source
                current_count = source_counts.get(source_filename, 0)

                if current_count < max_chunks_per_source:
                    diversified_results.append(result)
                    source_counts[source_filename] = current_count + 1

                    # Stop if we've reached the desired number of results
                    if len(diversified_results) >= top_k:
                        break

        # Display diversity information if transparency is enabled
        if transparency_enabled and len(diversified_results) != len(results[:top_k]):
            excluded_count = len(results[:top_k]) - len(diversified_results)
            diversity_info = "Result diversity applied:\n"
            diversity_info += f"    Limited to {max_chunks_per_source} chunks per source\n"
            diversity_info += f"    Excluded {excluded_count} results for diversity"
            print_md(diversity_info)

        return diversified_results

    def debug_configuration(self) -> Dict[str, Any]:
        """
        Get debug information about search configuration

        Returns:
            Dictionary with configuration and status information
        """
        return {
            "enabled": self.enabled,
            "has_query_analyzer": self.query_analyzer is not None,
            "recent_boost_months": self.recent_boost_months,
            "result_diversity_enabled": self.settings_manager.setting_get("rag_enable_result_diversity"),
            "max_chunks_per_source": self.settings_manager.setting_get("rag_max_chunks_per_source"),
        }