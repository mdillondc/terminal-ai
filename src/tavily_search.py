"""
Tavily Search Integration for Terminal AI Assistant

This module provides web search capabilities using the Tavily API,
allowing the AI to access current information from the internet.
"""

import os
from typing import List, Dict, Optional, Any
from tavily import TavilyClient
from print_helper import print_md


class TavilySearchError(Exception):
    """Custom exception for Tavily search errors"""
    pass


class TavilySearch:
    """
    Wrapper class for Tavily API search functionality.

    Provides methods to search the web and format results for AI consumption.
    """

    def __init__(self):
        """
        Initialize Tavily client.

        Args:
            api_key: Tavily API key. If not provided, looks for TAVILY_API_KEY env var.
        """
        self.api_key = os.getenv('TAVILY_API_KEY')
        if not self.api_key:
            raise TavilySearchError("TAVILY_API_KEY environment variable not set")

        try:
            self.client = TavilyClient(api_key=self.api_key)
        except Exception as e:
            raise TavilySearchError(f"Failed to initialize Tavily client: {e}")

    def search(self, query: str, max_results: int = 5, include_domains: Optional[List[str]] = None,
               exclude_domains: Optional[List[str]] = None, search_depth: str = "advanced",
               days: Optional[int] = None, topic: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform a web search using Tavily API.

        Args:
            query: Search query string
            max_results: Maximum number of results to return (default: 5)
            include_domains: List of domains to include in search
            exclude_domains: List of domains to exclude from search
            search_depth: "basic" or "advanced" search depth
            days: Number of days for freshness filtering (optional)
            topic: Topic category ("news", "finance", "sports", "general")

        Returns:
            Dictionary containing search results and metadata

        Raises:
            TavilySearchError: If search fails
        """
        try:
            # Prepare search parameters
            search_params = {
                "query": query,
                "max_results": max_results,
                "search_depth": search_depth,
                "include_answer": True,      # Include AI-generated answer
                "include_raw_content": False  # Don't include raw HTML content
            }

            # Add domain filters if provided
            if include_domains:
                search_params["include_domains"] = include_domains
            if exclude_domains:
                search_params["exclude_domains"] = exclude_domains

            # Add freshness filter if provided
            if days:
                search_params["days"] = days

            # Add topic filter if provided and not general
            if topic and topic != "general":
                search_params["topic"] = topic

            # Perform the search
            response = self.client.search(**search_params)

            return response

        except Exception as e:
            raise TavilySearchError(f"Search failed: {e}")

    def format_results_for_ai(self, search_results: Dict[str, Any], query: str) -> str:
        """
        Format search results into a readable format for AI consumption.

        Args:
            search_results: Raw search results from Tavily API
            query: Original search query

        Returns:
            Formatted string containing search results
        """
        if not search_results or 'results' not in search_results:
            return f"No search results found for query: '{query}'"

        formatted_output = []
        formatted_output.append(f"- **Web Search Results for:** '{query}'")
        formatted_output.append("=" * 50)

        # Include AI-generated answer if available
        if 'answer' in search_results and search_results['answer']:
            formatted_output.append("**AI Summary:**")
            formatted_output.append(search_results['answer'])
            formatted_output.append("\n" + "-" * 40)

        # Format individual search results
        results = search_results.get('results', [])
        for i, result in enumerate(results, 1):
            formatted_output.append(f"\n**{i}. {result.get('title', 'No Title')}**")
            formatted_output.append(f"- URL: {result.get('url', 'No URL')}")

            # Add content snippet if available
            content = result.get('content', '')
            if content:
                # Truncate very long content
                if len(content) > 500:
                    content = content[:500] + "..."
                formatted_output.append(f"- Content: {content}")

            # Add published date if available
            if 'published_date' in result:
                formatted_output.append(f"Published: {result['published_date']}")

            formatted_output.append("-" * 40)

        # Add search metadata
        formatted_output.append("- Search Metadata:")
        formatted_output.append(f"- Total results: {len(results)}")
        formatted_output.append(f"- Query: {query}")

        return "\n".join(formatted_output) + "\n"

    def get_source_metadata(self, search_results: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Extract structured source metadata from search results.

        Args:
            search_results: Raw search results from Tavily API

        Returns:
            List of dictionaries with keys: 'title', 'url', 'domain', 'published_date'
        """
        metadata = []

        if not search_results or 'results' not in search_results:
            return metadata

        results = search_results.get('results', [])
        for result in results:
            try:
                url = result.get('url', '')

                # Skip sources with invalid URLs
                if not url or not isinstance(url, str):
                    continue

                # Extract domain from URL
                domain = 'unknown'
                if url:
                    try:
                        from urllib.parse import urlparse
                        parsed = urlparse(url)
                        domain = parsed.netloc
                        if not domain:
                            # Skip sources with malformed URLs
                            continue
                    except:
                        # Skip sources with malformed URLs
                        continue

                # Get title with fallbacks
                title = result.get('title', '')
                if not title or not isinstance(title, str):
                    # Use domain name as fallback title
                    title = domain
                if not title:
                    # Final fallback
                    title = 'Unknown Source'

                source_info = {
                    'title': title,
                    'url': url,
                    'domain': domain,
                    'published_date': result.get('published_date', '')
                }

                metadata.append(source_info)

            except Exception as e:
                # Log warning for debugging but continue processing other sources
                from print_helper import print_md
                print_md(f"Warning: Skipping problematic source: {e}")
                continue

        return metadata

    def search_and_format(self, query: str, max_results: int = 5,
                         include_domains: Optional[List[str]] = None,
                         exclude_domains: Optional[List[str]] = None,
                         search_depth: str = "advanced", days: Optional[int] = None,
                         topic: Optional[str] = None, return_metadata: bool = False):
        """
        Convenience method that searches and formats results in one call.

        Args:
            query: Search query string
            max_results: Maximum number of results to return
            include_domains: List of domains to include in search
            exclude_domains: List of domains to exclude from search
            search_depth: "basic" or "advanced" search depth
            days: Number of days for freshness filtering (optional)
            topic: Topic category ("news", "finance", "sports", "general")
            return_metadata: Whether to return metadata along with formatted results

        Returns:
            Formatted search results string, or tuple of (formatted_results, source_metadata) if return_metadata=True

        Raises:
            TavilySearchError: If search fails
        """
        try:
            results = self.search(query, max_results, include_domains, exclude_domains,
                                search_depth, days, topic)
            formatted_results = self.format_results_for_ai(results, query)

            if return_metadata:
                source_metadata = self.get_source_metadata(results)
                return (formatted_results, source_metadata)
            else:
                return formatted_results
        except TavilySearchError:
            raise
        except Exception as e:
            raise TavilySearchError(f"- Search and format failed: {e}")


def create_tavily_search() -> Optional[TavilySearch]:
    """
    Factory function to create TavilySearch instance with error handling.

    Returns:
        TavilySearch instance if successful, None if API key is missing or invalid
    """
    try:
        return TavilySearch()
    except TavilySearchError as e:
        print_md(f"Tavily search error: {e}")
        return None
    except Exception as e:
        print_md(f"Unexpected error initializing Tavily search: {e}")
        return None


