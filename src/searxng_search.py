"""
SearXNG Search Integration for Terminal AI Assistant

This module provides web search capabilities using a local SearXNG instance,
allowing the AI to access current information from the internet while maintaining
privacy and control over the search infrastructure.
"""

import requests
from typing import List, Dict, Optional, Any
from urllib.parse import urljoin
from print_helper import print_md


class SearXNGSearchError(Exception):
    """Custom exception for SearXNG search errors"""
    pass


class SearXNGSearch:
    """
    Wrapper class for SearXNG API search functionality.

    Provides methods to search the web using a local SearXNG instance and format
    results for AI consumption. The interface mirrors TavilySearch for seamless
    integration with existing search workflows.
    """

    def __init__(self, base_url: str):
        """
        Initialize SearXNG client.

        Args:
            base_url: Base URL of the SearXNG instance (e.g., "http://localhost:8080")
        """
        self.base_url = base_url.rstrip('/')
        self.search_endpoint = urljoin(self.base_url + '/', 'search')

        # Test connection to SearXNG instance
        # try:
        #     response = requests.get(self.base_url, timeout=5)
        #     response.raise_for_status()
        # except requests.exceptions.RequestException as e:
        #     raise SearXNGSearchError(f"Failed to connect to SearXNG instance at {base_url}: {e}")

    def search(self, query: str, max_results: int = 5, include_domains: Optional[List[str]] = None,
               exclude_domains: Optional[List[str]] = None, search_depth: str = "advanced",
               days: Optional[int] = None, topic: Optional[str] = None,
               auto_parameters: bool = False) -> Dict[str, Any]:
        """
        Perform a web search using SearXNG API.

        Args:
            query: Search query string
            max_results: Maximum number of results to return (default: 5)
            include_domains: List of domains to include in search (ignored for SearXNG compatibility)
            exclude_domains: List of domains to exclude from search (ignored for SearXNG compatibility)
            search_depth: "basic" or "advanced" search depth (ignored for SearXNG compatibility)
            days: Number of days for freshness filtering (maps to time_range)
            topic: Topic category (maps to SearXNG categories)
            auto_parameters: Ignored for SearXNG compatibility

        Returns:
            Dictionary containing search results in Tavily-compatible format

        Raises:
            SearXNGSearchError: If search fails
        """
        try:
            # Prepare search parameters
            params = {
                "q": query,
                "format": "json"
            }

            # Map max_results to pageno if needed (SearXNG typically returns 10 per page)
            if max_results <= 10:
                params["pageno"] = "1"
            else:
                # For more results, we'll make multiple requests later
                params["pageno"] = "1"

            # Map time range
            if days:
                if days <= 1:
                    params["time_range"] = "day"
                elif days <= 30:
                    params["time_range"] = "month"
                elif days <= 365:
                    params["time_range"] = "year"

            # Map topic to categories
            if topic and topic != "general":
                category_map = {
                    "news": "news",
                    "finance": "news",
                    "sports": "news"
                }
                if topic in category_map:
                    params["categories"] = category_map[topic]

            # Perform the search
            response = requests.get(self.search_endpoint, params=params, timeout=10)
            response.raise_for_status()

            searxng_results = response.json()

            # Convert SearXNG format to Tavily-compatible format
            tavily_format = self._convert_to_tavily_format(searxng_results, query, max_results)

            return tavily_format

        except requests.exceptions.RequestException as e:
            raise SearXNGSearchError(f"HTTP request failed: {e}")
        except ValueError as e:
            raise SearXNGSearchError(f"Failed to parse JSON response: {e}")
        except Exception as e:
            raise SearXNGSearchError(f"Search failed: {e}")

    def _convert_to_tavily_format(self, searxng_results: Dict[str, Any], query: str, max_results: int) -> Dict[str, Any]:
        """
        Convert SearXNG response format to Tavily-compatible format.

        Args:
            searxng_results: Raw results from SearXNG
            query: Original search query
            max_results: Maximum number of results requested

        Returns:
            Dictionary in Tavily-compatible format
        """
        # Extract results and limit to max_results
        raw_results = searxng_results.get('results', [])
        limited_results = raw_results[:max_results]

        # Convert each result to Tavily format
        converted_results = []
        for result in limited_results:
            converted_result = {
                'title': result.get('title', ''),
                'url': result.get('url', ''),
                'content': result.get('content', ''),
                'score': result.get('score', 1.0)
            }

            # Add published date if available
            if 'publishedDate' in result:
                converted_result['published_date'] = result['publishedDate']

            converted_results.append(converted_result)

        # Create Tavily-compatible response structure
        tavily_response = {
            'query': query,
            'results': converted_results
        }

        # Add suggestions if available
        if 'suggestions' in searxng_results:
            tavily_response['suggestions'] = searxng_results['suggestions']

        # Generate a simple answer summary (SearXNG doesn't provide this like Tavily)
        if converted_results:
            first_result = converted_results[0]
            tavily_response['answer'] = f"Found {len(converted_results)} results for '{query}'. Top result: {first_result.get('title', 'Unknown')} - {first_result.get('content', '')[:200]}..."

        return tavily_response

    def format_results_for_ai(self, search_results: Dict[str, Any], query: str) -> str:
        """
        Format search results into a readable format for AI consumption.

        Args:
            search_results: Raw search results from SearXNG (in Tavily format)
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
        formatted_output.append("- Search engine: SearXNG")

        return "\n".join(formatted_output) + "\n"

    def get_source_metadata(self, search_results: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Extract structured source metadata from search results.

        Args:
            search_results: Search results in Tavily format

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
                print_md(f"Warning: Skipping problematic source: {e}")
                continue

        return metadata

    def search_and_format(self, query: str, max_results: int = 5,
                         include_domains: Optional[List[str]] = None,
                         exclude_domains: Optional[List[str]] = None,
                         search_depth: str = "advanced", days: Optional[int] = None,
                         topic: Optional[str] = None, return_metadata: bool = False,
                         auto_parameters: bool = False):
        """
        Convenience method that searches and formats results in one call.

        Args:
            query: Search query string
            max_results: Maximum number of results to return
            include_domains: List of domains to include in search (ignored for SearXNG compatibility)
            exclude_domains: List of domains to exclude from search (ignored for SearXNG compatibility)
            search_depth: "basic" or "advanced" search depth (ignored for SearXNG compatibility)
            days: Number of days for freshness filtering
            topic: Topic category
            return_metadata: Whether to return metadata along with formatted results
            auto_parameters: Ignored for SearXNG compatibility

        Returns:
            Formatted search results string, or tuple of (formatted_results, source_metadata) if return_metadata=True

        Raises:
            SearXNGSearchError: If search fails
        """
        try:
            results = self.search(query, max_results, include_domains, exclude_domains,
                                search_depth, days, topic, auto_parameters)
            formatted_results = self.format_results_for_ai(results, query)

            if return_metadata:
                source_metadata = self.get_source_metadata(results)
                return (formatted_results, source_metadata)
            else:
                return formatted_results
        except SearXNGSearchError:
            raise
        except Exception as e:
            raise SearXNGSearchError(f"- Search and format failed: {e}")


def create_searxng_search(base_url: str) -> Optional[SearXNGSearch]:
    """
    Factory function to create SearXNGSearch instance with error handling.

    Args:
        base_url: Base URL of the SearXNG instance

    Returns:
        SearXNGSearch instance if successful, None if connection fails
    """
    try:
        return SearXNGSearch(base_url)
    except SearXNGSearchError as e:
        print_md(f"SearXNG search error: {e}")
        return None
    except Exception as e:
        print_md(f"Unexpected error initializing SearXNG search: {e}")
        return None