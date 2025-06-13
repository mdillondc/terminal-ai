"""
Tavily Search Integration for Samantha AI Assistant

This module provides web search capabilities using the Tavily API,
allowing the AI to access current information from the internet.
"""

import os
from typing import List, Dict, Optional, Any
from tavily import TavilyClient


class TavilySearchError(Exception):
    """Custom exception for Tavily search errors"""
    pass


class TavilySearch:
    """
    Wrapper class for Tavily API search functionality.
    
    Provides methods to search the web and format results for AI consumption.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Tavily client.
        
        Args:
            api_key: Tavily API key. If not provided, looks for TAVILY_API_KEY env var.
        """
        self.api_key = api_key or os.getenv('TAVILY_API_KEY')
        if not self.api_key:
            raise TavilySearchError("TAVILY_API_KEY environment variable not set")
        
        try:
            self.client = TavilyClient(api_key=self.api_key)
        except Exception as e:
            raise TavilySearchError(f"Failed to initialize Tavily client: {e}")
    
    def search(self, query: str, max_results: int = 5, include_domains: Optional[List[str]] = None, 
               exclude_domains: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform a web search using Tavily API.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return (default: 5)
            include_domains: List of domains to include in search
            exclude_domains: List of domains to exclude from search
            
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
                "search_depth": "advanced",  # Use advanced search for better quality
                "include_answer": True,      # Include AI-generated answer
                "include_raw_content": False  # Don't include raw HTML content
            }
            
            # Add domain filters if provided
            if include_domains:
                search_params["include_domains"] = include_domains
            if exclude_domains:
                search_params["exclude_domains"] = exclude_domains
            
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
        formatted_output.append("=" * 60)
        
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
        formatted_output.append("\n- **Search Metadata:**")
        formatted_output.append(f"- Total results: {len(results)}")
        formatted_output.append(f"- Query: {query}")
        
        return "\n".join(formatted_output)
    
    def search_and_format(self, query: str, max_results: int = 5, 
                         include_domains: Optional[List[str]] = None,
                         exclude_domains: Optional[List[str]] = None) -> str:
        """
        Convenience method that searches and formats results in one call.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            include_domains: List of domains to include in search
            exclude_domains: List of domains to exclude from search
            
        Returns:
            Formatted search results string
            
        Raises:
            TavilySearchError: If search fails
        """
        try:
            results = self.search(query, max_results, include_domains, exclude_domains)
            return self.format_results_for_ai(results, query)
        except TavilySearchError:
            raise
        except Exception as e:
            raise TavilySearchError(f"Search and format failed: {e}")


def create_tavily_search() -> Optional[TavilySearch]:
    """
    Factory function to create TavilySearch instance with error handling.
    
    Returns:
        TavilySearch instance if successful, None if API key is missing or invalid
    """
    try:
        return TavilySearch()
    except TavilySearchError as e:
        print(f" - (!) Tavily search error: {e}")
        return None
    except Exception as e:
        print(f" - (!) Unexpected error initializing Tavily search: {e}")
        return None


# Convenience function for quick searches
def quick_search(query: str, max_results: int = 5) -> Optional[str]:
    """
    Perform a quick web search and return formatted results.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        
    Returns:
        Formatted search results string, or None if search fails
    """
    search_client = create_tavily_search()
    if not search_client:
        return None
    
    try:
        return search_client.search_and_format(query, max_results)
    except TavilySearchError as e:
        print(f" - (!) Search failed: {e}")
        return None