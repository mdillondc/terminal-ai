"""
Dynamic Search Intent Analyzer

This module provides intelligent intent detection for search queries using LLM analysis.
No hardcoded patterns - all analysis is done dynamically by the LLM.
"""

import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from print_helper import print_info


class SearchIntentAnalyzer:
    """
    Analyzes search queries using LLM to determine intent and optimal search parameters.
    Completely dynamic - no hardcoded patterns or terms.
    """

    def __init__(self, llm_client_manager, model: str = "gpt-4o-mini"):
        """
        Initialize the intent analyzer.

        Args:
            llm_client_manager: LLM client for analysis (required)
            model: Model to use for analysis
        """
        self.llm_client_manager = llm_client_manager
        self.model = model

        if not self.llm_client_manager:
            raise ValueError("LLM client manager is required for dynamic intent analysis")

    def analyze_query(self, query: str, conversation_context: str = "") -> Dict[str, Any]:
        """
        Analyze a search query to determine intent and optimal search parameters.

        Args:
            query: The search query to analyze
            conversation_context: Recent conversation context for better analysis

        Returns:
            Dictionary containing analysis results and search recommendations
        """
        try:
            # Create analysis prompt
            analysis_prompt = self._create_analysis_prompt(query, conversation_context)

            # Get LLM analysis
            response = self.llm_client_manager.create_chat_completion(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a search optimization expert. Analyze queries and provide structured recommendations for search parameters. Always respond with valid JSON only."
                    },
                    {
                        "role": "user",
                        "content": analysis_prompt
                    }
                ],
                temperature=0.1  # Low temperature for consistent analysis
            )

            # Parse JSON response
            analysis_text = response.choices[0].message.content.strip()

            # Try to clean up common JSON formatting issues
            analysis_text = self._clean_json_response(analysis_text)

            analysis = json.loads(analysis_text)

            # Validate the parsed JSON has required fields
            analysis = self._validate_and_fix_analysis(analysis, query)

            return analysis

        except json.JSONDecodeError as e:
            print_info(f"Failed to parse intent analysis JSON: {e}")
            print_info(f"Raw LLM response: {analysis_text[:200]}...")
            return self._fallback_analysis(query)
        except Exception as e:
            print_info(f"Intent analysis failed: {e}")
            return self._fallback_analysis(query)

    def _create_analysis_prompt(self, query: str, conversation_context: str) -> str:
        """
        Create the analysis prompt for the LLM with full context and noise-resistant prompting.

        Args:
            query: The search query to analyze
            conversation_context: Full conversation context (recent first, then earlier)

        Returns:
            Formatted prompt for LLM analysis
        """
        context_section = f"\n\nCONVERSATION CONTEXT (most recent first):\n{conversation_context}" if conversation_context else "\n\nCONVERSATION CONTEXT: No prior context."

        # Get current date for context
        current_date = datetime.now().strftime("%Y-%m-%d")
        current_year = datetime.now().year

        return f"""Analyze this search query and determine the optimal search parameters. Consider what type of information the user is seeking and how to get the most accurate, unbiased results.

TODAY'S DATE: {current_date} (Current year: {current_year})

CURRENT QUERY: "{query}"{context_section}

Instructions:
- Focus on recent conversation messages to understand the current topic and user intent
- Use the full conversation context to resolve unclear references (pronouns like "his", "their", "that event", company names, etc.)
- Consider the conversational flow to understand what the user is really asking about
- When analyzing temporal aspects, remember the current date is {current_date} and we are in {current_year}
- For recent events queries, consider how fresh the information needs to be based on the current date
- Determine if this query might contain claims that need fact-checking or verification

Provide your analysis as a JSON object with these fields:

{{
  "intent_type": "factual|controversial|recent_events|comparative|numerical|general",
  "search_depth": "basic|advanced",
  "topic_category": "news|finance|sports|general",
  "freshness_days": null or number (1, 7, 30),
  "needs_verification": true|false,
  "max_results": number (3-10),
  "search_strategy": "standard|verification|comprehensive|temporal",
  "confidence": number (0.0-1.0),
  "reasoning": "brief explanation of your analysis and what context informed your decision"
}}

Guidelines:
- Use "basic" search for simple, well-established factual queries
- Use "advanced" search for complex, controversial, or nuanced topics
- Set freshness_days for time-sensitive queries (1=today, 7=this week, 30=this month)
- Set needs_verification=true for claims that should be fact-checked or appear controversial
- Use higher max_results (6-10) for controversial topics needing multiple perspectives
- Choose search_strategy based on what approach will get the best results

Respond with ONLY the JSON object, no other text."""

    def _fallback_analysis(self, query: str) -> Dict[str, Any]:
        """
        Provide fallback analysis when LLM analysis fails.

        Args:
            query: The search query

        Returns:
            Basic analysis structure
        """
        return {
            'query': query,
            'intent_type': 'general',
            'search_depth': 'basic',
            'topic_category': 'general',
            'freshness_days': None,
            'needs_verification': False,
            'max_results': 5,
            'search_strategy': 'standard',
            'confidence': 0.5,
            'reasoning': 'Fallback analysis due to LLM analysis failure'
        }

    def get_search_parameters(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert analysis results into Tavily search parameters.

        Args:
            analysis: Results from analyze_query

        Returns:
            Dictionary of parameters for Tavily search
        """
        search_params = {
            'search_depth': analysis.get('search_depth', 'basic'),
            'max_results': analysis.get('max_results', 5),
            'include_answer': True,
            'include_raw_content': False
        }

        # Add topic if specified
        topic = analysis.get('topic_category')
        if topic and topic != 'general':
            search_params['topic'] = topic

        # Add freshness constraint if needed
        freshness_days = analysis.get('freshness_days')
        if freshness_days:
            search_params['days'] = freshness_days

        return search_params

    def should_use_verification_search(self, analysis: Dict[str, Any]) -> bool:
        """
        Determine if a verification search should be performed.

        Args:
            analysis: Results from analyze_query

        Returns:
            True if verification search is recommended
        """
        return analysis.get('needs_verification', False)

    def get_verification_queries(self, original_query: str, analysis: Dict[str, Any]) -> List[str]:
        """
        Generate verification queries for fact-checking.

        Args:
            original_query: The original search query
            analysis: Results from analyze_query

        Returns:
            List of verification queries
        """
        if not self.should_use_verification_search(analysis):
            return []

        try:
            # Get current date for context
            current_date = datetime.now().strftime("%Y-%m-%d")
            current_year = datetime.now().year

            prompt = f"""The user asked: "{original_query}"

TODAY'S DATE: {current_date} (Current year: {current_year})

This query appears to need fact-checking or verification. Generate 2-3 additional search queries that would help verify the accuracy of any claims or get more objective information about this topic.

Focus on:
- Finding authoritative sources
- Getting multiple perspectives
- Verifying specific claims or numbers
- Finding fact-checking information
- Use the correct current year ({current_year}) when generating queries about recent events

Respond with just the search queries, one per line, no explanations."""

            response = self.llm_client_manager.create_chat_completion(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Generate verification search queries to help fact-check information."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3
            )

            queries = response.choices[0].message.content.strip().split('\n')
            return [q.strip() for q in queries if q.strip()]

        except Exception as e:
            print_info(f"Failed to generate verification queries: {e}")
            return []

    def _clean_json_response(self, text: str) -> str:
        """
        Clean up common JSON formatting issues from LLM responses.

        Args:
            text: Raw LLM response text

        Returns:
            Cleaned text that's more likely to be valid JSON
        """
        # Remove markdown code blocks if present
        if text.startswith('```json'):
            text = text[7:]
        if text.startswith('```'):
            text = text[3:]
        if text.endswith('```'):
            text = text[:-3]

        # Strip whitespace
        text = text.strip()

        # Try to find JSON object bounds if there's extra text
        start_brace = text.find('{')
        end_brace = text.rfind('}')

        if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
            text = text[start_brace:end_brace + 1]

        return text

    def _validate_and_fix_analysis(self, analysis: Dict[str, Any], query: str) -> Dict[str, Any]:
        """
        Validate and fix the analysis results from LLM.

        Args:
            analysis: Parsed JSON analysis from LLM
            query: Original query for fallback

        Returns:
            Validated and fixed analysis
        """
        # Required fields with defaults
        defaults = {
            'query': query,
            'intent_type': 'general',
            'search_depth': 'basic',
            'topic_category': 'general',
            'freshness_days': None,
            'needs_verification': False,
            'max_results': 5,
            'search_strategy': 'standard',
            'confidence': 0.7,
            'reasoning': 'LLM analysis'
        }

        # Fill in missing fields with defaults
        for key, default_value in defaults.items():
            if key not in analysis:
                analysis[key] = default_value

        # Validate and fix field values
        if analysis['search_depth'] not in ['basic', 'advanced']:
            analysis['search_depth'] = 'basic'

        if analysis['topic_category'] not in ['news', 'finance', 'sports', 'general']:
            analysis['topic_category'] = 'general'

        if not isinstance(analysis['max_results'], int) or analysis['max_results'] < 3:
            analysis['max_results'] = 5
        elif analysis['max_results'] > 10:
            analysis['max_results'] = 10

        if analysis['search_strategy'] not in ['standard', 'verification', 'comprehensive', 'temporal']:
            analysis['search_strategy'] = 'standard'

        if not isinstance(analysis['confidence'], (int, float)) or analysis['confidence'] < 0:
            analysis['confidence'] = 0.5
        elif analysis['confidence'] > 1:
            analysis['confidence'] = 1.0

        return analysis