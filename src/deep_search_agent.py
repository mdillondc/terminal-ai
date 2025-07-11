"""
Deep Search Agent for Terminal AI Assistant

This module provides autonomous, intelligent web search capabilities that can
dynamically determine when sufficient information has been gathered to answer
complex queries comprehensively. Unlike basic search, this agent evaluates
its own progress and continues searching until it's confident it can provide
a thorough answer.
"""

import json
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from tavily_search import TavilySearch, TavilySearchError, create_tavily_search
from print_helper import print_md


class DeepSearchAgent:
    """
    Autonomous research agent that can intelligently evaluate its own progress
    and adapt search strategy dynamically to comprehensively answer complex queries.
    """

    def __init__(self, llm_client_manager, settings_manager):
        """
        Initialize the Deep Search Agent.

        Args:
            llm_client_manager: LLM client for AI operations
            settings_manager: Settings manager for configuration
        """
        self.llm_client_manager = llm_client_manager
        self.settings_manager = settings_manager
        self.search_client = create_tavily_search()

        if not self.search_client:
            raise Exception("Failed to initialize Tavily search client")

    def conduct_deep_search(self, query: str, context: str = "", model: str = "gpt-4o-mini") -> Tuple[List[str], List[Dict[str, str]]]:
        """
        Conduct an autonomous deep search that continues until comprehensive coverage is achieved.

        Args:
            query: The user's original query
            context: Conversation context for better understanding
            model: LLM model to use for evaluation and search strategy

        Returns:
            Tuple of (formatted_search_results, all_source_metadata)
        """
        print_md("**Deep Search Mode Activated**")
        print_md("AI will autonomously determine when sufficient information has been gathered...")

        all_search_results = []
        all_source_metadata = []
        seen_urls = set()
        search_iteration = 0
        max_searches = self.settings_manager.search_deep_max_queries

        # Generate initial search queries
        initial_queries = self._generate_initial_search_queries(query, context, model)
        strategy_text = "**Initial Research Strategy:**\n"
        for i, q in enumerate(initial_queries, 1):
            strategy_text += f"    {i}. {q}\n"
        print_md(strategy_text.rstrip())

        # Execute initial searches
        current_queries = initial_queries

        while search_iteration < max_searches:
            # Execute current batch of searches
            batch_results, batch_metadata = self._execute_search_batch(
                current_queries, seen_urls, search_iteration + 1
            )

            if batch_results:
                all_search_results.extend(batch_results)
                all_source_metadata.extend(batch_metadata)
                search_iteration += len(current_queries)
            else:
                break

            # Evaluate completeness
            if search_iteration >= max_searches:
                print_md(f"**Reached maximum search limit ({max_searches}). Concluding research.**")
                break

            combined_info = "\n".join(all_search_results)
            evaluation = self._evaluate_completeness(query, combined_info, context, model)

            evaluation_text = f"**Research Evaluation (after {search_iteration} searches):**\n"
            evaluation_text += f"    Completeness: {evaluation['completeness_score']}/10\n"
            evaluation_text += f"    Assessment: {evaluation['assessment']}"
            print_md(evaluation_text)

            if evaluation['is_sufficient']:
                print_md("    Decision: Research complete")
                break
            else:
                continue_text = f"    Decision: Continue research\n"
                continue_text += f"    Gaps identified: {', '.join(evaluation['gaps'])}"
                print_md(continue_text)

                # Generate targeted follow-up queries
                follow_up_queries = self._generate_follow_up_queries(
                    query, combined_info, evaluation['gaps'], model
                )

                if not follow_up_queries:
                    print_md("    No additional search strategies identified. Concluding research.")
                    break

                phase_text = "**Next Research Phase:**\n"
                for i, q in enumerate(follow_up_queries, 1):
                    phase_text += f"    {i}. {q}\n"
                print_md(phase_text.rstrip())

                current_queries = follow_up_queries

        # Final summary
        unique_sources = len(all_source_metadata)
        print_md(f"**Deep Search Complete:** {search_iteration} searches executed, {unique_sources} unique sources analyzed")

        return all_search_results, all_source_metadata

    def _generate_initial_search_queries(self, query: str, context: str, model: str) -> List[str]:
        """
        Generate initial broad search queries to begin research.

        Args:
            query: User's original query
            context: Conversation context
            model: LLM model to use

        Returns:
            List of initial search queries
        """
        current_date = datetime.now().strftime("%Y-%m-%d")
        current_year = datetime.now().year

        prompt = f"""You are a research strategist. Given a user's query, determine how many initial search queries are needed and generate them to gather the most important foundational information to answer their question thoroughly.

Today's date: {current_date} (year {current_year})

User's query: {query}

Context: {context}

Analyze the complexity and scope of this query, then generate the appropriate number of initial search queries that:
1. Cover the main topic from different angles
2. Include recent information when relevant (use {current_year} for current topics)
3. Are broad enough to gather foundational knowledge
4. Avoid redundancy
5. Match the complexity of the query (simple questions may need fewer queries, complex research may need more)

Respond with only the search queries, one per line, no explanations or numbering."""

        try:
            response = self.llm_client_manager.create_chat_completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )

            queries = response.choices[0].message.content.strip().split('\n')
            return [q.strip() for q in queries if q.strip()]

        except Exception as e:
            print_md(f"Warning: Error generating initial queries: {e}")
            # Fallback to user's original query
            return [query]

    def _execute_search_batch(self, queries: List[str], seen_urls: set, start_index: int) -> Tuple[List[str], List[Dict[str, str]]]:
        """
        Execute a batch of search queries and return results.

        Args:
            queries: List of search queries to execute
            seen_urls: Set of URLs already seen (for deduplication)
            start_index: Starting index for display numbering

        Returns:
            Tuple of (search_results, source_metadata)
        """
        batch_results = []
        batch_metadata = []

        for i, query in enumerate(queries):
            search_number = start_index + i

            try:
                # Execute search
                raw_results = self.search_client.search(
                    query=query,
                    max_results=self.settings_manager.search_deep_max_results_per_query,
                    auto_parameters=True
                )

                # Format results
                formatted_results = self.search_client.format_results_for_ai(raw_results, query)
                source_metadata = self.search_client.get_source_metadata(raw_results)

                if formatted_results:
                    batch_results.append(formatted_results)

                    # Add unique sources
                    unique_sources = []
                    for source in source_metadata:
                        if source.get('url') and source['url'] not in seen_urls:
                            unique_sources.append(source)
                            batch_metadata.append(source)
                            seen_urls.add(source['url'])

                    # Display results
                    if unique_sources:
                        search_text = f"**Search {search_number}:** {query}\n"
                        for source in unique_sources:
                            title = source.get('title', 'Unknown Source')
                            url = source.get('url', '')
                            if url:
                                # Remove square brackets from title for markdown links
                                clean_title = title.replace('[', '').replace(']', '')
                                search_text += f"    [{clean_title}]({url})\n"
                            else:
                                search_text += f"    {title}\n"
                        print_md(search_text.rstrip())

            except Exception as e:
                error_text = f"**Search {search_number}:** {query}\n"
                error_text += f"    Search failed: {e}"
                print_md(error_text)

        return batch_results, batch_metadata

    def _evaluate_completeness(self, original_query: str, gathered_info: str, context: str, model: str) -> Dict[str, Any]:
        """
        Evaluate whether gathered information is sufficient to comprehensively answer the query.

        Args:
            original_query: User's original query
            gathered_info: All information gathered so far
            context: Conversation context
            model: LLM model to use

        Returns:
            Dictionary with evaluation results
        """
        prompt = f"""You are a research evaluation expert. Analyze whether the gathered information is sufficient to comprehensively answer the user's query.

Original Query: {original_query}

Context: {context}

Gathered Information:
{gathered_info}

Evaluate the completeness and quality of information to answer the query. Consider:
1. Does the information directly address the main question?
2. Are there important aspects, perspectives, or details missing?
3. Is the information current and authoritative?
4. Are there potential counterarguments or alternative viewpoints not covered?
5. For complex topics, are mechanisms, causes, and effects explained?

Respond with a JSON object containing:
{{
    "completeness_score": (integer 1-10, where 10 is completely comprehensive),
    "is_sufficient": (boolean, true if ready to provide comprehensive answer),
    "assessment": (string, brief assessment of current information quality),
    "gaps": (array of strings, specific information gaps that need to be addressed),
    "confidence": (integer 1-10, confidence in this evaluation)
}}

Be thorough but practical - don't demand perfection if the information is reasonably comprehensive."""

        try:
            response = self.llm_client_manager.create_chat_completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )

            eval_text = response.choices[0].message.content.strip()

            # Try to parse JSON
            try:
                evaluation = json.loads(eval_text)

                # Validate required fields
                required_fields = ['completeness_score', 'is_sufficient', 'assessment', 'gaps', 'confidence']
                for field in required_fields:
                    if field not in evaluation:
                        raise ValueError(f"Missing required field: {field}")

                return evaluation

            except (json.JSONDecodeError, ValueError) as e:
                print_md(f"Warning: Could not parse evaluation JSON: {e}")
                # Fallback evaluation
                return {
                    "completeness_score": 6,
                    "is_sufficient": False,
                    "assessment": "Could not properly evaluate - continuing search",
                    "gaps": ["evaluation_error"],
                    "confidence": 3
                }

        except Exception as e:
            print_md(f"Warning: Error in completeness evaluation: {e}")
            # Conservative fallback
            return {
                "completeness_score": 5,
                "is_sufficient": False,
                "assessment": "Error in evaluation - continuing search to be safe",
                "gaps": ["evaluation_error"],
                "confidence": 2
            }

    def _generate_follow_up_queries(self, original_query: str, gathered_info: str, gaps: List[str], model: str) -> List[str]:
        """
        Generate targeted follow-up search queries to address identified gaps.

        Args:
            original_query: User's original query
            gathered_info: Information gathered so far
            gaps: List of identified information gaps
            model: LLM model to use

        Returns:
            List of targeted follow-up queries
        """
        current_year = datetime.now().year

        prompt = f"""You are a research strategist. Based on the information gathered so far and identified gaps, generate 1-3 highly targeted search queries that would address the most important missing information.

Original Query: {original_query}

Information Gathered So Far:
{gathered_info}

Identified Gaps:
{', '.join(gaps)}

Generate targeted search queries that:
1. Specifically address the most critical gaps
2. Are different from what has likely been searched already
3. Use precise terminology and focus
4. Include {current_year} for current information when relevant
5. Avoid redundancy with existing information

Respond with only the search queries, one per line, no explanations or numbering. If no meaningful follow-up queries can be generated, respond with just "NONE"."""

        try:
            response = self.llm_client_manager.create_chat_completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )

            result = response.choices[0].message.content.strip()

            if result.upper() == "NONE":
                return []

            queries = result.split('\n')
            return [q.strip() for q in queries if q.strip()]

        except Exception as e:
            print_md(f"Warning: Error generating follow-up queries: {e}")
            return []


def create_deep_search_agent(llm_client_manager, settings_manager) -> Optional[DeepSearchAgent]:
    """
    Factory function to create DeepSearchAgent with error handling.

    Args:
        llm_client_manager: LLM client manager instance
        settings_manager: Settings manager instance

    Returns:
        DeepSearchAgent instance if successful, None otherwise
    """
    try:
        return DeepSearchAgent(llm_client_manager, settings_manager)
    except Exception as e:
        print_md(f"Failed to initialize Deep Search Agent: {e}")
        return None