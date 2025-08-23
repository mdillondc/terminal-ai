"""
Deep Search Agent for Terminal AI Assistant

This module provides autonomous, intelligent web search capabilities that can
dynamically determine when sufficient information has been gathered to answer
complex queries comprehensively. Unlike basic search, this agent evaluates
its own progress and continues searching until it's confident it can provide
a thorough answer.
"""

import json
from typing import List, Dict, Optional, Any, Tuple, Union
from datetime import datetime
from json_repair import repair_json
from tavily_search import TavilySearch, create_tavily_search
from searxng_search import SearXNGSearch, create_searxng_search
from search_utils import extract_full_content_from_search_results
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

        # Initialize search client based on configured search engine
        search_engine = settings_manager.search_engine
        if search_engine == "searxng":
            search_client = create_searxng_search(settings_manager.searxng_base_url)
            if not search_client:
                raise Exception("Failed to initialize SearXNG search client")
            self.search_client: Union[TavilySearch, SearXNGSearch] = search_client
        else:
            # Default to Tavily
            search_client = create_tavily_search()
            if not search_client:
                raise Exception("Failed to initialize Tavily search client")
            self.search_client: Union[TavilySearch, SearXNGSearch] = search_client

    def conduct_deep_search(self, query: str, context: str = "", model: str = "gemini-2.5-flash") -> Tuple[List[str], List[Dict[str, str]]]:
        """
        Conduct an autonomous deep search that continues until comprehensive coverage is achieved.

        Args:
            query: The user's original query
            context: Conversation context for better understanding
            model: LLM model to use for evaluation and search strategy

        Returns:
            Tuple of (formatted_search_results, all_source_metadata)
        """
        search_engine_name = "Tavily" if self.settings_manager.search_engine == "tavily" else "SearXNG"
        activation_text = f"**Deep Search Mode Activated**\n"
        activation_text += f"Search engine: {search_engine_name}\n"
        activation_text += "AI will autonomously determine when sufficient information has been gathered..."
        print_md(activation_text)

        all_search_results = []
        all_source_metadata = []
        seen_urls = set()
        search_iteration = 0

        # Track evaluation progress to detect diminishing returns
        previous_completeness_scores = []
        highest_completeness_score = 0  # Track highest score to prevent illogical regression
        max_user_choice_iterations = 3

        # Generate initial search queries
        initial_queries = self._generate_initial_search_queries(query, context, model)
        strategy_text = "**Initial Research Strategy:**\n"
        for i, q in enumerate(initial_queries, 1):
            strategy_text += f"    {i}. {q}\n"
        print_md(strategy_text.rstrip())

        # Execute initial searches
        current_queries = initial_queries
        user_choice_iterations = 0

        while user_choice_iterations < max_user_choice_iterations:
            # Execute current batch of searches
            batch_results, batch_metadata = self._execute_search_batch(
                current_queries, seen_urls, search_iteration + 1
            )

            if batch_results:
                # Only add search results that contain unique sources
                unique_batch_results = []
                for i, result in enumerate(batch_results):
                    # Check if this search result corresponds to unique sources
                    if i < len(batch_metadata) and batch_metadata:
                        unique_batch_results.append(result)

                all_search_results.extend(unique_batch_results)
                all_source_metadata.extend(batch_metadata)
                search_iteration += len(current_queries)
            else:
                break

            # Evaluate completeness
            combined_info = "\n".join(all_search_results)
            evaluation = self._evaluate_completeness(query, combined_info, context, model)

            # Track highest score to prevent illogical regression due to LLM inconsistency
            raw_llm_score = evaluation['completeness_score']
            highest_completeness_score = max(highest_completeness_score, raw_llm_score)

            # Use highest score for display and logic (information only accumulates, scores shouldn't decrease)
            display_score = highest_completeness_score
            evaluation['completeness_score'] = display_score  # Update evaluation dict for consistency

            evaluation_text = f"**Research Evaluation (after {search_iteration} searches):**\n"
            evaluation_text += f"    Completeness: {display_score}/10\n"
            evaluation_text += f"    Assessment: {evaluation['assessment']}"
            print_md(evaluation_text)

            if evaluation['is_sufficient']:
                print_md("    **Decision:** Research complete")
                break
            else:
                # Check for diminishing returns
                current_score = display_score  # Use the highest score for diminishing returns logic
                previous_completeness_scores.append(current_score)

                # If we've had multiple evaluations and no improvement, detect diminishing returns
                if len(previous_completeness_scores) >= 2:
                    recent_scores = previous_completeness_scores[-2:]
                    if all(score == recent_scores[0] for score in recent_scores):
                        diminishing_text = f"**Diminishing returns detected:** Completeness remains at {current_score}/10 despite additional searches.\n"
                        diminishing_text += "    This suggests the missing information may not be available in current literature.\n"
                        diminishing_text += "    Concluding research with available information..."
                        print_md(diminishing_text)
                        break

                # Check for auto-stop on high quality scores
                current_score = evaluation['completeness_score']
                if current_score >= 9:
                    quality_desc, _ = self._get_quality_description(current_score)
                    print_md(f"**Decision:** Research quality is {quality_desc} ({current_score}/10). Auto-stopping with comprehensive coverage.")
                    break

                # Check for auto-continue on very poor scores
                if current_score <= 2:
                    quality_desc, _ = self._get_quality_description(current_score)
                    print_md(f"**Decision:** Research quality is {quality_desc} ({current_score}/10). Auto-continuing to gather more information...")
                    continue

                # Get quality assessment for user prompt
                quality_desc, recommendation = self._get_quality_description(current_score)
                continue_text = f"**Decision:** Research quality is {quality_desc} ({current_score}/10). {recommendation}\n"
                continue_text += f"**Gaps identified:** {', '.join(evaluation['gaps'])}"
                print_md(continue_text)

                # Ask user if they want to continue
                user_choice = self._get_user_continue_choice(evaluation['completeness_score'])

                if not user_choice:
                    print_md("**User chose to stop research.** Generating response with current information...")
                    break

                user_choice_iterations += 1

                # Generate targeted follow-up queries
                follow_up_queries = self._generate_follow_up_queries(
                    query, combined_info, evaluation['gaps'], model
                )

                if not follow_up_queries:
                    print_md("No additional search strategies identified. Concluding research.")
                    break

                phase_text = "**Next Research Phase:**\n"
                for i, q in enumerate(follow_up_queries, 1):
                    phase_text += f"    {i}. {q}\n"
                print_md(phase_text.rstrip())

                current_queries = follow_up_queries

        # Check if we hit the user choice iteration limit
        if user_choice_iterations >= max_user_choice_iterations:
            limit_text = f"**Research iteration limit reached ({max_user_choice_iterations} user choice cycles).**\n"
            limit_text += "    Concluding research to prevent excessive searches..."
            print_md(limit_text)

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
            Tuple of (search_results, source_metadata) - only unique results
        """
        batch_results = []
        batch_metadata = []
        stored_raw_results = []  # Store raw results for content extraction after display

        for i, query in enumerate(queries):
            search_number = start_index + i

            try:
                # Execute search
                raw_results = self.search_client.search(
                    query=query,
                    max_results=self.settings_manager.search_deep_max_results_per_query,
                    auto_parameters=True
                )

                # Format results and get source metadata
                formatted_results = self.search_client.format_results_for_ai(raw_results, query)
                source_metadata = self.search_client.get_source_metadata(raw_results)

                # Filter for unique sources only
                unique_sources = []
                for source in source_metadata:
                    if source.get('url') and source['url'] not in seen_urls:
                        unique_sources.append(source)
                        batch_metadata.append(source)
                        seen_urls.add(source['url'])

                # Only add search results to batch if they contain unique sources
                if unique_sources and formatted_results:
                    batch_results.append(formatted_results)

                    # Display results
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

                # Store raw results for content extraction after display (now that we know batch_index)
                stored_raw_results.append({
                    'raw_results': raw_results,
                    'query': query,
                    'search_number': search_number,
                    'batch_index': len(batch_results) - 1 if unique_sources and formatted_results else -1
                })

            except Exception as e:
                error_text = f"**Search {search_number}:** {query}\n"
                error_text += f"    Search failed: {e}"
                print_md(error_text)

        # Extract content if enabled for SearXNG - AFTER all search results are displayed
        if (self.settings_manager.search_engine == "searxng" and
            stored_raw_results):

            # Combine all URLs from all searches for ONE extraction
            all_results = []
            url_to_search_mapping = {}  # Map URL to search data for updating later

            for search_data in stored_raw_results:
                if search_data['batch_index'] >= 0:  # Only process successful searches
                    raw_results = search_data['raw_results']
                    for result in raw_results.get('results', []):
                        if result.get('url'):
                            all_results.append(result.copy())
                            url_to_search_mapping[result['url']] = search_data

            if all_results:
                # Create combined raw_results structure for ONE extraction
                combined_raw_results = {'results': all_results}

                # Do ONE content extraction for all URLs (shows one message)
                enhanced_combined_results = extract_full_content_from_search_results(
                    combined_raw_results, self.settings_manager, self.llm_client_manager
                )

                # Map enhanced results back to individual searches
                for enhanced_result in enhanced_combined_results.get('results', []):
                    url = enhanced_result.get('url')
                    if url in url_to_search_mapping:
                        search_data = url_to_search_mapping[url]
                        batch_idx = search_data['batch_index']
                        query = search_data['query']

                        # Update the original raw_results with enhanced content
                        raw_results = search_data['raw_results']
                        for i, original_result in enumerate(raw_results.get('results', [])):
                            if original_result.get('url') == url:
                                raw_results['results'][i] = enhanced_result
                                break

                        # Re-format results with enhanced content for AI
                        enhanced_results = self.search_client.format_results_for_ai(raw_results, query)

                        # Replace the corresponding entry in batch_results
                        batch_results[batch_idx] = enhanced_results

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

Only mark as sufficient (is_sufficient: true) if the completeness score is 10/10. For all other scores, mark as insufficient (is_sufficient: false) to allow user choice on whether to continue research."""

        try:
            response = self.llm_client_manager.create_chat_completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )

            eval_text = response.choices[0].message.content.strip()



            # Try to parse JSON with repair
            try:
                # First attempt to repair any malformed JSON (handles markdown blocks, etc.)
                cleaned_json = repair_json(eval_text)

                # Then parse the cleaned JSON
                evaluation = json.loads(cleaned_json)

                # Validate required fields
                required_fields = ['completeness_score', 'is_sufficient', 'assessment', 'gaps', 'confidence']
                for field in required_fields:
                    if field not in evaluation:
                        raise ValueError(f"Missing required field: {field}")

                return evaluation

            except Exception as repair_error:
                print_md(f"Warning: Could not repair/parse evaluation JSON: {repair_error}")
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

    def _get_quality_description(self, score: int) -> tuple[str, str]:
        """
        Return quality description and recommendation based on score.

        Args:
            score: Completeness score (1-10)

        Returns:
            Tuple of (quality_description, recommendation)
        """
        if score <= 3:
            return ("poor", "Continue searching to gather more comprehensive information?")
        elif score <= 5:
            return ("fair", "Continue searching to improve coverage?")
        elif score <= 7:
            return ("good", "Continue searching for higher completeness?")
        elif score <= 9:
            return ("very good", "Research is quite comprehensive. Continue for completeness?")
        else:
            return ("excellent", "Research complete - comprehensive coverage achieved")

    def _get_user_continue_choice(self, completeness_score: int) -> bool:
        """
        Get user choice on whether to continue deep search.

        Args:
            completeness_score: Current completeness score (1-10)

        Returns:
            True if user wants to continue, False if they want to stop
        """
        try:


            print_md("**[C]ontinue deep search or [S]top and generate response**")
            print("    Choice: ", end="", flush=True)

            while True:
                choice = input().strip().lower()

                if choice in ['c', 'continue']:
                    print_md("    Continuing research for higher completeness...")
                    return True
                elif choice in ['s', 'stop']:
                    return False
                else:
                    print("    Please enter 'c' to continue or 's' to stop: ", end="", flush=True)

        except (KeyboardInterrupt, EOFError):
            # Handle Ctrl+C or EOF gracefully
            print_md("    Research interrupted by user. Generating response with current information...")
            return False
        except Exception as e:
            print_md(f"    Error getting user input: {e}. Continuing research...")
            return True


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