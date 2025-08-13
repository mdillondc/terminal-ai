from datetime import datetime
import os
import select
import sys
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Any, List
from settings_manager import SettingsManager

from tavily_search import create_tavily_search, TavilySearchError
from searxng_search import create_searxng_search, SearXNGSearchError
from search_utils import extract_full_content_from_search_results
from deep_search_agent import create_deep_search_agent
from print_helper import print_md
from constants import ColorConstants, ConversationConstants
from llm_client_manager import LLMClientManager
from print_helper import print_md, print_lines
from rich.console import Console



class ConversationManager:
    def __init__(self, client: Any, model: Optional[str] = None) -> None:
        self.client = client
        self._original_openai_client = client  # Store original OpenAI client for switching back
        self._model = model
        self.conversation_history = []
        self.settings_manager = SettingsManager.getInstance()
        self.console = Console()
        self.log_renamed = False  # Track if we've already renamed the log with AI-generated title
        self._response_buffer = ""  # Buffer to accumulate response text for thinking coloring
        self._execution_buffer = ""  # Buffer to accumulate potential execution commands

        # Initialize LLM client manager for multi-provider support
        self.llm_client_manager = LLMClientManager(self._original_openai_client)

        # Initialize RAG engine - import here to avoid circular imports
        try:
            from rag_engine import RAGEngine
            self.rag_engine = RAGEngine(self._original_openai_client)
        except ImportError as e:
            print_md(f"Warning: Could not initialize RAG engine: {e}")
            self.rag_engine = None



    def log_context(self, content: str, role: str = "user") -> None:
        """Add content that LLM needs to see - goes to conversation history"""
        # Skip if incognito mode is enabled
        if self.settings_manager.setting_get("incognito"):
            return

        # Add to conversation history for LLM/JSON
        self.conversation_history.append({"role": role, "content": content})



    def _process_and_print_chunk(self, chunk: str) -> None:
        """
        Process a response chunk, applying thinking text coloring while maintaining streaming.
        Hides empty thinking blocks that contain only whitespace.
        """
        # ANSI color codes from constants
        GRAY = ColorConstants.THINKING_GRAY  # Light gray for thinking text
        RESET = ColorConstants.RESET

        # Initialize thinking state if needed
        if not hasattr(self, '_in_thinking_block'):
            self._in_thinking_block = False
        if not hasattr(self, '_thinking_buffer'):
            self._thinking_buffer = ""  # Buffer for potential empty thinking block
        if not hasattr(self, '_thinking_started_output'):
            self._thinking_started_output = False  # Track if we've output thinking content
        if not hasattr(self, '_skip_leading_whitespace'):
            self._skip_leading_whitespace = False



        # Combine with any buffered content
        text_to_process = self._response_buffer + chunk
        self._response_buffer = ""

        output = ""
        i = 0

        while i < len(text_to_process):
            # Look for complete <think> tag
            if text_to_process[i:i+7] == '<think>':
                if not self._in_thinking_block:
                    self._in_thinking_block = True
                    self._thinking_buffer = ""
                    self._thinking_started_output = False
                i += 7
                continue



            # Look for complete </think> tag
            if text_to_process[i:i+8] == '</think>':
                if self._in_thinking_block:
                    if self._thinking_started_output:
                        # We output thinking content, so close it normally
                        output += '</think>' + RESET
                    elif self._thinking_buffer.strip():
                        # We have non-empty buffered content, output it all at once
                        output += GRAY + '<think>' + self._thinking_buffer + '</think>' + RESET
                    else:
                        # Empty thinking block - skip it and following whitespace
                        self._skip_leading_whitespace = True
                    # Reset thinking state
                    self._in_thinking_block = False
                    self._thinking_buffer = ""
                    self._thinking_started_output = False
                else:
                    output += '</think>'
                i += 8
                continue





            # Check for potential partial tags at the end that we should buffer
            remaining = text_to_process[i:]
            if i == len(text_to_process) - len(remaining):  # At the end
                if (remaining.startswith('<think') and len(remaining) < ConversationConstants.THINKING_TAG_MIN_LENGTH) or \
                   (remaining.startswith('</think') and len(remaining) < ConversationConstants.CLOSE_TAG_MIN_LENGTH) or \
                   (remaining == '<' or remaining.startswith('<') and len(remaining) < ConversationConstants.PARTIAL_TAG_MAX_LENGTH):
                    # Keep potential partial tag in buffer
                    self._response_buffer = remaining
                    break

            # Regular character
            if self._in_thinking_block:
                # Check if this is the first non-whitespace character
                if not self._thinking_started_output and text_to_process[i] not in ' \t\n\r':
                    # First non-whitespace in thinking block - output buffered content and start streaming
                    output += GRAY + '<think>' + self._thinking_buffer + text_to_process[i]
                    self._thinking_started_output = True
                elif not self._thinking_started_output:
                    # Still in buffer phase - accumulate whitespace
                    self._thinking_buffer += text_to_process[i]
                else:
                    # Already started output - stream normally with color
                    output += text_to_process[i]

            else:
                # Check if we should skip leading whitespace after hidden thinking block
                if self._skip_leading_whitespace and text_to_process[i] in ' \t\n\r':
                    # Skip whitespace character
                    pass
                else:
                    # Regular text outside thinking blocks
                    if self._skip_leading_whitespace:
                        self._skip_leading_whitespace = False
                    output += text_to_process[i]
            i += 1

        # Print immediately to maintain streaming
        if output:
            print(output, end="", flush=True)

    def _adjust_markdown_headers_for_streamdown(self, chunk: str) -> str:
        """
        Adjust markdown headers for streamdown display by converting level 1 and 2 headers to level 3.
        """
        import re

        # Convert L1/L2 headers to L3, capturing title to rebuild the line correctly.
        chunk = re.sub(r'^[ \t]*(#{1,2})(?!#)[ \t]*(.*)', r'### \2', chunk, flags=re.MULTILINE)

        return chunk


    def _start_streamdown_process(self):
        """Starts and returns a streamdown subprocess for markdown rendering."""
        from settings_manager import SettingsManager
        streamdown_cmd = SettingsManager.getInstance().markdown_settings
        return subprocess.Popen(
            streamdown_cmd,
            stdin=subprocess.PIPE,
            text=True
        )

    @property
    def model(self) -> str:
        return self._model or self.settings_manager.setting_get("model")

    @model.setter
    def model(self, value: Optional[str]) -> None:
        self._model = value
        if value:
            self._update_client_for_model(value)

    def _is_ollama_available(self) -> bool:
        """Check if Ollama is available - delegated to LLMClientManager"""
        return self.llm_client_manager._is_ollama_available()

    def _is_ollama_model(self, model_name: str) -> bool:
        """Detect if a model is from Ollama - uses availability detection"""
        return self.llm_client_manager._get_provider_for_model(model_name) == 'ollama'

    def _is_google_model(self, model_name: str) -> bool:
        """Detect if a model is from Google - uses availability detection"""
        return self.llm_client_manager._get_provider_for_model(model_name) == 'google'

    def _update_client_for_model(self, model_name: str) -> None:
        """Update client based on model source - delegated to LLMClientManager"""
        try:
            self.client = self.llm_client_manager._get_client_for_model(model_name)
        except Exception as e:
            print_md(f"Warning: Could not get client for {model_name}: {e}")
            # Fall back to original OpenAI client
            self.client = self._original_openai_client

    def generate_response(self, instructions: Optional[str] = None) -> None:

        # Check if search is enabled and handle search workflow
        if self.settings_manager.setting_get("search") and self.conversation_history:
            self._handle_search_workflow()
        elif self.settings_manager.setting_get("search_deep") and self.conversation_history:
            self._handle_deep_search_workflow()

        # Check if RAG is active and inject context
        rag_sources = []
        if self.rag_engine and self.rag_engine.is_active() and self.conversation_history:
            # Get the last user message for RAG query
            user_messages = [msg for msg in self.conversation_history if msg['role'] == 'user']
            if user_messages:
                last_user_message = user_messages[-1]['content']
                rag_context, rag_sources = self.rag_engine.get_context_for_query(last_user_message)

                if rag_context:
                    # Insert RAG context as a system message before the AI response
                    self.log_context(rag_context, "system")



        # Display AI name to user
        print(f"\n{self.settings_manager.get_ai_name_with_instructions()}:")

        # Setup stream to receive response from AI
        stream = self.llm_client_manager.create_chat_completion(
            model=self.model, messages=self.conversation_history, stream=True
        )

        # Init variable to hold AI response in its entirety
        ai_response = ""

        # Check if markdown is enabled for response handling
        try:
            markdown_enabled = self.settings_manager.setting_get("markdown")
        except KeyError:
            markdown_enabled = False



        # Start streamdown process for real-time markdown rendering
        streamdown_process = None
        if markdown_enabled:
            streamdown_process = self._start_streamdown_process()

        # Process response stream with interrupt checking between chunks
        interrupted = False
        try:
            for chunk in stream:
                # Check for 'q + enter' interrupt before processing each chunk
                if self._check_for_interrupt():
                    print_md("Response interrupted by user")
                    interrupted = True
                    break

                if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'content') and delta.content is not None:
                        ai_response_chunk = delta.content
                        ai_response += ai_response_chunk
                        if markdown_enabled and streamdown_process and streamdown_process.stdin:
                            # Adjust headers for streamdown display and send chunk
                            adjusted_chunk = self._adjust_markdown_headers_for_streamdown(ai_response_chunk)
                            streamdown_process.stdin.write(adjusted_chunk)
                            streamdown_process.stdin.flush()
                        else:
                            # Normal processing for non-markdown mode
                            self._process_and_print_chunk(ai_response_chunk)
        except Exception as e:
            print_md(f"Error processing response stream: {e}")

        # Close streamdown process if it was used
        if markdown_enabled and streamdown_process and streamdown_process.stdin:
            # Send any remaining buffer content to streamdown
            if hasattr(self, '_response_buffer') and self._response_buffer:
                adjusted_buffer = self._adjust_markdown_headers_for_streamdown(self._response_buffer)
                streamdown_process.stdin.write(adjusted_buffer)
                streamdown_process.stdin.flush()

            # Add final newline to ensure streamdown processes the content
            streamdown_process.stdin.write('\n')
            streamdown_process.stdin.flush()

            # Close stdin to signal end of input
            streamdown_process.stdin.close()

            # Wait for streamdown to finish processing
            streamdown_process.wait()

        # Handle remaining buffer content for non-markdown mode
        complete_response = ai_response
        if hasattr(self, '_response_buffer') and self._response_buffer:
            complete_response += self._response_buffer
            if not markdown_enabled:
                print(self._response_buffer, end="", flush=True)
            self._response_buffer = ""



        # Reset thinking state for next response
        if hasattr(self, '_in_thinking_block'):
            self._in_thinking_block = False
        if hasattr(self, '_thinking_buffer'):
            self._thinking_buffer = ""
        if hasattr(self, '_thinking_started_output'):
            self._thinking_started_output = False
        if hasattr(self, '_skip_leading_whitespace'):
            self._skip_leading_whitespace = False

        # Reset markdown buffer and line counter for next response
        if hasattr(self, '_markdown_buffer'):
            self._markdown_buffer = ""
        if hasattr(self, '_markdown_raw_lines'):
            self._markdown_raw_lines = 0




        # Only save if we got a response
        if complete_response:
            # Append complete_response to the conversation_history array
            self.log_context(complete_response, "assistant")

            # Add newline after AI response for proper spacing
            print()

            # Display RAG sources if any were used
            if rag_sources and self.rag_engine:
                print_md(self.rag_engine.format_sources(rag_sources))

            self.log_save()

            # Check if this is the first user-AI exchange and rename log with descriptive title
            self._check_and_rename_log_after_first_exchange(interrupted)



        # Ensure proper spacing before next user prompt
        if complete_response:
            # Skip extra newline in markdown mode since we already add one for streamdown
            if not self.settings_manager.setting_get("markdown"):
                print()




    def _handle_search_workflow(self) -> None:
        """
        Handle the search workflow when search mode is enabled.
        Rewrites the user's question as a search query, performs search, and adds results to conversation.
        """
        try:
            # Get the last user message
            last_user_message = None
            for message in reversed(self.conversation_history):
                if message["role"] == "user":
                    last_user_message = message["content"]
                    break

            if not last_user_message:
                print_md("No user message found for search")
                return



            # Build full conversation context with recent messages first
            recent_messages = []
            earlier_messages = []

            # Get recent messages for immediate context
            recent_window = min(ConversationConstants.RECENT_MESSAGES_WINDOW, len(self.conversation_history))
            for message in self.conversation_history[-recent_window:]:
                if message["role"] in ["user", "assistant"]:
                    content = message["content"]
                    recent_messages.append(f"{message['role']}: {content}")

            # Get earlier messages for reference resolution
            if len(self.conversation_history) > recent_window:
                for message in self.conversation_history[:-recent_window]:
                    if message["role"] in ["user", "assistant"]:
                        content = message["content"]
                        earlier_messages.append(f"{message['role']}: {content}")

            # Structure context with recent first, then earlier
            context_parts = []
            if recent_messages:
                context_parts.append("RECENT CONVERSATION:")
                context_parts.extend(recent_messages)

            if earlier_messages:
                context_parts.append("\nEARLIER CONVERSATION (for reference resolution):")
                context_parts.extend(earlier_messages)

            context_text = "\n".join(context_parts) if context_parts else "No prior context."

            # Extract key topics/entities from conversation context for better search queries
            key_topics = self._extract_key_topics_from_context(context_text)
            topics_text = f"Key topics from conversation: {', '.join(key_topics)}" if key_topics else ""

            # Get current date for search context
            current_date = datetime.now().strftime("%Y-%m-%d")
            current_year = datetime.now().year

            # Create a temporary conversation to generate search query
            search_query_conversation = [
                {
                    "role": "system",
                    "content": f"You are a search query optimizer. Today's date is {current_date} (year {current_year}). Given a user's question or statement and the conversation context, rewrite it as 1-3 optimal search queries that would find the most relevant and current information to answer their question. If the question is very basic, such as a persons age, simple arithmetic, or you otherwise think the question can be answered with only one query, do not generate more than one query. Consider the conversation context to understand what the user is really asking about. When generating queries about recent events, use the correct current year ({current_year}). Respond with only the search queries, one per line, no explanations."
                },
                {
                    "role": "user",
                    "content": f"CONVERSATION CONTEXT:\n{context_text}\n\n{topics_text}\n\nCURRENT QUESTION: {last_user_message}\n\nRewrite the current question as optimal search queries, considering the conversation context and key topics:"
                }
            ]

            # Generate search query using AI
            query_response = self.llm_client_manager.create_chat_completion(
                model=self.model,
                messages=search_query_conversation,
                temperature=0.3
            )

            search_queries = query_response.choices[0].message.content.strip().split('\n')
            search_queries = [q.strip() for q in search_queries if q.strip()]

            # Perform searches - route to appropriate search engine
            search_engine = self.settings_manager.search_engine
            if search_engine == "searxng":
                search_client = create_searxng_search(self.settings_manager.searxng_base_url)
                if not search_client:
                    print_md("Failed to initialize SearXNG search client. Continuing without search")
                    return
            else:
                # Default to Tavily
                search_client = create_tavily_search()
                if not search_client:
                    print_md("Failed to initialize Tavily search client. Continuing without search")
                    return

            all_search_results = []
            max_queries = self.settings_manager.search_max_queries

            # Print queries
            query_content = "Generating search queries...\n"
            for query in search_queries[:max_queries]:
                query_content += f"  {query}\n"
            if query_content:
                print_md(query_content.rstrip())

            # Set search parameters based on search engine
            if search_engine == "searxng":
                # SearXNG doesn't support auto_parameters
                search_params = {
                    'max_results': self.settings_manager.search_max_results
                }
            else:
                # Use Tavily's auto-parameters for intelligent search optimization
                search_params = {
                    'auto_parameters': True,
                    'max_results': self.settings_manager.search_max_results  # Must be set manually per Tavily docs
                }

            # Inform user about search configuration
            search_engine_name = "Tavily" if search_engine == "tavily" else "SearXNG"
            print_md(f"Search engine: {search_engine_name}")

            all_source_metadata = []
            seen_urls = set()
            display_seen_urls = set()
            stored_raw_results = []  # Store raw results for content extraction after display



            # Execute all searches in parallel
            with ThreadPoolExecutor(max_workers=self.settings_manager.concurrent_workers) as executor:
                # Start all searches at once
                future_to_query = {
                    executor.submit(search_client.search, query, **search_params): query
                    for query in search_queries[:max_queries]
                }

                # Process results as they complete
                for i, future in enumerate(as_completed(future_to_query), 1):
                    query = future_to_query[future]
                    search_section = f"Searching ({i}/{min(len(search_queries), max_queries)}): {query}\n"

                    try:
                        # Get search results
                        raw_results = future.result()

                        # Store raw results for potential content extraction after display
                        stored_raw_results.append({
                            'raw_results': raw_results,
                            'query': query,
                            'search_number': i
                        })

                        # Format results for display
                        results = search_client.format_results_for_ai(raw_results, query)
                        source_metadata = search_client.get_source_metadata(raw_results)

                        if results:
                            all_search_results.append(results)
                            # Add sources, avoiding duplicates by URL
                            for source in source_metadata:
                                if source.get('url') and source['url'] not in seen_urls:
                                    all_source_metadata.append(source)
                                    seen_urls.add(source['url'])

                            # Build source lines, avoiding duplicates
                            source_lines = []
                            if source_metadata:
                                for source in source_metadata:
                                    title = source.get('title', 'Unknown Source')
                                    url = source.get('url', '')

                                    if url and url not in display_seen_urls:
                                        # Remove square brackets from title for markdown links
                                        clean_title = title.replace('[', '').replace(']', '')
                                        source_lines.append(f"    [{clean_title}]({url})")
                                        display_seen_urls.add(url)
                                    elif not url:
                                        source_lines.append(f"    {title}")

                            # Only display search section if it contributed unique sources
                            if source_lines:
                                search_section += "\n".join(source_lines)
                                # Print the complete section
                                print_md(search_section)
                            # If no unique sources found, skip displaying this search section entirely

                    except (TavilySearchError, SearXNGSearchError) as e:
                        search_section += f"    Search failed: {e}"
                        # Always print error sections
                        print_md(search_section)
                    except Exception as e:
                        search_section += f"    Search failed: {e}"
                        # Always print error sections
                        print_md(search_section)

            # Extract full content if enabled for SearXNG - AFTER all search results are displayed
            if (search_engine == "searxng" and
                self.settings_manager.searxng_extract_full_content and
                stored_raw_results):

                # Combine all URLs from all searches for ONE extraction
                all_results = []
                url_to_search_mapping = {}  # Map URL to search data for updating later

                for search_data in stored_raw_results:
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
                            query = search_data['query']

                            # Update the original raw_results with enhanced content
                            raw_results = search_data['raw_results']
                            for i, original_result in enumerate(raw_results.get('results', [])):
                                if original_result.get('url') == url:
                                    raw_results['results'][i] = enhanced_result
                                    break

                            # Update the corresponding entry in all_search_results with enhanced content
                            enhanced_results = search_client.format_results_for_ai(raw_results, query)

                            # Find and replace the matching result in all_search_results
                            for idx, existing_result in enumerate(all_search_results):
                                if query in existing_result:
                                    all_search_results[idx] = enhanced_results
                                    break




            if all_search_results:
                # Combine all search results
                combined_results = "\n\n" + "="*80 + "\n".join(all_search_results) + "\n" + "="*80 + "\n"

                # Add search results as a system message to provide context
                search_content = f"SEARCH RESULTS FOR USER'S QUERY:\n{combined_results}\n\nUse this information to provide a comprehensive and current answer to the user's question. MANDATORY: You MUST always conclude your response with a 'Sources:' section that includes:\n\n1. A numbered list of the sources used in your answer\n2. Each source must include the full URL as a clickable link in markdown format: [Title](URL)\n3. Prefer recent, authoritative sources over older or less credible ones\n4. Always include this sources section even if you only reference one source\n\nExample format:\n\n## Sources:\n1. [Article Title](https://example.com/article)\n2. [Another Source](https://example.com/source2)"

                # Add search context to conversation
                self.log_context(search_content, "system")
                unique_sources = len(all_source_metadata)
                print_md(f"Synthesizing {unique_sources} sources...")
            else:
                print_md("No search results found. Continuing without search data")

        except Exception as e:
            print_md(f"Search workflow error: {e}. Continuing without search")
            return

    def _handle_deep_search_workflow(self) -> None:
        """
        Handle the deep search workflow when deep search mode is enabled.
        Uses autonomous research agent to intelligently gather comprehensive information.
        """
        try:
            # Get the last user message
            last_user_message = ""
            for message in reversed(self.conversation_history):
                if message.get("role") == "user":
                    last_user_message = message.get("content", "")
                    break

            if not last_user_message:
                return

            # Build context from recent conversation
            context_parts = []
            recent_messages = self.conversation_history[-self.settings_manager.search_context_window:]

            for message in recent_messages:
                if message.get("role") in ["user", "assistant"]:
                    content = message.get("content", "")
                    # No truncation - pass full content to LLM
                    context_parts.append(f"{message['role']}: {content}")

            context_text = "\n".join(context_parts) if context_parts else ""

            # Create and execute deep search
            deep_search_agent = create_deep_search_agent(self.llm_client_manager, self.settings_manager)
            if not deep_search_agent:
                print_md("Failed to initialize Deep Search Agent. Continuing without search")
                return

            # Execute autonomous deep search
            search_results, source_metadata = deep_search_agent.conduct_deep_search(
                query=last_user_message,
                context=context_text,
                model=self.model
            )

            if search_results:
                # Combine all search results
                combined_results = "\n\n" + "="*80 + "\n".join(search_results) + "\n" + "="*80 + "\n"

                # Add search results as a system message to provide context
                search_content = f"COMPREHENSIVE RESEARCH RESULTS FOR USER'S QUERY:\n{combined_results}\n\nUse this extensive research to provide a thorough, well-sourced answer to the user's question. MANDATORY: You MUST always conclude your response with a 'Sources:' section that includes:\n\n1. A numbered list of the sources used in your answer\n2. Each source must include the full URL as a clickable link in markdown format: [Title](URL)\n3. Prioritize the most authoritative and recent sources\n4. Always include this sources section even if you only reference one source\n\nExample format:\n\n## Sources:\n1. [Article Title](https://example.com/article)\n2. [Another Source](https://example.com/source2)"

                # Add search context to conversation
                self.log_context(search_content, "system")
                print_md(f"**Research synthesis complete** - generating comprehensive response...\n")
            else:
                print_md("No search results found. Continuing without search data")

        except Exception as e:
            print_md(f"Deep search workflow error: {e}. Continuing without search")
            return




    def _extract_key_topics_from_context(self, context_text: str) -> List[str]:
        """
        Extract key topics/entities from conversation context using LLM analysis.

        Args:
            context_text: The conversation context text

        Returns:
            List of key topics/entities found in the context
        """
        if not context_text or context_text == "No prior context.":
            return []

        try:
            # Use LLM to dynamically extract key topics
            extraction_prompt = f"""Analyze this conversation context and extract the 3-5 most important topics, entities, people, places, or concepts that would be relevant for search queries.

CONVERSATION CONTEXT:
{context_text}

Extract key topics that would help understand what the conversation is about. Focus on:
- People's names
- Organizations, companies, locations
- Important concepts or subjects being discussed
- Events or situations mentioned

Respond with just the key topics, one per line, no explanations. Maximum 5 topics."""

            response = self.llm_client_manager.create_chat_completion(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Extract key topics from conversation context to help with search queries."
                    },
                    {
                        "role": "user",
                        "content": extraction_prompt
                    }
                ],
                temperature=0.1
            )

            topics_text = response.choices[0].message.content.strip()
            topics = [topic.strip() for topic in topics_text.split('\n') if topic.strip()]

            return topics[:5]  # Limit to top 5 topics

        except Exception as e:
            print_md(f"Topic extraction failed: {e}")
            return []




    def _check_for_interrupt(self) -> bool:
        """Check for 'q' + Enter. Returns True if 'q' was entered."""
        # Skip if not in interactive terminal
        if not sys.stdin.isatty():
            return False

        try:
            # Check if input is available with small timeout
            if select.select([sys.stdin], [], [], 0.05)[0]:
                try:
                    line = sys.stdin.readline().strip()
                    if line.lower() == 'q':
                        return True
                    return False
                except (EOFError, KeyboardInterrupt):
                    return True
            return False
        except Exception:
            return False

    def apply_instructions(self, file_name: Optional[str], old_file_name: Optional[str] = None) -> None:
        if file_name is None:
            print_md("Please specify the instructions file to use")
            return

        new_file_path = self.settings_manager.setting_get("working_dir") + "/instructions/" + file_name

        if not os.path.exists(new_file_path):
            print_md(f"{new_file_path} does not exist")
        else:
            if old_file_name:
                # Remove old instructions from conversation_history
                self.conversation_history = [entry for entry in self.conversation_history if f'instructions:{old_file_name}' not in entry['content']]

            # Update settings
            self.settings_manager.setting_set("instructions", file_name)

            # Read instructions
            instructions = self.read_file(self.settings_manager.setting_get("working_dir") + "/instructions/" + self.settings_manager.setting_get("instructions"))
            today = "\n\nFYI - Today's date and time: " + datetime.now().strftime('%A %Y-%m-%d %H:%M')

            # Append new instructions to conversation_history
            # TODO: Consider if it'd be better to prepend instead of append (evaluate GPT response performance over time)
            self.log_context(f"instructions:{file_name}\n" + instructions + today, "system")

            # Inform user
            notice = f"Instructions {file_name}"
            if old_file_name:
                print_md(notice)


    @staticmethod
    def read_file(file_path: str) -> str:
        with open(file_path, "r") as file:
            return file.read()

    def log_save(self) -> None:
        # Skip saving if incognito mode is enabled
        if self.settings_manager.setting_get("incognito"):
            return

        current_log_file_location = self.settings_manager.setting_get("log_file_location")
        if current_log_file_location:
            if os.path.exists(current_log_file_location):
                os.remove(current_log_file_location)

        instructions_file_name = self.settings_manager.setting_get("instructions").rsplit('.', 1)[0] # strip extension
        log_file_name = self.settings_manager.setting_get("log_file_name")
        log_file_path = os.path.join(self.settings_manager.setting_get("working_dir"), f"logs/{instructions_file_name}/")

        # Remove .md extension if present for base name
        if log_file_name.endswith('.md'):
            log_file_name = log_file_name[:-3]

        log_file_location = os.path.join(log_file_path, log_file_name)

        if not os.path.exists(log_file_path):
            os.makedirs(log_file_path)

        # Save conversation history as JSON only
        conversation_history = json.dumps(self.conversation_history, indent=4)
        json_file_location = log_file_location + ".json"
        with open(json_file_location, 'w') as file:
            file.write(conversation_history)

        self.settings_manager.setting_set("log_file_location", json_file_location)

    def _check_and_rename_log_after_first_exchange(self, interrupted: bool = False) -> None:
        """Check if this is the first user-AI exchange and rename log with AI-generated title"""
        if self.log_renamed:
            return  # Already renamed

        # Skip renaming if incognito mode is enabled
        if self.settings_manager.setting_get("incognito"):
            return

        # Count user and assistant messages (excluding system/instructions)
        user_messages = [msg for msg in self.conversation_history if msg['role'] == 'user']
        assistant_messages = [msg for msg in self.conversation_history if msg['role'] == 'assistant']

        # Need at least 1 user message and 1 assistant response
        if len(user_messages) >= 1 and len(assistant_messages) >= 1:
            try:
                # Generate descriptive title using AI
                descriptive_title = self._generate_log_title()
                if descriptive_title:
                    self._rename_log_files_with_title(descriptive_title, interrupted)
                    self.log_renamed = True
            except Exception as e:
                print_md(f"Could not generate descriptive log title: {e}")
                # Continue without renaming - not critical functionality

    def _create_title_generation_prompt(self, context: str) -> str:
        """Create the standardized prompt for AI title generation"""
        return """You are an expert at creating precise, descriptive filenames that capture the ACTUAL CONTENT and SUBSTANCE of conversations, not the format or source.

Context: "{}"

CRITICAL INSTRUCTIONS:
- Focus on WHAT is being discussed, NOT HOW it was obtained
- Extract the core topic, subject matter, or key concepts
- Ignore format words like: transcript, video, analysis, summary, document, file, etc.
- 3-8 words maximum
- Use lowercase letters, numbers, and hyphens only

CONTENT-FOCUSED EXAMPLES:
- If analyzing React performance → "react-component-optimization"
- If explaining database queries → "sql-join-query-design"
- If about Docker deployment → "docker-container-deployment"

AVOID FORMAT WORDS:
❌ "youtube-video-transcript-analysis" → Focus on what the video discusses
❌ "document-summary-review" → Focus on the document's topic
❌ "file-processing-tutorial" → Focus on what's being processed
❌ "conversation-about-coding" → Focus on the specific coding topic

EXTRACT THE SUBSTANCE: What is the core subject, technology, concept, person, event, or topic being discussed?

Generate only the filename focusing on content substance:""".format(context[:1000])

    def _generate_title_from_context(self, context: str) -> Optional[str]:
        """Generate a title using AI given conversation context"""
        try:
            title_prompt = self._create_title_generation_prompt(context)

            # Get title from AI
            response = self.llm_client_manager.create_chat_completion(
                model=self.model,
                messages=[{"role": "user", "content": title_prompt}],
                max_tokens=1000,
                temperature=0.1
            )

            if response.choices and response.choices[0].message.content:
                title = response.choices[0].message.content.strip()
                # Clean up the title - ensure it's filename-safe
                title = title.lower().replace(' ', '-').replace('_', '-')
                # Remove any non-alphanumeric characters except hyphens
                title = re.sub(r'[^a-z0-9-]', '', title)
                # Remove multiple consecutive hyphens
                title = re.sub(r'-+', '-', title)
                # Remove leading/trailing hyphens
                title = title.strip('-')
                # Ensure it's not empty
                if title and len(title) >= 3:
                    return title

        except Exception as e:
            print(f"Error generating title: {e}")

        return None

    def _generate_log_title(self) -> Optional[str]:
        """Generate a 3-8 word descriptive title for the conversation using AI"""
        try:
            # Get user messages to check if we have meaningful content
            user_messages = [msg for msg in self.conversation_history if msg['role'] == 'user']
            if not user_messages:
                return "general-conversation"

            # Get the first user message content for analysis
            first_user_msg = user_messages[0]['content']

            # If the message is too short or generic, use fallback
            if len(first_user_msg.strip()) < 10 or first_user_msg.lower().strip() in ['hello', 'hi', 'hey', 'test']:
                return "general-conversation"

            # Build context with first user message and AI response
            context_parts = [f"User: {first_user_msg[:1500]}"]

            # Find the first assistant response
            for msg in self.conversation_history:
                if msg['role'] == 'assistant':
                    context_parts.append(f"Assistant: {msg['content'][:1500]}")
                    break

            context = "\n".join(context_parts)

            # Use the shared title generation method
            title = self._generate_title_from_context(context)
            if title:
                return title

        except Exception as e:
            print_md(f"Error generating title: {e}")

        # Fallback to generic name if all else fails
        return "general-conversation"

    def _rename_log_files_with_title(self, title: str, interrupted: bool = False) -> None:
        """Rename the current log file to include the descriptive title"""
        try:
            current_log_location = self.settings_manager.setting_get("log_file_location")
            if not current_log_location:
                return

            # Extract date from current filename (base name without extension)
            current_filename = os.path.basename(current_log_location)

            # Extract date part (format: 2025-06-08_timestamp)
            date_part = current_filename.split('_')[0] if '_' in current_filename else current_filename

            # Add unix timestamp to prevent conflicts
            import time
            timestamp = int(time.time())

            # Create new base filename: date_descriptive-title_timestamp
            new_filename = f"{date_part}_{title}_{timestamp}"

            # Get directory path
            log_directory = os.path.dirname(current_log_location)
            new_log_location = os.path.join(log_directory, new_filename)

            # Rename JSON file
            json_current = current_log_location
            json_new = new_log_location + ".json"
            if os.path.exists(json_current):
                os.rename(json_current, json_new)

            # Update settings with new location
            self.settings_manager.setting_set("log_file_location", new_log_location + ".json")
            self.settings_manager.setting_set("log_file_name", new_filename)

            # Adjust spacing based on whether response was interrupted
            print_md(f"Log renamed to: {new_filename}")

            # Add spacing after log rename in markdown mode
            if self.settings_manager.setting_get("markdown"):
                print()

        except Exception as e:
            print_md(f"Error renaming log file: {e}")

    def generate_ai_suggested_title(self) -> str:
        """Generate AI-suggested title using full conversation context (for --logmv command)"""
        try:
            # Build context from recent conversation history
            context_parts = []
            for msg in self.conversation_history[-6:]:  # Last 6 messages for context
                role = msg['role']
                if role == 'user':
                    context_parts.append(f"User: {msg['content'][:200]}")
                elif role == 'assistant':
                    context_parts.append(f"Assistant: {msg['content'][:150]}")

            if not context_parts:
                return "general-conversation"

            context = "\n".join(context_parts)
            title = self._generate_title_from_context(context)

            return title if title else "general-conversation"

        except Exception as e:
            print_md(f"Error generating AI suggested title: {e}")
            return "general-conversation"

    def manual_log_rename(self, title: str) -> str:
        """Manually rename log with user-provided title while preserving date/timestamp"""
        # Skip renaming if incognito mode is enabled
        if self.settings_manager.setting_get("incognito"):
            return "incognito-mode"  # Return placeholder name

        try:
            # Sanitize the title first
            title = title.replace(" ", "-").replace('"', "").replace("'", "")

            current_log_location = self.settings_manager.setting_get("log_file_location")
            if not current_log_location:
                # If no current log, create filename with current date/timestamp
                import time
                import datetime
                date_part = datetime.datetime.now().strftime('%Y-%m-%d')
                timestamp = int(time.time())
                new_filename = f"{date_part}_{title}_{timestamp}"
                self.settings_manager.setting_set("log_file_name", new_filename)
                return new_filename

            # Extract date from current filename (base name without extension)
            current_filename = os.path.basename(current_log_location)

            # Extract date part (format: 2025-06-08_timestamp or 2025-06-08_title_timestamp)
            date_part = current_filename.split('_')[0] if '_' in current_filename else current_filename

            # Add unix timestamp to prevent conflicts
            import time
            timestamp = int(time.time())

            # Create new base filename: date_descriptive-title_timestamp
            new_filename = f"{date_part}_{title}_{timestamp}"

            # Get directory path
            log_directory = os.path.dirname(current_log_location)
            new_log_location = os.path.join(log_directory, new_filename)

            # Rename JSON file
            json_new = new_log_location + ".json"
            if os.path.exists(current_log_location):
                os.rename(current_log_location, json_new)

            # Update settings with new location and filename
            self.settings_manager.setting_set("log_file_location", new_log_location + ".json")
            self.settings_manager.setting_set("log_file_name", new_filename)

            return new_filename

        except Exception as e:
            print_md(f"Error renaming log file: {e}")
            # Fallback to simple filename if rename fails
            fallback_name = f"{title}"
            self.settings_manager.setting_set("log_file_name", fallback_name)
            return fallback_name

    def log_delete(self) -> bool:
        """Delete the current conversation's log file (JSON)"""
        try:
            current_log_location = self.settings_manager.setting_get("log_file_location")
            if not current_log_location:
                return False

            # Delete JSON file
            if os.path.exists(current_log_location):
                os.remove(current_log_location)

            # Generate new log file name for next conversation (like start_new_conversation_log)
            new_log_name = self.settings_manager.generate_new_log_filename()
            self.settings_manager.setting_set("log_file_name", new_log_name)

            # Clear current log location to force new file creation
            self.settings_manager.setting_set("log_file_location", None)

            # Clear conversation history since log is deleted
            self.conversation_history = []

            # Reset logging state for new conversation (like start_new_conversation_log)
            self.log_renamed = False

            return True

        except Exception as e:
            print_md(f"Error deleting log: {e}")
            return False

    def log_resume(self) -> None:
        path = self.settings_manager.setting_get("working_dir") + "/logs"
        file = self.settings_manager.setting_get("log_file_name") + ".json"
        path_to_log_json = None

        for root, dirs, files in os.walk(path):
            if file in files:
                path_to_log_json = os.path.join(root, file)
                path_to_log_json = root + "/" + file
                break

        if path_to_log_json:
            self.settings_manager.setting_set("log_file_location", path_to_log_json)

            with open(path_to_log_json) as file:
                self.conversation_history = json.load(file)

            # Mark log as already renamed to prevent auto-renaming when resuming
            self.log_renamed = True

            # Display the conversation history to the user
            self._display_conversation_history()

            log_resume_text = "Conversation history replaced with " + self.settings_manager.setting_get('log_file_name') + "\n"
            log_resume_text += "    Now logging to " + self.settings_manager.setting_get('log_file_name')
            print_md(log_resume_text)
        else:
            print_md("Log file not found")

    def start_new_conversation_log(self) -> None:
        """
        Start a new conversation log with a completely fresh start.
        Clears everything including search results and injected context,
        then re-injects fresh instructions from settings.
        """
        # Clear EVERYTHING for a truly fresh start
        self.conversation_history = []

        # Re-inject current instructions from settings
        current_instructions = self.settings_manager.setting_get("instructions")
        self.apply_instructions(current_instructions)

        # Reset logging state for new conversation
        self.log_renamed = False

        # Generate new log file name using existing logic
        new_log_name = self.settings_manager.generate_new_log_filename()
        self.settings_manager.setting_set("log_file_name", new_log_name)

        # Clear current log location to force new file creation
        self.settings_manager.setting_set("log_file_location", None)

    def _display_conversation_history(self) -> None:
        """Display the conversation history formatted from JSON data."""
        if not self.conversation_history:
            print_md("No conversation history available")
            return

        try:
            print_md("Resuming conversation history:")
            print_lines()

            # Count messages for summary
            message_count = len([msg for msg in self.conversation_history if msg.get('role') in ['user', 'assistant']])
            if message_count > 0:
                print_md(f"Conversation Summary: {message_count} messages")
                print_lines()

            # Build markdown content from conversation history
            markdown_lines = []
            for message in self.conversation_history:
                role = message.get('role', '')
                content = message.get('content', '')

                if role == 'system':
                    # Show full system messages
                    summary = content
                    markdown_lines.append(f"**System:** {summary}")
                    markdown_lines.append("")
                elif role in ['user', 'assistant']:
                    # Show full user and assistant messages
                    markdown_lines.append(f"**{role}:**  ")
                    markdown_lines.append(f"{content}")
                    markdown_lines.append("")

            md_content = "\n".join(markdown_lines)

            # Display the content
            if self.settings_manager.setting_get("markdown"):
                # Use streamdown for markdown rendering
                streamdown_process = self._start_streamdown_process()
                if streamdown_process.stdin:
                    streamdown_process.stdin.write(md_content)
                    streamdown_process.stdin.close()
                streamdown_process.wait()
            else:
                # Plain text fallback
                print(md_content)

            print_lines()

        except Exception as e:
            print_md(f"Error displaying conversation history: {e}")


