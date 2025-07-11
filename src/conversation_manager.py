from datetime import datetime
import os
import select
import sys
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Any
from settings_manager import SettingsManager
from tts_service import get_tts_service, interrupt_tts, is_tts_playing
from tavily_search import create_tavily_search, TavilySearchError
from deep_search_agent import create_deep_search_agent
from print_helper import print_md
from llm_client_manager import LLMClientManager
from print_helper import print_md, print_lines
from rich.console import Console
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import black, gray


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

    def _process_and_print_chunk(self, chunk: str) -> None:
        """
        Process a response chunk, applying thinking text coloring while maintaining streaming.
        Hides empty thinking blocks that contain only whitespace.
        """
        # ANSI color codes
        GRAY = '\033[38;2;204;204;204m'  # Light gray (#cccccc)
        RESET = '\033[0m'

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
                if (remaining.startswith('<think') and len(remaining) < 7) or \
                   (remaining.startswith('</think') and len(remaining) < 8) or \
                   (remaining == '<' or remaining.startswith('<') and len(remaining) < 10):
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
        """Detect if a model is from Ollama - delegated to LLMClientManager"""
        return self.llm_client_manager._is_ollama_model(model_name)

    def _is_google_model(self, model_name: str) -> bool:
        """Detect if a model is from Google - delegated to LLMClientManager"""
        return self.llm_client_manager._is_google_model(model_name)

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
                    self.conversation_history.append({
                        "role": "system",
                        "content": rag_context
                    })



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
            self.conversation_history.append({"role": "assistant", "content": complete_response})

            # Add newline after AI response for proper spacing
            print()

            # Display RAG sources if any were used
            if rag_sources and self.rag_engine:
                print(f"{self.rag_engine.format_sources(rag_sources)}")

            self.log_save()

            # Check if this is the first user-AI exchange and rename log with descriptive title
            self._check_and_rename_log_after_first_exchange(interrupted)

        # Generate and play TTS audio if enabled
        if self.settings_manager.setting_get("tts") and complete_response:
            self._handle_tts_playback(complete_response, interrupted)

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
            char_limit = self.settings_manager.search_context_char_limit
            recent_messages = []
            earlier_messages = []

            # Get recent messages (last 10) for immediate context
            recent_window = min(10, len(self.conversation_history))
            for message in self.conversation_history[-recent_window:]:
                if message["role"] in ["user", "assistant"]:
                    content = message["content"]
                    if len(content) > char_limit:
                        content = content[:char_limit] + "..."
                    recent_messages.append(f"{message['role']}: {content}")

            # Get earlier messages for reference resolution
            if len(self.conversation_history) > recent_window:
                for message in self.conversation_history[:-recent_window]:
                    if message["role"] in ["user", "assistant"]:
                        content = message["content"]
                        if len(content) > char_limit:
                            content = content[:char_limit] + "..."
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

            # Perform searches
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

            # Use Tavily's auto-parameters for intelligent search optimization
            search_params = {
                'auto_parameters': True,
                'max_results': self.settings_manager.search_max_results  # Must be set manually per Tavily docs
            }

            # Inform user about search configuration
            print_md(f"Search mode: Auto-optimized")

            all_source_metadata = []
            seen_urls = set()
            display_seen_urls = set()

            # Execute all searches in parallel
            with ThreadPoolExecutor(max_workers=3) as executor:
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
                                        source_lines.append(f"    [{title}]({url})")
                                        display_seen_urls.add(url)
                                    elif not url:
                                        source_lines.append(f"    {title}")

                            # Only display search section if it contributed unique sources
                            if source_lines:
                                search_section += "\n".join(source_lines)
                                # Print the complete section
                                print_md(search_section)
                            # If no unique sources found, skip displaying this search section entirely

                    except TavilySearchError as e:
                        search_section += f"    Search failed: {e}"
                        # Always print error sections
                        print_md(search_section)
                    except Exception as e:
                        search_section += f"    Search failed: {e}"
                        # Always print error sections
                        print_md(search_section)



            if all_search_results:
                # Combine all search results
                combined_results = "\n\n" + "="*80 + "\n".join(all_search_results) + "\n" + "="*80 + "\n"

                # Add search results as a system message to provide context
                search_context = {
                    "role": "system",
                    "content": f"SEARCH RESULTS FOR USER'S QUERY:\n{combined_results}\n\nUse this information to provide a comprehensive and current answer to the user's question. MANDATORY: You MUST always conclude your response with a 'Sources:' section that includes:\n\n1. A numbered list of the sources used in your answer\n2. Each source must include the full URL as a clickable link in markdown format: [Title](URL)\n3. Prefer recent, authoritative sources over older or less credible ones\n4. Always include this sources section even if you only reference one source\n\nExample format:\n\n## Sources:\n1. [Article Title](https://example.com/article)\n2. [Another Source](https://example.com/source2)"
                }

                # Insert search context before the last user message
                self.conversation_history.insert(-1, search_context)
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
                    if len(content) > self.settings_manager.search_context_char_limit:
                        content = content[:self.settings_manager.search_context_char_limit] + "..."
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
                search_context = {
                    "role": "system",
                    "content": f"COMPREHENSIVE RESEARCH RESULTS FOR USER'S QUERY:\n{combined_results}\n\nUse this extensive research to provide a thorough, well-sourced answer to the user's question. MANDATORY: You MUST always conclude your response with a 'Sources:' section that includes:\n\n1. A numbered list of the sources used in your answer\n2. Each source must include the full URL as a clickable link in markdown format: [Title](URL)\n3. Prioritize the most authoritative and recent sources\n4. Always include this sources section even if you only reference one source\n\nExample format:\n\n## Sources:\n1. [Article Title](https://example.com/article)\n2. [Another Source](https://example.com/source2)"
                }

                # Insert search context before the last user message
                self.conversation_history.insert(-1, search_context)
                print_md(f"**Research synthesis complete** - generating comprehensive response...\n")
            else:
                print_md("No search results found. Continuing without search data")

        except Exception as e:
            print_md(f"Deep search workflow error: {e}. Continuing without search")
            return

    def _extract_key_topics_from_context(self, context_text: str) -> list:
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


    def _handle_tts_playback(self, text: str, was_interrupted: bool = False):
        """
        Handle TTS playback of AI response.

        Args:
            text: Text to convert to speech
            was_interrupted: Whether the AI response was interrupted
        """
        try:


            if was_interrupted:
                print("Skipping TTS due to interrupted response")
                return

            # Get TTS service and generate speech
            tts_service = get_tts_service(self.client)
            tts_service.generate_and_play_speech(text)

        except Exception as e:
            print_md(f"TTS error: {e}")

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
                        # Also interrupt any playing TTS
                        if is_tts_playing():
                            interrupt_tts()
                        return True
                    return False
                except (EOFError, KeyboardInterrupt):
                    # Also interrupt TTS on keyboard interrupt
                    if is_tts_playing():
                        interrupt_tts()
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
            self.conversation_history.append(
                {"role": "system", "content": f"instructions:{file_name}\n" + instructions + today}
            )

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
            os.remove(current_log_file_location)
            os.remove(current_log_file_location + ".json")

        instructions_file_name = self.settings_manager.setting_get("instructions").rsplit('.', 1)[0] # strip extension
        log_file_name = self.settings_manager.setting_get("log_file_name")
        log_file_path = os.path.join(self.settings_manager.setting_get("working_dir"), f"logs/{instructions_file_name}/")
        log_file_location = log_file_path + log_file_name

        if not os.path.exists(log_file_path):
            os.makedirs(log_file_path)

        conversation_history_as_markdown = ""
        for conversation in self.conversation_history:
            # Format each conversation piece
            formatted_piece = f"**{conversation['role']}:**  \n{conversation['content']}\n\n"
            conversation_history_as_markdown += formatted_piece

        # Save as markdown for user
        with open(log_file_location, 'w') as file:
            file.write(conversation_history_as_markdown)
            self.settings_manager.setting_set("log_file_location", log_file_location)

        # Save as JSON in case user wants to resume conversation later
        conversation_history = json.dumps(self.conversation_history, indent=4)
        with open(log_file_location + ".json", 'w') as file:
            file.write(conversation_history)

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
        """Rename the current log files to include the descriptive title"""
        try:
            current_log_location = self.settings_manager.setting_get("log_file_location")
            if not current_log_location or not os.path.exists(current_log_location):
                return

            # Extract date from current filename
            current_filename = os.path.basename(current_log_location)
            if current_filename.endswith('.md'):
                current_filename = current_filename[:-3]  # Remove .md extension

            # Extract date part (format: 2025-06-08_timestamp)
            date_part = current_filename.split('_')[0] if '_' in current_filename else current_filename

            # Add unix timestamp to prevent conflicts
            import time
            timestamp = int(time.time())

            # Create new filename: date_descriptive-title_timestamp.md
            new_filename = f"{date_part}_{title}_{timestamp}.md"

            # Get directory path
            log_directory = os.path.dirname(current_log_location)
            new_log_location = os.path.join(log_directory, new_filename)

            # Rename both .md and .json files
            if os.path.exists(current_log_location):
                os.rename(current_log_location, new_log_location)

            json_current = current_log_location + ".json"
            json_new = new_log_location + ".json"
            if os.path.exists(json_current):
                os.rename(json_current, json_new)

            # Update settings with new location
            self.settings_manager.setting_set("log_file_location", new_log_location)
            self.settings_manager.setting_set("log_file_name", new_filename)

            # Adjust spacing based on whether response was interrupted
            print_md(f"Log renamed to: {new_filename}")

            # Add spacing after log rename in markdown mode
            if self.settings_manager.setting_get("markdown"):
                print()

        except Exception as e:
            print_md(f"Error renaming log files: {e}")

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
            return "incognito-mode.md"  # Return placeholder name

        try:
            # Sanitize the title first
            title = title.replace(" ", "-").replace('"', "").replace("'", "")

            current_log_location = self.settings_manager.setting_get("log_file_location")
            if not current_log_location or not os.path.exists(current_log_location):
                # If no current log, create filename with current date/timestamp
                import time
                import datetime
                date_part = datetime.datetime.now().strftime('%Y-%m-%d')
                timestamp = int(time.time())
                new_filename = f"{date_part}_{title}_{timestamp}.md"
                self.settings_manager.setting_set("log_file_name", new_filename)
                return new_filename

            # Extract date from current filename
            current_filename = os.path.basename(current_log_location)
            if current_filename.endswith('.md'):
                current_filename = current_filename[:-3]  # Remove .md extension

            # Extract date part (format: 2025-06-08_timestamp or 2025-06-08_title_timestamp)
            date_part = current_filename.split('_')[0] if '_' in current_filename else current_filename

            # Add unix timestamp to prevent conflicts
            import time
            timestamp = int(time.time())

            # Create new filename: date_descriptive-title_timestamp.md
            new_filename = f"{date_part}_{title}_{timestamp}.md"

            # Get directory path
            log_directory = os.path.dirname(current_log_location)
            new_log_location = os.path.join(log_directory, new_filename)

            # Rename both .md and .json files
            if os.path.exists(current_log_location):
                os.rename(current_log_location, new_log_location)

            json_current = current_log_location + ".json"
            json_new = new_log_location + ".json"
            if os.path.exists(json_current):
                os.rename(json_current, json_new)

            # Update settings with new location and filename
            self.settings_manager.setting_set("log_file_location", new_log_location)
            self.settings_manager.setting_set("log_file_name", new_filename)

            return new_filename

        except Exception as e:
            print_md(f"Error renaming log files: {e}")
            # Fallback to simple filename if rename fails
            fallback_name = f"{title}.md"
            self.settings_manager.setting_set("log_file_name", fallback_name)
            return fallback_name

    def log_delete(self) -> bool:
        """Delete the current conversation's log files (.md and .json)"""
        try:
            current_log_location = self.settings_manager.setting_get("log_file_location")
            if not current_log_location:
                return False

            # Delete .md file
            if os.path.exists(current_log_location):
                os.remove(current_log_location)

            # Delete .json file
            json_log_location = current_log_location + ".json"
            if os.path.exists(json_log_location):
                os.remove(json_log_location)

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
            self.settings_manager.setting_set("log_file_location", path_to_log_json[:-5]) # strip .json part

            with open(path_to_log_json) as file:
                self.conversation_history = json.load(file)

            # Mark log as already renamed to prevent auto-renaming when resuming
            self.log_renamed = True

            # Display the conversation history to the user
            self._display_conversation_history()

            print_md("Conversation history replaced with " + self.settings_manager.setting_get('log_file_name'))
            print_md("Now logging to " + self.settings_manager.setting_get('log_file_name'))
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
        """Display the loaded conversation history, applying markdown if enabled."""
        if not self.conversation_history:
            print_md("No conversation history to display")
            return

        # Filter out system messages for display
        display_messages = [msg for msg in self.conversation_history if msg.get('role') != 'system']

        if not display_messages:
            print_md("No user conversation to display (only system messages found)")
            return

        print_md("Resuming conversation history:")
        print_lines()

        user_name = self.settings_manager.setting_get('name_user') or "User"
        ai_name = self.settings_manager.get_ai_name_with_instructions() or "Assistant"
        markdown_enabled = self.settings_manager.setting_get("markdown")

        # Show summary for long conversations
        total_messages = len(display_messages)
        if total_messages > 0:
            print_md(f"Conversation Summary: {total_messages} messages")
            print_lines()

        if markdown_enabled:
            # Pipe the conversation through streamdown for rendering
            streamdown_process = self._start_streamdown_process()
            if streamdown_process.stdin:
                for i, entry in enumerate(display_messages, 1):
                    role = entry.get('role', 'unknown')
                    content = entry.get('content', '')

                    if role == 'user':
                        output = f"**[{i}] {user_name}{self.settings_manager.get_enabled_toggles()}:**\n\n{content}\n\n"
                        streamdown_process.stdin.write(output)
                    elif role == 'assistant':
                        adjusted_content = self._adjust_markdown_headers_for_streamdown(content)
                        output = f"**[{i}] {ai_name}:**\n\n{adjusted_content}\n\n"
                        streamdown_process.stdin.write(output)

                streamdown_process.stdin.close()
            streamdown_process.wait()
        else:
            # Original plain text display
            for i, entry in enumerate(display_messages, 1):
                role = entry.get('role', 'unknown')
                content = entry.get('content', '')

                if role == 'user':
                    print(f"\n[{i}] {user_name}{self.settings_manager.get_enabled_toggles()}:")
                    print(content)
                elif role == 'assistant':
                    print(f"\n[{i}] {ai_name}:")
                    print(content)

        print_lines()

    def generate_pdf(self) -> str:
        """
        Generate a PDF of the current conversation and save it to pdf/ directory.
        Returns the path to the generated PDF file.
        """
        # Check if we have a conversation to export
        if not self.conversation_history:
            raise ValueError("No conversation history to export")

        # Get log name for PDF filename
        log_file_name = self.settings_manager.setting_get("log_file_name")
        if not log_file_name:
            raise ValueError("No log file name available for PDF generation")

        # Remove .md extension if present and create PDF filename
        pdf_filename = log_file_name.replace('.md', '') + '.pdf'

        # Create pdf directory if it doesn't exist
        working_dir = self.settings_manager.setting_get("working_dir")
        pdf_dir = os.path.join(working_dir, "pdf")
        os.makedirs(pdf_dir, exist_ok=True)

        # Full path to PDF file
        pdf_path = os.path.join(pdf_dir, pdf_filename)

        # Set up PDF document
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        styles = getSampleStyleSheet()

        # Custom styles
        normal_style = styles['Normal']
        normal_style.fontName = 'Helvetica'
        normal_style.fontSize = 10
        normal_style.spaceAfter = 12

        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontName='Helvetica-Bold',
            fontSize=12,
            spaceAfter=6,
            spaceBefore=12,
            textColor=black
        )

        system_style = ParagraphStyle(
            'SystemMessage',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=9,
            textColor=gray,
            spaceAfter=8,
            leftIndent=20
        )

        # Build PDF content
        story = []

        # Add title
        title = f"Conversation Export: {pdf_filename.replace('.pdf', '')}"
        story.append(Paragraph(title, styles['Title']))
        story.append(Spacer(1, 20))

        # Add generation timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        story.append(Paragraph(f"Generated: {timestamp}", system_style))
        story.append(Spacer(1, 12))

        # Get user and AI names
        user_name = self.settings_manager.setting_get('name_user') or "User"
        ai_name = self.settings_manager.get_ai_name_with_instructions() or "Assistant"

        # Process conversation history
        for msg in self.conversation_history:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')

            if role == 'system':
                # Handle system messages (including instructions) in light gray
                if content.startswith('instructions:'):
                    # Extract instruction file name and content
                    lines = content.split('\n', 1)
                    header = lines[0]  # "instructions:filename"
                    instruction_content = lines[1] if len(lines) > 1 else ""

                    story.append(Paragraph(f"<b>System Instructions:</b> {header}", system_style))
                    if instruction_content.strip():
                        # Add instruction content with additional indentation
                        instruction_paragraphs = instruction_content.split('\n\n')
                        for para in instruction_paragraphs:
                            if para.strip():
                                story.append(Paragraph(para.strip(), system_style))
                else:
                    # Other system messages
                    story.append(Paragraph(f"<b>System:</b> {content}", system_style))

            elif role == 'user':
                story.append(Paragraph(f"<b>{user_name}:</b>", heading_style))
                # Split content into paragraphs for better formatting
                paragraphs = content.split('\n\n')
                for para in paragraphs:
                    if para.strip():
                        story.append(Paragraph(para.strip(), normal_style))

            elif role == 'assistant':
                story.append(Paragraph(f"<b>{ai_name}:</b>", heading_style))
                # Split content into paragraphs for better formatting
                paragraphs = content.split('\n\n')
                for para in paragraphs:
                    if para.strip():
                        story.append(Paragraph(para.strip(), normal_style))

            # Add some space between messages
            story.append(Spacer(1, 6))

        # Build and save PDF
        doc.build(story)

        return pdf_path
