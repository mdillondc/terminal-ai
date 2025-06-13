from datetime import datetime
import os

import select
import sys
import json
import urllib.request
import urllib.error
import subprocess
import re

from typing import Optional, Any
from settings_manager import SettingsManager
from tts_service import get_tts_service, is_tts_available, interrupt_tts, is_tts_playing
from tavily_search import create_tavily_search, TavilySearchError

class ConversationManager:
    def __init__(self, client: Any, api: Optional[str] = None, model: Optional[str] = None) -> None:
        self.client = client
        self._original_openai_client = client  # Store original OpenAI client for switching back
        self.api = api
        self._model = model
        self.conversation_history = []
        self.settings_manager = SettingsManager.getInstance()
        self.log_renamed = False  # Track if we've already renamed the log with AI-generated title
        self._ollama_available = None  # Cache Ollama availability check
        self._response_buffer = ""  # Buffer to accumulate response text for thinking coloring
        self._execution_buffer = ""  # Buffer to accumulate potential execution commands

        # Initialize RAG engine - import here to avoid circular imports
        try:
            from rag_engine import RAGEngine
            self.rag_engine = RAGEngine(self._original_openai_client)
        except ImportError as e:
            print(f"Warning: Could not initialize RAG engine: {e}")
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

        # Initialize execute state if needed
        if not hasattr(self, '_in_execute_block'):
            self._in_execute_block = False
        if not hasattr(self, '_execute_buffer'):
            self._execute_buffer = ""  # Buffer for execute block content

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

            # Look for complete <execute> tag
            if text_to_process[i:i+9] == '<execute>':
                if not self._in_execute_block:
                    self._in_execute_block = True
                    self._execute_buffer = ""
                    output += '<execute>\n'
                i += 9
                continue

            # Look for complete </execute> tag
            if text_to_process[i:i+10] == '</execute>':
                if self._in_execute_block:
                    # Format the execute block properly
                    output += self._execute_buffer + '\n</execute>\n\n'
                    # Reset execute state
                    self._in_execute_block = False
                    self._execute_buffer = ""
                else:
                    output += '</execute>'
                i += 10
                continue



            # Check for potential partial tags at the end that we should buffer
            remaining = text_to_process[i:]
            if i == len(text_to_process) - len(remaining):  # At the end
                if (remaining.startswith('<think') and len(remaining) < 7) or \
                   (remaining.startswith('</think') and len(remaining) < 8) or \
                   (remaining.startswith('<execute') and len(remaining) < 9) or \
                   (remaining.startswith('</execute') and len(remaining) < 10) or \
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
            elif self._in_execute_block:
                # Buffer execute block content
                self._execute_buffer += text_to_process[i]
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

    @property
    def model(self) -> Optional[str]:
        """Get current model"""
        return self._model

    @model.setter
    def model(self, value: Optional[str]) -> None:
        """Set model and update client if necessary"""
        self._model = value
        if value:
            self._update_client_for_model(value)

    def _is_ollama_available(self) -> bool:
        """Check if Ollama is available with caching"""
        if self._ollama_available is not None:
            return self._ollama_available

        try:
            base_url = self.settings_manager.setting_get("ollama_base_url")
            url = f"{base_url}/api/version"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=1) as response:
                self._ollama_available = response.status == 200
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
            self._ollama_available = False
        except Exception:
            self._ollama_available = False

        return self._ollama_available

    def _is_ollama_model(self, model_name: str) -> bool:
        """Detect if a model is from Ollama based on naming patterns"""
        if not self._is_ollama_available():
            return False

        # Common Ollama model patterns
        ollama_patterns = [
            ':',  # Most Ollama models have tags like "llama3.2:latest"
            'llama', 'mistral', 'qwen', 'codellama', 'phi', 'gemma',
            'tinyllama', 'vicuna', 'orca', 'openchat', 'starling'
        ]

        model_lower = model_name.lower()
        return any(pattern in model_lower for pattern in ollama_patterns)

    def _is_google_model(self, model_name: str) -> bool:
        """Detect if a model is from Google based on naming patterns"""
        # Check if GOOGLE_API_KEY is available
        if not os.environ.get("GOOGLE_API_KEY"):
            return False

        # Common Google model patterns
        google_patterns = ['gemini', 'palm', 'bard']
        model_lower = model_name.lower()
        return any(pattern in model_lower for pattern in google_patterns)

    def _create_ollama_client(self):
        """Create OpenAI-compatible client configured for Ollama"""
        try:
            from openai import OpenAI
            base_url = self.settings_manager.setting_get("ollama_base_url")
            return OpenAI(
                base_url=f"{base_url}/v1",
                api_key="ollama"  # Ollama doesn't require a real API key
            )
        except Exception:
            return None

    def _create_google_client(self):
        """Create OpenAI-compatible client configured for Google Gemini"""
        try:
            from openai import OpenAI
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                return None
            return OpenAI(
                api_key=api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/"
            )
        except Exception:
            return None

    def _update_client_for_model(self, model_name: str) -> None:
        """Update client based on model source"""
        if self._is_ollama_model(model_name):
            ollama_client = self._create_ollama_client()
            if ollama_client:
                self.client = ollama_client
                print(f" - Using Ollama for model: {model_name}")
            else:
                print(f" - Warning: Could not create Ollama client for {model_name}")
        elif self._is_google_model(model_name):
            google_client = self._create_google_client()
            if google_client:
                self.client = google_client
                print(f" - Using Google Gemini for model: {model_name}")
            else:
                print(f" - Warning: Could not create Google client for {model_name}")
                print(f" - Make sure GOOGLE_API_KEY environment variable is set")
        else:
            # Use stored OpenAI client
            self.client = self._original_openai_client

    def generate_response(self, instructions: Optional[str] = None) -> None:
        # print(self.conversation_history)

        # Check if search is enabled and handle search workflow
        if self.settings_manager.setting_get("search") and self.conversation_history:
            self._handle_search_workflow()

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

        # Inject execute mode system prompt if execute mode is enabled
        if self.settings_manager.setting_get("execute_enabled") and self.conversation_history:
            execute_prompt = self._get_execute_mode_prompt()
            self.conversation_history.append({
                "role": "system",
                "content": execute_prompt
            })

        # Display AI name to user
        print(f"\n{self.settings_manager.setting_get('name_ai')} (`q` + `Enter` to interrupt):")

        # Setup stream to receive response from AI
        stream = self.client.chat.completions.create(
            model=self.model, messages=self.conversation_history, stream=True
        )

        # Init variable to hold AI response in its entirety
        ai_response = ""

        # Process response stream with interrupt checking between chunks
        interrupted = False
        try:
            for chunk in stream:
                # Check for 'q + enter' interrupt before processing each chunk
                if self._check_for_interrupt():
                    print("\n - Response interrupted by user.")
                    interrupted = True
                    break

                if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'content') and delta.content is not None:
                        ai_response_chunk = delta.content
                        ai_response += ai_response_chunk

                        # Process chunk with thinking text coloring
                        self._process_and_print_chunk(ai_response_chunk)
        except Exception as e:
            print(f"\n - Error processing response stream: {e}")

        # Flush any remaining buffer content and reset state at the end
        if hasattr(self, '_response_buffer') and self._response_buffer:
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
        if hasattr(self, '_in_execute_block'):
            self._in_execute_block = False
        if hasattr(self, '_execute_buffer'):
            self._execute_buffer = ""

        # Only save if we got a response
        if ai_response:
            # Process any command execution requests in the response
            processed_response = self._process_execution_requests(ai_response)

            # Append processed_response to the conversation_history array
            self.conversation_history.append({"role": "assistant", "content": processed_response})

            # Display RAG sources if any were used
            if rag_sources and self.rag_engine:
                print(f"\n\n- {self.rag_engine.format_sources(rag_sources)}")

            self.log_save()

            # Check if this is the first user-AI exchange and rename log with descriptive title
            self._check_and_rename_log_after_first_exchange(interrupted)

        # Generate and play TTS audio if enabled
        if self.settings_manager.setting_get("tts") and ai_response:
            # Use processed response for TTS to avoid reading execution tags
            processed_response = self._process_execution_requests(ai_response) if ai_response else ""
            self._handle_tts_playback(processed_response, interrupted)

    def _process_execution_requests(self, response: str) -> str:
        """
        Process command execution requests in AI response.
        Looks for <execute>command</execute> tags and executes them if execute mode is enabled.
        Returns the response with execution tags removed and results added.
        """
        if not self.settings_manager.setting_get("execute_enabled"):
            # If execution is disabled, just remove the tags and return
            return re.sub(r'<execute>(.*?)</execute>', r'[Command execution disabled: \1]', response, flags=re.DOTALL)

        # Find all execution requests (formatting already handled during streaming)
        execution_pattern = r'<execute>(.*?)</execute>'
        matches = list(re.finditer(execution_pattern, response, flags=re.DOTALL))

        if not matches:
            return response

        # Process each execution request from end to start to preserve indices
        processed_response = response
        for match in reversed(matches):
            command = match.group(1).strip()

            # Execute the command
            execution_result = self._execute_system_command(command)

            # Replace the execution tag with the result
            if execution_result:
                replacement = f"\n[Executed: {command}]\n```\n{execution_result}\n```"
            else:
                replacement = f"\n[Command execution failed or denied: {command}]"

            processed_response = processed_response[:match.start()] + replacement + processed_response[match.end():]

        return processed_response

    def _execute_system_command(self, command: str) -> Optional[str]:
        """
        Execute a system command with proper permission handling.
        Returns the command output or None if execution was denied/failed.
        """
        if not self.settings_manager.setting_get("execute_enabled"):
            return None

        # Check if permission is required
        if self.settings_manager.setting_get("execute_require_permission"):
            response = input(" - Allow execution? (Y/n): ").strip().lower()
            if response not in ['', 'y', 'yes']:
                print(" - Command execution denied by user")
                return None

        try:
            # Execute the command safely
            print(f"\n - Running: {command}")

            # Use shell=True for complex commands but be aware of security implications
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True
            )

            output = ""
            if result.stdout:
                output += f"STDOUT:\n{result.stdout}"
            if result.stderr:
                output += f"STDERR:\n{result.stderr}"
            if not result.stdout and not result.stderr:
                output = f"Command completed successfully (no output)"

            output += f"\n\nReturn code: {result.returncode}"

            # Display the command output immediately
            if result.returncode == 0:
                if result.stdout:
                    print(result.stdout.rstrip())
                else:
                    print(" - Command ran successfully (no output produced)")
            else:
                if result.stderr:
                    print(f" - Command failed: {result.stderr.rstrip()}")
                else:
                    print(f" - Command failed (exit code: {result.returncode})")

            return output

        except Exception as e:
            error_msg = f"Could not run command: {str(e)}"
            print(f" - Error: {error_msg}")
            return error_msg

    def _get_execute_mode_prompt(self) -> str:
        """
        Get the system prompt for execute mode to make AI more proactive about command execution.
        """
        return """EXECUTE MODE ACTIVE: You can now run system commands using <execute>command</execute> tags.

EXECUTION GUIDELINES:
- BE DIRECT: Don't ask "would you like me to..." - just run the appropriate command
- ONE COMMAND PER RESPONSE: Only include one <execute> tag per response
- BE CONCISE: Give brief explanations (1-2 sentences), then execute
- IMMEDIATE ACTION: Jump straight to running relevant commands
- PRACTICAL FOCUS: Choose commands that directly answer the user's question
- USE PYTHON FOR COMPLEXITY: For file conversion, data processing, calculations, or complex operations, use Python

COMMAND PRIORITY:
1. Simple system commands: ls, pwd, cat, grep, find, etc.
2. Python for complex tasks: file conversion, data analysis, calculations, text processing, web scraping
3. Development tools: git, npm, pip, docker, etc.
4. System utilities: curl, wget, tar, zip, etc.

TOOL AVAILABILITY:
- ALWAYS check if required tools/packages are available before using them
- For Python packages: <execute>python3 -c "import package_name; print('Available')"</execute>
- For system commands: <execute>which command_name</execute> or <execute>command_name --version</execute>
- If missing, suggest installation commands immediately

INSTALLATION SUGGESTIONS:
- Python packages: <execute>pip3 install package_name</execute>
- System packages (Ubuntu/Debian): <execute>sudo apt update && sudo apt install package_name</execute>
- System packages (MacOS): <execute>brew install package_name</execute>
- Node packages: <execute>npm install -g package_name</execute>

PYTHON ONE-LINERS FOR COMMON TASKS:
- File conversion (JSON): python3 -c "import json; print(json.dumps({'converted': True}, indent=2))"
- CSV analysis: python3 -c "import csv; rows=list(csv.reader(open('file.csv'))); print(f'Rows: {len(rows)}')"
- Image resize: python3 -c "from PIL import Image; Image.open('in.jpg').resize((800,600)).save('out.jpg')"
- Text processing: python3 -c "with open('file.txt') as f: print(len(f.read().split()))"
- Math/calculations: python3 -c "import math; print(f'Result: {math.sqrt(64)}')"
- Web requests: python3 -c "import requests; print(requests.get('http://httpbin.org/ip').json())"
- Data structures: python3 -c "data=[1,2,3,4]; print(f'Sum: {sum(data)}, Avg: {sum(data)/len(data)}')"
- File operations: python3 -c "import shutil; shutil.copy2('source.txt', 'backup.txt')"
- These are just examples to help you understand how to use python, however you should use your own judgement

RESPONSE PATTERN:
1. Brief explanation of what you're doing
2. Execute the most relevant command (prefer Python for data/file operations)
3. Follow up with additional commands in next responses if needed

The system will handle permission prompts - your job is to suggest the right commands directly."""

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
                print(" - No user message found for search.")
                return

            print(" - Search mode enabled. Generating optimal search query...")

            # Get recent conversation context using configurable window size
            context_window = self.settings_manager.search_context_window
            char_limit = self.settings_manager.search_context_char_limit
            context_messages = []
            for message in self.conversation_history[-context_window:]:
                if message["role"] in ["user", "assistant"]:
                    # Truncate long messages to keep context manageable
                    content = message["content"]
                    if len(content) > char_limit:
                        content = content[:char_limit] + "..."
                    context_messages.append(f"{message['role']}: {content}")

            context_text = "\n".join(context_messages) if context_messages else "No prior context."

            # Extract key topics/entities from conversation context for better search queries
            key_topics = self._extract_key_topics_from_context(context_text)
            topics_text = f"Key topics from conversation: {', '.join(key_topics)}" if key_topics else ""

            # Create a temporary conversation to generate search query
            search_query_conversation = [
                {
                    "role": "system",
                    "content": "You are a search query optimizer. Given a user's question or statement and the conversation context, rewrite it as 1-3 optimal search queries that would find the most relevant and current information to answer their question. Consider the conversation context to understand what the user is really asking about. Respond with only the search queries, one per line, no explanations."
                },
                {
                    "role": "user",
                    "content": f"CONVERSATION CONTEXT:\n{context_text}\n\n{topics_text}\n\nCURRENT QUESTION: {last_user_message}\n\nRewrite the current question as optimal search queries, considering the conversation context and key topics:"
                }
            ]

            # Generate search query using AI
            query_response = self.client.chat.completions.create(
                model=self.model,
                messages=search_query_conversation,
                temperature=0.3
            )

            search_queries = query_response.choices[0].message.content.strip().split('\n')
            search_queries = [q.strip() for q in search_queries if q.strip()]

            print(f" - Generated search queries: {', '.join(search_queries)}")

            # Perform searches
            search_client = create_tavily_search()
            if not search_client:
                print(" - (!) Failed to initialize Tavily search client. Continuing without search.")
                return

            all_search_results = []
            max_queries = self.settings_manager.search_max_queries
            for query in search_queries[:max_queries]:  # Limit queries to avoid overwhelming
                print(f" - Searching: {query}")
                try:
                    results = search_client.search_and_format(query, max_results=3)
                    if results:
                        all_search_results.append(results)
                except TavilySearchError as e:
                    print(f" - Search failed for '{query}': {e}")
                    continue

            if all_search_results:
                # Combine all search results
                combined_results = "\n\n" + "="*80 + "\n".join(all_search_results) + "\n" + "="*80 + "\n"

                # Add search results as a system message to provide context
                search_context = {
                    "role": "system",
                    "content": f"SEARCH RESULTS FOR USER'S QUERY:\n{combined_results}\n\nUse this information to provide a comprehensive and current answer to the user's question. Cite sources when relevant."
                }

                # Insert search context before the last user message
                self.conversation_history.insert(-1, search_context)
                print(" - Search completed. Analyzing results...")
            else:
                print(" - No search results found. Continuing without search data.")

        except Exception as e:
            print(f" - (!) Search workflow error: {e}. Continuing without search.")
            return

    def _extract_key_topics_from_context(self, context_text: str) -> list:
        """
        Extract key topics and entities from conversation context to improve search queries.

        Args:
            context_text: The conversation context string

        Returns:
            List of key topics/entities found in the context
        """
        if not context_text or context_text == "No prior context.":
            return []

        # Simple keyword extraction - look for important terms
        # This could be enhanced with more sophisticated NLP techniques
        important_keywords = []

        # Look for proper nouns and important terms (simple heuristic approach)
        words = context_text.split()
        for i, word in enumerate(words):
            # Capitalized words that might be names, places, organizations
            if word[0].isupper() and len(word) > 2 and word not in ['The', 'This', 'That', 'When', 'Where', 'What', 'How', 'Why']:
                # Avoid adding common sentence starters
                if i == 0 or words[i-1].endswith('.') or words[i-1].endswith(':'):
                    continue
                important_keywords.append(word.strip('.,!?:;'))

        # Look for specific patterns that indicate important topics
        topic_patterns = [
            'Trump', 'Marines', 'mobilization', 'deployment', 'firearms', 'military',
            'President', 'LA', 'California', 'video', 'YouTube', 'allegations',
            'political', 'claims', 'Marines', 'weapons', 'policy'
        ]

        for pattern in topic_patterns:
            if pattern.lower() in context_text.lower():
                important_keywords.append(pattern)

        # Remove duplicates and return unique topics
        return list(set(important_keywords))[:5]  # Limit to top 5 topics


    def _handle_tts_playback(self, text: str, was_interrupted: bool = False):
        """
        Handle TTS playback of AI response.

        Args:
            text: Text to convert to speech
            was_interrupted: Whether the AI response was interrupted
        """
        try:
            if not is_tts_available():
                print("- TTS not available - OpenAI or audio system unavailable")
                return

            if was_interrupted:
                print("- Skipping TTS due to interrupted response")
                return

            # Get TTS service and generate speech
            tts_service = get_tts_service(self.client)
            tts_service.generate_and_play_speech(text)

        except Exception as e:
            print(f"- TTS error: {e}")

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
            print(" - Please specify the instructions file to use.")
            return

        new_file_path = self.settings_manager.setting_get("working_dir") + "/instructions/" + file_name

        if not os.path.exists(new_file_path):
            print(f" - (!) {new_file_path} does not exist.")
        else:
            if old_file_name:
                # Remove old instructions from conversation_history
                print(old_file_name)
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
            notice = f" - Applied instructions {file_name}"
            if old_file_name:
                print(notice)


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
                print(f" - Note: Could not generate descriptive log title: {e}")
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
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": title_prompt}],
                max_tokens=25,
                temperature=0.1
            )

            if response.choices and response.choices[0].message.content:
                title = response.choices[0].message.content.strip()
                # Clean up the title - ensure it's filename-safe
                title = title.lower().replace(' ', '-').replace('_', '-')
                # Remove any non-alphanumeric characters except hyphens
                import re
                title = re.sub(r'[^a-z0-9-]', '', title)
                # Remove multiple consecutive hyphens
                title = re.sub(r'-+', '-', title)
                # Remove leading/trailing hyphens
                title = title.strip('-')
                # Ensure it's not empty
                if title and len(title) >= 3:
                    return title

        except Exception as e:
            print(f" - Error generating title: {e}")

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
            print(f" - Error generating title: {e}")

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
            if interrupted:
                print(f" - Log renamed to: {new_filename}")
            else:
                print(f"\n - Log renamed to: {new_filename}")

        except Exception as e:
            print(f" - Error renaming log files: {e}")

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
            print(f" - Error generating AI suggested title: {e}")
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
            print(f" - Error renaming log files: {e}")
            # Fallback to simple filename if rename fails
            fallback_name = f"{title}.md"
            self.settings_manager.setting_set("log_file_name", fallback_name)
            return fallback_name

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

            # Display the conversation history to the user
            self._display_conversation_history()

            print(" - Conversation history replaced with " + self.settings_manager.setting_get('log_file_name'))
            print(" - Now logging to " + self.settings_manager.setting_get('log_file_name'))
        else:
            print(" - (!) Log file not found.")

    def _display_conversation_history(self) -> None:
        """Display the loaded conversation history to the user in a readable format"""
        if not self.conversation_history:
            print(" - No conversation history to display.")
            return

        # Filter out system messages for display
        display_messages = []
        for msg in self.conversation_history:
            if msg.get('role') != 'system':
                display_messages.append(msg)

        if not display_messages:
            print(" - No user conversation to display (only system messages found).")
            return

        print("\n" + "=" * 70)
        print("RESUMING CONVERSATION HISTORY")
        print("=" * 70)

        user_name = self.settings_manager.setting_get('name_user') or "User"
        ai_name = self.settings_manager.setting_get('name_ai') or "Assistant"

        # Show summary for long conversations
        total_messages = len(display_messages)
        if total_messages > 10:
            print("\nConversation Summary: " + str(total_messages) + " messages")
            print("   Showing last 8 messages (use full log file to see complete history)")
            print("-" * 70)
            display_messages = display_messages[-8:]

        for i, entry in enumerate(display_messages, 1):
            role = entry.get('role', 'unknown')
            content = entry.get('content', '')

            # Display user and assistant messages with numbering
            if role == 'user':
                print("\n[" + str(i) + "] " + user_name + self.settings_manager.get_enabled_toggles() + ":")
                print(content)
            elif role == 'assistant':
                print("\n[" + str(i) + "] " + ai_name + ":")
                print(content)

        print("\n" + "=" * 70)
        print("END OF CONVERSATION HISTORY - Resuming from here...")
        print("=" * 70 + "\n")
