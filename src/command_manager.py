import os
import re
import requests
from typing import Optional, Any, Dict, List, Tuple
import yt_dlp
import clipboard

import tiktoken
from settings_manager import SettingsManager
from export_manager import ExportManager
from openai import OpenAI
from command_registry import CommandRegistry
from command_completer import CommandCompleter
from tavily_search import create_tavily_search, TavilySearchError
from web_content_extractor import WebContentExtractor
from document_processor import DocumentProcessor
from rag_config import is_supported_file, get_supported_extensions_display
from print_helper import print_md, start_capturing_print_info, stop_capturing_print_info
from constants import NetworkConstants, ModelPricingConstants


class CommandManager:
    def __init__(self, conversation_manager: Any) -> None:
        """
        Initialize the CommandManager with dependencies and configuration.

        Sets up the command manager with access to settings, conversation management,
        command registry, RAG engine, and so on.

        Args:
            conversation_manager: The conversation manager instance that handles
                                 AI interactions and conversation history
        """
        self.settings_manager = SettingsManager.getInstance()
        self.conversation_manager = conversation_manager
        self.working_dir = self.settings_manager.setting_get("working_dir")
        self.command_registry = CommandRegistry(self.working_dir)
        self.available_commands = self.command_registry.get_available_commands()
        self.completer = CommandCompleter(self.command_registry)

        # Use the RAG engine that was already created by conversation_manager
        # This ensures we don't create a duplicate RAG engine with a hardcoded OpenAI client
        self.rag_engine = self.conversation_manager.rag_engine

        # Initialize export manager for markdown export functionality
        self.export_manager = ExportManager()

    def _log_command_with_output(self, command: str, captured_output: list) -> None:
        """
        Log a command to LLM context and its verbose output to .md only.

        Args:
            command: The command that was executed (goes to LLM context)
            captured_output: List of captured print_info messages (goes to .md only)
        """
        if self.settings_manager.setting_get("incognito"):
            return

        # Log the command itself to LLM context (clean, no noise)
        self.conversation_manager.log_context(command, "user")

        # Verbose output is no longer logged to files (JSON-only system)
        # Users can export conversations to markdown using --export-markdown if needed

    def _extract_valid_commands(self, user_input: str) -> Tuple[List[Dict[str, Any]], str]:
        """
        Extract valid commands from user input using the command registry.

        Args:
            user_input: Raw user input string that may contain commands

        Returns:
            tuple: (list of command dictionaries, remaining text)
                   Each command dict contains: {'command': str, 'argument': str|None}
        """
        # Known command flags that should not be sent to AI
        known_flags = ["--force-full"]

        valid_commands = self.command_registry.get_available_commands()
        extracted_commands = []
        remaining_text = user_input

        # Sort commands by length (longest first) to avoid partial matches
        sorted_commands = sorted(valid_commands, key=len, reverse=True)

        # Process text to find and extract commands
        for command_name in sorted_commands:
            # Find all occurrences of this command
            pattern = re.escape(command_name) + r'(?=\s|$)'

            matches = list(re.finditer(pattern, remaining_text))

            for match in reversed(matches):  # Process in reverse order to maintain indices
                start_pos = match.start()
                end_pos = match.end()

                # Check if this command requires an argument
                requires_arg = self.command_registry.requires_argument(command_name)

                # Always try to extract arguments (both required and optional)
                argument = None
                rest_of_text = remaining_text[end_pos:].lstrip()
                if rest_of_text:
                    if rest_of_text.startswith('"'):
                        # Handle quoted argument - extract everything until closing quote
                        closing_quote = rest_of_text.find('"', 1)
                        if closing_quote != -1:
                            argument = rest_of_text[1:closing_quote]
                            # Update end position to include quotes and any whitespace after
                            arg_end = closing_quote + 1
                            end_pos = end_pos + len(remaining_text[end_pos:]) - len(rest_of_text) + arg_end
                    else:
                        # Handle unquoted argument - take first word
                        words = rest_of_text.split()
                        if words:
                            argument = words[0]
                            # Update end position to include the argument and any whitespace after it
                            arg_end = rest_of_text.find(argument) + len(argument)
                            end_pos = end_pos + len(remaining_text[end_pos:]) - len(rest_of_text) + arg_end

                extracted_commands.append({
                    'command': command_name,
                    'argument': argument
                })

                # Remove the extracted command (and its argument if any) from remaining text
                remaining_text = remaining_text[:start_pos] + remaining_text[end_pos:]

        # Strip known command flags from remaining text
        for flag in known_flags:
            remaining_text = remaining_text.replace(flag, "")

        # Clean up remaining text (remove extra spaces)
        remaining_text = ' '.join(remaining_text.split())

        # Reverse to get original order
        extracted_commands.reverse()

        return extracted_commands, remaining_text

    def process_commands(self, user_input: str) -> tuple[bool, str]:
        """
        Process and execute commands from user input.

        DRY IMPLEMENTATION: All commands are automatically logged with their output.
        Any new command added to this method will be automatically logged without
        additional code changes.

        Args:
            user_input: Raw user input string that may contain commands

        Returns:
            tuple: (command_processed: bool, remaining_text: str)
                   command_processed: True if any commands were processed, False otherwise
                   remaining_text: Text remaining after command extraction
        """
        # Extract valid commands from user input
        extracted_commands, remaining_text = self._extract_valid_commands(user_input)

        command_processed = False

        # Process each extracted command
        for cmd_info in extracted_commands:
            command_name = cmd_info['command']
            arg = cmd_info['argument']

            # Validate command using registry
            is_valid, error_msg = self.command_registry.validate_command_input(command_name, arg)
            if not is_valid:
                print(f"{error_msg}")
                command_processed = True
                continue

            # Reconstruct full command for logging and processing
            command = command_name + (f" {arg}" if arg else "")

            # DRY SOLUTION: Start capturing for ANY valid command
            start_capturing_print_info()
            command_executed = False

            try:
                if command_name == "--model-clear-cache":
                    self.clear_model_cache()
                    command_executed = True
                elif command_name == "--model":
                    self.set_model(arg)
                    command_executed = True
                elif command_name == "--instructions":
                    self.conversation_manager.apply_instructions(
                        arg, self.settings_manager.setting_get("instructions")
                    )
                    command_executed = True
                elif command_name == "--logmv":
                    # Special case: log BEFORE renaming to capture in current log
                    captured_so_far = stop_capturing_print_info()
                    self._log_command_with_output(command, captured_so_far)

                    # Now handle the rename
                    start_capturing_print_info()

                    # Check if incognito mode is enabled
                    if self.settings_manager.setting_get("incognito"):
                        print_md("Cannot rename log: incognito mode is enabled (no logging active)")
                    elif not self.settings_manager.setting_get("log_file_location") or not os.path.exists(self.settings_manager.setting_get("log_file_location")):
                        print_md("No log file exists yet to rename. Log files are created after the first AI response.")
                    else:
                        title = None
                        if arg is None:
                            print_md("No log file name specified. AI will suggest log filename for you")
                            title = self.conversation_manager.generate_ai_suggested_title()
                        else:
                            title = arg

                        actual_filename = self.conversation_manager.manual_log_rename(title)
                        print_md(f"Log renamed to: {actual_filename}")

                    # Log rename output to NEW log file
                    rename_output = stop_capturing_print_info()
                    # Rename output is no longer logged to files (JSON-only system)
                    command_processed = True
                    continue  # Skip the normal logging flow
                elif command_name == "--logrm":
                    if self.conversation_manager.log_delete():
                        print_md("Log deleted")
                    else:
                        print_md("No log file to delete or deletion failed")
                    command_executed = True
                elif command_name == "--export-markdown":
                    export_path = self.export_manager.export_current_conversation()
                    if export_path:
                        print_md(f"Conversation exported to: {export_path}")
                    else:
                        print_md("Export failed: No active conversation or export error")
                    command_executed = True
                elif command_name == "--log":
                    if self.settings_manager.setting_get("incognito"):
                        print_md("Cannot load log: incognito mode is enabled (no logging active)")
                    elif arg is None:
                        print_md("Please specify the log you want to use")
                    else:
                        # Strip any file extensions (.md or .json) for compatibility
                        base_name = arg
                        if base_name.endswith('.md'):
                            base_name = base_name[:-3]
                        elif base_name.endswith('.json'):
                            base_name = base_name[:-5]
                        self.settings_manager.setting_set("log_file_name", base_name)
                        self.conversation_manager.log_resume()
                    command_executed = True
                elif command_name == "--cbl":
                    latest_reply = None
                    for message in reversed(self.conversation_manager.conversation_history):
                        if message["role"] == "assistant":
                            latest_reply = message["content"]
                            break

                    if latest_reply:
                        try:
                            clipboard.copy(latest_reply)
                            print_md("Latest AI reply copied to clipboard")
                        except Exception as e:
                            print_md(f"Failed to copy to clipboard: {e}")
                    else:
                        print_md("No AI reply found to copy")
                    command_executed = True
                elif command_name == "--cb":
                    clipboard_content = clipboard.paste()
                    if clipboard_content:
                        print_md("Clipboard content added to conversation context")
                        self.conversation_manager.log_context(clipboard_content, "user")
                    else:
                        print_md("Clipboard is empty. Please type your input")
                    command_executed = True
                elif command_name == "--youtube":
                    if arg is None:
                        print_md("Please specify a youtube url")
                    else:
                        self.extract_youtube_content(arg)
                    command_executed = True
                elif command_name == "--url":
                    if arg is None:
                        print_md("Please specify a URL")
                    else:
                        self.extract_url_content(arg)
                    command_executed = True
                elif command_name == "--file":
                    if arg is None:
                        print_md("Please specify a file path")
                    else:
                        self.extract_file_content(arg)
                    command_executed = True
                elif command_name == "--folder":
                    if arg is None:
                        print_md("Please specify a directory path")
                    else:
                        self.extract_folder_content(arg)
                    command_executed = True
                elif command_name == "--folder-recursive":
                    if arg is None:
                        print_md("Please specify a directory path")
                    else:
                        self.extract_folder_content_recursive(arg)
                    command_executed = True
                elif command_name == "--search":
                    if self.settings_manager.setting_get("search"):
                        self.settings_manager.setting_set("search", False)
                        print_md("Web search disabled")
                    else:
                        # Disable search-deep if it's currently enabled
                        if self.settings_manager.setting_get("search_deep"):
                            self.settings_manager.setting_set("search_deep", False)
                            print_md("Deep search disabled")
                        self.settings_manager.setting_set("search", True)
                        print_md("Web search enabled")
                    command_executed = True
                elif command_name == "--search-deep":
                    if self.settings_manager.setting_get("search_deep"):
                        self.settings_manager.setting_set("search_deep", False)
                        print_md("Deep search disabled")
                    else:
                        # Disable regular search if it's currently enabled
                        if self.settings_manager.setting_get("search"):
                            self.settings_manager.setting_set("search", False)
                            print_md("Web search disabled")
                        self.settings_manager.setting_set("search_deep", True)
                        print_md("Deep search enabled - AI will autonomously determine research completeness")
                    command_executed = True
                elif command_name == "--search-engine":
                    self.set_search_engine(arg)
                    command_executed = True

                elif command_name == "--nothink":
                    nothink = self.settings_manager.setting_get("nothink")
                    if nothink:
                        self.settings_manager.setting_set("nothink", False)
                        print_md("Nothink mode disabled")
                    else:
                        self.settings_manager.setting_set("nothink", True)
                        print_md("Nothink mode enabled")
                    command_executed = True
                elif command_name == "--markdown":
                    markdown = self.settings_manager.setting_get("markdown")
                    if markdown:
                        self.settings_manager.setting_set("markdown", False)
                        print_md("Markdown rendering disabled")
                    else:
                        self.settings_manager.setting_set("markdown", True)
                        print_md("Markdown rendering enabled")
                    command_executed = True
                elif command_name == "--incognito":
                    incognito = self.settings_manager.setting_get("incognito")
                    if incognito:
                        self.settings_manager.setting_set("incognito", False)
                        print_md("Incognito mode disabled - logging resumed")
                    else:
                        self.settings_manager.setting_set("incognito", True)
                        print_md("Incognito mode enabled - no data will be saved to logs")
                    command_executed = True
                elif command_name == "--clear":
                    self.conversation_manager.start_new_conversation_log()
                    clear_text = "Conversation history cleared - will create new log file after first AI response\n"
                    clear_text += "    AI instructions preserved"
                    print_md(clear_text)
                    command_executed = True
                elif command_name == "--usage":
                    self.display_token_usage()
                    command_executed = True

                elif command_name == "--rag-deactivate":
                    self.rag_off()
                    print_md("RAG deactivated")
                    command_executed = True
                elif command_name == "--rag-status":
                    self.rag_status()
                    command_executed = True
                elif command_name == "--rag-rebuild":
                    if arg is None:
                        print_md("Please specify a collection name to rebuild")
                    else:
                        # Check if --force-full flag is present
                        force_full = False
                        if user_input and "--force-full" in user_input:
                            force_full = True
                        self.rag_rebuild(arg, force_full)
                    command_executed = True
                elif command_name == "--rag-show":
                    if arg is None:
                        print_md("Please specify a filename to show")
                    else:
                        self.rag_show(arg)
                    command_executed = True
                elif command_name == "--rag-test":
                    self.rag_test_connection()
                    command_executed = True
                elif command_name == "--rag-info":
                    self.rag_model_info()
                    command_executed = True
                elif command_name == "--rag":
                    if arg is None:
                        print_md("Please specify a collection name to activate")
                    else:
                        # Activate specific collection
                        self.rag_activate(arg)
                    command_executed = True



            finally:
                # DRY LOGGING: This handles ALL commands automatically
                if command_executed:
                    captured_output = stop_capturing_print_info()
                    self._log_command_with_output(command, captured_output)
                    command_processed = True
                else:
                    # Command not recognized - don't log it
                    stop_capturing_print_info()  # Clean up capture

        # Send remaining text to AI if there's any non-command content
        if remaining_text.strip():
            # Check if nothink mode is enabled and prepend /nothink prefix
            final_user_input = remaining_text.strip()
            if self.settings_manager.setting_get("nothink"):
                final_user_input = "/nothink " + final_user_input

            # Add remaining text to conversation and generate response
            self.conversation_manager.log_context(final_user_input, "user")

            self.conversation_manager.generate_response()

        # Ensure proper spacing before next user prompt
        if command_processed:
            print()

        return command_processed, remaining_text

    def rag_list(self, from_toggle: bool = False) -> None:
        """
        List all available RAG collections.

        Displays collections found in the rag/ directory.
        """
        if not self.rag_engine:
            print_md("RAG engine not available")
            return

        if from_toggle:
            print_md("RAG is not active. Available collections:")
        else:
            print_md("Available RAG collections:")

        collections = self.rag_engine.list_collections()
        if not collections:
            no_collections_text = "No RAG collections found in rag/ directory\n"
            no_collections_text += "    Create collections by making directories in rag/collection_name/"
            print_md(no_collections_text)
            return
        for i, collection in enumerate(collections, 1):
            status_info = []
            if collection.get("has_index"):
                if collection.get("cache_valid"):
                    status_info.append("- ready")
                else:
                    status_info.append("- needs rebuild")
            else:
                status_info.append("- not built")

            status = f"({', '.join(status_info)})" if status_info else ""
            print_md(f"{i}. {collection['name']} - {collection['file_count']} files {status}")

        print_md("Use --rag <collection_name>  # Activate collection (builds automatically if needed)")

    def rag_activate(self, collection_name: str) -> None:
        """
        Activate a specific RAG collection for use in conversations.

        Loads the specified collection and makes it available for context
        retrieval during AI conversations. If the collection hasn't been
        built yet, it will be built automatically. If files on disk have
        changed since last build, it will be rebuilt. Only one collection
        can be active at a time.

        Args:
            collection_name: Name of the collection to activate (directory name in rag/)
        """
        if not self.rag_engine:
            print_md("RAG engine not available")
            return

        success = self.rag_engine.activate_collection(collection_name)
        if not success:
            # Error messages are printed by the RAG engine
            available = self.rag_engine.vector_store.get_available_collections()
            if available:
                print_md(f"Available collections: {', '.join(available)}")

    def rag_off(self) -> None:
        """
        Deactivate the currently active RAG collection.

        Turns off RAG mode, which means no document context will be
        retrieved and added to AI conversations. The conversation
        will proceed with only the standard context.
        """
        if not self.rag_engine:
            print_md("RAG engine not available")
            return

        self.rag_engine.deactivate_collection()

    def rag_rebuild(self, collection_name: str, force_full: bool = False) -> None:
        """
        Rebuild a RAG collection's vector index.

        By default, performs a smart rebuild that only processes changed files.
        Use --force-full to force a complete rebuild from scratch.

        Args:
            collection_name: Name of the collection to rebuild
            force_full: If True, force complete rebuild ignoring smart rebuild
        """
        if not self.rag_engine:
            print_md("RAG engine not available")
            return

        self.rag_engine.build_collection(collection_name, force_rebuild=True, force_full=force_full)

    def rag_show(self, filename: str) -> None:
        """
        Display the document chunks for a specific file in the active collection.

        Shows how a file has been split into chunks for vector storage,
        which is useful for debugging and understanding how documents
        are processed for RAG retrieval.

        Args:
            filename: Name of the file to display chunks for
        """
        if not self.rag_engine:
            print_md("RAG engine not available")
            return

        result = self.rag_engine.show_chunk_in_file(filename)
        print_md(result)

    def rag_status(self) -> None:
        """
        Display comprehensive RAG system status and configuration.

        Shows current RAG state including active collection, chunk counts,
        available collections, embedding provider configuration, and
        relevant settings. Useful for troubleshooting and system monitoring.
        """
        if not self.rag_engine:
            print_md("RAG engine not available")
            return

        status = self.rag_engine.get_status()
        status_text = "RAG Status:\n"
        status_text += f"    Active: {status['active']}\n"
        if status['active']:
            status_text += f"    Collection: {status['active_collection']}\n"
            status_text += f"    Chunks loaded: {status['chunk_count']}"
        print_md(status_text)

        print_md(f"Available collections: {status['available_collections']}")

        # Show embedding provider information
        try:
            provider = self.settings_manager.setting_get("embedding_provider")
            print_md(f"Embedding provider: {provider}")

            if provider == "openai":
                model = self.settings_manager.setting_get("openai_embedding_model")
                print_md(f"OpenAI model: {model}")
            elif provider == "ollama":
                model = self.settings_manager.setting_get("ollama_embedding_model")
                ollama_url = self.settings_manager.setting_get("ollama_base_url")
                ollama_text = f"    Ollama model: {model}\n"
                ollama_text += f"    Ollama URL: {ollama_url}"
                print_md(ollama_text)
        except Exception as e:
            print_md(f"Provider info: Error getting provider details")

        print_md(f"Settings:")
        for key, value in status['settings'].items():
            print_md(f"{key}: {value}")

    def rag_test_connection(self) -> None:
        """
        Test connectivity and authentication with the current embedding provider.

        Verifies that the embedding service (OpenAI or Ollama) is accessible
        and properly configured. Shows model information, dimensions, and
        any provider-specific details. Useful for troubleshooting.
        """
        if not self.rag_engine:
            print_md("RAG engine not available")
            return

        try:
            provider = self.settings_manager.setting_get("embedding_provider")
            print_md(f"Testing connection to {provider} embedding service...")

            # Test using the embedding service
            success = self.rag_engine.embedding_service.test_connection()

            if success:
                success_text = f"Connection to {provider} successful!\n"

                # Show additional info
                model_info = self.rag_engine.embedding_service.get_embedding_model_info()
                model = model_info.get("model", "unknown")
                dimensions = self.rag_engine.embedding_service.get_embedding_dimensions()
                success_text += f"    Model: {model}\n"
                success_text += f"    Dimensions: {dimensions}"
                print_md(success_text)

                if provider == "ollama":
                    info = model_info.get("info", {})
                    if info.get("multilingual"):
                        print_md(f"Languages: {info.get('languages', 'Multiple languages supported')}")

            else:
                print_md(f"Connection to {provider} failed!")

                if provider == "ollama":
                    ollama_url = self.settings_manager.setting_get("ollama_base_url")
                    print_md(f"Check that Ollama is running at: {ollama_url}")
                    model = self.settings_manager.setting_get("ollama_embedding_model")
                    print_md(f"Check that model '{model}' is available in Ollama")
                elif provider == "openai":
                    print_md("Check your OpenAI API key and internet connection")

        except Exception as e:
            print_md(f"Error testing connection: {e}")

    def rag_model_info(self) -> None:
        """
        Display detailed information about the current embedding model and RAG configuration.

        Shows embedding model specifications including provider, model name,
        dimensions, token limits, and costs. Also displays current RAG
        settings like chunk size, overlap, and retrieval parameters.
        """
        if not self.rag_engine:
            print_md("RAG engine not available")
            return

        try:
            model_info = self.rag_engine.embedding_service.get_embedding_model_info()
            provider = model_info.get("provider", "unknown")
            model = model_info.get("model", "unknown")
            info = model_info.get("info", {})

            model_text = "Embedding Model Information:\n"
            model_text += f"    Provider: {provider}\n"
            model_text += f"    Model: {model}\n"
            model_text += f"    Dimensions: {info.get('dimensions', 'unknown')}\n"
            model_text += f"    Max tokens: {info.get('max_tokens', 'unknown')}"
            print_md(model_text)

            if provider == "openai":
                cost = info.get('cost_per_1k_tokens', 0)
                print_md(f"Cost per 1K tokens: ${cost}")
            elif provider == "ollama":
                print_md(f"Cost per 1K tokens: Free (local)")
                if info.get('multilingual'):
                    print_md(f"Multilingual: Yes")
                    languages = info.get('languages')
                    if languages:
                        print_md(f"Languages: {languages}")

            # Show current settings
            settings_text = "Current RAG Settings:\n"
            settings_text += f"    Chunk size: {self.settings_manager.setting_get('rag_chunk_size')} tokens\n"
            settings_text += f"    Chunk overlap: {self.settings_manager.setting_get('rag_chunk_overlap')} tokens\n"
            settings_text += f"    Top K results: {self.settings_manager.setting_get('rag_top_k')}"
            print_md(settings_text)

        except Exception as e:
            print_md(f"Error getting model info: {e}")

    def _validate_model(self, model_name: str) -> bool:
        """
        Validate if a model name is available from OpenAI, Google, or Ollama APIs.

        Args:
            model_name: The model name to validate

        Returns:
            bool: True if model is valid, False otherwise
        """
        try:
            # Get available models using the same logic as command completer
            available_models = self.completer._get_available_models()

            # Check if model exists in available models
            for model_info in available_models:
                if model_info["name"] == model_name:
                    return True

            return False

        except Exception:
            # If we can't fetch models, allow the model (fallback behavior)
            return True

    def clear_model_cache(self) -> None:
        """
        Clear the cached model list.

        This is useful when new models have been added to OpenAI, Google,
        or Ollama.
        """
        print_md("Clearing model cache...")
        self.completer.clear_models_cache()
        print_md("Model cache cleared. Fresh models will be fetched on next use")

    def set_model(self, arg: Optional[str]) -> None:
        """
        Set the active AI model for conversations.

        Validates the model name against available models from OpenAI, Google,
        and Ollama APIs.

        Args:
            arg: Model name to set (e.g., 'gpt-5').
                 If None, displays available models and usage information.
        """
        if arg == None:
            ollama_url = self.settings_manager.setting_get("ollama_base_url")
            model_help_text = "Please specify the model to use\n"
            model_help_text += "Available sources:\n"
            model_help_text += "    OpenAI (https://platform.openai.com/docs/models)\n"
            model_help_text += f"    Ollama ({ollama_url} if running)\n"
            model_help_text += "    Google (https://ai.google.dev/gemini-api/docs/models)\n"

            print_md(model_help_text)
            return

        model = arg

        # Validate the model before setting it
        if not self._validate_model(model):
            invalid_model_text = f"Invalid model: {model}\n"
            invalid_model_text += "    Model not found in OpenAI, Google, or Ollama APIs"
            print_md(invalid_model_text)

            # Show available models
            try:
                available_models = self.completer._get_available_models()
                if available_models:
                    print_md("Available models:")

                    # Group by source
                    openai_models = [m["name"] for m in available_models if m["source"] == "OpenAI"]
                    ollama_models = [m["name"] for m in available_models if m["source"] == "Ollama"]
                    google_models = [m["name"] for m in available_models if m["source"] == "Google"]


                    if openai_models:
                        print_md(f"OpenAI: {', '.join(openai_models[:5])}" +
                              (f" (and {len(openai_models)-5} more)" if len(openai_models) > 5 else ""))

                    if google_models:
                        print_md(f"Google: {', '.join(google_models)}")



                    if ollama_models:
                        print_md(f"Ollama: {', '.join(ollama_models)}")

                    if not openai_models and not ollama_models and not google_models:
                        print_md("(No models available - check API keys and network connection)")
                else:
                    print_md("(Unable to fetch available models)")
            except Exception:
                print_md("(Unable to fetch available models)")

            return

        # Set the model (this will trigger client update in conversation manager)
        self.conversation_manager.model = model
        self.settings_manager.setting_set("model", model)

        # Determine model source
        provider = self.conversation_manager.llm_client_manager.get_provider_for_model(model)



        # Show appropriate message using centralized method
        self.settings_manager.display_model_info("switch", provider, self.conversation_manager.llm_client_manager)

    def set_search_engine(self, arg: Optional[str]) -> None:
        """
        Set the active search engine for current session.

        Args:
            arg: Search engine name to set ('tavily' or 'searxng').
                 If None, displays current engine and available options.
        """
        if arg is None:
            current_engine = self.settings_manager.search_engine
            print_md(f"Current search engine: {current_engine}")
            engine_help_text = "Available search engines:\n"
            engine_help_text += "    tavily   - Tavily API (commercial search with AI-powered results)\n"
            engine_help_text += "    searxng  - SearXNG (privacy-focused open-source metasearch)\n"
            engine_help_text += "Usage: --search-engine tavily"
            print_md(engine_help_text)
            return

        search_engine = arg.lower()

        # Validate search engine
        valid_engines = ["tavily", "searxng"]
        if search_engine not in valid_engines:
            invalid_engine_text = f"Invalid search engine: {search_engine}\n"
            invalid_engine_text += f"    Valid options: {', '.join(valid_engines)}"
            print_md(invalid_engine_text)
            return

        # Set the search engine
        self.settings_manager.setting_set("search_engine", search_engine)

        # Show confirmation with additional info
        if search_engine == "tavily":
            print_md("Search engine: Tavily (commercial API with AI-powered results)")
        else:  # searxng
            searxng_url = self.settings_manager.searxng_base_url
            print_md(f"Search engine: SearXNG (privacy-focused, using {searxng_url})")

    def extract_youtube_content(self, arg: str) -> None:
        """
        Extract transcript and metadata from a YouTube video for AI analysis.

        Attempts to extract video transcripts using multiple strategies:
        1. Manual English subtitles (highest quality)
        2. Auto-generated captions in English
        3. Manual subtitles in other languages (as fallback)
        4. Auto-generated captions in other languages (as fallback)
        5. Fallback to video metadata only if no transcript available

        When using non-English subtitles, a language note is added to the context
        to inform the AI about the transcript language.

        Supports various YouTube URL formats and automatically extracts video ID.
        The extracted content (title, channel, transcript) is added to the
        conversation context for AI analysis.

        Args:
            arg: YouTube URL (supports youtube.com/watch?v=, youtu.be/, and other formats)
        """
        print_md("Extracting info from YouTube...")

        video_url = None

        # Check if 'watch?v=' is in the URL, otherwise resolve URL
        if "watch?v=" in arg or "10.13.0.200:8090/embed" in arg:
            video_url = arg
        else:
            print_md("Video ID missing from URL")
            try:
                print_md("Attempting to determine ID")
                response = requests.get(arg)
                video_url = response.url
            except requests.RequestException as e:
                print_md(f"Error resolving URL: {e}")
                return

        if video_url:
            try:
                if "watch?v=" in video_url:
                    video_id = video_url.split("watch?v=")[1]
                    # Handle additional parameters after video ID
                    if "&" in video_id:
                        video_id = video_id.split("&")[0]
                elif "10.13.0.200:8090/embed" in video_url:
                    # Extract video ID from embed URL
                    video_id = video_url.split("/embed/")[1]
                    # Handle additional parameters after video ID
                    if "?" in video_id:
                        video_id = video_id.split("?")[0]
                    # Convert embed URL to standard YouTube URL for yt-dlp
                    video_url = f"https://www.youtube.com/watch?v={video_id}"
                else:
                    print_md("Unsupported URL format")
                    return
                print_md(f"Video ID found: {video_id}")
            except IndexError:
                print_md("Invalid YouTube URL provided")
                return

            try:
                # Get video info using yt-dlp
                print_md("Fetching video info...")
                ydl_opts = {'quiet': True, 'no_warnings': True}
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(video_url, download=False)
                    if info:
                        video_title = "[" + info.get('title', 'Unknown Title') + "](" + video_url + ")"
                        channel_title = "[" + info.get('uploader', 'Unknown Channel') + "](" + info.get('uploader_url', video_url) + ")"
                        print_md(f"Title: {info.get('title', 'Unknown Title')}")
                    else:
                        print_md("Could not extract video information")
                        return

                # Extract subtitles using yt-dlp
                print_md("Fetching video transcript using yt-dlp...")
                try:
                    # Configure yt-dlp to extract subtitles
                    subtitle_opts = {
                        'quiet': True,
                        'no_warnings': True,
                        'writesubtitles': False,
                        'writeautomaticsub': True,
                        'subtitleslangs': ['en', 'en-US', 'en-GB'],
                        'skip_download': True,
                        'extract_flat': False
                    }

                    # Get subtitle information
                    with yt_dlp.YoutubeDL(subtitle_opts) as ydl:
                        subtitle_info = ydl.extract_info(video_url, download=False)

                        transcript_text = None
                        subtitles = subtitle_info.get('subtitles', {}) if subtitle_info else {}
                        automatic_captions = subtitle_info.get('automatic_captions', {}) if subtitle_info else {}

                        # Try to find subtitles in order of preference
                        subtitle_data = None
                        subtitle_source = None

                        # First try manual English subtitles
                        for lang in ['en', 'en-US', 'en-GB', 'en-CA', 'en-AU']:
                            if lang in subtitles and subtitles[lang]:
                                subtitle_data = subtitles[lang]
                                subtitle_source = f"manual {lang}"
                                print_md(f"Found manual {lang} subtitles")
                                break

                        # If no manual English subtitles, try automatic English captions
                        if not subtitle_data:
                            for lang in ['en', 'en-US', 'en-GB', 'en-CA', 'en-AU']:
                                if lang in automatic_captions and automatic_captions[lang]:
                                    subtitle_data = automatic_captions[lang]
                                    subtitle_source = f"auto-generated {lang}"
                                    print_md(f"Found auto-generated {lang} captions")
                                    break

                        # If no English subtitles found, try other languages as fallback
                        if not subtitle_data:
                            available_langs = list(subtitles.keys()) + list(automatic_captions.keys())
                            if available_langs:
                                subtitle_fallback_text = f"Available subtitle languages: {', '.join(set(available_langs))}\n"
                                subtitle_fallback_text += "    No English subtitles found\n"
                                subtitle_fallback_text += "    Attempting to use other available subtitle languages..."
                                print_md(subtitle_fallback_text)

                                # Try manual subtitles first (any language)
                                for lang in subtitles.keys():
                                    if subtitles[lang]:
                                        subtitle_data = subtitles[lang]
                                        subtitle_source = f"manual {lang}"
                                        print_md(f"Using manual {lang} subtitles as fallback")
                                        break

                                # If no manual subtitles, try automatic captions (any language)
                                if not subtitle_data:
                                    for lang in automatic_captions.keys():
                                        if automatic_captions[lang]:
                                            subtitle_data = automatic_captions[lang]
                                            subtitle_source = f"auto-generated {lang}"
                                            print_md(f"Using auto-generated {lang} captions as fallback")
                                            break

                        if subtitle_data:
                            # Find the best subtitle format (prefer vtt, then srv3, then srv2, then srv1)
                            preferred_formats = ['vtt', 'srv3', 'srv2', 'srv1']
                            subtitle_url = None

                            for fmt in preferred_formats:
                                for sub in subtitle_data:
                                    if sub.get('ext') == fmt:
                                        subtitle_url = sub.get('url')
                                        break
                                if subtitle_url:
                                    break

                            if not subtitle_url and subtitle_data:
                                # Use first available subtitle format
                                subtitle_url = subtitle_data[0].get('url')

                            if subtitle_url:
                                # Download and parse the subtitle file
                                try:
                                    response = requests.get(subtitle_url, timeout=NetworkConstants.SUBTITLE_FETCH_TIMEOUT)
                                    if response.status_code == 200:
                                        subtitle_content = response.text


                                        # Parse VTT or SRV format to extract text
                                        import re

                                        # Remove VTT headers and timing information
                                        lines = subtitle_content.split('\n')
                                        text_lines = []

                                        for line in lines:
                                            line = line.strip()
                                            # Skip empty lines, headers, and timing lines
                                            if (line and
                                                not line.startswith('WEBVTT') and
                                                not line.startswith('NOTE') and
                                                not '-->' in line and
                                                not re.match(r'^\d+$', line) and
                                                not line.startswith('<') and
                                                not line.startswith('{')):

                                                # Clean up HTML tags and formatting
                                                clean_line = re.sub(r'<[^>]+>', '', line)
                                                clean_line = re.sub(r'&[a-zA-Z]+;', '', clean_line)
                                                clean_line = clean_line.strip()

                                                if clean_line:
                                                    text_lines.append(clean_line)

                                        transcript_text = ' '.join(text_lines)


                                        if transcript_text:
                                            transcript_success_text = f"Successfully extracted transcript from {subtitle_source}\n"
                                            transcript_success_text += f"    Processing {len(transcript_text)} characters as input..."
                                            print_md(transcript_success_text)

                                            # Add language note if not English
                                            language_note = ""
                                            if not any(lang_code in subtitle_source for lang_code in ['en', 'en-US', 'en-GB', 'en-CA', 'en-AU']):
                                                lang_code = subtitle_source.split()[-1]
                                                language_note = f"\nNote: Transcript is in {lang_code} language. Please respond in the users preferred language regardless of the transcript language (infer preferred language based on users prompts)."

                                            user_input = (
                                                "Channel title: " + channel_title +
                                                "\nVideo title: " + video_title +
                                                language_note +
                                                "\nVideo transcript: " + transcript_text
                                            )
                                            self.conversation_manager.log_context(user_input, "user")
                                            print_md("YouTube content added to conversation context")
                                        else:
                                            raise Exception("Transcript text was empty after parsing")
                                    else:
                                        raise Exception(f"Failed to download subtitles: HTTP {response.status_code}")

                                except Exception as subtitle_error:
                                    print_md(f"Error processing subtitle file: {subtitle_error}")
                                    raise subtitle_error
                            else:
                                raise Exception("No subtitle URL found")
                        else:
                            # No subtitles found at all
                            print_md("No subtitles available for this video")
                            raise Exception("No subtitles available")

                except Exception as e:
                    transcript_error_text = f"Could not extract transcript: {str(e)}\n"
                    transcript_error_text += "    Continuing with video info only..."
                    print_md(transcript_error_text)
                    # Still provide video info even without transcript
                    user_input = (
                        "Channel title: " + channel_title +
                        "\nVideo title: " + video_title +
                        "\nNote: No transcript could be extracted for this video."
                    )
                    self.conversation_manager.log_context(user_input, "user")
                    print_md("YouTube content added to conversation context")

            except Exception as e:
                print_md("An error occurred:", e)
                return
        else:
            print_md("Could not determine the YouTube video URL")
            return

    def extract_url_content(self, url: str) -> None:
        """
        Extract and process content from any web URL for AI analysis.

        Uses advanced web scraping techniques to extract readable content from
        web pages, handling various content types, paywalls, and other content.
        Automatically detects YouTube URLs and redirects them to the specialized
        YouTube transcript extraction for better results.

        The extracted content (title, URL, main text) is cleaned and added to
        the conversation context for AI analysis.

        Args:
            url: Web URL to extract content from (any valid HTTP/HTTPS URL)
        """
        # Check if this is a YouTube URL and redirect to YouTube command
        # Detect YouTube URLs by common patterns including watch?v= parameter and embed URLs
        url_lower = url.lower()
        if ("youtube.com" in url_lower or "youtu.be" in url_lower or "watch?v=" in url_lower or "10.13.0.200:8090/embed" in url_lower):
            print_md("YouTube URL detected - redirecting to --youtube command for better transcript extraction...")
            self.extract_youtube_content(url)
            return

        print_md("Extracting content from URL...")

        extractor = WebContentExtractor(self.conversation_manager.llm_client_manager)
        result = extractor.extract_content(url)

        if result['error']:
            print_md(f"Error: {result['error']}")
            return

        if not result['content']:
            print_md("No content could be extracted from the URL")
            return

        # Display warning if paywall was encountered
        if result.get('warning'):
            print_md(f"{result['warning']}")

        # Format the content for the conversation
        title = result['title'] or "Web Content"
        formatted_content = f"Website: {title}\n\nSource: {url}\n\n{result['content']}"

        # Add to conversation history
        self.conversation_manager.log_context(formatted_content, "user")

        # Only show success messages if bypass didn't fail
        # (WebContentExtractor already explained bypass failure to user)
        if not result.get('bypass_failed'):
            url_success_text = "Content added to conversation context\n"
            url_success_text += "    You can now ask questions about this content"
            print_md(url_success_text)

    def extract_file_content(self, file_path: str) -> None:
        """
        Load and process file contents for AI analysis.

        Supports multiple file formats including PDF, DOCX, TXT, Markdown, and more.
        Uses the same document processing pipeline as the RAG system to ensure
        consistent text extraction and cleaning. The processed content is added
        to the conversation context with metadata (filename, path, word count).

        Args:
            file_path: Path to the file to process (relative or absolute path)

        Note:
            File must exist and be of a supported type. See get_supported_extensions_display()
            or rag_config.py for a list of supported file formats.
        """
        import os

        # Expand user home directory (handle tilde paths)
        file_path = os.path.expanduser(file_path)

        # Check if file exists
        if not os.path.exists(file_path):
            print_md(f"Error: File not found: {file_path}")
            return

        # Check if file type is supported
        if not is_supported_file(file_path):
            supported_types = get_supported_extensions_display()
            print_md(f"Error: Unsupported file type. Supported types: {supported_types}")
            return

        print_md(f"Loading file: {os.path.basename(file_path)}")

        # Use DocumentProcessor to load the file content
        processor = DocumentProcessor()
        try:
            content = processor.load_file(file_path)

            if not content or not content.strip():
                print_md("Error: No content could be extracted from the file")
                return

            # Clean the content
            content = processor.clean_text(content)

            # Get file info for display
            filename = os.path.basename(file_path)
            file_ext = os.path.splitext(filename)[1].lower()
            word_count = len(content.split())

            print_md(f"Extracted content from {filename} ({word_count} words)")

            # Format the content for the conversation
            formatted_content = f"File: {filename}\n\nPath: {file_path}\n\n{content}"

            # Add to conversation history
            self.conversation_manager.log_context(formatted_content, "user")

            file_success_text = "File content added to conversation context\n"
            file_success_text += "    You can now ask questions about this content"
            print_md(file_success_text)

        except Exception as e:
            print_md(f"Error loading file: {e}")

    def extract_folder_content(self, folder_path: str) -> None:
        """
        Load and process contents of all supported files in a directory.
        Non-recursive - only processes files in the specified directory.

        Args:
            folder_path: Path to the directory to process (relative or absolute path)
        """
        import os

        # Expand user home directory (handle tilde paths)
        folder_path = os.path.expanduser(folder_path)

        # Check if directory exists
        if not os.path.exists(folder_path):
            print_md(f"Error: Directory not found: {folder_path}")
            return

        # Check if path is actually a directory
        if not os.path.isdir(folder_path):
            print_md(f"Error: Path is not a directory: {folder_path}")
            return

        print_md(f"Loading files from: {os.path.basename(folder_path)}")

        # Get all files in the directory
        try:
            files = os.listdir(folder_path)
            supported_files = []

            # Filter for supported files
            for filename in files:
                file_path = os.path.join(folder_path, filename)
                if os.path.isfile(file_path) and is_supported_file(file_path):
                    supported_files.append(file_path)

            if not supported_files:
                supported_types = get_supported_extensions_display()
                print_md(f"No supported files found in directory. Supported types: {supported_types}")
                return

            print_md(f"Found {len(supported_files)} supported file(s)")

            # Process each supported file
            files_processed = 0
            for file_path in supported_files:
                try:
                    self.extract_file_content(file_path)
                    files_processed += 1
                except Exception as e:
                    print_md(f"Error processing {os.path.basename(file_path)}: {e}")

            print_md(f"Successfully processed {files_processed} file(s) from folder")

        except Exception as e:
            print_md(f"Error reading directory: {e}")

    def extract_folder_content_recursive(self, folder_path: str) -> None:
        """
        Load and process contents of all supported files in a directory and its subdirectories.
        Recursive - processes files in the specified directory and all subdirectories.

        Args:
            folder_path: Path to the directory to process recursively (relative or absolute path)
        """
        import os

        # Expand user home directory (handle tilde paths)
        folder_path = os.path.expanduser(folder_path)

        # Check if directory exists
        if not os.path.exists(folder_path):
            print_md(f"Error: Directory not found: {folder_path}")
            return

        # Check if path is actually a directory
        if not os.path.isdir(folder_path):
            print_md(f"Error: Path is not a directory: {folder_path}")
            return

        print_md(f"Loading files recursively from: {os.path.basename(folder_path)}")

        # Walk directory tree recursively
        try:
            supported_files = []

            for root, dirs, files in os.walk(folder_path):
                for filename in files:
                    file_path = os.path.join(root, filename)
                    if is_supported_file(file_path):
                        supported_files.append(file_path)

            if not supported_files:
                supported_types = get_supported_extensions_display()
                print_md(f"No supported files found in directory tree. Supported types: {supported_types}")
                return

            print_md(f"Found {len(supported_files)} supported file(s) recursively")

            # Process each supported file
            files_processed = 0
            for file_path in supported_files:
                try:
                    # Show relative path for better readability
                    relative_path = os.path.relpath(file_path, folder_path)
                    print_md(f"Processing: {relative_path}")
                    self.extract_file_content(file_path)
                    files_processed += 1
                except Exception as e:
                    relative_path = os.path.relpath(file_path, folder_path)
                    print_md(f"Error processing {relative_path}: {e}")

            print_md(f"Successfully processed {files_processed} file(s) from folder tree")

        except Exception as e:
            print_md(f"Error reading directory tree: {e}")

    def estimate_tokens_for_text(self, text: str, model: str) -> int:
        """
        Estimate token count for text using tiktoken.
        Works best for OpenAI models, provides approximation for others.
        """
        try:
            # Try to get encoding for the specific model
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback to a common encoding if model not recognized
            encoding = tiktoken.get_encoding("cl100k_base")

        return len(encoding.encode(text))

    def get_openai_pricing(self, model: str) -> Optional[Dict[str, float]]:
        """
        Get pricing data for a model from any provider.
        Returns costs per 1M tokens for input and output.
        """
        return ModelPricingConstants.get_model_pricing(model)

    def calculate_cost(self, input_tokens: int, output_tokens: int, model: str) -> dict:
        """
        Calculate estimated cost for token usage with an OpenAI model.
        Returns cost breakdown and total.
        """
        pricing = self.get_openai_pricing(model)

        if not pricing:
            return None

        # Convert to cost (pricing is per 1M tokens)
        input_cost = (input_tokens / 1_000_000) * pricing['input']
        output_cost = (output_tokens / 1_000_000) * pricing['output']
        total_cost = input_cost + output_cost

        return {
            'input_cost': input_cost,
            'output_cost': output_cost,
            'total_cost': total_cost,
            'input_rate': pricing['input'],
            'output_rate': pricing['output']
        }

    def estimate_conversation_tokens(self) -> dict:
        """
        Calculate token usage breakdown for the entire conversation history.

        Analyzes all messages in the current conversation and estimates
        token counts by role (system, user, assistant). Uses tiktoken
        for accurate OpenAI model estimates, with approximations for
        other providers.

        Returns:
            dict: Token breakdown with keys 'system', 'user', 'assistant', 'total'
        """
        model = self.conversation_manager.model
        conversation_history = self.conversation_manager.conversation_history

        token_breakdown = {
            'system': 0,
            'user': 0,
            'assistant': 0,
            'total': 0
        }

        for message in conversation_history:
            role = message.get('role', 'unknown')
            content = message.get('content', '')

            tokens = self.estimate_tokens_for_text(content, model)

            if role in token_breakdown:
                token_breakdown[role] += tokens

            token_breakdown['total'] += tokens

        return token_breakdown

    def display_token_usage(self) -> None:
        """
        Display detailed token usage statistics and cost estimates.

        Shows comprehensive token analysis including:
        - Token counts by message type (system, user, assistant)
        - Cost estimates for OpenAI models
        - Last exchange token usage
        - Current model information and pricing rates

        Provides both conversation-wide and recent exchange statistics
        to help users understand their API usage and costs.
        """
        model = self.conversation_manager.model

        # Check if this is an OpenAI model for accuracy warning
        is_openai_model = model.startswith(('gpt-', 'text-embedding-', 'whisper-', 'dall-e'))

        if not is_openai_model:
            print_md("Note: Token counts are rough estimates for non-OpenAI models")
            print("")

        # Get token breakdown
        breakdown = self.estimate_conversation_tokens()

        # Build complete token usage section
        token_section = f"Token Usage Summary:\n"
        token_section += f"    Current model: {model}\n"
        token_section += f"    System messages (e.g. instructions, search, RAG): {breakdown['system']:,} tokens\n"
        token_section += f"    User messages: {breakdown['user']:,} tokens\n"
        token_section += f"    Assistant responses: {breakdown['assistant']:,} tokens\n"
        token_section += f"    Total conversation: {breakdown['total']:,} tokens"
        print_md(token_section)

        # Calculate and display costs for OpenAI models
        if is_openai_model:
            input_tokens = breakdown['system'] + breakdown['user']
            output_tokens = breakdown['assistant']

            cost_info = self.calculate_cost(input_tokens, output_tokens, model)

            if cost_info:
                # Build complete cost section
                cost_section = f"\nEstimated Cost:\n"
                cost_section += f"    Input tokens ({input_tokens:,}): ${cost_info['input_cost']:.4f}\n"
                cost_section += f"    Output tokens ({output_tokens:,}): ${cost_info['output_cost']:.4f}\n"
                cost_section += f"    Total conversation cost: ${cost_info['total_cost']:.4f}\n"
                cost_section += f"    Rate: ${cost_info['input_rate']:.2f}/${cost_info['output_rate']:.2f} per 1M tokens (input/output)"
                print_md(cost_section)

        # Show last exchange if available
        conversation_history = self.conversation_manager.conversation_history

        # Find last user and assistant messages
        last_user = None
        last_assistant = None

        for message in reversed(conversation_history):
            if message.get('role') == 'user' and last_user is None:
                last_user = message.get('content', '')
            elif message.get('role') == 'assistant' and last_assistant is None:
                last_assistant = message.get('content', '')

            if last_user and last_assistant:
                break

        # Only show Last Exchange section if we have both user and assistant messages
        if last_user and last_assistant:
            user_tokens = self.estimate_tokens_for_text(last_user, model)
            assistant_tokens = self.estimate_tokens_for_text(last_assistant, model)

            # Build complete Last Exchange section
            exchange_section = f"\nLast Exchange:\n"
            exchange_section += f"    User message: {user_tokens:,} tokens\n"
            exchange_section += f"    AI response: {assistant_tokens:,} tokens"

            # Show cost for last exchange if OpenAI model
            if is_openai_model:
                exchange_cost = self.calculate_cost(user_tokens, assistant_tokens, model)
                if exchange_cost:
                    exchange_section += f"\n    Exchange cost: ${exchange_cost['total_cost']:.4f}"

            print_md(exchange_section)


