import requests
from typing import Optional, Any
import yt_dlp
import clipboard
import subprocess
import shlex
import tiktoken
from settings_manager import SettingsManager
from openai import OpenAI
from command_completer import CommandCompleter
from tavily_search import create_tavily_search, TavilySearchError
from web_content_extractor import WebContentExtractor
from document_processor import DocumentProcessor
from rag_config import is_supported_file, get_supported_extensions_display
from print_helper import print_info, print_lines


class CommandManager:
    def __init__(self, conversation_manager: Any) -> None:
        self.settings_manager = SettingsManager.getInstance()
        self.conversation_manager = conversation_manager
        self.command_registry = self.settings_manager.command_registry
        self.available_commands = self.command_registry.get_available_commands()
        self.completer = CommandCompleter(self.command_registry)
        self.working_dir = self.settings_manager.setting_get("working_dir")

        # Use the RAG engine that was already created by conversation_manager
        # This ensures we don't create a duplicate RAG engine with a hardcoded OpenAI client
        self.rag_engine = self.conversation_manager.rag_engine

    def execute_system_command(self, command: str) -> Optional[str]:
        """
        Execute a system command with proper permission handling.
        Returns the command output or None if execution was denied/failed.
        """
        if not self.settings_manager.setting_get("execute_enabled"):
            print_info("Execute mode is disabled")
            return None

        # Check if permission is required
        if self.settings_manager.setting_get("execute_require_permission"):
            response = input("- Allow execution? (Y/n): ").strip().lower()
            if response not in ['', 'y', 'yes']:
                print_info("Command execution denied by user")
                return None

        try:
            # Execute the command safely
            print_info(f"Running: {command}")

            # Use shell=True for complex commands but be aware of security implications
            # In a production environment, you might want to restrict this further
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True
            )

            output = ""
            if result.stdout:
                output += f"STDOUT:\n{result.stdout}\n"
            if result.stderr:
                output += f"STDERR:\n{result.stderr}\n"

            output += f"Return code: {result.returncode}"

            if result.returncode == 0:
                print_info("Command ran successfully")
            else:
                print_info(f"Command finished with exit code: {result.returncode}")

            return output

        except Exception as e:
            error_msg = f"Could not run command: {str(e)}"
            print_info(error_msg)
            return error_msg

    def parse_commands(self, user_input: str) -> bool:
        command_processed = False
        commands = [
            "--" + command.strip()
            for command in user_input.split("--")
            if command.strip()
        ]

        for command in commands:
            parts = command.split(" ", 1)
            command_name = parts[0]
            arg = parts[1] if len(parts) > 1 else None

            # Validate command using registry
            is_valid, error_msg = self.command_registry.validate_command_input(command_name, arg)
            if not is_valid:
                print(f"{error_msg}")
                command_processed = True
                continue

            if command.startswith("--model"):
                self.model(arg)
                command_processed = True
            elif command.startswith("--refresh-models"):
                self.models_refresh()
                command_processed = True
            elif command.startswith("--instructions"):
                self.conversation_manager.apply_instructions(
                    arg, self.settings_manager.setting_get("instructions")
                )

                self.conversation_manager.log_save()
                command_processed = True
            elif command.startswith("--logmv"):
                # Check if incognito mode is enabled
                if self.settings_manager.setting_get("incognito"):
                    print_info("Cannot rename log: incognito mode is enabled (no logging active)")
                    command_processed = True
                    continue

                title = None
                if arg is None:
                    print_info("No log file name specified. AI will suggest log filename for you")
                    title = self.conversation_manager.generate_ai_suggested_title()
                else:
                    # Title will be sanitized in manual_log_rename method
                    title = arg

                # Use the proper renaming method that preserves date/timestamp
                actual_filename = self.conversation_manager.manual_log_rename(title)
                self.conversation_manager.log_save()
                print_info(f"Log renamed to: {actual_filename}")

                command_processed = True
            elif command.startswith("--logrm"):
                # Delete the current log files
                if self.conversation_manager.log_delete():
                    print_info("Log deleted")
                else:
                    print_info("No log file to delete or deletion failed")

                command_processed = True
            elif command.startswith("--log"):
                # Check if incognito mode is enabled
                if self.settings_manager.setting_get("incognito"):
                    print_info("Cannot load log: incognito mode is enabled (no logging active)")
                    command_processed = True

                if arg is None:
                    print_info("Please specify the log you want to use")
                else:
                    self.settings_manager.setting_set("log_file_name", arg)
                    self.conversation_manager.log_resume()

                command_processed = True
            elif command.startswith("--cb"):
                clipboard_content = clipboard.paste()
                if clipboard_content:
                    print_info("Clipboard content added to conversation context")
                    self.conversation_manager.conversation_history.append(
                        {"role": "user", "content": clipboard_content}
                    )
                else:
                    print_info("Clipboard is empty. Please type your input")

                command_processed = True
            elif command.startswith("--youtube"):
                if arg is None:
                    print_info("Please specify a youtube url")
                    command_processed = True
                else:
                    self.youtube(arg)
                    command_processed = True
            elif command.startswith("--url"):
                if arg is None:
                    print_info("Please specify a URL")
                    command_processed = True
                else:
                    self.url(arg)
                    command_processed = True
            elif command.startswith("--file"):
                if arg is None:
                    print_info("Please specify a file path")
                    command_processed = True
                else:
                    self.file(arg)
                    command_processed = True
            elif command == "--search":
                if self.settings_manager.setting_get("search"):
                    self.settings_manager.setting_set("search", False)
                    print_info("Web search disabled")
                else:
                    self.settings_manager.setting_set("search", True)
                    print_info("Web search enabled")
                command_processed = True
            elif command.startswith("--nothink"):
                nothink = self.settings_manager.setting_get("nothink")
                if nothink:
                    self.settings_manager.setting_set("nothink", False)
                    print_info("Nothink mode disabled")
                else:
                    self.settings_manager.setting_set("nothink", True)
                    print_info("Nothink mode enabled")

                command_processed = True
            elif command.startswith("--incognito"):
                incognito = self.settings_manager.setting_get("incognito")
                if incognito:
                    self.settings_manager.setting_set("incognito", False)
                    print_info("Incognito mode disabled - logging resumed")
                else:
                    self.settings_manager.setting_set("incognito", True)
                    print_info("Incognito mode enabled - no data will be saved to logs")

                command_processed = True
            elif command.startswith("--execute"):
                execute_enabled = self.settings_manager.setting_get("execute_enabled")
                if execute_enabled:
                    self.settings_manager.setting_set("execute_enabled", False)
                    print_info("Execute mode disabled - AI cannot run system commands")
                else:
                    self.settings_manager.setting_set("execute_enabled", True)
                    require_permission = self.settings_manager.setting_get("execute_require_permission")
                    permission_text = " (requires permission for each command)" if require_permission else " (automatic execution enabled)"
                    print_info(f"Execute mode enabled - AI can run system commands{permission_text}")

                command_processed = True
            elif command.startswith("--clear"):
                self.conversation_manager.start_new_conversation_log()
                print_info("Conversation history cleared - will create new log file after first AI response")
                print_info("AI instructions preserved")
                command_processed = True
            elif command.startswith("--usage"):
                self.display_token_usage()
                command_processed = True
            elif command.startswith("--tts-model"):
                if arg is None:
                    print_info("Please specify a TTS model. Available models: tts-1, tts-1-hd, gpt-4o-mini-tts")
                    command_processed = True
                else:
                    valid_models = ["tts-1", "tts-1-hd", "gpt-4o-mini-tts"]
                    if arg in valid_models:
                        self.settings_manager.setting_set("tts_model", arg)
                        print_info(f"TTS model set to: {arg}")
                    else:
                        print_info(f"Invalid TTS model: {arg}. Available models: {', '.join(valid_models)}")
                    command_processed = True
            elif command.startswith("--tts-voice"):
                if arg is None:
                    print_info("Please specify a TTS voice. Available voices: alloy, echo, fable, onyx, nova, shimmer")
                    command_processed = True
                else:
                    valid_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
                    if arg in valid_voices:
                        self.settings_manager.setting_set("tts_voice", arg)
                        print_info(f"TTS voice set to: {arg}")
                    else:
                        print_info(f"Invalid TTS voice: {arg}. Available voices: {', '.join(valid_voices)}")
                    command_processed = True
            elif command.startswith("--tts-save-as-mp3"):
                tts_save_mp3 = self.settings_manager.setting_get("tts_save_mp3")
                if tts_save_mp3:
                    self.settings_manager.setting_set("tts_save_mp3", False)
                    print_info("TTS save as MP3 disabled")
                else:
                    self.settings_manager.setting_set("tts_save_mp3", True)
                    print_info("TTS save as MP3 enabled")

                command_processed = True
            elif command.startswith("--tts"):
                tts = self.settings_manager.setting_get("tts")
                if tts:
                    self.settings_manager.setting_set("tts", False)
                    print_info("TTS disabled")
                else:
                    # Check for privacy: don't enable TTS when using Ollama models
                    current_model = self.settings_manager.setting_get("model")
                    if current_model and self.conversation_manager.llm_client_manager.get_provider_for_model(current_model) == "ollama":
                        print_info("TTS not available when using Ollama models")
                        print_info("TTS would send your text to OpenAI, breaking local privacy")
                    else:
                        self.settings_manager.setting_set("tts", True)
                        print_info("TTS enabled")

                command_processed = True
            elif command.startswith("--rag-status"):
                self.rag_status()
                command_processed = True
            elif command.startswith("--rag-debug"):
                if arg is None:
                    print_info("Please specify a query to test")
                else:
                    self.rag_debug(arg)
                command_processed = True
            elif command.startswith("--rag-rebuild"):
                if arg is None:
                    print_info("Please specify a collection name to rebuild")
                else:
                    self.rag_rebuild(arg)
                command_processed = True
            elif command.startswith("--rag-show"):
                if arg is None:
                    print_info("Please specify a filename to show")
                else:
                    self.rag_show(arg)
                command_processed = True
            elif command.startswith("--rag-test"):
                self.rag_test_connection()
                command_processed = True
            elif command.startswith("--rag-info"):
                self.rag_model_info()
                command_processed = True
            elif command.startswith("--rag"):
                if arg is None:
                    # Toggle RAG on/off
                    if self.rag_engine and self.rag_engine.is_active():
                        self.rag_off()
                    else:
                        self.rag_list(from_toggle=True)
                else:
                    # Activate specific collection
                    self.rag_activate(arg)
                command_processed = True

        # Ensure proper spacing before next user prompt
        if command_processed:
            print()

        return command_processed

    def rag_list(self, from_toggle: bool = False) -> None:
        """List available RAG collections"""
        if not self.rag_engine:
            print_info("RAG engine not available")
            return

        if from_toggle:
            print_info("RAG is not active. Available collections:")
        else:
            print_info("Available RAG collections:")

        collections = self.rag_engine.list_collections()
        if not collections:
            print_info("No RAG collections found in rag/ directory")
            print_info("Create collections by making directories in rag/collection_name/")
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
            print_info(f"{i}. {collection['name']} - {collection['file_count']} files {status}")

        print_info("Use --rag <collection_name>  # Activate collection (builds automatically if needed)")

    def rag_activate(self, collection_name: str) -> None:
        """Activate a RAG collection"""
        if not self.rag_engine:
            print_info("RAG engine not available")
            return

        success = self.rag_engine.activate_collection(collection_name)
        if not success:
            # Error messages are printed by the RAG engine
            available = self.rag_engine.vector_store.get_available_collections()
            if available:
                print_info(f"Available collections: {', '.join(available)}")

    def rag_off(self) -> None:
        """Deactivate RAG mode"""
        if not self.rag_engine:
            print_info("RAG engine not available")
            return

        self.rag_engine.deactivate_collection()

    def rag_rebuild(self, collection_name: str) -> None:
        """Force rebuild a RAG collection"""
        if not self.rag_engine:
            print_info("RAG engine not available")
            return

        self.rag_engine.build_collection(collection_name, force_rebuild=True)

    def rag_show(self, filename: str) -> None:
        """Show relevant chunks in a file"""
        if not self.rag_engine:
            print_info("RAG engine not available")
            return

        result = self.rag_engine.show_chunk_in_file(filename)
        print_info(result)

    def rag_status(self) -> None:
        """Show RAG status"""
        if not self.rag_engine:
            print_info("RAG engine not available")
            return

        status = self.rag_engine.get_status()
        print_info("RAG Status:")
        print_info(f"Active: {status['active']}")
        if status['active']:
            print_info(f"Collection: {status['active_collection']}")
            print_info(f"Chunks loaded: {status['chunk_count']}")

        print_info(f"Available collections: {status['available_collections']}")

        # Show embedding provider information
        try:
            provider = self.settings_manager.setting_get("embedding_provider")
            print_info(f"Embedding provider: {provider}")

            if provider == "openai":
                model = self.settings_manager.setting_get("openai_embedding_model")
                print_info(f"OpenAI model: {model}")
            elif provider == "ollama":
                model = self.settings_manager.setting_get("ollama_embedding_model")
                ollama_url = self.settings_manager.setting_get("ollama_base_url")
                print_info(f"Ollama model: {model}")
                print_info(f"Ollama URL: {ollama_url}")
        except Exception as e:
            print_info(f"Provider info: Error getting provider details")

        print_info(f"Settings:")
        for key, value in status['settings'].items():
            print_info(f"{key}: {value}")

    def rag_debug(self, query: str) -> None:
        """Debug RAG querying to see what content would be retrieved"""
        if not self.rag_engine:
            print_info("RAG engine not available")
            return

        if not self.rag_engine.is_active():
            print_info("No RAG collection is active")
            print_info("Use --rag <collection> to activate a collection first")
            return

        print_info(f"Testing RAG query: '{query}'")
        print_lines()

        try:
            # Test the full RAG context generation
            rag_context, rag_sources = self.rag_engine.get_context_for_query(query)

            if rag_context:
                print_info(f"Found {len(rag_sources)} relevant chunks")
                print_info("Context that would be sent to AI:")
                print_info("" + "-"*30)
                print_info(f"{rag_context}")
                print_info("" + "-"*30)

                if rag_sources:
                    print_info("Source details:")
                    for i, source in enumerate(rag_sources, 1):
                        score = source.get('similarity_score', 0)
                        print_info(f"{i}. {source['filename']} (score: {score:.3f})")
                        print_info(f"Content preview: {source['content'][:100]}...")
            else:
                print_info("No relevant content found for this query")
                print_info("This means the query didn't match any document content")

        except Exception as e:
            print_info(f"Error during RAG query: {e}")

    def rag_test_connection(self) -> None:
        """Test connection to current embedding provider"""
        if not self.rag_engine:
            print_info("RAG engine not available")
            return

        try:
            provider = self.settings_manager.setting_get("embedding_provider")
            print_info(f"Testing connection to {provider} embedding service...")

            # Test using the embedding service
            success = self.rag_engine.embedding_service.test_connection()

            if success:
                print_info(f"Connection to {provider} successful!")

                # Show additional info
                model_info = self.rag_engine.embedding_service.get_embedding_model_info()
                model = model_info.get("model", "unknown")
                dimensions = self.rag_engine.embedding_service.get_embedding_dimensions()
                print_info(f"Model: {model}")
                print_info(f"Dimensions: {dimensions}")

                if provider == "ollama":
                    info = model_info.get("info", {})
                    if info.get("multilingual"):
                        print_info(f"Languages: {info.get('languages', 'Multiple languages supported')}")

            else:
                print_info(f"Connection to {provider} failed!")

                if provider == "ollama":
                    ollama_url = self.settings_manager.setting_get("ollama_base_url")
                    print_info(f"Check that Ollama is running at: {ollama_url}")
                    model = self.settings_manager.setting_get("ollama_embedding_model")
                    print_info(f"Check that model '{model}' is available in Ollama")
                elif provider == "openai":
                    print_info("Check your OpenAI API key and internet connection")

        except Exception as e:
            print_info(f"Error testing connection: {e}")

    def rag_model_info(self) -> None:
        """Show detailed information about current embedding model"""
        if not self.rag_engine:
            print_info("RAG engine not available")
            return

        try:
            model_info = self.rag_engine.embedding_service.get_embedding_model_info()
            provider = model_info.get("provider", "unknown")
            model = model_info.get("model", "unknown")
            info = model_info.get("info", {})

            print_info("Embedding Model Information:")
            print_info(f"Provider: {provider}")
            print_info(f"Model: {model}")
            print_info(f"Dimensions: {info.get('dimensions', 'unknown')}")
            print_info(f"Max tokens: {info.get('max_tokens', 'unknown')}")

            if provider == "openai":
                cost = info.get('cost_per_1k_tokens', 0)
                print_info(f"Cost per 1K tokens: ${cost}")
            elif provider == "ollama":
                print_info(f"Cost per 1K tokens: Free (local)")
                if info.get('multilingual'):
                    print_info(f"Multilingual: Yes")
                    languages = info.get('languages')
                    if languages:
                        print_info(f"Languages: {languages}")

            # Show current settings
            print_info("Current RAG Settings:")
            print_info(f"Chunk size: {self.settings_manager.setting_get('rag_chunk_size')} tokens")
            print_info(f"Chunk overlap: {self.settings_manager.setting_get('rag_chunk_overlap')} tokens")
            print_info(f"Top K results: {self.settings_manager.setting_get('rag_top_k')}")

        except Exception as e:
            print_info(f"Error getting model info: {e}")

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

    def models_refresh(self) -> None:
        """Clear model cache and force fresh fetch from APIs"""
        print_info("Refreshing model cache...")
        self.completer.clear_models_cache()
        print_info("Model cache cleared. Fresh models will be fetched on next use")

    def model(self, arg: Optional[str]) -> None:
        if arg == None:
            ollama_url = self.settings_manager.setting_get("ollama_base_url")
            print_info("Please specify the model to use")
            print_info("Available sources: OpenAI (https://platform.openai.com/docs/models)")
            print_info(f"Ollama ({ollama_url} if running)")
            print_info("Google (https://ai.google.dev/gemini-api/docs/models)")
            print_info("Anthropic (https://docs.anthropic.com/en/docs/about-claude/models)")
            return

        model = arg

        # Validate the model before setting it
        if not self._validate_model(model):
            print_info(f"Invalid model: {model}")
            print_info("Model not found in OpenAI, Google, or Ollama APIs")

            # Show available models
            try:
                available_models = self.completer._get_available_models()
                if available_models:
                    print_info("Available models:")

                    # Group by source
                    openai_models = [m["name"] for m in available_models if m["source"] == "OpenAI"]
                    ollama_models = [m["name"] for m in available_models if m["source"] == "Ollama"]
                    google_models = [m["name"] for m in available_models if m["source"] == "Google"]
                    anthropic_models = [m["name"] for m in available_models if m["source"] == "Anthropic"]

                    if openai_models:
                        print_info(f"OpenAI: {', '.join(openai_models[:5])}" +
                              (f" (and {len(openai_models)-5} more)" if len(openai_models) > 5 else ""))

                    if google_models:
                        print_info(f"Google: {', '.join(google_models)}")

                    if anthropic_models:
                        print_info(f"Anthropic: {', '.join(anthropic_models)}")

                    if ollama_models:
                        print_info(f"Ollama: {', '.join(ollama_models)}")

                    if not openai_models and not ollama_models and not google_models and not anthropic_models:
                        print_info("(No models available - check API keys and network connection)")
                else:
                    print_info("(Unable to fetch available models)")
            except Exception:
                print_info("(Unable to fetch available models)")

            return

        # Set the model (this will trigger client update in conversation manager)
        self.conversation_manager.model = model
        self.settings_manager.setting_set("model", model)

        # Determine model source
        provider = self.conversation_manager.llm_client_manager.get_provider_for_model(model)

        # Check if TTS needs to be disabled for privacy when switching to Ollama
        if provider == "ollama":
            if self.settings_manager.setting_get("tts"):
                self.settings_manager.setting_set("tts", False)
                print_info("TTS automatically disabled for privacy (Ollama models)")

        # Show appropriate message based on provider

        if provider == "ollama":
            ollama_url = self.settings_manager.setting_get("ollama_base_url")
            if self.conversation_manager.llm_client_manager._is_ollama_available():
                print_info(f"Switched to Ollama model: {model}")
                print_info(f"Running locally via Ollama at {ollama_url}")
            else:
                print_info(f"Warning: Selected Ollama model '{model}' but Ollama not available")
                print_info(f"Make sure Ollama is running at {ollama_url}")
        elif provider == "google":
            print_info(f"Switched to Google Gemini model: {model}")
            print_info(f"https://ai.google.dev/gemini-api/docs/models")
        elif provider == "anthropic":
            print_info(f"Switched to Anthropic model: {model}")
            print_info(f"https://docs.anthropic.com/en/docs/about-claude/models")
        else:
            print_info(f"Switched to OpenAI model: {model}")
            print_info(f"https://platform.openai.com/docs/models")

    def youtube(self, arg: str) -> None:
        """
        Extract transcript from YouTube video and send to conversation manager as input.
        """
        print_info("Extracting info from YouTube...")

        video_url = None

        # Check if 'watch?v=' is in the URL, otherwise resolve URL
        if "watch?v=" in arg:
            video_url = arg
        else:
            print_info("Video ID missing from URL")
            try:
                print_info("Attempting to determine ID")
                response = requests.get(arg)
                video_url = response.url
            except requests.RequestException as e:
                print_info(f"Error resolving URL: {e}")
                return

        if video_url:
            try:
                video_id = video_url.split("watch?v=")[1]
                print_info(f"Video ID found: {video_id}")
            except IndexError:
                print_info("Invalid YouTube URL provided")
                return

            try:
                # Get video info using yt-dlp
                print_info("Fetching video info...")
                ydl_opts = {'quiet': True, 'no_warnings': True}
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(video_url, download=False)
                    if info:
                        video_title = "[" + info.get('title', 'Unknown Title') + "](" + video_url + ")"
                        channel_title = "[" + info.get('uploader', 'Unknown Channel') + "](" + info.get('uploader_url', video_url) + ")"
                    else:
                        print_info("Could not extract video information")
                        return

                # Extract subtitles using yt-dlp
                print_info("Fetching video transcript using yt-dlp...")
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

                        # Try to find English subtitles in order of preference
                        subtitle_data = None
                        subtitle_source = None

                        # First try manual English subtitles
                        for lang in ['en', 'en-US', 'en-GB', 'en-CA', 'en-AU']:
                            if lang in subtitles and subtitles[lang]:
                                subtitle_data = subtitles[lang]
                                subtitle_source = f"manual {lang}"
                                print_info(f"Found manual {lang} subtitles")
                                break

                        # If no manual subtitles, try automatic captions
                        if not subtitle_data:
                            for lang in ['en', 'en-US', 'en-GB', 'en-CA', 'en-AU']:
                                if lang in automatic_captions and automatic_captions[lang]:
                                    subtitle_data = automatic_captions[lang]
                                    subtitle_source = f"auto-generated {lang}"
                                    print_info(f"Found auto-generated {lang} captions")
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
                                    response = requests.get(subtitle_url, timeout=10)
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
                                            print_info(f"Successfully extracted transcript from {subtitle_source}")
                                            print_info(f"Processing {len(transcript_text)} characters as input...")

                                            user_input = (
                                                "Channel title: " + channel_title +
                                                "\nVideo title: " + video_title +
                                                "\nVideo transcript: " + transcript_text
                                            )
                                            self.conversation_manager.conversation_history.append(
                                                {"role": "user", "content": user_input}
                                            )
                                            print_info("YouTube content added to conversation context")
                                        else:
                                            raise Exception("Transcript text was empty after parsing")
                                    else:
                                        raise Exception(f"Failed to download subtitles: HTTP {response.status_code}")

                                except Exception as subtitle_error:
                                    print_info(f"Error processing subtitle file: {subtitle_error}")
                                    raise subtitle_error
                            else:
                                raise Exception("No subtitle URL found")
                        else:
                            # List available subtitle languages for debugging
                            available_langs = list(subtitles.keys()) + list(automatic_captions.keys())
                            if available_langs:
                                print_info(f"Available subtitle languages: {', '.join(set(available_langs))}")
                                print_info("No English subtitles found")
                            else:
                                print_info("No subtitles available for this video")
                            raise Exception("No English subtitles available")

                except Exception as e:
                    print_info(f"Could not extract transcript: {str(e)}")
                    print_info("Continuing with video info only...")
                    # Still provide video info even without transcript
                    user_input = (
                        "Channel title: " + channel_title +
                        "\nVideo title: " + video_title +
                        "\nNote: No transcript could be extracted for this video."
                    )
                    self.conversation_manager.conversation_history.append(
                        {"role": "user", "content": user_input}
                    )
                    print_info("YouTube content added to conversation context")

            except Exception as e:
                print_info("An error occurred:", e)
                return
        else:
            print_info("Could not determine the YouTube video URL")
            return

    def url(self, url: str) -> None:
        """
        Extract content from a website URL and send to conversation manager as input.
        """
        # Check if this is a YouTube URL and redirect to YouTube command
        if "youtube.com" in url.lower() or "youtu.be" in url.lower():
            print_info("YouTube URL detected - redirecting to --youtube command for better transcript extraction...")
            self.youtube(url)
            return

        print_info("Extracting content from URL...")

        extractor = WebContentExtractor()
        result = extractor.extract_content(url)

        if result['error']:
            print_info(f"Error: {result['error']}")
            return

        if not result['content']:
            print_info("No content could be extracted from the URL")
            return

        # Display warning if paywall was encountered
        if result.get('warning'):
            print_info(f"{result['warning']}")

        # Format the content for the conversation
        title = result['title'] or "Web Content"
        formatted_content = f"Website: {title}\n\nSource: {url}\n\n{result['content']}"

        # Add to conversation history
        self.conversation_manager.conversation_history.append(
            {"role": "user", "content": formatted_content}
        )

        print_info("Content added to conversation context")
        print_info("You can now ask questions about this content")

    def file(self, file_path: str) -> None:
        """
        Load file contents and add to conversation context using RAG document processing logic.
        """
        import os

        # Check if file exists
        if not os.path.exists(file_path):
            print_info(f"Error: File not found: {file_path}")
            return

        # Check if file type is supported
        if not is_supported_file(file_path):
            supported_types = get_supported_extensions_display()
            print_info(f"Error: Unsupported file type. Supported types: {supported_types}")
            return

        print_info(f"Loading file: {os.path.basename(file_path)}")

        # Use DocumentProcessor to load the file content
        processor = DocumentProcessor()
        try:
            content = processor.load_file(file_path)

            if not content or not content.strip():
                print_info("Error: No content could be extracted from the file")
                return

            # Clean the content
            content = processor.clean_text(content)

            # Get file info for display
            filename = os.path.basename(file_path)
            file_ext = os.path.splitext(filename)[1].lower()
            word_count = len(content.split())

            print_info(f"Extracted content from {filename} ({word_count} words)")

            # Format the content for the conversation
            formatted_content = f"File: {filename}\n\nPath: {file_path}\n\n{content}"

            # Add to conversation history
            self.conversation_manager.conversation_history.append(
                {"role": "user", "content": formatted_content}
            )

            print_info("File content added to conversation context")
            print_info("You can now ask questions about this content")

        except Exception as e:
            print_info(f"Error loading file: {e}")

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

    def get_openai_pricing(self, model: str) -> dict:
        """
        Get OpenAI pricing data for a model.
        Returns costs per 1M tokens for input and output.
        """
        # OpenAI pricing (per 1M tokens) as of January 2025
        pricing_data = {
            # GPT-4o models
            'gpt-4o': {'input': 5.00, 'output': 15.00},
            'gpt-4o-2024-11-20': {'input': 2.50, 'output': 10.00},
            'gpt-4o-2024-08-06': {'input': 2.50, 'output': 10.00},
            'gpt-4o-2024-05-13': {'input': 5.00, 'output': 15.00},

            # GPT-4o-mini models
            'gpt-4o-mini': {'input': 0.15, 'output': 0.60},
            'gpt-4o-mini-2024-07-18': {'input': 0.15, 'output': 0.60},

            # GPT-4 models
            'gpt-4': {'input': 30.00, 'output': 60.00},
            'gpt-4-turbo': {'input': 10.00, 'output': 30.00},
            'gpt-4-turbo-2024-04-09': {'input': 10.00, 'output': 30.00},
            'gpt-4-0125-preview': {'input': 10.00, 'output': 30.00},
            'gpt-4-1106-preview': {'input': 10.00, 'output': 30.00},

            # GPT-3.5 models
            'gpt-3.5-turbo': {'input': 3.00, 'output': 6.00},
            'gpt-3.5-turbo-0125': {'input': 0.50, 'output': 1.50},

            # New GPT-4.1 models
            'gpt-4.1': {'input': 2.00, 'output': 8.00},
            'gpt-4.1-mini': {'input': 0.40, 'output': 1.60},
            'gpt-4.1-nano': {'input': 0.10, 'output': 0.40},

            # GPT-4.1 model variations
            'gpt-4.1-2024-12-05': {'input': 2.00, 'output': 8.00},
            'gpt-4.1-preview': {'input': 2.00, 'output': 8.00},
            'gpt-4.1-mini-2024-12-05': {'input': 0.40, 'output': 1.60},
            'gpt-4.1-nano-2024-12-05': {'input': 0.10, 'output': 0.40},

            # Reasoning models
            'o3': {'input': 2.00, 'output': 8.00},
            'o3-mini': {'input': 1.10, 'output': 4.40},
            'o1': {'input': 15.00, 'output': 60.00},
            'o1-mini': {'input': 3.00, 'output': 12.00},
            'o1-preview': {'input': 15.00, 'output': 60.00},
        }

        # Try exact match first
        if model in pricing_data:
            return pricing_data[model]

        # Try partial matching for model families
        for known_model, prices in pricing_data.items():
            if model.startswith(known_model):
                return prices

        # Return None if no pricing found
        return None

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
        Estimate total tokens for current conversation history.
        Returns breakdown of tokens by message type.
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
        """Display comprehensive token usage information"""
        model = self.conversation_manager.model

        # Check if this is an OpenAI model for accuracy warning
        is_openai_model = model.startswith(('gpt-', 'o1', 'o3', 'text-embedding-', 'tts-', 'whisper-', 'dall-e'))

        if not is_openai_model:
            print_info("Note: Token counts are rough estimates for non-OpenAI models")
            print("")

        # Get token breakdown
        breakdown = self.estimate_conversation_tokens()

        print_info("Token Usage Summary:")
        print_info("─────────────────────")
        print_info(f"Current model: {model}")
        print_info(f"System messages (e.g. instructions, search, RAG): {breakdown['system']:,} tokens")
        print_info(f"User messages: {breakdown['user']:,} tokens")
        print_info(f"Assistant responses: {breakdown['assistant']:,} tokens")
        print_info(f"Total conversation: {breakdown['total']:,} tokens")

        # Calculate and display costs for OpenAI models
        if is_openai_model:
            input_tokens = breakdown['system'] + breakdown['user']
            output_tokens = breakdown['assistant']

            cost_info = self.calculate_cost(input_tokens, output_tokens, model)

            if cost_info:
                print_info("")
                print_info("Estimated Cost (OpenAI):")
                print_info("─────────────────────")
                print_info(f"Input tokens ({input_tokens:,}): ${cost_info['input_cost']:.4f}")
                print_info(f"Output tokens ({output_tokens:,}): ${cost_info['output_cost']:.4f}")
                print_info(f"Total conversation cost: ${cost_info['total_cost']:.4f}")
                print_info(f"Rate: ${cost_info['input_rate']:.2f}/${cost_info['output_rate']:.2f} per 1M tokens (input/output)")

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
            print_info("")
            print_info("Last Exchange:")
            print_info("─────────────")

            user_tokens = self.estimate_tokens_for_text(last_user, model)
            print_info(f"Last user message: {user_tokens:,} tokens")

            assistant_tokens = self.estimate_tokens_for_text(last_assistant, model)
            print_info(f"Last AI response: {assistant_tokens:,} tokens")

            # Show cost for last exchange if OpenAI model
            if is_openai_model:
                exchange_cost = self.calculate_cost(user_tokens, assistant_tokens, model)
                if exchange_cost:
                    print_info(f"Last exchange cost: ${exchange_cost['total_cost']:.4f}")


