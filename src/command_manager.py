import requests
from typing import Optional, Any
import yt_dlp
# Removed youtube_transcript_api - using yt-dlp for more reliable transcript extraction
import clipboard
import subprocess
import shlex
from settings_manager import SettingsManager
from openai import OpenAI

from command_completer import CommandCompleter
from tavily_search import create_tavily_search, TavilySearchError
from web_content_extractor import WebContentExtractor
from document_processor import DocumentProcessor
from rag_config import is_supported_file, get_supported_extensions_display


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

    def print_command_result(self, message: str) -> None:
        """Helper method to print command results with consistent spacing"""
        print(message)  # The message

    def execute_system_command(self, command: str) -> Optional[str]:
        """
        Execute a system command with proper permission handling.
        Returns the command output or None if execution was denied/failed.
        """
        if not self.settings_manager.setting_get("execute_enabled"):
            self.print_command_result(" - Execute mode is disabled. Use --execute to enable.")
            return None

        # Check if permission is required
        if self.settings_manager.setting_get("execute_require_permission"):
            response = input(" - Allow execution? (Y/n): ").strip().lower()
            if response not in ['', 'y', 'yes']:
                self.print_command_result(" - Command execution denied by user")
                return None

        try:
            # Execute the command safely
            self.print_command_result(f"- Running: {command}")

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
                self.print_command_result(" - Command ran successfully")
            else:
                self.print_command_result(f"- Command finished with exit code: {result.returncode}")

            return output

        except Exception as e:
            error_msg = f"- Could not run command: {str(e)}"
            self.print_command_result(error_msg)
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
                print(f"- {error_msg}")
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
                    self.print_command_result(" - Cannot rename log: incognito mode is enabled (no logging active)")
                    command_processed = True
                    continue

                title = None
                if arg is None:
                    print("\n - No log file name specified. AI will suggest log filename for you.")
                    title = self.conversation_manager.generate_ai_suggested_title()
                else:
                    # Title will be sanitized in manual_log_rename method
                    title = arg

                # Use the proper renaming method that preserves date/timestamp
                actual_filename = self.conversation_manager.manual_log_rename(title)
                self.conversation_manager.log_save()
                self.print_command_result(f"\n - Log renamed to: {actual_filename}")

                command_processed = True
            elif command.startswith("--log"):
                # Check if incognito mode is enabled
                if self.settings_manager.setting_get("incognito"):
                    self.print_command_result(" - Cannot load log: incognito mode is enabled (no logging active)")
                    command_processed = True
                    continue

                if arg is None:
                    print(" - (!) Please specify the log you want to use.")
                else:
                    self.settings_manager.setting_set("log_file_name", arg)
                    self.conversation_manager.log_resume()

                command_processed = True
            elif command.startswith("--cb"):
                clipboard_content = clipboard.paste()
                if clipboard_content:
                    print(" - Clipboard content added to conversation context.")
                    self.conversation_manager.conversation_history.append(
                        {"role": "user", "content": clipboard_content}
                    )
                else:
                    print(" - Clipboard is empty. Please type your input.")

                command_processed = True
            elif command.startswith("--youtube"):
                if arg is None:
                    print(" - (!) please specify a youtube url.")
                else:
                    self.youtube(arg)

                command_processed = True
            elif command.startswith("--url"):
                if arg is None:
                    print(" - (!) please specify a URL.")
                else:
                    self.url(arg)

                command_processed = True
            elif command.startswith("--file"):
                if arg is None:
                    print(" - (!) please specify a file path.")
                else:
                    self.file(arg)

                command_processed = True
            elif command == "--search":
                if self.settings_manager.setting_get("search"):
                    self.settings_manager.setting_set("search", False)
                    self.print_command_result(" - Web search disabled")
                else:
                    self.settings_manager.setting_set("search", True)
                    self.print_command_result(" - Web search enabled")
                command_processed = True
            elif command.startswith("--nothink"):
                nothink = self.settings_manager.setting_get("nothink")
                if nothink:
                    self.settings_manager.setting_set("nothink", False)
                    self.print_command_result(" - Nothink mode disabled")
                else:
                    self.settings_manager.setting_set("nothink", True)
                    self.print_command_result(" - Nothink mode enabled")

                command_processed = True
            elif command.startswith("--incognito"):
                incognito = self.settings_manager.setting_get("incognito")
                if incognito:
                    self.settings_manager.setting_set("incognito", False)
                    self.print_command_result(" - Incognito mode disabled - logging resumed")
                else:
                    self.settings_manager.setting_set("incognito", True)
                    self.print_command_result(" - Incognito mode enabled - no data will be saved to logs")

                command_processed = True
            elif command.startswith("--execute"):
                execute_enabled = self.settings_manager.setting_get("execute_enabled")
                if execute_enabled:
                    self.settings_manager.setting_set("execute_enabled", False)
                    self.print_command_result(" - Execute mode disabled - AI cannot run system commands")
                else:
                    self.settings_manager.setting_set("execute_enabled", True)
                    require_permission = self.settings_manager.setting_get("execute_require_permission")
                    permission_text = " (requires permission for each command)" if require_permission else " (automatic execution enabled)"
                    self.print_command_result(f"- Execute mode enabled - AI can run system commands{permission_text}")

                command_processed = True
            elif command.startswith("--clear"):
                self.conversation_manager.conversation_history.clear()
                self.print_command_result(" - Conversation history cleared")
                command_processed = True
            elif command.startswith("--tts-model"):
                if arg is None:
                    self.print_command_result(" - (!) Please specify a TTS model.\n - Available models: tts-1, tts-1-hd, gpt-4o-mini-tts")
                else:
                    valid_models = ["tts-1", "tts-1-hd", "gpt-4o-mini-tts"]
                    if arg in valid_models:
                        self.settings_manager.setting_set("tts_model", arg)
                        self.print_command_result(f"- TTS model set to: {arg}")
                    else:
                        self.print_command_result(f"- (!) Invalid TTS model: {arg}\n - Available models: {', '.join(valid_models)}")

                command_processed = True
            elif command.startswith("--tts-voice"):
                if arg is None:
                    self.print_command_result(" - (!) Please specify a TTS voice.\n - Available voices: alloy, echo, fable, onyx, nova, shimmer")
                else:
                    valid_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
                    if arg in valid_voices:
                        self.settings_manager.setting_set("tts_voice", arg)
                        self.print_command_result(f"- TTS voice set to: {arg}")
                    else:
                        self.print_command_result(f"- (!) Invalid TTS voice: {arg}\n - Available voices: {', '.join(valid_voices)}")

                command_processed = True
            elif command.startswith("--tts-save-as-mp3"):
                tts_save_mp3 = self.settings_manager.setting_get("tts_save_mp3")
                if tts_save_mp3:
                    self.settings_manager.setting_set("tts_save_mp3", False)
                    self.print_command_result(" - TTS save as MP3 disabled")
                else:
                    self.settings_manager.setting_set("tts_save_mp3", True)
                    self.print_command_result(" - TTS save as MP3 enabled")

                command_processed = True
            elif command.startswith("--tts"):
                tts = self.settings_manager.setting_get("tts")
                if tts:
                    self.settings_manager.setting_set("tts", False)
                    self.print_command_result(" - TTS disabled")
                else:
                    self.settings_manager.setting_set("tts", True)
                    self.print_command_result(" - TTS enabled")

                command_processed = True
            elif command.startswith("--rag-status"):
                self.rag_status()
                command_processed = True
            elif command.startswith("--rag-debug"):
                if arg is None:
                    print(" - (!) Please specify a query to test.")
                else:
                    self.rag_debug(arg)
                command_processed = True
            elif command.startswith("--rag-build"):
                if arg is None:
                    print(" - (!) Please specify a collection name to build.")
                else:
                    self.rag_build(arg)
                command_processed = True
            elif command.startswith("--rag-refresh"):
                if arg is None:
                    print(" - (!) Please specify a collection name to refresh.")
                else:
                    self.rag_refresh(arg)
                command_processed = True
            elif command.startswith("--rag-show"):
                if arg is None:
                    print(" - (!) Please specify a filename to show.")
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
                    self.rag_list()
                elif arg.lower() == "off":
                    self.rag_off()
                else:
                    self.rag_activate(arg)
                command_processed = True

        return command_processed

    def rag_list(self) -> None:
        """List available RAG collections"""
        if not self.rag_engine:
            print(" - RAG engine not available")
            return

        collections = self.rag_engine.list_collections()
        if not collections:
            print(" - No RAG collections found in rag/ directory")
            print(" - Create collections by making directories in rag/collection_name/")
            return

        print(" - Available RAG collections:")
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
            print(f"   {i}. {collection['name']} - {collection['file_count']} files {status}")

        print("\n - Usage:")
        print("   --rag <collection_name>  # Activate collection")
        print("   --rag-build <name>       # Build/rebuild collection index")

    def rag_activate(self, collection_name: str) -> None:
        """Activate a RAG collection"""
        if not self.rag_engine:
            print(" - RAG engine not available")
            return

        success = self.rag_engine.activate_collection(collection_name)
        if not success:
            # Error messages are printed by the RAG engine
            available = self.rag_engine.vector_store.get_available_collections()
            if available:
                print(f"- Available collections: {', '.join(available)}")

    def rag_off(self) -> None:
        """Deactivate RAG mode"""
        if not self.rag_engine:
            print(" - RAG engine not available")
            return

        self.rag_engine.deactivate_collection()

    def rag_build(self, collection_name: str) -> None:
        """Build/rebuild a RAG collection"""
        if not self.rag_engine:
            print(" - RAG engine not available")
            return

        self.rag_engine.build_collection(collection_name, force_rebuild=True)

    def rag_refresh(self, collection_name: str) -> None:
        """Refresh/rebuild a RAG collection"""
        if not self.rag_engine:
            print(" - RAG engine not available")
            return

        self.rag_engine.refresh_collection(collection_name)

    def rag_show(self, filename: str) -> None:
        """Show relevant chunks in a file"""
        if not self.rag_engine:
            print(" - RAG engine not available")
            return

        result = self.rag_engine.show_chunk_in_file(filename)
        print(result)

    def rag_status(self) -> None:
        """Show RAG status"""
        if not self.rag_engine:
            print(" - RAG engine not available")
            return

        status = self.rag_engine.get_status()
        print(" - RAG Status:")
        print(f"   Active: {status['active']}")
        if status['active']:
            print(f"   Collection: {status['active_collection']}")
            print(f"   Chunks loaded: {status['chunk_count']}")

        print(f"   Available collections: {status['available_collections']}")

        # Show embedding provider information
        try:
            provider = self.settings_manager.setting_get("embedding_provider")
            print(f"   Embedding provider: {provider}")

            if provider == "openai":
                model = self.settings_manager.setting_get("openai_embedding_model")
                print(f"   OpenAI model: {model}")
            elif provider == "ollama":
                model = self.settings_manager.setting_get("ollama_embedding_model")
                ollama_url = self.settings_manager.setting_get("ollama_base_url")
                print(f"   Ollama model: {model}")
                print(f"   Ollama URL: {ollama_url}")
        except Exception as e:
            print(f"   Provider info: Error getting provider details")

        print(f"   Settings:")
        for key, value in status['settings'].items():
            print(f"     {key}: {value}")

    def rag_debug(self, query: str) -> None:
        """Debug RAG querying to see what content would be retrieved"""
        if not self.rag_engine:
            print(" - RAG engine not available")
            return

        if not self.rag_engine.is_active():
            print(" - No RAG collection is active")
            print(" - Use --rag <collection> to activate a collection first")
            return

        print(f"- Testing RAG query: '{query}'")
        print(" - " + "="*50)

        try:
            # Test the full RAG context generation
            rag_context, rag_sources = self.rag_engine.get_context_for_query(query)

            if rag_context:
                print(f"- Found {len(rag_sources)} relevant chunks")
                print(" - Context that would be sent to AI:")
                print(" - " + "-"*30)
                print(f"- {rag_context}")
                print(" - " + "-"*30)

                if rag_sources:
                    print(" - Source details:")
                    for i, source in enumerate(rag_sources, 1):
                        score = source.get('similarity_score', 0)
                        print(f"- {i}. {source['filename']} (score: {score:.3f})")
                        print(f"- Content preview: {source['content'][:100]}...")
            else:
                print("- No relevant content found for this query")
                print(" - This means the query didn't match any document content")

        except Exception as e:
            print(f"- Error during RAG query: {e}")



    def rag_test_connection(self) -> None:
        """Test connection to current embedding provider"""
        if not self.rag_engine:
            print(" - RAG engine not available")
            return

        try:
            provider = self.settings_manager.setting_get("embedding_provider")
            print(f"- Testing connection to {provider} embedding service...")

            # Test using the embedding service
            success = self.rag_engine.embedding_service.test_connection()

            if success:
                print(f"- Connection to {provider} successful!")

                # Show additional info
                model_info = self.rag_engine.embedding_service.get_embedding_model_info()
                model = model_info.get("model", "unknown")
                dimensions = self.rag_engine.embedding_service.get_embedding_dimensions()
                print(f"- Model: {model}")
                print(f"- Dimensions: {dimensions}")

                if provider == "ollama":
                    info = model_info.get("info", {})
                    if info.get("multilingual"):
                        print(f"- Languages: {info.get('languages', 'Multiple languages supported')}")

            else:
                print(f"- Connection to {provider} failed!")

                if provider == "ollama":
                    ollama_url = self.settings_manager.setting_get("ollama_base_url")
                    print(f"- Check that Ollama is running at: {ollama_url}")
                    model = self.settings_manager.setting_get("ollama_embedding_model")
                    print(f"- Check that model '{model}' is available in Ollama")
                elif provider == "openai":
                    print(" - Check your OpenAI API key and internet connection")

        except Exception as e:
            print(f"- Error testing connection: {e}")

    def rag_model_info(self) -> None:
        """Show detailed information about current embedding model"""
        if not self.rag_engine:
            print(" - RAG engine not available")
            return

        try:
            model_info = self.rag_engine.embedding_service.get_embedding_model_info()
            provider = model_info.get("provider", "unknown")
            model = model_info.get("model", "unknown")
            info = model_info.get("info", {})

            print("- Embedding Model Information:")
            print(f"- Provider: {provider}")
            print(f"- Model: {model}")
            print(f"- Dimensions: {info.get('dimensions', 'unknown')}")
            print(f"- Max tokens: {info.get('max_tokens', 'unknown')}")

            if provider == "openai":
                cost = info.get('cost_per_1k_tokens', 0)
                print(f"- Cost per 1K tokens: ${cost}")
            elif provider == "ollama":
                print(f"- Cost per 1K tokens: Free (local)")
                if info.get('multilingual'):
                    print(f"- Multilingual: Yes")
                    languages = info.get('languages')
                    if languages:
                        print(f"- Languages: {languages}")

            # Show current settings
            print("- Current RAG Settings:")
            print(f"- Chunk size: {self.settings_manager.setting_get('rag_chunk_size')} tokens")
            print(f"- Chunk overlap: {self.settings_manager.setting_get('rag_chunk_overlap')} tokens")
            print(f"- Top K results: {self.settings_manager.setting_get('rag_top_k')}")

        except Exception as e:
            print(f"- Error getting model info: {e}")

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
        print(" - Refreshing model cache...")
        self.completer.clear_models_cache()
        print(" - Model cache cleared. Fresh models will be fetched on next use.")

    def model(self, arg: Optional[str]) -> None:
        if arg == None:
            ollama_url = self.settings_manager.setting_get("ollama_base_url")
            print(" - Please specify the model to use")
            print("   Available sources: OpenAI (https://platform.openai.com/docs/models)")
            print(f"                     Ollama ({ollama_url} if running)")
            print("                     Google (https://ai.google.dev/gemini-api/docs/models)")
            return

        model = arg

        # Validate the model before setting it
        if not self._validate_model(model):
            print(f"- (!) Invalid model: {model}")
            print(" - Model not found in OpenAI, Google, or Ollama APIs")

            # Show available models
            try:
                available_models = self.completer._get_available_models()
                if available_models:
                    print(" - Available models:")

                    # Group by source
                    openai_models = [m["name"] for m in available_models if m["source"] == "OpenAI"]
                    ollama_models = [m["name"] for m in available_models if m["source"] == "Ollama"]
                    google_models = [m["name"] for m in available_models if m["source"] == "Google"]

                    if openai_models:
                        print(f"   OpenAI: {', '.join(openai_models[:5])}" +
                              (f" (and {len(openai_models)-5} more)" if len(openai_models) > 5 else ""))

                    if google_models:
                        print(f"   Google: {', '.join(google_models)}")

                    if ollama_models:
                        print(f"   Ollama: {', '.join(ollama_models)}")

                    if not openai_models and not ollama_models and not google_models:
                        print("   (No models available - check API keys and network connection)")
                else:
                    print("   (Unable to fetch available models)")
            except Exception:
                print("   (Unable to fetch available models)")

            return

        # Set the model (this will trigger client update in conversation manager)
        self.conversation_manager.model = model
        self.settings_manager.setting_set("model", model)

        # Determine model source and show appropriate message
        if self.conversation_manager._is_ollama_model(model):
            ollama_url = self.settings_manager.setting_get("ollama_base_url")
            if self.conversation_manager._is_ollama_available():
                print(f"- Switched to Ollama model: {model}")
                print(f"- Running locally via Ollama at {ollama_url}")
            else:
                print(f"- Warning: Selected Ollama model '{model}' but Ollama not available")
                print(f"   Make sure Ollama is running at {ollama_url}")
        elif self.conversation_manager._is_google_model(model):
            print(f"- Switched to Google Gemini model: {model}")
            print(f"- https://ai.google.dev/gemini-api/docs/models")
        else:
            print(f"- Switched to OpenAI model: {model}")
            print(f"- https://platform.openai.com/docs/models")

    def youtube(self, arg: str) -> None:
        """
        Extract transcript from YouTube video and send to conversation manager as input.
        """
        print(" - Extracting info from YouTube...")

        video_url = None

        # Check if 'watch?v=' is in the URL, otherwise resolve URL
        if "watch?v=" in arg:
            video_url = arg
        else:
            print(" - Video ID missing from URL.")
            try:
                print(" - Attempting to determine ID.")
                response = requests.get(arg)
                video_url = response.url
            except requests.RequestException as e:
                print(f"- Error resolving URL: {e}")
                return

        if video_url:
            try:
                video_id = video_url.split("watch?v=")[1]
                print(f"- Video ID found: {video_id}.")
            except IndexError:
                print(" - Invalid YouTube URL provided.")
                return

            try:
                # Get video info using yt-dlp
                print(" - Fetching video info...")
                ydl_opts = {'quiet': True, 'no_warnings': True}
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(video_url, download=False)
                    if info:
                        video_title = "[" + info.get('title', 'Unknown Title') + "](" + video_url + ")"
                        channel_title = "[" + info.get('uploader', 'Unknown Channel') + "](" + info.get('uploader_url', video_url) + ")"
                    else:
                        print(" - Could not extract video information.")
                        return

                # Extract subtitles using yt-dlp
                print(" - Fetching video transcript using yt-dlp...")
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
                                print(f"- Found manual {lang} subtitles")
                                break

                        # If no manual subtitles, try automatic captions
                        if not subtitle_data:
                            for lang in ['en', 'en-US', 'en-GB', 'en-CA', 'en-AU']:
                                if lang in automatic_captions and automatic_captions[lang]:
                                    subtitle_data = automatic_captions[lang]
                                    subtitle_source = f"auto-generated {lang}"
                                    print(f"- Found auto-generated {lang} captions")
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
                                            print(f"- Successfully extracted transcript from {subtitle_source}")
                                            print(f"- Processing {len(transcript_text)} characters as input...")

                                            user_input = (
                                                "Channel title: " + channel_title +
                                                "\nVideo title: " + video_title +
                                                "\nVideo transcript: " + transcript_text
                                            )
                                            self.conversation_manager.conversation_history.append(
                                                {"role": "user", "content": user_input}
                                            )
                                            print(" - YouTube content added to conversation context.")
                                        else:
                                            raise Exception("Transcript text was empty after parsing")
                                    else:
                                        raise Exception(f"Failed to download subtitles: HTTP {response.status_code}")

                                except Exception as subtitle_error:
                                    print(f"- Error processing subtitle file: {subtitle_error}")
                                    raise subtitle_error
                            else:
                                raise Exception("No subtitle URL found")
                        else:
                            # List available subtitle languages for debugging
                            available_langs = list(subtitles.keys()) + list(automatic_captions.keys())
                            if available_langs:
                                print(f"- Available subtitle languages: {', '.join(set(available_langs))}")
                                print(" - No English subtitles found")
                            else:
                                print(" - No subtitles available for this video")
                            raise Exception("No English subtitles available")

                except Exception as e:
                    print(f"- Could not extract transcript: {str(e)}")
                    print(" - Continuing with video info only...")
                    # Still provide video info even without transcript
                    user_input = (
                        "Channel title: " + channel_title +
                        "\nVideo title: " + video_title +
                        "\nNote: No transcript could be extracted for this video."
                    )
                    self.conversation_manager.conversation_history.append(
                        {"role": "user", "content": user_input}
                    )
                    print(" - YouTube content added to conversation context.")

            except Exception as e:
                print(" - An error occurred:", e)
                return
        else:
            print(" - Could not determine the YouTube video URL.")
            return

    def url(self, url: str) -> None:
        """
        Extract content from a website URL and send to conversation manager as input.
        """
        # Check if this is a YouTube URL and redirect to YouTube command
        if "youtube.com" in url.lower() or "youtu.be" in url.lower():
            print(" - YouTube URL detected - redirecting to --youtube command for better transcript extraction...")
            self.youtube(url)
            return

        print(" - Extracting content from URL...")

        extractor = WebContentExtractor()
        result = extractor.extract_content(url)

        if result['error']:
            print(f"- Error: {result['error']}")
            return

        if not result['content']:
            print(" - No content could be extracted from the URL.")
            return

        # Display warning if paywall was encountered
        if result.get('warning'):
            print(f"- {result['warning']}")

        # Format the content for the conversation
        title = result['title'] or "Web Content"
        formatted_content = f"Website: {title}\n\nSource: {url}\n\n{result['content']}"

        # Add to conversation history
        self.conversation_manager.conversation_history.append(
            {"role": "user", "content": formatted_content}
        )

        print(" - Content added to conversation context.")
        print(" - You can now ask questions about this content.")

    def file(self, file_path: str) -> None:
        """
        Load file contents and add to conversation context using RAG document processing logic.
        """
        import os

        # Check if file exists
        if not os.path.exists(file_path):
            print(f"- Error: File not found: {file_path}")
            return

        # Check if file type is supported
        if not is_supported_file(file_path):
            supported_types = get_supported_extensions_display()
            print(f"- Error: Unsupported file type. Supported types: {supported_types}")
            return

        print(f"- Loading file: {os.path.basename(file_path)}")

        # Use DocumentProcessor to load the file content
        processor = DocumentProcessor()
        try:
            content = processor.load_file(file_path)

            if not content or not content.strip():
                print(" - Error: No content could be extracted from the file.")
                return

            # Clean the content
            content = processor.clean_text(content)

            # Get file info for display
            filename = os.path.basename(file_path)
            file_ext = os.path.splitext(filename)[1].lower()
            word_count = len(content.split())

            print(f"- Extracted content from {filename} ({word_count} words)")

            # Format the content for the conversation
            formatted_content = f"File: {filename}\n\nPath: {file_path}\n\n{content}"

            # Add to conversation history
            self.conversation_manager.conversation_history.append(
                {"role": "user", "content": formatted_content}
            )

            print(" - File content added to conversation context.")
            print(" - You can now ask questions about this content.")

        except Exception as e:
            print(f"- Error loading file: {e}")







