"""
Improved Command Completer for Samantha AI Assistant

This module provides intelligent tab completion for commands using the
centralized command registry system. It supports various completion types
including file paths, model names, and custom suggestions.
"""

from typing import Generator, List, Optional, Dict, Any
import os
import glob
import time
import json
import urllib.request
import urllib.error
from datetime import datetime, timezone
from functools import lru_cache
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from openai import OpenAI
from vector_store import VectorStore

from command_registry import CommandRegistry, CompletionType, CompletionRules
from settings_manager import SettingsManager


class CompletionCache:
    """Simple cache for file system completions and API calls to improve performance"""
    
    def __init__(self, cache_duration: float = 2.0):
        self.cache_duration = cache_duration
        self._cache: Dict[str, tuple[float, List[str]]] = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached results if they're still valid"""
        if key in self._cache:
            timestamp, results = self._cache[key]
            if time.time() - timestamp < self.cache_duration:
                return results
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Cache results with current timestamp"""
        self._cache[key] = (time.time(), value)
    
    def clear(self) -> None:
        """Clear all cached results"""
        self._cache.clear()


class CommandCompleter(Completer):
    """
    Intelligent command completer that uses the CommandRegistry for
    context-aware completion suggestions.
    """
    
    def __init__(self, command_registry: CommandRegistry):
        self.registry = command_registry
        self.settings_manager = SettingsManager.getInstance()
        self.cache = CompletionCache(cache_duration=300.0)  # Cache API calls for 5 minutes
        
        # Pre-compute available commands for faster access
        self.available_commands = self.registry.get_available_commands()
        
        # Initialize OpenAI client for dynamic model fetching
        try:
            self.openai_client = OpenAI()
        except Exception:
            self.openai_client = None
    
    def get_completions(self, document: Document, complete_event: Any) -> Generator[Completion, None, None]:
        """
        Generate completion suggestions based on current input context.
        
        Args:
            document: The current document/input
            complete_event: Completion event from prompt_toolkit
            
        Yields:
            Completion objects for matching suggestions
        """
        try:
            text_before_cursor = document.text_before_cursor
            
            # Find the last command in the input
            last_command_info = self._parse_current_command(text_before_cursor)
            
            if not last_command_info:
                return
            
            command_name, argument_part = last_command_info
            
            if not command_name:
                # Complete command names
                yield from self._complete_command_names(argument_part)
            else:
                # Complete arguments for specific command
                yield from self._complete_command_arguments(command_name, argument_part)
                
        except Exception:
            # Graceful degradation - don't break the completion system
            # In production, you might want to log this error
            pass
    
    def _parse_current_command(self, text: str) -> Optional[tuple[Optional[str], str]]:
        """
        Parse the current command context from input text.
        
        Returns:
            tuple: (command_name, argument_part) or None if no command context
        """
        text_lower = text.lower()
        last_double_dash = text_lower.rfind("--")
        
        if last_double_dash == -1:
            return None
        
        # Extract everything after the last "--"
        after_dash = text[last_double_dash + 2:]
        
        # Split into command and argument parts
        parts = after_dash.split(" ", 1)
        
        if len(parts) == 1:
            # Still typing command name
            return None, parts[0]
        else:
            # Have command name, completing argument
            command_name = "--" + parts[0]
            argument_part = parts[1] if len(parts) > 1 else ""
            return command_name, argument_part
    
    def _complete_command_names(self, partial_command: str) -> Generator[Completion, None, None]:
        """Complete command names based on partial input"""
        for command in self.available_commands:
            command_keyword = command[2:]  # Remove "--" prefix
            
            if command_keyword.startswith(partial_command.lower()):
                description = self.registry.get_command_description(command)
                yield Completion(
                    text=command_keyword,
                    start_position=-len(partial_command),
                    display_meta=description
                )
    
    def _complete_command_arguments(self, command_name: str, argument_part: str) -> Generator[Completion, None, None]:
        """Complete arguments for a specific command"""
        completion_rules = self.registry.get_completion_rules(command_name)
        
        if not completion_rules:
            return
        
        completion_type = completion_rules.completion_type
        
        if completion_type == CompletionType.NONE:
            return
        elif completion_type == CompletionType.FILE_PATH:
            yield from self._complete_file_paths(argument_part, completion_rules)
        elif completion_type == CompletionType.LOG_FILE:
            yield from self._complete_log_files(argument_part, completion_rules)
        elif completion_type == CompletionType.INSTRUCTION_FILE:
            yield from self._complete_instruction_files(argument_part, completion_rules)
        elif completion_type == CompletionType.MODEL_NAME:
            yield from self._complete_model_names(argument_part, completion_rules)
        elif completion_type == CompletionType.TTS_MODEL:
            yield from self._complete_tts_models(argument_part, completion_rules)
        elif completion_type == CompletionType.TTS_VOICE:
            yield from self._complete_tts_voices(argument_part, completion_rules)
        elif completion_type == CompletionType.RAG_COLLECTION:
            yield from self._complete_rag_collections(argument_part, completion_rules)
        elif completion_type == CompletionType.RAG_COLLECTION_FILE:
            yield from self._complete_rag_collection_files(argument_part, completion_rules)
        elif completion_type == CompletionType.SIMPLE:
            yield from self._complete_simple_suggestions(argument_part, completion_rules)
    
    def _complete_file_paths(self, partial_path: str, rules: CompletionRules) -> Generator[Completion, None, None]:
        """Complete file paths with optional extension filtering"""
        try:
            # Expand user home directory
            expanded_path = os.path.expanduser(partial_path)
            
            # Determine directory and filename parts
            if os.path.isdir(expanded_path):
                directory = expanded_path
                filename_prefix = ""
            else:
                directory = os.path.dirname(expanded_path) or "."
                filename_prefix = os.path.basename(expanded_path)
            
            # Get file suggestions (convert list to tuple for caching)
            extensions_tuple = tuple(rules.file_extensions) if rules.file_extensions else None
            files = self._get_files_in_directory(directory, filename_prefix, extensions_tuple)
            
            for file_path in files:
                # Calculate the completion text
                if partial_path.endswith("/") or not filename_prefix:
                    completion_text = os.path.basename(file_path)
                    start_pos = 0
                else:
                    completion_text = os.path.basename(file_path)
                    start_pos = -len(filename_prefix)
                
                yield Completion(
                    text=completion_text,
                    start_position=start_pos,
                    display_meta=self._get_file_display_meta(file_path)
                )
                
        except (OSError, PermissionError):
            # Handle file system errors gracefully
            pass
    
    def _complete_log_files(self, partial_name: str, rules: CompletionRules) -> Generator[Completion, None, None]:
        """Complete log file names from the logs directory"""
        if not rules.base_directory or not os.path.exists(rules.base_directory):
            return
        
        cache_key = f"logs:{partial_name}:{rules.base_directory}"
        cached_results = self.cache.get(cache_key)
        
        if cached_results is None:
            # Search recursively for log files
            pattern = os.path.join(rules.base_directory, "**", "*.md")
            all_log_files = glob.glob(pattern, recursive=True)
            
            # Filter by partial name
            matching_files = [
                f for f in all_log_files
                if partial_name.lower() in os.path.basename(f).lower()
            ]
            
            self.cache.set(cache_key, matching_files)
            cached_results = matching_files
        
        for log_file in cached_results:
            filename = os.path.basename(log_file)
            yield Completion(
                text=filename,
                start_position=-len(partial_name),
                display_meta=f"Log: {self._get_relative_path(log_file, rules.base_directory)}"
            )
    
    def _complete_instruction_files(self, partial_name: str, rules: CompletionRules) -> Generator[Completion, None, None]:
        """Complete instruction file names from the instructions directory"""
        if not rules.base_directory or not os.path.exists(rules.base_directory):
            return
        
        try:
            files = os.listdir(rules.base_directory)
            markdown_files = [f for f in files if f.endswith('.md')]
            
            for file in markdown_files:
                if partial_name.lower() in file.lower():
                    yield Completion(
                        text=file,
                        start_position=-len(partial_name),
                        display_meta="Instruction set"
                    )
        except OSError:
            pass
    
    def _complete_model_names(self, partial_name: str, rules: CompletionRules) -> Generator[Completion, None, None]:
        """Complete model names from OpenAI, Google, and Ollama APIs with fallback to predefined suggestions"""
        # Try to get dynamic models from all sources
        models = self._get_available_models()
        
        # If we have models from APIs, use those
        if models:
            for model_info in models:
                model_name = model_info["name"]
                model_source = model_info["source"]
                
                if partial_name.lower() in model_name.lower():
                    yield Completion(
                        text=model_name,
                        start_position=-len(partial_name),
                        display_meta=f"{model_source} Model"
                    )
        # Fallback to predefined suggestions if APIs are unavailable
        elif rules.custom_suggestions:
            for suggestion in rules.custom_suggestions:
                if partial_name.lower() in suggestion.lower():
                    yield Completion(
                        text=suggestion,
                        start_position=-len(partial_name),
                        display_meta="Fallback Model"
                    )
    
    def _complete_tts_models(self, partial_name: str, rules: CompletionRules) -> Generator[Completion, None, None]:
        """Complete TTS model names with OpenAI TTS models"""
        # OpenAI TTS models
        tts_models = [
            ("tts-1", "Standard TTS model, optimized for real-time use"),
            ("tts-1-hd", "High-definition TTS model, optimized for quality"),
            ("gpt-4o-mini-tts", "Latest GPT-based TTS model with improved prosody")
        ]
        
        for model_name, description in tts_models:
            if partial_name.lower() in model_name.lower():
                yield Completion(
                    text=model_name,
                    start_position=-len(partial_name),
                    display_meta=description
                )
    
    def _complete_tts_voices(self, partial_name: str, rules: CompletionRules) -> Generator[Completion, None, None]:
        """Complete TTS voice names with OpenAI TTS voices"""
        # OpenAI TTS voices with descriptions
        tts_voices = [
            ("alloy", "Neutral, versatile voice"),
            ("echo", "Clear, professional voice"),
            ("fable", "Warm, storytelling voice"),
            ("onyx", "Deep, authoritative voice"),
            ("nova", "Bright, energetic voice"),
            ("shimmer", "Soft, gentle voice")
        ]
        
        for voice_name, description in tts_voices:
            if partial_name.lower() in voice_name.lower():
                yield Completion(
                    text=voice_name,
                    start_position=-len(partial_name),
                    display_meta=description
                )
    
    def _complete_rag_collections(self, partial_name: str, rules: CompletionRules) -> Generator[Completion, None, None]:
        """Complete RAG collection names using VectorStore (single source of truth)"""
        try:
            # Special completion for 'off' option
            if "off".startswith(partial_name.lower()):
                yield Completion(
                    text="off",
                    start_position=-len(partial_name),
                    display_meta="Deactivate RAG mode"
                )
            
            # Use VectorStore for collection discovery (DRY principle)
            vector_store = VectorStore()
            collections = vector_store.get_available_collections()
            
            for collection in collections:
                if collection.startswith(partial_name):
                    yield Completion(
                        text=collection,
                        start_position=-len(partial_name),
                        display_meta="RAG collection"
                    )
        except Exception:
            # Silently fail if there are any issues with directory access
            pass
    
    def _complete_rag_collection_files(self, partial_filename: str, rules: CompletionRules) -> Generator[Completion, None, None]:
        """Complete with files from the active RAG collection"""
        try:
            # Check if there's an active RAG collection
            active_collection = self.settings_manager.setting_get("rag_active_collection")
            if not active_collection:
                return
                
            # Get the collection path
            working_dir = self.settings_manager.setting_get("working_dir")
            collection_path = os.path.join(working_dir, "rag", active_collection)
            
            if not os.path.exists(collection_path) or not os.path.isdir(collection_path):
                return
                
            # Get all supported files in the collection (recursively)
            supported_extensions = {'.txt', '.md'}
            files = []
            
            for root, dirs, filenames in os.walk(collection_path):
                for filename in filenames:
                    file_ext = os.path.splitext(filename)[1].lower()
                    
                    if file_ext in supported_extensions:
                        # Get relative path from collection root
                        file_path = os.path.join(root, filename)
                        relative_path = os.path.relpath(file_path, collection_path)
                        files.append(relative_path)
            
            # Filter files based on partial input and yield completions
            partial_lower = partial_filename.lower()
            for file_path in sorted(files):
                if file_path.lower().startswith(partial_lower):
                    yield Completion(
                        text=file_path,
                        start_position=-len(partial_filename),
                        display_meta="RAG file"
                    )
                    
        except Exception:
            # Graceful degradation - don't break completion
            return

    def _complete_simple_suggestions(self, partial_text: str, rules: CompletionRules) -> Generator[Completion, None, None]:
        """Complete with simple static suggestions"""
        if not rules.custom_suggestions:
            return
            
        partial_lower = partial_text.lower()
        for suggestion in rules.custom_suggestions:
            if suggestion.lower().startswith(partial_lower):
                yield Completion(
                    text=suggestion,
                    start_position=-len(partial_text)
                )
    
    @lru_cache(maxsize=100)
    def _get_files_in_directory(self, directory: str, prefix: str, extensions: Optional[tuple] = None) -> List[str]:
        """Get files in directory with optional prefix and extension filtering (cached)"""
        try:
            if not os.path.exists(directory):
                return []
            
            files = []
            for item in os.listdir(directory):
                item_path = os.path.join(directory, item)
                
                # Skip if doesn't match prefix
                if prefix and not item.lower().startswith(prefix.lower()):
                    continue
                
                # Include directories for navigation
                if os.path.isdir(item_path):
                    files.append(item_path + "/")
                    continue
                
                # Filter by extensions if specified
                if extensions:
                    if any(item.lower().endswith(ext.lower()) for ext in extensions):
                        files.append(item_path)
                else:
                    files.append(item_path)
            
            return sorted(files)
            
        except (OSError, PermissionError):
            return []
    
    def _get_file_display_meta(self, file_path: str) -> str:
        """Get display metadata for instruction and log files"""
        try:
            if os.path.isdir(file_path):
                return "Directory"
            
            # Get file size
            size = os.path.getsize(file_path)
            if size < 1024:
                size_str = f"{size}B"
            elif size < 1024 * 1024:
                size_str = f"{size // 1024}KB"
            else:
                size_str = f"{size // (1024 * 1024)}MB"
            
            # Get file extension - only handle relevant types
            ext = os.path.splitext(file_path)[1].lower()
            if ext == '.md':
                file_type = 'Markdown'
            elif ext == '.json':
                file_type = 'JSON'
            else:
                file_type = 'File'
            
            return f"{file_type} ({size_str})"
            
        except OSError:
            return "File"
    
    def _get_relative_path(self, file_path: str, base_dir: str) -> str:
        """Get relative path from base directory"""
        try:
            return os.path.relpath(file_path, base_dir)
        except ValueError:
            return os.path.basename(file_path)
    
    def _get_available_models(self) -> List[Dict[str, str]]:
        """Get available models from OpenAI, Google, and Ollama APIs with persistent caching"""
        all_models = []
        
        # Get OpenAI models (with persistent cache)
        openai_models = self._get_openai_models_cached()
        for model in openai_models:
            all_models.append({"name": model, "source": "OpenAI"})
        
        # Get Google models (with persistent cache)
        google_models = self._get_google_models_cached()
        for model in google_models:
            all_models.append({"name": model, "source": "Google"})
        
        # Get Ollama models (always live, no cache)
        ollama_models = self._get_ollama_models()
        for model in ollama_models:
            all_models.append({"name": model, "source": "Ollama"})
        
        return all_models
    
    def _get_openai_models_cached(self) -> List[str]:
        """Get available models from OpenAI API with persistent caching"""
        # Try to load from cache first
        cached_models = self._load_models_from_cache("openai")
        if cached_models is not None:
            return cached_models
        
        # Cache miss or expired, try to fetch from API
        fresh_models = self._get_openai_models()
        if fresh_models:
            # Save to cache if API call succeeded
            self._save_models_to_cache("openai", fresh_models)
            return fresh_models
        
        # API call failed, try to use expired cache as fallback
        cache_path = self._get_cache_file_path("openai")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    cache_data = json.load(f)
                return cache_data.get("models", [])
            except (json.JSONDecodeError, KeyError, OSError):
                pass
        
        return []

    def _get_openai_models(self) -> List[str]:
        """Get available models from OpenAI API (no caching)"""
        if not self.openai_client:
            return []
        
        try:
            # Fetch models from OpenAI API
            models_response = self.openai_client.models.list()
            
            # Extract model IDs and filter for relevant ones
            all_models = [model.id for model in models_response.data]
            
            # Filter for GPT models
            filtered_models = []
            
            for model in all_models:
                if any(keyword in model.lower() for keyword in ['gpt', 'o1']):
                    filtered_models.append(model)
            
            return filtered_models
            
        except Exception:
            # If API call fails, return empty list (will fallback to predefined)
            return []
    
    def _get_ollama_models(self) -> List[str]:
        """Get available models from Ollama with graceful error handling"""
        try:
            # Get Ollama base URL from settings
            settings = SettingsManager.getInstance()
            base_url = settings.setting_get("ollama_base_url")
            url = f"{base_url}/api/tags"
            
            # Create request with short timeout
            req = urllib.request.Request(url)
            req.add_header('Content-Type', 'application/json')
            
            # Very short timeout - we don't want to block if Ollama isn't available
            with urllib.request.urlopen(req, timeout=2) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode('utf-8'))
                    
                    models = []
                    if 'models' in data:
                        for model in data['models']:
                            model_name = model.get('name', '')
                            if model_name:
                                models.append(model_name)
                    

                    return models
                    
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, TimeoutError, OSError):
            # Ollama not available, connection failed, or invalid response - silently continue
            pass
        except Exception:
            # Any other error - silently continue to not break the application
            pass
        
        return []
    
    def _get_google_models_cached(self) -> List[str]:
        """Get available models from Google Gemini API with persistent caching"""
        # Try to load from cache first
        cached_models = self._load_models_from_cache("google")
        if cached_models is not None:
            return cached_models
        
        # Cache miss or expired, try to fetch from API
        fresh_models = self._get_google_models()
        if fresh_models:
            # Save to cache if API call succeeded
            self._save_models_to_cache("google", fresh_models)
            return fresh_models
        
        # API call failed, try to use expired cache as fallback
        cache_path = self._get_cache_file_path("google")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    cache_data = json.load(f)
                return cache_data.get("models", [])
            except (json.JSONDecodeError, KeyError, OSError):
                pass
        
        return []

    def _get_google_models(self) -> List[str]:
        """Get available models from Google Gemini API (no caching)"""
        try:
            import os
            
            # Check if API key is available
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                return []
            
            # Get Google models from API
            url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
            req = urllib.request.Request(url)
            req.add_header('Content-Type', 'application/json')
            
            with urllib.request.urlopen(req, timeout=3) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode('utf-8'))
                    
                    models = []
                    if 'models' in data:
                        for model_info in data['models']:
                            model_name = model_info.get('name', '')
                            if model_name:
                                # Extract model name from "models/gemini-1.5-flash" format
                                if model_name.startswith('models/'):
                                    model_name = model_name[7:]  # Remove "models/" prefix
                                
                                # Filter for chat-capable models (exclude embeddings, vision-only, etc.)
                                if any(keyword in model_name.lower() for keyword in ['gemini', 'palm', 'bard']):
                                    # Skip deprecated or vision-only models
                                    if not any(skip in model_name.lower() for skip in ['vision', 'embedding', 'deprecated']):
                                        models.append(model_name)
                    
                    return models
                    
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, TimeoutError, OSError):
            # Google API not available, connection failed, or invalid response
            pass
        except Exception:
            # Any other error - silently continue to not break the application
            pass
        
        return []

    def refresh_cache(self) -> None:
        """Clear completion cache to force refresh"""
        self.cache.clear()
        # Clear the lru_cache as well
        self._get_files_in_directory.cache_clear()

    def clear_models_cache(self) -> None:
        """Clear persistent model cache files"""
        cache_dir = os.path.join(os.path.dirname(self.registry.working_dir), "cache", "models")
        for cache_file in ["openai.json", "google.json"]:
            cache_path = os.path.join(cache_dir, cache_file)
            if os.path.exists(cache_path):
                try:
                    os.remove(cache_path)
                except OSError:
                    pass  # Ignore errors if file can't be removed

    def _get_cache_file_path(self, source: str) -> str:
        """Get the cache file path for a given model source"""
        cache_dir = os.path.join(os.path.dirname(self.registry.working_dir), "cache", "models")
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f"{source.lower()}.json")

    def _load_models_from_cache(self, source: str) -> Optional[List[str]]:
        """Load models from cache file if valid"""
        cache_path = self._get_cache_file_path(source)
        
        if not os.path.exists(cache_path):
            return None

        try:
            with open(cache_path, 'r') as f:
                cache_data = json.load(f)
            
            if not self._is_cache_valid(cache_data, source):
                return None
                
            return cache_data.get("models", [])
            
        except (json.JSONDecodeError, KeyError, OSError):
            return None

    def _save_models_to_cache(self, source: str, models: List[str]) -> None:
        """Save models to cache file"""
        cache_path = self._get_cache_file_path(source)
        
        cache_data = {
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "models": models
        }
        
        try:
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f, indent=2)
        except OSError:
            pass  # Ignore cache write errors

    def _is_cache_valid(self, cache_data: Dict[str, Any], source: str) -> bool:
        """Check if cache data is still valid based on expiration time"""
        try:
            last_updated_str = cache_data.get("last_updated")
            if not last_updated_str:
                return False
                
            last_updated = datetime.fromisoformat(last_updated_str.replace('Z', '+00:00'))
            now = datetime.now(timezone.utc)
            
            # Get cache expiration hours from settings
            from settings_manager import SettingsManager
            settings = SettingsManager.getInstance()
            if source.lower() == "openai":
                cache_hours = settings.setting_get("openai_models_cache_hours")
            elif source.lower() == "google":
                cache_hours = settings.setting_get("google_models_cache_hours")
            else:
                return False
                
            cache_duration_seconds = cache_hours * 3600
            return (now - last_updated).total_seconds() < cache_duration_seconds
            
        except (ValueError, TypeError):
            return False