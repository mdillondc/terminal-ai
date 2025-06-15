"""
Improved Command Completer for Terminal AI Assistant

This module provides intelligent tab completion for commands using the
centralized command registry system. It supports various completion types
including file paths, model names, and custom suggestions.
"""

from typing import Generator, List, Optional, Dict, Any
import os
import glob
import time
from functools import lru_cache
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from openai import OpenAI
from vector_store import VectorStore
from command_registry import CommandRegistry, CompletionType, CompletionRules
from settings_manager import SettingsManager
from rag_config import get_file_type_info, is_supported_file
from model_manager import ModelManager


class CompletionCache:
    """Enhanced cache for completions with better state management"""

    def __init__(self, cache_duration: float = 2.0):
        self.cache_duration = cache_duration
        self._cache: Dict[str, tuple[float, List[str]]] = {}
        self._last_completion_text: str = ""

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
        self._last_completion_text = ""

    def should_refresh(self, current_text: str) -> bool:
        """Check if completion should be refreshed based on text changes"""
        # If text got shorter (backspace), we should refresh
        if len(current_text) < len(self._last_completion_text):
            return True
        self._last_completion_text = current_text
        return False

    def invalidate_pattern(self, pattern: str) -> None:
        """Invalidate cache entries that match a pattern"""
        keys_to_remove = [key for key in self._cache.keys() if pattern in key]
        for key in keys_to_remove:
            del self._cache[key]


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

        # Initialize ModelManager for model fetching and filtering
        self.model_manager = ModelManager(self.openai_client)

    def get_completions(self, document: Document, complete_event: Any) -> Generator[Completion, None, None]:
        """
        Generate completion suggestions with enhanced backspace handling.

        Args:
            document: The current document/input
            complete_event: Completion event from prompt_toolkit

        Yields:
            Completion objects for matching suggestions
        """
        try:
            text_before_cursor = document.text_before_cursor

            # Always clear cache to ensure fresh completions
            self.cache.clear()

            # Find the last command in the input
            last_command_info = self._parse_current_command(text_before_cursor)

            if not last_command_info:
                return

            command_name, argument_part = last_command_info

            if not command_name:
                # Complete command names - always generate fresh results
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

    def _fuzzy_match(self, partial: str, target: str) -> tuple[bool, int]:
        """
        Enhanced fuzzy matching function with improved scoring.

        Args:
            partial: Partial input to match against
            target: Target string to match

        Returns:
            tuple: (matches, score) where higher score = better match
        """
        partial_lower = partial.lower()
        target_lower = target.lower()

        if not partial_lower:
            return True, 0

        # Strategy 1: Exact start match (highest score)
        if target_lower.startswith(partial_lower):
            return True, 1000 + len(partial) * 2

        # Strategy 2: Word boundary match (after dash, underscore, space)
        import re
        word_boundary_pattern = r'[\-_\s]' + re.escape(partial_lower)
        if re.search(word_boundary_pattern, target_lower):
            return True, 800 + len(partial)

        # Strategy 3: Subsequence match (e.g., "ragb" matches "rag-build")
        partial_idx = 0
        last_match_pos = -1
        for i, char in enumerate(target_lower):
            if partial_idx < len(partial_lower) and char == partial_lower[partial_idx]:
                partial_idx += 1
                last_match_pos = i

        if partial_idx == len(partial_lower):
            # All characters found in order - bonus for consecutive matches
            consecutive_bonus = 50 if last_match_pos - len(partial) >= 0 else 0
            return True, 500 + consecutive_bonus + len(partial)

        # Strategy 4: Contains substring anywhere
        if partial_lower in target_lower:
            return True, 300 + len(partial)

        # Strategy 5: Character overlap with minimum threshold
        matching_chars = sum(1 for c in partial_lower if c in target_lower)
        overlap_ratio = matching_chars / len(partial_lower) if partial_lower else 0

        if overlap_ratio >= 0.6:  # Lowered threshold for better matching
            return True, 100 + matching_chars

        return False, 0

    def _complete_command_names(self, partial_command: str) -> Generator[Completion, None, None]:
        """Complete command names with enhanced fuzzy matching - always fresh results"""
        matches = []

        for command in self.available_commands:
            command_keyword = command[2:]  # Remove "--" prefix

            is_match, score = self._fuzzy_match(partial_command, command_keyword)

            if is_match:
                description = self.registry.get_command_description(command)
                matches.append((score, command_keyword, description))

        # Sort by score (higher is better), then alphabetically
        matches.sort(key=lambda x: (-x[0], x[1]))

        # Yield completions in order of relevance
        for score, command_keyword, description in matches:
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
        """Complete file paths with support for special paths and supported file types"""
        try:
            # Handle special paths
            if partial_path.strip() == "":
                # Show current working directory contents
                directory = rules.base_directory or "."
                filename_prefix = ""
            elif partial_path.strip() in ["/", "~"]:
                # Handle root and home directory
                if partial_path.strip() == "/":
                    directory = "/"
                else:  # "~"
                    directory = os.path.expanduser("~")
                filename_prefix = ""
            else:
                # Expand user home directory and handle paths
                expanded_path = os.path.expanduser(partial_path)

                # Determine directory and filename parts
                if os.path.isdir(expanded_path):
                    directory = expanded_path
                    filename_prefix = ""
                else:
                    directory = os.path.dirname(expanded_path) or (rules.base_directory or ".")
                    filename_prefix = os.path.basename(expanded_path)

            # Use supported file extensions from rag_config for --file command
            if rules.file_extensions:
                use_file_filtering = True
            else:
                use_file_filtering = False

            files = self._get_files_in_directory(directory, filename_prefix, use_file_filtering)

            for file_path in files:
                # Calculate the completion text and start position
                completion_text = os.path.basename(file_path.rstrip("/"))

                if partial_path.endswith("/"):
                    # Directory path - append to the existing path
                    start_pos = 0
                elif filename_prefix:
                    # Partial filename - replace the filename part
                    start_pos = -len(filename_prefix)
                else:
                    # Empty or special paths - handle appropriately
                    if partial_path.strip() in ["/", "~"]:
                        start_pos = -len(partial_path)
                    else:
                        start_pos = 0

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

            # Sort by modification time (most recent first)
            matching_files.sort(key=lambda f: os.path.getmtime(f), reverse=True)

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
        """Complete model names from OpenAI, Google, and Ollama APIs with fuzzy matching"""
        # Try to get dynamic models from all sources
        models = self.model_manager.get_available_models()

        # If we have models from APIs, use those with fuzzy matching
        if models:
            matches = []
            for model_info in models:
                model_name = model_info["name"]
                model_source = model_info["source"]

                is_match, score = self._fuzzy_match(partial_name, model_name)
                if is_match:
                    matches.append((model_source, model_name, score, f"{model_source} Model"))

            # Sort by fuzzy match score (higher is better), then by source, then by model name
            matches.sort(key=lambda x: (-x[2], x[0], x[1]))

            for model_source, model_name, score, display_meta in matches:
                yield Completion(
                    text=model_name,
                    start_position=-len(partial_name),
                    display_meta=display_meta
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
        """Complete TTS model names with fuzzy matching"""
        # OpenAI TTS models
        tts_models = [
            ("tts-1", "Standard TTS model, optimized for real-time use"),
            ("tts-1-hd", "High-definition TTS model, optimized for quality"),
            ("gpt-4o-mini-tts", "Latest GPT-based TTS model with improved prosody")
        ]

        matches = []
        for model_name, description in tts_models:
            is_match, score = self._fuzzy_match(partial_name, model_name)
            if is_match:
                matches.append((score, model_name, description))

        # Sort by score (higher is better), then alphabetically
        matches.sort(key=lambda x: (-x[0], x[1]))

        for score, model_name, description in matches:
            yield Completion(
                text=model_name,
                start_position=-len(partial_name),
                display_meta=description
            )

    def _complete_tts_voices(self, partial_name: str, rules: CompletionRules) -> Generator[Completion, None, None]:
        """Complete TTS voice names with fuzzy matching"""
        # OpenAI TTS voices with descriptions
        tts_voices = [
            ("alloy", "Neutral, versatile voice"),
            ("echo", "Clear, professional voice"),
            ("fable", "Warm, storytelling voice"),
            ("onyx", "Deep, authoritative voice"),
            ("nova", "Bright, energetic voice"),
            ("shimmer", "Soft, gentle voice")
        ]

        matches = []
        for voice_name, description in tts_voices:
            is_match, score = self._fuzzy_match(partial_name, voice_name)
            if is_match:
                matches.append((score, voice_name, description))

        # Sort by score (higher is better), then alphabetically
        matches.sort(key=lambda x: (-x[0], x[1]))

        for score, voice_name, description in matches:
            yield Completion(
                text=voice_name,
                start_position=-len(partial_name),
                display_meta=description
            )

    def _complete_rag_collections(self, partial_name: str, rules: CompletionRules) -> Generator[Completion, None, None]:
        """Complete RAG collection names using VectorStore with fuzzy matching"""
        try:
            matches = []

            # Use VectorStore for collection discovery (DRY principle)
            vector_store = VectorStore()
            collections = vector_store.get_available_collections()

            for collection in collections:
                is_match, score = self._fuzzy_match(partial_name, collection)
                if is_match:
                    matches.append((score, collection, "RAG collection"))

            # Sort by score (higher is better), then alphabetically
            matches.sort(key=lambda x: (-x[0], x[1]))

            for score, text, display_meta in matches:
                yield Completion(
                    text=text,
                    start_position=-len(partial_name),
                    display_meta=display_meta
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
            files = []

            for root, dirs, filenames in os.walk(collection_path):
                for filename in filenames:
                    # Get full path for proper file type detection (including MIME type for text files)
                    file_path = os.path.join(root, filename)

                    if is_supported_file(file_path):
                        # Get relative path from collection root
                        relative_path = os.path.relpath(file_path, collection_path)
                        files.append(relative_path)

            # Filter files based on partial input with smart fuzzy matching
            partial_lower = partial_filename.lower()
            for file_path in sorted(files):
                file_path_lower = file_path.lower()
                filename = os.path.basename(file_path).lower()

                # Try multiple matching strategies for better UX:
                # 1. Full path starts with partial (original behavior)
                # 2. Filename starts with partial (user-friendly for nested files)
                # 3. Any part of the path contains partial (fuzzy matching)
                if (file_path_lower.startswith(partial_lower) or
                    filename.startswith(partial_lower) or
                    partial_lower in file_path_lower):

                    yield Completion(
                        text=file_path,
                        start_position=-len(partial_filename),
                        display_meta="RAG file"
                    )

        except Exception:
            # Graceful degradation - don't break completion
            return

    def _complete_simple_suggestions(self, partial_text: str, rules: CompletionRules) -> Generator[Completion, None, None]:
        """Complete with simple static suggestions using fuzzy matching"""
        if not rules.custom_suggestions:
            return

        matches = []
        for suggestion in rules.custom_suggestions:
            is_match, score = self._fuzzy_match(partial_text, suggestion)
            if is_match:
                matches.append((score, suggestion))

        # Sort by score (higher is better), then alphabetically
        matches.sort(key=lambda x: (-x[0], x[1]))

        for score, suggestion in matches:
            yield Completion(
                text=suggestion,
                start_position=-len(partial_text)
            )

    @lru_cache(maxsize=100)
    def _get_files_in_directory(self, directory: str, prefix: str, filter_supported: bool = False) -> List[str]:
        """Get files in directory with optional prefix and supported file filtering (cached)"""
        try:
            if not os.path.exists(directory):
                return []

            candidates = []
            for item in os.listdir(directory):
                item_path = os.path.join(directory, item)

                # Include directories for navigation
                if os.path.isdir(item_path):
                    candidates.append((item_path + "/", True))  # (path, is_directory)
                    continue

                # Filter by supported file types if specified
                if filter_supported:
                    if is_supported_file(item_path):
                        candidates.append((item_path, False))
                else:
                    candidates.append((item_path, False))

            # Apply fuzzy matching if prefix is provided
            if prefix:
                matches = []
                for item_path, is_dir in candidates:
                    item_name = os.path.basename(item_path.rstrip("/"))
                    is_match, score = self._fuzzy_match(prefix, item_name)
                    if is_match:
                        matches.append((score, item_path))

                # Sort by score (higher is better), then alphabetically
                matches.sort(key=lambda x: (-x[0], x[1]))
                return [item_path for score, item_path in matches]
            else:
                # No prefix - return all candidates sorted
                return sorted([item_path for item_path, is_dir in candidates])

        except (OSError, PermissionError):
            return []

    def _get_file_display_meta(self, file_path: str) -> str:
        """Get display metadata for files using rag_config file type information"""
        try:
            # Clean up file path (remove trailing slash for directories)
            clean_path = file_path.rstrip("/")

            if os.path.isdir(clean_path):
                return "Directory"

            # Get file size
            size = os.path.getsize(clean_path)
            if size < 1024:
                size_str = f"{size}B"
            elif size < 1024 * 1024:
                size_str = f"{size // 1024}KB"
            else:
                size_str = f"{size // (1024 * 1024)}MB"

            # Get file type information from rag_config - only show supported types
            ext = os.path.splitext(clean_path)[1].lower()
            file_type_info = get_file_type_info(ext)

            if file_type_info:
                file_type = file_type_info.get('name', 'File')
            else:
                file_type = 'File'

            return f"{file_type} ({size_str})"

        except (OSError, PermissionError):
            return "File"

    def _get_relative_path(self, file_path: str, base_dir: str) -> str:
        """Get relative path from base directory"""
        try:
            return os.path.relpath(file_path, base_dir)
        except ValueError:
            return os.path.basename(file_path)





    def refresh_cache(self) -> None:
        """Clear completion cache to force refresh"""
        self.cache.clear()
        # Clear the lru_cache as well
        self._get_files_in_directory.cache_clear()
        # Clear model cache as well
        self.model_manager.refresh_cache()

    def clear_models_cache(self) -> None:
        """Clear persistent model cache files"""
        self.model_manager.clear_models_cache()

