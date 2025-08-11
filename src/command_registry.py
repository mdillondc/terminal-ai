"""
Command Registry System for Terminal AI Assistant

This module provides a centralized registry for all commands, their metadata,
and completion rules. It separates command definition from execution and
completion logic for better maintainability and extensibility.
"""

from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from dataclasses import dataclass
import os


class CompletionType(Enum):
    """Defines different types of completion behavior for commands"""
    SIMPLE = "simple"  # Just command name completion
    FILE_PATH = "file_path"  # File path completion
    LOG_FILE = "log_file"  # Log file completion
    INSTRUCTION_FILE = "instruction_file"  # Instruction file completion
    MODEL_NAME = "model_name"  # Model name suggestions

    RAG_COLLECTION = "rag_collection"  # RAG collection name suggestions
    RAG_COLLECTION_FILE = "rag_collection_file"  # Files in active RAG collection
    SEARCH_ENGINE = "search_engine"  # Search engine suggestions
    NONE = "none"  # No additional completion after command name


@dataclass
class CompletionRules:
    """Rules for how a command should be completed"""
    completion_type: CompletionType
    file_extensions: Optional[List[str]] = None  # For file completions
    base_directory: Optional[str] = None  # Base directory for file searches
    custom_suggestions: Optional[List[str]] = None  # Static suggestions
    validator: Optional[Callable[[str], bool]] = None  # Custom validation


@dataclass
class CommandInfo:
    """Complete information about a command"""
    name: str  # Command name (e.g., "--model")
    description: str  # Human-readable description
    usage: str  # Usage example
    execution_order: int  # Order of execution when chaining commands
    completion_rules: CompletionRules  # How to complete this command
    requires_argument: bool = False  # Whether command requires an argument
    aliases: Optional[List[str]] = None  # Alternative command names


class CommandRegistry:
    """
    Central registry for all commands in the application.

    This class maintains a single source of truth for command definitions,
    their metadata, and completion behavior.
    """

    def __init__(self, working_dir: str):
        self.working_dir = working_dir
        self._commands: Dict[str, CommandInfo] = {}
        self._initialize_default_commands()

    def _initialize_default_commands(self) -> None:
        """Initialize all default commands with their metadata and completion rules"""

        # Core commands
        self.register_command(CommandInfo(
            name="--model",
            description="Switches the underlying AI model. Use this to select a different AI model for processing commands (dynamically fetched from OpenAI, Google, and Ollama APIs).",
            usage="--model gpt-5",
            execution_order=1,
            completion_rules=CompletionRules(
                CompletionType.MODEL_NAME,
                custom_suggestions=["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"]  # Fallback suggestions prioritizing Google over OpenAI
            ),
            requires_argument=True
        ))

        self.register_command(CommandInfo(
            name="--model-clear-cache",
            description="Clears the model cache and forces a fresh fetch of available models from connected APIs.<",
            usage="--model-clear-cache",
            execution_order=1,
            completion_rules=CompletionRules(CompletionType.NONE),
            requires_argument=False
        ))

        # Content processing commands
        self.register_command(CommandInfo(
            name="--instructions",
            description="Activates a specific skill for your AI. Skills are specialized functions like summarizing YouTube videos.",
            usage="--instructions wisdom.md",
            execution_order=1,
            completion_rules=CompletionRules(
                CompletionType.INSTRUCTION_FILE,
                file_extensions=[".md"],
                base_directory=os.path.join(self.working_dir, "instructions")
            ),
            requires_argument=True
        ))

        self.register_command(CommandInfo(
            name="--youtube",
            description="Extract video transcript and add to conversation context.",
            usage="--youtube https://youtube.com/watch?v=example",
            execution_order=2,
            completion_rules=CompletionRules(CompletionType.NONE),
            requires_argument=True
        ))

        self.register_command(CommandInfo(
            name="--url",
            description="Extract content from any website URL and send to AI for analysis.",
            usage="--url https://example.com/article",
            execution_order=2,
            completion_rules=CompletionRules(CompletionType.NONE),
            requires_argument=True
        ))

        self.register_command(CommandInfo(
            name="--file",
            description="Load and add file contents to conversation context. Supports text and multiple binary formats (see rag_config.py).",
            usage="--file path/to/document.pdf",
            execution_order=2,
            completion_rules=CompletionRules(
                CompletionType.FILE_PATH,
                file_extensions=True,
                base_directory=self.working_dir
            ),
            requires_argument=True
        ))

        self.register_command(CommandInfo(
            name="--folder",
            description="Load and add contents of all supported files from a directory to conversation context. Non-recursive.",
            usage="--folder path/to/directory/",
            execution_order=2,
            completion_rules=CompletionRules(
                CompletionType.FILE_PATH,
                file_extensions=True,
                base_directory=self.working_dir
            ),
            requires_argument=True
        ))

        self.register_command(CommandInfo(
            name="--folder-recursive",
            description="Load and add contents of all supported files from a directory and its subdirectories to conversation context. Recursive.",
            usage="--folder-recursive path/to/directory/",
            execution_order=2,
            completion_rules=CompletionRules(
                CompletionType.FILE_PATH,
                file_extensions=True,
                base_directory=self.working_dir
            ),
            requires_argument=True
        ))

        self.register_command(CommandInfo(
            name="--search",
            description="Toggles web search mode on or off. When enabled, user prompts are automatically enhanced with web search results. Search engine (Tavily or SearXNG) can be configured via settings.",
            usage="--search",
            execution_order=1,
            completion_rules=CompletionRules(CompletionType.NONE),
            requires_argument=False
        ))

        self.register_command(CommandInfo(
            name="--search-deep",
            description="Toggles autonomous deep search mode. AI intelligently evaluates research completeness and continues searching until comprehensive coverage is achieved. Search engine (Tavily or SearXNG) can be configured via settings.",
            usage="--search-deep",
            execution_order=1,
            completion_rules=CompletionRules(CompletionType.NONE),
            requires_argument=False
        ))

        self.register_command(CommandInfo(
            name="--search-engine",
            description="Switch search engine for current session. Changes take effect immediately for --search and --search-deep commands.",
            usage="--search-engine tavily",
            execution_order=1,
            completion_rules=CompletionRules(
                CompletionType.SEARCH_ENGINE,
                custom_suggestions=["tavily", "searxng"]
            ),
            requires_argument=True
        ))

        self.register_command(CommandInfo(
            name="--nothink",
            description="Toggles thinking mode off/on for Ollama models.",
            usage="--nothink",
            execution_order=1,
            completion_rules=CompletionRules(CompletionType.NONE),
            requires_argument=False
        ))

        self.register_command(CommandInfo(
            name="--markdown",
            description="Toggle markdown rendering for AI responses.",
            usage="--markdown",
            execution_order=1,
            completion_rules=CompletionRules(CompletionType.NONE),
            requires_argument=False
        ))

        self.register_command(CommandInfo(
            name="--incognito",
            description="Toggles incognito mode on or off. When enabled, no conversation data is saved to log files for privacy.",
            usage="--incognito",
            execution_order=1,
            completion_rules=CompletionRules(CompletionType.NONE),
            requires_argument=False
        ))





        self.register_command(CommandInfo(
            name="--clear",
            description="Clears the current conversation history, starting fresh without any previous messages.",
            usage="--clear",
            execution_order=1,
            completion_rules=CompletionRules(CompletionType.NONE),
            requires_argument=False
        ))

        self.register_command(CommandInfo(
            name="--usage",
            description="Display comprehensive usage statistics including token counts, costs (for OpenAI models), and conversation metrics.",
            usage="--usage",
            execution_order=1,
            completion_rules=CompletionRules(CompletionType.NONE),
            requires_argument=False
        ))

        self.register_command(CommandInfo(
            name="--cb",
            description="Add clipboard contents to conversation context.",
            usage="--cb",
            execution_order=2,
            completion_rules=CompletionRules(CompletionType.NONE),
            requires_argument=False
        ))

        self.register_command(CommandInfo(
            name="--cbl",
            description="Copy latest AI reply to clipboard.",
            usage="--cbl",
            execution_order=2,
            completion_rules=CompletionRules(CompletionType.NONE),
            requires_argument=False
        ))

        # Conversation management commands
        self.register_command(CommandInfo(
            name="--log",
            description="Resume previous conversation. Replaces current conversation.",
            usage="--log name-of-log",
            execution_order=2,
            completion_rules=CompletionRules(
                CompletionType.LOG_FILE,
                file_extensions=[".json"],
                base_directory=os.path.join(self.working_dir, "logs")
            ),
            requires_argument=True
        ))

        self.register_command(CommandInfo(
            name="--logmv",
            description="Renames the current conversation's log file.",
            usage='--logmv (for AI-suggested title) | --logmv your-title (for custom title) | --logmv "your title with spaces"',
            execution_order=2,
            completion_rules=CompletionRules(CompletionType.NONE),
            requires_argument=False
        ))

        self.register_command(CommandInfo(
            name="--logrm",
            description="Deletes the current conversation's log file and clears the conversation history.",
            usage="--logrm",
            execution_order=2,
            completion_rules=CompletionRules(CompletionType.NONE),
            requires_argument=False
        ))

        self.register_command(CommandInfo(
            name="--export-markdown",
            description="Exports the current conversation to markdown format in logs/{instruction-set}/export/ directory.",
            usage="--export-markdown",
            execution_order=2,
            completion_rules=CompletionRules(CompletionType.NONE),
            requires_argument=False
        ))



        # Utility commands



        # RAG (Retrieval-Augmented Generation) commands
        self.register_command(CommandInfo(
            name="--rag",
            description="Activate a specific RAG collection.",
            usage="--rag [collection_name]",
            execution_order=1,
            completion_rules=CompletionRules(CompletionType.RAG_COLLECTION),
            requires_argument=True
        ))

        self.register_command(CommandInfo(
            name="--rag-deactivate",
            description="Deactivate RAG",
            usage="--rag-deactivate",
            execution_order=1,
            completion_rules=CompletionRules(CompletionType.NONE),
            requires_argument=False
        ))

        self.register_command(CommandInfo(
            name="--rag-rebuild",
            description="Rebuild embeddings index for a RAG collection. Uses smart rebuild by default (only processes changed files). Add --force-full to force complete rebuild from scratch.",
            usage="--rag-rebuild collection_name [--force-full]",
            execution_order=1,
            completion_rules=CompletionRules(CompletionType.RAG_COLLECTION),
            requires_argument=True
        ))

        self.register_command(CommandInfo(
            name="--rag-show",
            description="Show relevant chunks from a file in the active RAG collection.",
            usage="--rag-show filename.txt",
            execution_order=1,
            completion_rules=CompletionRules(CompletionType.RAG_COLLECTION_FILE),
            requires_argument=True
        ))

        self.register_command(CommandInfo(
            name="--rag-status",
            description="Show current RAG status and configuration.",
            usage="--rag-status",
            execution_order=1,
            completion_rules=CompletionRules(CompletionType.NONE),
            requires_argument=False
        ))

        self.register_command(CommandInfo(
            name="--rag-test",
            description="Test connection to the current embedding provider.",
            usage="--rag-test",
            execution_order=1,
            completion_rules=CompletionRules(CompletionType.NONE),
            requires_argument=False
        ))

        self.register_command(CommandInfo(
            name="--rag-info",
            description="Show detailed information about the current embedding model and provider configuration.",
            usage="--rag-info",
            execution_order=1,
            completion_rules=CompletionRules(CompletionType.NONE),
            requires_argument=False
        ))



    def register_command(self, command_info: CommandInfo) -> None:
        """Register a new command in the registry"""
        self._commands[command_info.name] = command_info

        # Also register any aliases
        if command_info.aliases:
            for alias in command_info.aliases:
                self._commands[alias] = command_info

    def get_available_commands(self) -> List[str]:
        """Get list of all available command names"""
        # Return only the primary command names (not aliases)
        return sorted([
            cmd for cmd, info in self._commands.items()
            if cmd == info.name  # Only primary names, not aliases
        ])

    def get_command_info(self, command_name: str) -> Optional[CommandInfo]:
        """Get complete information about a command"""
        return self._commands.get(command_name)

    def get_completion_rules(self, command_name: str) -> Optional[CompletionRules]:
        """Get completion rules for a specific command"""
        command_info = self.get_command_info(command_name)
        return command_info.completion_rules if command_info else None

    def command_exists(self, command_name: str) -> bool:
        """Check if a command exists in the registry"""
        return command_name in self._commands

    def requires_argument(self, command_name: str) -> bool:
        """Check if a command requires an argument"""
        command_info = self.get_command_info(command_name)
        return command_info.requires_argument if command_info else False

    def get_commands_by_execution_order(self) -> Dict[int, List[CommandInfo]]:
        """Get commands grouped by their execution order"""
        by_order: Dict[int, List[CommandInfo]] = {}

        # Only include primary commands (not aliases)
        for command_info in self._commands.values():
            if command_info.name in self._commands:  # Is primary command
                order = command_info.execution_order
                if order not in by_order:
                    by_order[order] = []
                if command_info not in by_order[order]:  # Avoid duplicates
                    by_order[order].append(command_info)

        return by_order

    def get_command_description(self, command_name: str) -> str:
        """Get description for a command, with fallback"""
        command_info = self.get_command_info(command_name)
        return command_info.description if command_info else "No description available"

    def get_command_usage(self, command_name: str) -> str:
        """Get usage example for a command, with fallback"""
        command_info = self.get_command_info(command_name)
        return command_info.usage if command_info else f"{command_name} <args>"

    def validate_command_input(self, command_name: str, argument: Optional[str] = None) -> tuple[bool, Optional[str]]:
        """
        Validate command input and return (is_valid, error_message)

        Returns:
            tuple: (True, None) if valid, (False, error_message) if invalid
        """
        command_info = self.get_command_info(command_name)

        if not command_info:
            return False, f"Unknown command: {command_name}"

        if command_info.requires_argument and not argument:
            return False, f"Command {command_name} requires an argument. Usage: {command_info.usage}"

        # Check custom validator if present
        if (command_info.completion_rules.validator and
            argument and
            not command_info.completion_rules.validator(argument)):
            return False, f"Invalid argument for {command_name}: {argument}"

        return True, None

    def get_legacy_command_descriptions(self) -> Dict[str, Dict[str, Any]]:
        """
        Get command descriptions in the legacy format for backward compatibility.
        This allows existing code to continue working while we transition.
        """
        legacy_format = {}

        for command_name, command_info in self._commands.items():
            if command_name == command_info.name:  # Only primary commands
                legacy_format[command_name] = {
                    "description": command_info.description,
                    "usage": command_info.usage,
                    "execution_order": command_info.execution_order
                }

        return legacy_format