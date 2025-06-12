"""
Command Registry System for Samantha AI Assistant

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
    TTS_MODEL = "tts_model"  # TTS model suggestions
    TTS_VOICE = "tts_voice"  # TTS voice suggestions
    RAG_COLLECTION = "rag_collection"  # RAG collection name suggestions
    RAG_COLLECTION_FILE = "rag_collection_file"  # Files in active RAG collection
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
    name: str  # Command name (e.g., "--help")
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
            name="--help",
            description="Displays a list of all available commands along with brief descriptions. You can chain multiple commands together.",
            usage="--help",
            execution_order=1,
            completion_rules=CompletionRules(CompletionType.NONE),
            requires_argument=False
        ))

        self.register_command(CommandInfo(
            name="--model",
            description="Switches the underlying AI model. Use this to select a different AI model for processing commands (dynamically fetched from OpenAI, Google, and Ollama APIs).",
            usage="--model gpt-4",
            execution_order=1,
            completion_rules=CompletionRules(
                CompletionType.MODEL_NAME,
                custom_suggestions=["gpt-4", "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]  # Fallback suggestions if APIs unavailable
            ),
            requires_argument=True
        ))

        self.register_command(CommandInfo(
            name="--refresh-models",
            description="Clears the model cache and forces a fresh fetch of available models from OpenAI and Google APIs.",
            usage="--refresh-models",
            execution_order=1,
            completion_rules=CompletionRules(CompletionType.NONE),
            requires_argument=False
        ))

        # Content processing commands
        self.register_command(CommandInfo(
            name="--instructions",
            description="Activates a specific skill for Samantha. Skills are specialized functions like summarizing YouTube videos.",
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
            description="Extract transcript, send to AI. Use appropriate instructions to e.g. summarize video.",
            usage="--youtube https://www.youtube.com/watch?v=9gCHMuC7T40",
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
            name="--search",
            description="Toggles web search mode on or off. When enabled, user prompts are automatically enhanced with Tavily web search results.",
            usage="--search",
            execution_order=1,
            completion_rules=CompletionRules(CompletionType.NONE),
            requires_argument=False
        ))

        self.register_command(CommandInfo(
            name="--nothink",
            description="Toggles nothink mode on or off. When enabled, user messages are prepended with '/nothink ' to disable thinking on thinking Ollama models.",
            usage="--nothink",
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
            name="--cb",
            description="Use clipboard contents as input.",
            usage="--cb",
            execution_order=2,
            completion_rules=CompletionRules(CompletionType.NONE),
            requires_argument=False
        ))

        # Conversation management commands
        self.register_command(CommandInfo(
            name="--log",
            description="Resume previous conversation. Replaces current conversation.",
            usage="--log name-of-log.md",
            execution_order=2,
            completion_rules=CompletionRules(
                CompletionType.LOG_FILE,
                file_extensions=[".md"],
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

        # TTS commands
        self.register_command(CommandInfo(
            name="--tts",
            description="Toggles the Text-to-Speech feature on or off, enabling or disabling spoken responses.",
            usage="--tts",
            execution_order=1,
            completion_rules=CompletionRules(CompletionType.NONE),
            requires_argument=False
        ))

        self.register_command(CommandInfo(
            name="--tts-model",
            description="Changes the Text-to-Speech model to alter the voice synthesis.",
            usage="--tts-model tts-1-hd",
            execution_order=1,
            completion_rules=CompletionRules(
                CompletionType.TTS_MODEL,
                custom_suggestions=["tts-1", "tts-1-hd", "gpt-4o-mini-tts"]
            ),
            requires_argument=True
        ))

        self.register_command(CommandInfo(
            name="--tts-voice",
            description="Selects a different voice for Text-to-Speech responses.",
            usage="--tts-voice onyx",
            execution_order=1,
            completion_rules=CompletionRules(
                CompletionType.TTS_VOICE,
                custom_suggestions=["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
            ),
            requires_argument=True
        ))

        self.register_command(CommandInfo(
            name="--tts-save-as-mp3",
            description="[EXPERIMENTAL] Save TTS responses to a MP3 files.",
            usage="--tts-save-as-mp3",
            execution_order=1,
            completion_rules=CompletionRules(CompletionType.NONE),
            requires_argument=False
        ))

        # Utility commands
        self.register_command(CommandInfo(
            name="--text",
            description="Allows you to send a text input directly as a command, useful for scripting or automation.",
            usage="python3 main.py --input '--instructions summary.md --youtube https://www.youtube.com/watch?v=9gCHMuC7T40'",
            execution_order=2,
            completion_rules=CompletionRules(CompletionType.NONE),
            requires_argument=True
        ))

        # RAG (Retrieval-Augmented Generation) commands
        self.register_command(CommandInfo(
            name="--rag",
            description="List available RAG collections or activate a specific collection. Use 'off' to deactivate.",
            usage="--rag [collection_name|off]",
            execution_order=1,
            completion_rules=CompletionRules(CompletionType.RAG_COLLECTION),
            requires_argument=False
        ))

        self.register_command(CommandInfo(
            name="--rag-build",
            description="Build/rebuild embeddings index for a RAG collection.",
            usage="--rag-build collection_name",
            execution_order=1,
            completion_rules=CompletionRules(CompletionType.RAG_COLLECTION),
            requires_argument=True
        ))

        self.register_command(CommandInfo(
            name="--rag-refresh",
            description="Force refresh/rebuild embeddings index for a RAG collection.",
            usage="--rag-refresh collection_name",
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