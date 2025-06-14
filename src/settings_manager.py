import os
import time
import datetime
from typing import Optional, Any, ClassVar
from command_registry import CommandRegistry


class SettingsManager:
    _instance: ClassVar[Optional['SettingsManager']] = None

    @staticmethod
    def getInstance() -> 'SettingsManager':
        if SettingsManager._instance is None:
            SettingsManager._instance = SettingsManager()
        return SettingsManager._instance

    def __init__(self) -> None:
        if SettingsManager._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            SettingsManager._instance = self
            self.assign_defaults()

    def assign_defaults(self) -> None:
        # Global settings
        self.working_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

        # API Settings
        self.model = "gpt-4.1-mini"  # Default model (provider detected automatically)
        self.ollama_base_url = "http://localhost:11434"

        # Cache Settings
        self.openai_models_cache_hours = 168  # 7 days
        self.google_models_cache_hours = 72   # 3 days

        # Voice Settings
        self.tts = False
        self.tts_model = "gpt-4o-mini-tts"
        self.tts_voice = "shimmer"
        self.tts_save_mp3 = False  # Experimental
        self.stt = False
        self.stt_waiting_msg = True  # Do not change

        # Search Settings
        self.search = False

        # RAG Settings
        self.embedding_provider = "ollama"  # "openai" or "ollama"
        self.openai_embedding_model = "text-embedding-3-small"
        self.ollama_embedding_model = "snowflake-arctic-embed2:latest"
        self.rag_chunk_size = 400
        self.rag_chunk_overlap = 80
        self.rag_top_k = 8
        self.rag_active_collection = None

        # Hybrid Search Settings
        self.rag_enable_hybrid_search = True
        self.rag_semantic_weight = 0.6  # Weight for semantic similarity (0.0-1.0)
        self.rag_temporal_weight = 0.3  # Weight for temporal recency (0.0-1.0)
        self.rag_keyword_weight = 0.1   # Weight for BM25 keyword search (0.0-1.0)
        self.rag_temporal_boost_months = 6  # Boost chunks from last N months
        self.rag_temporal_detection_threshold = 0.15  # Confidence threshold for temporal query detection

        # LLM Query Analysis Settings (uses self.model)
        self.rag_enable_llm_query_analysis = True  # Use LLM for intelligent query analysis
        self.rag_query_analysis_temperature = 0.1  # Temperature for analysis (low for consistency)
        self.rag_query_analysis_cache_duration = 300  # Cache duration in seconds (5 minutes)
        self.rag_analysis_confidence_threshold = 0.7  # Minimum confidence for LLM analysis

        # Context Management Settings
        self.rag_max_context_tokens = 10000
        self.rag_enable_context_management = True
        self.conversation_history_limit = 50000
        self.context_management_strategy = "generous"  # "strict", "balanced", or "generous"

        # Search Context Settings
        self.search_context_window = 6  # Number of recent messages to include in search context
        self.search_context_char_limit = 1000  # Character limit for truncating long messages in search context
        self.search_max_queries = 2  # Maximum number of search queries to generate and execute

        # Thinking Settings
        self.nothink = False

        # Privacy Settings
        self.incognito = False

        # Execution Settings
        self.execute_enabled = False
        self.execute_require_permission = True

        # General Settings
        self.name_ai = "Samantha"
        self.name_user = "User"
        # self.default_input = "Hi"
        self.instructions = "samantha.md"
        # self.log_file_name = f"{int(time.time())}.md"
        # self.log_file_name = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S.md")
        self.log_file_name = (
            f"{datetime.datetime.now().strftime('%Y-%m-%d')}_{int(time.time())}.md"
        )
        self.log_file_location = None  # Do not change

        # Command Registry - centralized command management
        self.command_registry = CommandRegistry(self.working_dir)

    def setting_set(self, key: str, value: Any) -> None:
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            raise KeyError(f"Setting {key} not found in Settings")

    def setting_get(self, key: str) -> Any:
        if hasattr(self, key):
            value = getattr(self, key)
            if key == "abc":
                return self.read_file(value)
            else:
                return value
        elif key == "command_descriptions":
            # Backward compatibility: return legacy format from command registry
            return self.command_registry.get_legacy_command_descriptions()
        else:
            raise KeyError(f"Setting {key} not found in Settings")

    def get_enabled_toggles(self) -> str:
        """Get a formatted string of enabled toggles for display next to user name"""
        enabled_toggles = []

        if self.tts:
            enabled_toggles.append("tts")
        if self.stt:
            enabled_toggles.append("stt")
        if self.search:
            enabled_toggles.append("search")
        if self.nothink:
            enabled_toggles.append("nothink")
        if self.incognito:
            enabled_toggles.append("incognito")
        if self.execute_enabled:
            enabled_toggles.append("execute")
        if self.rag_active_collection:
            enabled_toggles.append(f"rag {self.rag_active_collection}")

        if enabled_toggles:
            return f" ({', '.join(enabled_toggles)})"
        return ""

    @staticmethod
    def read_file(filename: str) -> str:
        with open(filename, "r") as file:
            return file.read()



