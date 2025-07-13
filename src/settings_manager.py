
import os
import time
import datetime
from typing import Optional, Any, ClassVar, Dict, List
from command_registry import CommandRegistry
from print_helper import print_md


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

        # Voice Settings
        self.tts = False
        self.tts_model = "gpt-4o-mini-tts"
        self.tts_voice = "echo"
        self.tts_save_mp3 = False
        self.stt = False
        self.stt_waiting_msg = True  # Do not change

        # Search Settings
        self.search = False
        self.search_max_results = 3  # Maximum number of search results per query
        self.search_engine = "tavily"  # "tavily" or "searxng"
        self.searxng_base_url = "http://10.13.0.200:8095"  # URL for SearXNG instance

        # Deep Search Settings
        self.search_deep = False
        self.search_deep_max_queries = 35  # Maximum number of search queries for deep search (safety net)
        self.search_deep_max_results_per_query = 5  # Maximum number of results per query for deep search (safety net)
    
        # Markdown streamdown settings
        self.markdown_settings = ['sd', '-b', '0.1,0.5,0.5', '-c', '[style]\nMargin = 1']
        self.rag_enable_hybrid_search = True
        self.rag_temporal_boost_months = 6  # Boost chunks from last N months for recent queries

        # RAG Settings
        self.embedding_provider = "ollama"  # "openai" or "ollama"
        self.openai_embedding_model = "text-embedding-3-small"
        self.ollama_embedding_model = "snowflake-arctic-embed2:latest"
        self.rag_chunk_size = 400
        self.rag_chunk_overlap = 80
        self.rag_batch_size = 16
        self.rag_top_k = 8
        self.rag_active_collection = None
        self.rag_enable_search_transparency = True  # Show search process information
        self.rag_enable_result_diversity = True  # Prevent over-representation from single sources
        self.rag_max_chunks_per_source = 4  # Maximum chunks to return from same source document

        # Search Context Settings
        self.search_context_window = 6  # Number of recent messages to include in search context
        self.search_context_char_limit = 2000  # Character limit for truncating long messages in search context
        self.search_max_queries = 2  # Maximum number of search queries to generate and execute

        # Thinking Settings
        self.nothink = False

        # Privacy Settings
        self.incognito = False

        # Navigation Settings
        self.scroll = False
        self.scroll_lines = 1

        # General Settings
        self.name_ai = "AI"
        self.name_user = "User"
        # self.default_input = "Hi"
        self.instructions = "samantha.md"
        # self.log_file_name = f"{int(time.time())}.md"
        # self.log_file_name = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S.md")
        self.log_file_name = self.generate_new_log_filename()

        # Display Settings
        self.markdown = True  # Enable markdown parsing and rendering
        self.log_file_location = None  # Do not change

        # Command Registry - centralized command management
        self.command_registry = CommandRegistry(self.working_dir)

    def generate_new_log_filename(self) -> str:
        """Generate a new log filename using date + timestamp format"""
        return f"{datetime.datetime.now().strftime('%Y-%m-%d')}_{int(time.time())}.md"

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

        # Instructions
        if self.instructions:
            instruction_name = self.instructions.rsplit('.', 1)[0]
            enabled_toggles.append(instruction_name)

        if self.tts:
            enabled_toggles.append("tts")
        if self.stt:
            enabled_toggles.append("stt")
        if self.search:
            enabled_toggles.append("search")
        if self.search_deep:
            enabled_toggles.append("search-deep")
        if self.nothink:
            enabled_toggles.append("nothink")
        if self.incognito:
            enabled_toggles.append("incognito")
        if self.markdown:
            enabled_toggles.append("markdown")

        if self.rag_active_collection:
            enabled_toggles.append(f"rag {self.rag_active_collection}")

        if enabled_toggles:
            return f" ({', '.join(enabled_toggles)})"
        return ""

    def get_ai_name_with_instructions(self) -> str:
        """Get AI name (instructions now displayed in user prompt via get_enabled_toggles)"""
        return self.name_ai

    def display_model_info(self, context: str = "simple", provider: str = None, llm_client_manager = None) -> None:
        """
        Display model information with appropriate formatting for different contexts.

        Args:
            context: Display context - "switch", "simple", or "config"
            provider: Model provider (openai, google, anthropic, ollama)
            llm_client_manager: Optional client manager for Ollama availability checking
        """
        model = self.setting_get("model")

        if context == "switch":
            # Full messaging for command switches
            if provider == "ollama":
                ollama_url = self.setting_get("ollama_base_url")
                if llm_client_manager and llm_client_manager._is_ollama_available():
                    ollama_text = f"Model: {model}\n"
                    ollama_text += f"    Running locally via Ollama at {ollama_url}"
                    print_md(ollama_text)
                else:
                    warning_text = f"Warning: Selected Ollama model '{model}' but Ollama not available\n"
                    warning_text += f"    Make sure Ollama is running at {ollama_url}"
                    print_md(warning_text)
            elif provider == "google":
                google_text = f"Model: {model}\n"
                google_text += "    https://ai.google.dev/gemini-api/docs/models"
                print_md(google_text)
            elif provider == "anthropic":
                anthropic_text = f"Model: {model}\n"
                anthropic_text += "    https://docs.anthropic.com/en/docs/about-claude/models"
                print_md(anthropic_text)
            else:
                openai_text = f"Model: {model}\n"
                openai_text += "    https://platform.openai.com/docs/models"
                print_md(openai_text)
        else:
            # Simplified messaging for startup/config
            print_md(f"Model: {model}")

    def get_config_path(self) -> str:
        """Get the path to the config file"""
        home = os.path.expanduser("~")
        return os.path.join(home, ".config", "terminal-ai", "config")

    def parse_config_file(self, config_path: str) -> Dict[str, str]:
        """Parse the plain text config file with comments support"""
        config_data = {}

        try:
            with open(config_path, 'r') as file:
                for line_num, line in enumerate(file, 1):
                    line = line.strip()

                    # Skip empty lines and comments
                    if not line or line.startswith('#'):
                        continue

                    # Handle inline comments
                    if '#' in line:
                        line = line.split('#')[0].strip()

                    # Parse key = value
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()

                        if key and value:
                            config_data[key] = value

        except FileNotFoundError:
            # Config file doesn't exist - that's fine, use defaults
            pass
        except Exception as e:
            print_md(f"Error reading config file: {e}")

        return config_data

    def get_valid_settings(self) -> List[str]:
        """Get list of valid setting names that can be configured"""
        valid_settings = []

        # Get all attributes that don't start with underscore and aren't methods
        for attr_name in dir(self):
            if not attr_name.startswith('_') and not callable(getattr(self, attr_name)):
                # Exclude special attributes that shouldn't be configurable
                if attr_name not in ['command_registry', 'working_dir', 'log_file_location']:
                    valid_settings.append(attr_name)

        return valid_settings

    def convert_config_value(self, setting_name: str, config_value: str) -> Any:
        """Convert config string value to appropriate type based on original setting"""
        original_value = getattr(self, setting_name)
        original_type = type(original_value)

        # Handle None values - keep as string
        if original_value is None:
            return config_value

        # Convert based on original type
        if original_type == bool:
            # Handle various boolean representations
            lower_value = config_value.lower()
            if lower_value in ['true', '1', 'yes', 'on']:
                return True
            elif lower_value in ['false', '0', 'no', 'off']:
                return False
            else:
                return bool(config_value)  # Fallback

        elif original_type == int:
            try:
                return int(config_value)
            except ValueError:
                return original_value  # Keep original if conversion fails

        elif original_type == float:
            try:
                return float(config_value)
            except ValueError:
                return original_value  # Keep original if conversion fails

        else:
            # String or other types
            return config_value

    def load_config(self) -> None:
        """Load and apply config file overrides with validation (public method for main.py)"""
        config_path = self.get_config_path()
        config_data = self.parse_config_file(config_path)

        if not config_data:
            return  # No config to process

        # print_md(f"Apply settings from: {config_path}:")

        valid_settings = self.get_valid_settings()
        invalid_settings = []

        # Process each config setting and collect output
        config_outputs = []
        for setting_name, config_value in config_data.items():
            if setting_name in valid_settings:
                # Valid setting - convert and apply
                converted_value = self.convert_config_value(setting_name, config_value)
                setattr(self, setting_name, converted_value)

                # Collect what's being overridden with source attribution
                config_source = "~/.config/terminal-ai/config"
                if setting_name == "model":
                    config_outputs.append(f"Model: {converted_value} ({config_source})")
                elif setting_name == "instructions":
                    instruction_name = converted_value.rsplit('.', 1)[0] if converted_value else converted_value
                    config_outputs.append(f"Instructions: {instruction_name} ({config_source})")
                else:
                    config_outputs.append(f"Set {setting_name}: {converted_value} ({config_source})")
            else:
                # Invalid setting - track for warning
                invalid_settings.append(setting_name)

        # Print all config settings at once for faster output
        if config_outputs:
            print_md('\n'.join(config_outputs))

        # Warn about invalid settings
        if invalid_settings:
            invalid_list = ', '.join(invalid_settings)
            print_md(f"Config contains invalid setting(s): {invalid_list}")



    @staticmethod
    def read_file(filename: str) -> str:
        with open(filename, "r") as file:
            return file.read()



