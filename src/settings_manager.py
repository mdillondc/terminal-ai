import os
import datetime
from typing import Optional, Any, ClassVar, Dict, List
from print_helper import print_md
from constants import FilenameConstants, USER_PROMPT_MODEL_MAX_CHARS


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
        self.name_ai = "AI"  # Name of the AI, example: "Samantha"
        self.name_user = "User"  # Name of the user, example: "John"
        self.instructions = "samantha.md"  # Instructions file for the AI letting it know how to behave
        self.log_file_name = self.generate_new_log_filename()  # Keep track of the log file name for the current session
        self.log_file_location = None  # Full path to current JSON log file (including .json extension), set after first AI response

        # API Settings
        self.model = "gemini-2.5-flash"  # Default model (Google Gemini - reliable and cheaper than OpenAI)
        self.ollama_base_url = "http://localhost:11434"  # Base URL for Ollama API, won't be different unless you specifically configure Ollama to be different
        self.nothink = False  # Disable thinking mode on Ollama models that support it
        self.gpt5_reasoning_effort = "medium"  # GPT-5 reasoning effort level: minimal, low, medium, high (higher effort = slower, but more intelligent response)
        self.gpt5_display_full_reasoning = False  # Whether to display full reasoning summaries during OpenAI Responses streaming for GPT-5 models. If False, show a generic working indicator until visible output starts.

        # Vision Settings
        self.cloud_vision_model = "gpt-5-mini"  # Vision model when using cloud providers (e.g., OpenAI)
        self.ollama_vision_model = "qwen2.5-vl:7b"  # Vision model when using Ollama locally
        self.vision_debug = False  # Print raw vision model output for debugging image analysis
        self.folder_include_images = True  # Include image files (jpg, jpeg, png) when using --folder/--folder-recursive
        self.folder_image_prompt_threshold = 5  # If more than this many images are found, prompt the user to confirm inclusion (Y/n)

        # Search Settings
        self.search = False  # Enable or disable search by default
        self.search_auto = False  # Auto web search toggle (LLM-gated). When enabled, the AI decides per prompt whether to run a regular web search.
        self.search_max_results = 3  # Balance between comprehensive results and API cost/speed
        self.search_engine = "tavily"  # "tavily" or "searxng"
        self.search_deep = False  # Enable or disable deep search
        self.search_deep_auto = False  # Auto deep web search toggle (LLM-gated). When enabled, the AI decides per prompt whether to run autonomous deep research.
        self.search_deep_max_results_per_query = 5  # Maximum number of results per query for deep search
        self.search_context_window = 6  # Number of recent messages to include in search context
        self.search_max_queries = 2  # Maximum number of search queries to generate and execute
        self.searxng_base_url = "http://10.13.0.200:8095, https://some.instance"  # URL(s) for SearXNG instance. NB! Instances must have JSON API enabled. System will iterate until it finds an instance that works, or exhaust the list

        self.searxng_extract_full_content_truncate = 10000  # Maximum words to keep from each extracted URL content (SearXNG always extracts full content, prevents context window overflow)
        self.extraction_method_timeout_seconds = 10  # Maximum seconds each extraction method is allowed before moving to the next one
        self.concurrent_workers = 20  # Number of concurrent threads for URL extraction and search operations
        # Control Jina extraction when using local Ollama models:
        # If False (default), Jina will NOT be used when the active provider is Ollama (preserves local-only privacy).
        # Set to True to allow Jina fallback even with Ollama models (you accept the third‑party privacy tradeoff).
        self.allow_jina_with_ollama = False

        # Image Generation Settings
        self.image_engine = "nano-banana"  # Default image engine for image generation/editing
        self.image_revision_mode = "original"  # "original" or "iterative" - how image edits build on each other
        self.image_generate_mode = False  # True when --image-generate mode is active
        self.image_edit_mode = False  # True when --image-edit mode is active
        self.image_current_working_file = None  # Current working image file path for iterative mode

        # Markdown streamdown settings
        self.markdown = True  # Enable markdown parsing and rendering
        self.markdown_settings = ['sd', '-b', '0.1,0.5,0.5', '-c', '[style]\nMargin = 1']  # Gruvbox theme for streamdown/markdown formatting
        self.rag_enable_hybrid_search = True   # Combine semantic (embedding) and keyword (BM25) search for better document retrieval
        self.rag_temporal_boost_months = 6  # Boost chunks from last N months for recent queries

        # RAG Settings
        self.cloud_embedding_model = "text-embedding-3-large"  # Embedding model for cloud providers (e.g., OpenAI)
        self.ollama_embedding_model = "snowflake-arctic-embed2:latest"  # Embedding model for Ollama
        self.rag_chunk_size = 400   # Number of tokens per document chunk - balance between context and precision
        self.rag_chunk_overlap = 80  # Tokens shared between adjacent chunks to preserve context across boundaries
        self.rag_batch_size = 16  # Number of text chunks to process simultaneously for embedding generation
        self.rag_top_k = 8   # Maximum number of most relevant document chunks to return per query
        self.rag_active_collection = None  # Currently active document collection name (None = no RAG active)
        self.rag_enable_search_transparency = True  # Show search process information
        self.rag_enable_result_diversity = True  # Prevent over-representation from single sources
        self.rag_max_chunks_per_source = 4  # Maximum chunks to return from same source document

        # RAG PDF Vision Extraction (scanned/image-only pages)
        self.rag_pdf_vision_max_pages = 999  # Maximum number of PDF pages per file to process via vision fallback (0 disables)
        self.rag_pdf_vision_dpi = 180  # Rendering DPI for PDF pages converted to images for vision

        # Privacy Settings
        self.incognito = False  # Enable or disable conversation logging

    def generate_new_log_filename(self) -> str:
        """Generate a new log filename using standardized YYYYMMDD-HHMMSS format"""
        return f"{datetime.datetime.now().strftime(FilenameConstants.TIMESTAMP_FORMAT)}.md"

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

        else:
            raise KeyError(f"Setting {key} not found in Settings")

    def get_enabled_toggles(self) -> str:
        """Get a formatted string of enabled toggles for display next to user name"""
        enabled_toggles = []

        # Instructions
        if self.instructions:
            instruction_name = self.instructions.rsplit('.', 1)[0]
            enabled_toggles.append(instruction_name)

        if self.search:
            enabled_toggles.append("search")
        if self.search_auto:
            enabled_toggles.append("search-auto")
        if self.search_deep:
            enabled_toggles.append("search-deep")
        if self.search_deep_auto:
            enabled_toggles.append("search-deep-auto")
        if self.nothink:
            enabled_toggles.append("nothink")
        if self.incognito:
            enabled_toggles.append("incognito")
        if self.markdown:
            enabled_toggles.append("markdown")

        if self.rag_active_collection:
            enabled_toggles.append(f"rag {self.rag_active_collection}")

        # Image mode indicators
        if getattr(self, 'image_edit_mode', False):
            # Show current working image basename if available
            try:
                import os
                filename = os.path.basename(self.image_current_working_file) if self.image_current_working_file else None
            except Exception:
                filename = None
            if filename:
                enabled_toggles.append(f"image-edit ({filename})")
            else:
                enabled_toggles.append("image-edit")
        elif getattr(self, 'image_generate_mode', False):
            enabled_toggles.append("image-generate")

        # Add model name as the last element
        if self.model:
            model_display = self.model
            if len(model_display) > USER_PROMPT_MODEL_MAX_CHARS:
                model_display = model_display[:USER_PROMPT_MODEL_MAX_CHARS] + "..."
            # If using GPT-5, show reasoning effort next to model
            try:
                if model_display.lower().startswith('gpt-5'):
                    effort = getattr(self, 'gpt5_reasoning_effort', None)
                    if effort:
                        model_display = f"{model_display} {effort}"
            except Exception:
                pass
            enabled_toggles.append(model_display)

        if enabled_toggles:
            return f" ({', '.join(enabled_toggles)})"
        return ""

    def get_ai_name_with_instructions(self) -> str:
        """Get AI name and append current model (with GPT-5 effort) similar to user toggles."""
        ai_name = self.name_ai

        try:
            model = self.setting_get("model")
            if model:
                model_display = model
                if len(model_display) > USER_PROMPT_MODEL_MAX_CHARS:
                    model_display = model_display[:USER_PROMPT_MODEL_MAX_CHARS] + "..."
                # If using GPT-5, show reasoning effort next to model
                try:
                    if model_display.lower().startswith('gpt-5'):
                        effort = getattr(self, 'gpt5_reasoning_effort', None)
                        if effort:
                            model_display = f"{model_display} {effort}"
                except Exception:
                    pass
                ai_name = f"{ai_name} ({model_display})"
        except Exception:
            pass

        return ai_name

    def display_model_info(self, context: str = "simple", provider: str = None, llm_client_manager = None) -> None:
        """
        Display model information with appropriate formatting for different contexts.

        Args:
            context: Display context - "switch", "simple", or "config"
            provider: Model provider (openai, google, ollama)
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
                if attr_name not in ['working_dir', 'log_file_location']:
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



