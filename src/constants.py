"""
Central constants for Terminal AI Assistant.

All (dumb) magic numbers and configuration values that affect system behavior
are defined here with clear documentation and rationale.
"""

from typing import Optional, List
import re
from datetime import datetime



class CompletionScoringConstants:
    """
    Fuzzy matching score weights for command completion.

    Higher scores indicate better matches. The scoring system uses these weights
    to rank command completion suggestions by relevance.
    """

    # Base scores for different match types (higher = better match)
    EXACT_MATCH_BASE = 1000         # Perfect match at start of string (highest priority)
    WORD_BOUNDARY_BASE = 800        # Match at word boundary (after -, _, space)
    SUBSEQUENCE_BASE = 500          # All characters found in sequence
    SUBSTRING_BASE = 300            # Simple substring match anywhere
    FUZZY_BASE = 100                # Basic fuzzy/overlap match (lowest priority)

    # Bonus multipliers and additions
    EXACT_MATCH_MULTIPLIER = 2      # Extra points for longer exact matches
    CONSECUTIVE_BONUS = 50          # Bonus for consecutive character matches
    FUZZY_THRESHOLD = 0.6           # Minimum overlap ratio for fuzzy matches (60%)


class AudioSystemConstants:
    """
    Audio system configuration values.

    These values control audio quality, performance, and timing behavior.
    Values chosen for balance between quality and system compatibility.
    """

    # Pygame mixer initialization - CD quality settings
    SAMPLE_RATE = 22050           # CD-quality audio sample rate (Hz)
    SAMPLE_SIZE = -16             # 16-bit signed samples (negative = signed)
    CHANNELS = 2                  # Stereo audio channels
    BUFFER_SIZE = 512             # Low-latency buffer size (smaller = more responsive)

    # Timing and monitoring intervals
    PLAYBACK_POLL_INTERVAL = 0.1  # How often to check playback status (seconds)
    CLEANUP_DELAY = 0.5           # Time to wait before cleaning temp files (seconds)
    DEFAULT_VOLUME = 1.0          # Maximum volume level (0.0 to 1.0)


class NetworkConstants:
    """
    Network request timeouts and limits.

    Timeouts prevent hanging requests while allowing reasonable response times.
    Values chosen based on typical network conditions and service response times.
    """

    OLLAMA_CHECK_TIMEOUT = 2     # Seconds to wait for Ollama health check (fast local)
    SUBTITLE_FETCH_TIMEOUT = 10  # Seconds to wait for subtitle downloads
    WEB_REQUEST_TIMEOUT = 30     # Default web request timeout (generous for slow sites)


class ConversationConstants:
    """
    Conversation handling limits and thresholds.

    These values control conversation context management and parsing behavior.
    """

    RECENT_MESSAGES_WINDOW = 10  # Recent messages to include in search context
    THINKING_TAG_MIN_LENGTH = 7  # Minimum chars for complete <think> tag
    CLOSE_TAG_MIN_LENGTH = 8     # Minimum chars for complete </think> tag
    PARTIAL_TAG_MAX_LENGTH = 10  # Maximum partial tag length to buffer


class FilenameConstants:
    """
    Filename and timestamp formatting constants.

    Standardized formats used across the application for consistent file naming.
    """

    # Standardized timestamp format: YYYYMMDD-HHMMSS (e.g., 20241229-143022)
    TIMESTAMP_FORMAT = '%Y%m%d-%H%M%S'


class ColorConstants:
    """
    ANSI color codes for terminal output.

    Provides consistent color scheme across the application.
    Using 24-bit RGB codes for better color accuracy.
    """

    # Thinking text styling (muted gray for less distraction)
    THINKING_GRAY = '\033[38;2;204;204;204m'  # Light gray (#cccccc)
    RESET = '\033[0m'                         # Reset to default color

    # Status and message colors
    ERROR_RED = '\033[91m'                    # Bright red for errors
    WARNING_YELLOW = '\033[93m'               # Bright yellow for warnings
    SUCCESS_GREEN = '\033[92m'                # Bright green for success
    INFO_BLUE = '\033[94m'                    # Bright blue for information


class CacheConstants:
    """
    Caching configuration values.

    Controls cache behavior and timing across different components.
    """

    COMPLETION_CACHE_DURATION = 300.0  # Command completion cache TTL (5 minutes)
    MODEL_CACHE_TTL = 3600             # Model list cache TTL (1 hour)
    COMPLETION_CACHE_SHORT = 2.0       # Short-term completion cache (2 seconds)


class UIConstants:
    """
    User interface configuration values.

    Controls display behavior, formatting, and user interaction.
    """

    MAX_COMPLETION_CACHE_ENTRIES = 100  # LRU cache size for completions
    GRUVBOX_STYLE_MARGIN = 1            # Markdown formatting margin
    RESPONSE_WAIT_SEC = 3               # Seconds to wait before showing "working" indicator
    RESPONSE_WORKING_LABEL = "Working"  # Label used by the animated/static status indicator
    USER_PROMPT_MODEL_MAX_CHARS = 20    # Truncate model name in prompt display


class LLMSettingConstants:
    """
    Model-specific API parameter configurations.

    Different model series have varying API parameter requirements and restrictions.
    GPT-5 models in particular have several parameter changes from earlier models.
    """

    # Temperature settings
    DEFAULT_TEMPERATURE = 0.7     # Default for most models (more focused)
    GPT5_TEMPERATURE = 1.0        # GPT-5 models only support temperature=1.0

    # Token limit parameter names (GPT-5 uses different parameter name)
    DEFAULT_MAX_TOKENS_PARAM = "max_tokens"              # Standard parameter name
    GPT5_MAX_TOKENS_PARAM = "max_completion_tokens"      # GPT-5 parameter name

    # GPT-5 model prefix for detection (all GPT-5 models start with "gpt-5")
    GPT5_MODEL_PREFIX = 'gpt-5'

    # GPT-5 reasoning effort levels
    GPT5_REASONING_EFFORTS = ["minimal", "low", "medium", "high"]
    GPT5_DEFAULT_REASONING_EFFORT = "medium"

    @classmethod
    def get_temperature_for_model(cls, model_name: str, requested_temp: float) -> float:
        """Get appropriate temperature for model, handling GPT-5 restrictions."""
        if model_name.lower().startswith(cls.GPT5_MODEL_PREFIX):
            return cls.GPT5_TEMPERATURE
        return requested_temp

    @classmethod
    def get_max_tokens_param_name(cls, model_name: str) -> str:
        """Get correct max tokens parameter name for model."""
        if model_name.lower().startswith(cls.GPT5_MODEL_PREFIX):
            return cls.GPT5_MAX_TOKENS_PARAM
        return cls.DEFAULT_MAX_TOKENS_PARAM

    @classmethod
    def is_gpt5_model(cls, model_name: str) -> bool:
        """Check if model is from GPT-5 series."""
        return model_name.lower().startswith(cls.GPT5_MODEL_PREFIX)

    @classmethod
    def is_valid_reasoning_effort(cls, effort: str) -> bool:
        """Check if reasoning effort level is valid for GPT-5."""
        return effort.lower() in cls.GPT5_REASONING_EFFORTS

    @classmethod
    def get_reasoning_effort_for_model(cls, model_name: str, requested_effort: str) -> Optional[str]:
        """Get appropriate reasoning effort for model, defaulting for non-GPT-5 models."""
        if cls.is_gpt5_model(model_name) and cls.is_valid_reasoning_effort(requested_effort):
            return requested_effort.lower()
        return cls.GPT5_DEFAULT_REASONING_EFFORT if cls.is_gpt5_model(model_name) else None


class ModelPricingConstants:
    """
    Multi-provider API pricing data (per 1M tokens) as of January 2025.
    Includes OpenAI, Google Gemini, and other providers.

    All prices are in USD per 1 million tokens.
    Sources:
    - OpenAI: https://openai.com/pricing
    - Google: https://ai.google.dev/pricing
    """

    # Multi-provider pricing data (per 1M tokens)
    PRICING_DATA = {
        # Google Gemini models - Cost-effective with excellent performance
        'gemini-2.5-flash': {'input': 0.15, 'output': 0.60},
        'gemini-2.5-pro': {'input': 1.25, 'output': 10.00},
        'gemini-2.0-flash': {'input': 0.10, 'output': 0.40},
        'gemini-1.5-flash': {'input': 0.075, 'output': 0.30},
        'gemini-1.5-pro': {'input': 1.25, 'output': 5.00},

        # GPT-3.5 models - Legacy models for cost-sensitive applications
        'gpt-3.5-turbo': {'input': 3.00, 'output': 6.00},
        'gpt-3.5-turbo-0125': {'input': 0.50, 'output': 1.50},

        # GPT-4o models - Latest generation with vision and multimodal capabilities
        'gpt-4o': {'input': 5.00, 'output': 15.00},
        'gpt-4o-2024-11-20': {'input': 2.50, 'output': 10.00},
        'gpt-4o-2024-08-06': {'input': 2.50, 'output': 10.00},
        'gpt-4o-2024-05-13': {'input': 5.00, 'output': 15.00},

        # GPT-4o-mini models - Cost-effective version of GPT-4o
        'gpt-4o-mini': {'input': 0.15, 'output': 0.60},
        'gpt-4o-mini-2024-07-18': {'input': 0.15, 'output': 0.60},

        # GPT-4 models - Previous generation high-capability models
        'gpt-4': {'input': 30.00, 'output': 60.00},
        'gpt-4-turbo': {'input': 10.00, 'output': 30.00},
        'gpt-4-turbo-2024-04-09': {'input': 10.00, 'output': 30.00},
        'gpt-4-0125-preview': {'input': 10.00, 'output': 30.00},
        'gpt-4-1106-preview': {'input': 10.00, 'output': 30.00},

        # GPT-5 models
        'gpt-5': {'input': 2.00, 'output': 8.00},
        'gpt-5-mini': {'input': 0.40, 'output': 1.60},
        'gpt-5-nano': {'input': 0.10, 'output': 0.40},
        'gpt-5-2024-12-05': {'input': 2.00, 'output': 8.00},
        'gpt-5-preview': {'input': 2.00, 'output': 8.00},
        'gpt-5-mini-2024-12-05': {'input': 0.40, 'output': 1.60},
        'gpt-5-nano-2024-12-05': {'input': 0.10, 'output': 0.40},
    }

    @classmethod
    def get_model_pricing(cls, model_name: str) -> dict:
        """Get pricing information for a specific model."""
        return cls.PRICING_DATA.get(model_name)

    @classmethod
    def calculate_cost(cls, model_name: str, input_tokens: int, output_tokens: int) -> dict:
        """Calculate cost for token usage with a specific model."""
        pricing = cls.get_model_pricing(model_name)
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
            'currency': 'USD'
        }