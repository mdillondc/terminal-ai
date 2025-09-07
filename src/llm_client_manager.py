"""
LLM Client Manager Module

Unified service for handling multiple LLM providers (OpenAI, Ollama, Google Gemini).
Provides a single interface for making chat completions regardless of the underlying provider.
"""

import os
import urllib.request
from typing import Any, Optional, Dict, List
from openai import OpenAI

from settings_manager import SettingsManager
from constants import LLMSettingConstants
from model_manager import ModelManager


class LLMClientManager:
    """
    Manages multiple LLM providers and provides a unified interface for chat completions.
    Automatically detects the provider based on model names and routes requests appropriately.
    """

    def __init__(self, original_openai_client: OpenAI):
        self.original_openai_client = original_openai_client
        self.settings_manager = SettingsManager.getInstance()

        # Cache for provider availability checks
        self._ollama_available = None
        self._google_available = None

        # Cache for created clients
        self._ollama_client = None
        self._google_client = None

        # Model manager for availability-based detection
        self.model_manager = ModelManager(original_openai_client)

    def create_chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = LLMSettingConstants.DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Any:
        """
        Create a chat completion using the appropriate provider for the given model.

        Args:
            model: The model name to use
            messages: List of message dictionaries
            temperature: Temperature for response generation
            max_tokens: Maximum tokens in response
            **kwargs: Additional parameters to pass to the completion API

        Returns:
            Response object from the completion API

        Raises:
            Exception: If the provider is not available or the request fails
        """




        client = self._get_client_for_model(model)

        # Build parameters
        params = {
            "model": model,
            "messages": messages,
            **kwargs
        }

        # Set temperature for all models, with GPT-5 specific handling
        params["temperature"] = LLMSettingConstants.get_temperature_for_model(model, temperature)

        # Handle max_tokens parameter (GPT-5 uses different parameter name)
        if max_tokens is not None:
            max_tokens_param = LLMSettingConstants.get_max_tokens_param_name(model)
            params[max_tokens_param] = max_tokens

        # Handle reasoning_effort parameter (GPT-5 specific)
        if LLMSettingConstants.is_gpt5_model(model):
            reasoning_effort = self.settings_manager.setting_get("gpt5_reasoning_effort")
            if reasoning_effort:
                params["reasoning_effort"] = reasoning_effort

        # Ollama-only: pass context window via extra_body.options.num_ctx
        if self._get_provider_for_model(model) == "ollama":
            try:
                num_ctx = int(self.settings_manager.setting_get("ollama_context_window"))
                # Merge with any provided extra_body/options
                extra_body = dict(params.get("extra_body") or {})
                options = dict(extra_body.get("options") or {})
                options["num_ctx"] = num_ctx
                extra_body["options"] = options
                params["extra_body"] = extra_body
            except Exception:
                # Silently ignore if setting missing or invalid; fall back to model default
                pass

        return client.chat.completions.create(**params)

    def _get_client_for_model(self, model_name: str) -> OpenAI:
        """
        Get the appropriate client for the given model based on availability.

        Args:
            model_name: The model name

        Returns:
            OpenAI client configured for the appropriate provider
        """
        provider = self._get_provider_for_model(model_name)

        if provider == 'ollama':
            client = self._get_ollama_client()
            if not client:
                raise Exception(f"Ollama client not available for model: {model_name}")
            return client
        elif provider == 'google':
            client = self._get_google_client()
            if not client:
                raise Exception(f"Google client not available for model: {model_name}")
            return client

        else:
            # Default to OpenAI
            return self.original_openai_client

    def _get_provider_for_model(self, model_name: str) -> str:
        """Find which provider has this model using availability-based detection"""
        try:
            # Get all available models using existing cache system
            all_models = self.model_manager.get_available_models()

            # Parse the results to find our model
            for model_info in all_models:
                if model_info.get('name') == model_name:
                    source = model_info.get('source', '').lower()

                    # Map source names to our provider names
                    provider_map = {
                        'openai': 'openai',
                        'google': 'google',
                        'ollama': 'ollama'
                    }
                    return provider_map.get(source, 'openai')

        except Exception:
            # Graceful fallback if cache system fails
            pass

        # Default to OpenAI if not found or on error
        return 'openai'



    def _is_ollama_available(self) -> bool:
        """Check if Ollama is available with caching"""
        if self._ollama_available is not None:
            return self._ollama_available

        try:
            base_url = self.settings_manager.setting_get("ollama_base_url")
            url = f"{base_url}/api/version"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=1) as response:
                self._ollama_available = response.status == 200
        except Exception:
            self._ollama_available = False

        return self._ollama_available

    def _is_google_available(self) -> bool:
        """Check if Google API is available with caching"""
        if self._google_available is not None:
            return self._google_available

        self._google_available = bool(os.environ.get("GOOGLE_API_KEY"))
        return self._google_available



    def _get_ollama_client(self) -> Optional[OpenAI]:
        """Get or create Ollama client with caching"""
        if self._ollama_client is not None:
            return self._ollama_client

        try:
            base_url = self.settings_manager.setting_get("ollama_base_url")
            self._ollama_client = OpenAI(
                base_url=f"{base_url}/v1",
                api_key="ollama"  # Ollama doesn't require a real API key
            )
        except Exception:
            self._ollama_client = None

        return self._ollama_client

    def _get_google_client(self) -> Optional[OpenAI]:
        """Get or create Google client with caching"""
        if self._google_client is not None:
            return self._google_client

        try:
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                return None

            self._google_client = OpenAI(
                api_key=api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/"
            )
        except Exception:
            self._google_client = None

        return self._google_client



    def get_provider_for_model(self, model_name: str) -> str:
        """
        Get the provider name for a given model.

        Args:
            model_name: The model name

        Returns:
            Provider name: "ollama", "google", or "openai"
        """
        return self._get_provider_for_model(model_name)

    def create_responses_stream(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = LLMSettingConstants.DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        include_reasoning_summary: bool = True,
        **kwargs
    ) -> Any:
        """
        OpenAI Responses API streaming call (for OpenAI models only).
        Returns the SDK stream iterator (semantic events). This does NOT transform events.
        Callers who need chat-like deltas should adapt their streaming loop accordingly.

        Args:
            model: Model name (must be provided by OpenAI)
            messages: Chat-style messages [{"role": "...", "content": "..."}]
            temperature: Optional temperature (GPT-5 is effectively fixed to 1.0)
            max_tokens: Upper bound for visible output tokens (Responses: max_output_tokens)
            reasoning_effort: minimal|low|medium|high (defaults to configured gpt5_reasoning_effort)
            include_reasoning_summary: If True, request reasoning summaries ("summary": "auto")
            **kwargs: Extra params passed through

        Returns:
            An iterator of Responses API streaming events
        """
        # Must route only to OpenAI
        provider = self._get_provider_for_model(model)
        if provider != "openai":
            raise Exception("create_responses_stream is only supported for OpenAI models")

        client = self._get_client_for_model(model)

        # Map chat messages -> Responses input items
        input_items = self._map_messages_to_responses_input(messages)

        # Reasoning config
        effort = reasoning_effort or self.settings_manager.setting_get("gpt5_reasoning_effort")
        reasoning_obj: Dict[str, Any] = {}
        if effort:
            reasoning_obj["effort"] = effort
        if include_reasoning_summary:
            # Ask API to include reasoning summaries in output
            reasoning_obj["summary"] = "auto"

        # Build parameters for Responses API
        params: Dict[str, Any] = {
            "model": model,
            "input": input_items,
            "stream": True,
            # Avoid server-side storage unless explicitly desired elsewhere
            "store": False,
        }

        # Set temperature (GPT-5 uses 1.0)
        params["temperature"] = LLMSettingConstants.get_temperature_for_model(model, temperature)

        # Map max tokens -> Responses param name
        if max_tokens is not None:
            # Responses uses max_output_tokens for the visible output limit
            params["max_output_tokens"] = max_tokens

        if reasoning_obj:
            params["reasoning"] = reasoning_obj

        # Pass through any additional kwargs (e.g., top_p, etc.) if provided
        params.update(kwargs or {})

        # Return the streaming iterator from the Responses API
        return client.responses.create(**params)

    def _map_messages_to_responses_input(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Convert Chat Completions-style messages to Responses 'input' items.
        - Map 'system' role to 'developer' to align with Responses semantics.
        - Preserve 'user' and 'assistant' as-is.
        """
        input_items: List[Dict[str, str]] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                mapped_role = "developer"
            else:
                mapped_role = role
            input_items.append({"role": mapped_role, "content": content})
        return input_items

    def is_model_available(self, model_name: str) -> bool:
        """
        Check if a model is available from its provider.

        Args:
            model_name: The model name to check

        Returns:
            True if the model's provider is available, False otherwise
        """
        provider = self.get_provider_for_model(model_name)

        if provider == "ollama":
            return self._is_ollama_available()
        elif provider == "google":
            return self._is_google_available()
        else:
            return True  # Assume OpenAI is always available if we have a client

    def clear_cache(self):
        """Clear all cached availability checks and clients"""
        self._ollama_available = None
        self._google_available = None
        self._ollama_client = None
        self._google_client = None

    def get_status(self) -> Dict[str, Any]:
        """
        Get status information about all providers.

        Returns:
            Dictionary with provider availability and configuration info
        """
        status = {
            "openai": {
                "available": True,
                "client": self.original_openai_client is not None
            },
            "ollama": {
                "available": self._is_ollama_available(),
                "base_url": self.settings_manager.setting_get("ollama_base_url")
            },
            "google": {
                "available": self._is_google_available(),
                "api_key_set": bool(os.environ.get("GOOGLE_API_KEY"))
            },

        }

        return status