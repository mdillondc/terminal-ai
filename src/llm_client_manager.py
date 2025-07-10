"""
LLM Client Manager Module

Unified service for handling multiple LLM providers (OpenAI, Ollama, Google Gemini).
Provides a single interface for making chat completions regardless of the underlying provider.
"""

import os
import urllib.request
from typing import Any, Optional, Dict, List
from openai import OpenAI
from openai import BadRequestError
from settings_manager import SettingsManager
from print_helper import print_md


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
        self._anthropic_available = None

        # Cache for created clients
        self._ollama_client = None
        self._google_client = None
        self._anthropic_client = None

    def create_chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
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
        # Check if this is an o3 model that requires Responses API
        if self._is_o3_model(model):
            return self._create_response_completion(model, messages, max_tokens, **kwargs)

        # Handle Anthropic models differently
        if self._is_anthropic_model(model):
            return self._create_anthropic_completion(model, messages, temperature, max_tokens, **kwargs)

        client = self._get_client_for_model(model)

        # Build parameters
        params = {
            "model": model,
            "messages": messages,
            **kwargs
        }

        # Handle temperature parameter based on model type
        if self._is_o1_model(model):
            # o1 models don't support temperature parameter at all
            pass
        else:
            # Other models support custom temperature
            params["temperature"] = temperature

        if max_tokens is not None:
            params["max_tokens"] = max_tokens

        return client.chat.completions.create(**params)

    def _get_client_for_model(self, model_name: str) -> OpenAI:
        """
        Get the appropriate client for the given model.

        Args:
            model_name: The model name

        Returns:
            OpenAI client configured for the appropriate provider
        """
        if self._is_ollama_model(model_name):
            client = self._get_ollama_client()
            if not client:
                raise Exception(f"Ollama client not available for model: {model_name}")
            return client
        elif self._is_google_model(model_name):
            client = self._get_google_client()
            if not client:
                raise Exception(f"Google client not available for model: {model_name}")
            return client
        elif self._is_anthropic_model(model_name):
            client = self._get_anthropic_client()
            if not client:
                raise Exception(f"Anthropic client not available for model: {model_name}")
            return client
        else:
            # Default to OpenAI
            return self.original_openai_client

    def _is_ollama_model(self, model_name: str) -> bool:
        """Detect if a model is from Ollama based on naming patterns"""
        if not self._is_ollama_available():
            return False

        # Common Ollama model patterns
        ollama_patterns = [
            ':',  # Most Ollama models have tags like "llama3.2:latest"
            'llama', 'mistral', 'qwen', 'codellama', 'phi', 'gemma',
            'tinyllama', 'vicuna', 'orca', 'openchat', 'starling'
        ]

        model_lower = model_name.lower()
        return any(pattern in model_lower for pattern in ollama_patterns)

    def _is_google_model(self, model_name: str) -> bool:
        """Detect if a model is from Google based on naming patterns"""
        if not self._is_google_available():
            return False

        # Common Google model patterns
        google_patterns = ['gemini', 'palm', 'bard']
        model_lower = model_name.lower()
        return any(pattern in model_lower for pattern in google_patterns)

    def _is_anthropic_model(self, model_name: str) -> bool:
        """Detect if a model is from Anthropic based on naming patterns"""
        if not self._is_anthropic_available():
            return False

        # Common Anthropic model patterns
        anthropic_patterns = ['claude']
        model_lower = model_name.lower()
        return any(pattern in model_lower for pattern in anthropic_patterns)

    def _is_claude_extended_thinking_model(self, model_name: str) -> bool:
        """Detect if a Claude model supports extended thinking"""
        model_lower = model_name.lower()
        # Extended thinking is supported in Claude 4, Claude 3.7
        extended_thinking_patterns = [
            'claude-opus-4', 'claude-sonnet-4', 'claude-4',
            'claude-3-7-sonnet'
        ]
        return any(pattern in model_lower for pattern in extended_thinking_patterns)

    def _is_o1_model(self, model_name: str) -> bool:
        """Detect if a model is from OpenAI's o1 series (doesn't support temperature)"""
        model_lower = model_name.lower()
        # Known o1 model names and patterns
        o1_exact_names = ['o1', 'o1-preview', 'o1-mini', 'o1-pro']
        o1_patterns = ['o1-2024-', 'o1_2024_', 'o1-mini-', 'o1_mini_', 'o1-pro-', 'o1_pro_']

        return (model_lower in o1_exact_names or
                any(pattern in model_lower for pattern in o1_patterns))

    def _is_o3_model(self, model_name: str) -> bool:
        """Detect if a model is from OpenAI's o3 series (uses Responses API)"""
        model_lower = model_name.lower()
        # Known o3 model names and patterns
        o3_exact_names = ['o3', 'o3-mini', 'o3-pro']
        o3_patterns = ['o3-2025-', 'o3_2025_', 'o3-mini-2025-', 'o3_mini_2025_', 'o3-pro-', 'o3_pro_']

        return (model_lower in o3_exact_names or
                any(pattern in model_lower for pattern in o3_patterns))

    def _create_response_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Any:
        """
        Create a response using OpenAI's Responses API for o3 models.

        Args:
            model: The model name to use
            messages: List of message dictionaries
            max_tokens: Maximum tokens in response
            **kwargs: Additional parameters

        Returns:
            Response object from the Responses API, wrapped to be compatible with Chat Completions API
        """
        client = self._get_client_for_model(model)

        # Extract stream parameter
        is_stream = kwargs.pop('stream', False)

        # Convert messages to the input format expected by Responses API
        input_messages = []
        for msg in messages:
            if msg['role'] == 'system':
                # Convert system messages to developer messages
                input_messages.append({
                    "role": "developer",
                    "content": [{"type": "input_text", "text": msg['content']}]
                })
            else:
                input_messages.append({
                    "role": msg['role'],
                    "content": [{"type": "input_text", "text": msg['content']}]
                })

        # Build parameters for Responses API
        params = {
            "model": model,
            "input": input_messages,
            "reasoning": {"effort": "medium"},  # Default reasoning effort
            "store": False,  # Don't store the conversation
            "stream": is_stream,
            **kwargs
        }

        if max_tokens is not None:
            params["max_output_tokens"] = max_tokens

        try:
            response = client.responses.create(**params)
        except BadRequestError as e:
            if "organization must be verified" in str(e).lower():
                print_md(f"Organization verification required for {model} model streaming.")
                print_md("To enable streaming with o3 models:")
                print_md("1. Go to https://platform.openai.com/settings/organization/general")
                print_md("2. Click 'Verify Organization'")
                print_md("3. Wait up to 15 minutes for access to propagate")

                # Return a mock response indicating the error
                class MockErrorResponse:
                    def __init__(self):
                        self.content = "Organization verification required for this model. Please verify your organization in OpenAI settings."

                return MockErrorResponse()
            else:
                # Re-raise other BadRequestErrors
                raise

        if is_stream:
            # Wrap the streaming response to be compatible with Chat Completions API format
            return self._wrap_responses_stream(response)
        else:
            # Wrap the non-streaming response
            return self._wrap_responses_response(response)

    def _wrap_responses_stream(self, response_stream):
        """
        Wrap Responses API streaming response to be compatible with Chat Completions API format.
        """
        # Define mock classes once to avoid duplication
        class MockDelta:
            def __init__(self):
                self.content = None

        class MockChoice:
            def __init__(self):
                self.delta = MockDelta()

        class MockChunk:
            def __init__(self):
                self.choices = [MockChoice()]

        class ChatCompletionWrapper:
            def __init__(self, response_stream):
                self.response_stream = response_stream

            def __iter__(self):
                return self

            def __next__(self):
                try:
                    event = next(self.response_stream)
                    chunk = MockChunk()

                    # Extract text content from various event types
                    if hasattr(event, 'type'):
                        if event.type == 'response.output_text.delta' and hasattr(event, 'delta'):
                            chunk.choices[0].delta.content = event.delta
                        elif hasattr(event, 'data') and hasattr(event.data, 'delta'):
                            chunk.choices[0].delta.content = event.data.delta

                    return chunk

                except StopIteration:
                    raise StopIteration
                except Exception:
                    # Return empty chunk on error
                    return MockChunk()

        return ChatCompletionWrapper(response_stream)

    def _create_anthropic_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Any:
        """
        Create a completion using Anthropic's API with proper message format conversion.
        """
        client = self._get_anthropic_client()
        if not client:
            raise Exception(f"Anthropic client not available for model: {model}")

        # Convert OpenAI format messages to Anthropic format
        system_message = None
        anthropic_messages = []

        for msg in messages:
            if msg['role'] == 'system':
                system_message = msg['content']
            else:
                anthropic_messages.append({
                    "role": msg['role'],
                    "content": msg['content']
                })

        # Build parameters for Anthropic API
        params = {
            "model": model,
            "messages": anthropic_messages,
            "temperature": temperature,
            **kwargs
        }

        if system_message:
            params["system"] = system_message

        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        else:
            # Anthropic requires max_tokens to be set
            params["max_tokens"] = 4096

        # Enable extended thinking for supported models
        if self._is_claude_extended_thinking_model(model):
            params["thinking"] = {
                "type": "enabled",
                "budget_tokens": 12000  # Allow up to 12k tokens for thinking
            }

        # Check if streaming is requested
        is_stream = kwargs.get('stream', False)

        if is_stream:
            response = client.messages.create(**params)
            return self._wrap_anthropic_stream(response)
        else:
            response = client.messages.create(**params)
            return self._wrap_anthropic_response(response)

    def _wrap_anthropic_stream(self, response_stream):
        """
        Wrap Anthropic streaming response to be compatible with OpenAI format.
        """
        class MockDelta:
            def __init__(self):
                self.content = None

        class MockChoice:
            def __init__(self):
                self.delta = MockDelta()

        class MockChunk:
            def __init__(self):
                self.choices = [MockChoice()]

        class AnthropicStreamWrapper:
            def __init__(self, response_stream):
                self.response_stream = response_stream

            def __iter__(self):
                return self

            def __next__(self):
                try:
                    event = next(self.response_stream)
                    chunk = MockChunk()

                    # Handle different Anthropic streaming event types
                    if hasattr(event, 'type'):
                        if event.type == 'content_block_delta':
                            if hasattr(event, 'delta') and hasattr(event.delta, 'text'):
                                chunk.choices[0].delta.content = event.delta.text
                        elif event.type == 'message_delta':
                            # Handle other delta types if needed
                            pass

                    return chunk

                except StopIteration:
                    raise StopIteration
                except Exception:
                    # Return empty chunk on error
                    return MockChunk()

        return AnthropicStreamWrapper(response_stream)

    def _wrap_anthropic_response(self, response):
        """
        Wrap Anthropic non-streaming response to be compatible with OpenAI format.
        """
        # Extract text content from Anthropic response
        text_content = ""
        if hasattr(response, 'content') and response.content:
            for content_block in response.content:
                if hasattr(content_block, 'text'):
                    text_content += content_block.text

        # Create a mock response that matches OpenAI's response format
        class MockMessage:
            def __init__(self, content):
                self.content = content

        class MockChoice:
            def __init__(self, content):
                self.message = MockMessage(content)

        class MockResponse:
            def __init__(self, content):
                self.choices = [MockChoice(content)]
                self.content = content  # Keep for backward compatibility

        return MockResponse(text_content)

    def _wrap_responses_response(self, response):
        """
        Wrap Responses API non-streaming response to extract text content.
        """
        # Extract text content from the response output
        text_content = ""
        if hasattr(response, 'output'):
            for output_item in response.output:
                if hasattr(output_item, 'content'):
                    for content_item in output_item.content:
                        if hasattr(content_item, 'text'):
                            text_content += content_item.text

        # Create a mock response that matches OpenAI's response format
        class MockMessage:
            def __init__(self, content):
                self.content = content

        class MockChoice:
            def __init__(self, content):
                self.message = MockMessage(content)

        class MockResponse:
            def __init__(self, content):
                self.choices = [MockChoice(content)]
                self.content = content  # Keep for backward compatibility

        return MockResponse(text_content)

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

    def _is_anthropic_available(self) -> bool:
        """Check if Anthropic API is available with caching"""
        if self._anthropic_available is not None:
            return self._anthropic_available

        self._anthropic_available = bool(os.environ.get("ANTHROPIC_API_KEY"))
        return self._anthropic_available

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

    def _get_anthropic_client(self) -> Optional[Any]:
        """Get or create Anthropic client with caching"""
        if self._anthropic_client is not None:
            return self._anthropic_client

        try:
            import anthropic
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                return None

            self._anthropic_client = anthropic.Anthropic(api_key=api_key)
        except Exception:
            self._anthropic_client = None

        return self._anthropic_client

    def get_provider_for_model(self, model_name: str) -> str:
        """
        Get the provider name for a given model.

        Args:
            model_name: The model name

        Returns:
            Provider name: "ollama", "google", or "openai"
        """
        if self._is_ollama_model(model_name):
            return "ollama"
        elif self._is_google_model(model_name):
            return "google"
        elif self._is_anthropic_model(model_name):
            return "anthropic"
        else:
            return "openai"

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
        elif provider == "anthropic":
            return self._is_anthropic_available()
        else:
            return True  # Assume OpenAI is always available if we have a client

    def clear_cache(self):
        """Clear all cached availability checks and clients"""
        self._ollama_available = None
        self._google_available = None
        self._anthropic_available = None
        self._ollama_client = None
        self._google_client = None
        self._anthropic_client = None

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
            "anthropic": {
                "available": self._is_anthropic_available(),
                "api_key_set": bool(os.environ.get("ANTHROPIC_API_KEY"))
            }
        }

        return status