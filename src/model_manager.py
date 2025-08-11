import json
import os
import time
import urllib.request
import urllib.error
from typing import List, Dict, Optional
from settings_manager import SettingsManager
from constants import CacheConstants


class ModelCache:
    """Simple cache for model data with TTL support"""

    def __init__(self, ttl_seconds: int = CacheConstants.MODEL_CACHE_TTL):
        self.ttl_seconds = ttl_seconds
        self._cache = {}

    def get(self, key: str) -> Optional[List[str]]:
        """Get cached models if not expired"""
        if key in self._cache:
            data, timestamp = self._cache[key]
            if time.time() - timestamp < self.ttl_seconds:
                return data
            else:
                del self._cache[key]
        return None

    def set(self, key: str, models: List[str]) -> None:
        """Cache models with current timestamp"""
        self._cache[key] = (models, time.time())

    def clear(self) -> None:
        """Clear all cached data"""
        self._cache.clear()


class ModelManager:
    """Manages discovery, filtering, and caching of models from multiple providers"""

    def __init__(self, openai_client=None):
        self.openai_client = openai_client
        self.cache = ModelCache()

    def get_available_models(self) -> List[Dict[str, str]]:
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

        # Sort by source alphabetically, then by model name alphabetically within each source
        all_models.sort(key=lambda x: (x["source"], x["name"]))

        return all_models

    def _get_openai_models_cached(self) -> List[str]:
        """Get OpenAI models with persistent file-based caching"""
        # Try to load from persistent cache first
        cached_models = self._load_models_from_cache("openai")
        if cached_models:
            return cached_models

        # If no valid cache, fetch from API
        models = self._get_openai_models()
        if models:
            # Save to persistent cache
            self._save_models_to_cache("openai", models)

        return models

    def _get_openai_models(self) -> List[str]:
        """Get available models from OpenAI API (no caching)"""
        if not self.openai_client:
            return []

        try:
            # Fetch models from OpenAI API
            models_response = self.openai_client.models.list()

            # Extract model IDs and filter out non-chat models
            all_models = [model.id for model in models_response.data]

            # Filter out models that are not suitable for chat
            filtered_models = []
            for model in all_models:
                model_lower = model.lower()
                # Exclude specialized non-chat models and older models
                if not any(keyword in model_lower for keyword in [
                    'whisper', 'tts', 'embedding', 'moderation',
                    'edit', 'search', 'similarity', 'code-search',
                    'dall-e', 'davinci', 'gpt-3.5', 'transcribe'
                ]):
                    filtered_models.append(model)

            return filtered_models

        except Exception:
            # If API call fails, return empty list (will fallback to predefined)
            return []

    def _get_google_models_cached(self) -> List[str]:
        """Get Google models with persistent file-based caching"""
        # Try to load from persistent cache first
        cached_models = self._load_models_from_cache("google")
        if cached_models:
            return cached_models

        # If no valid cache, fetch from API
        models = self._get_google_models()
        if models:
            # Save to persistent cache
            self._save_models_to_cache("google", models)

        return models

    def _get_google_models(self) -> List[str]:
        """Get available models from Google Generative AI API with smart filtering"""
        try:
            import os

            # Check if API key is available
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                return []

            # Get Google models from API
            url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"

            # Create request with timeout
            req = urllib.request.Request(url)
            req.add_header('Content-Type', 'application/json')

            with urllib.request.urlopen(req, timeout=3) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode('utf-8'))

                    models = []
                    if 'models' in data:
                        for model_info in data['models']:
                            model_name = model_info.get('name', '')
                            supported_methods = model_info.get('supportedGenerationMethods', [])

                            if model_name and supported_methods:
                                # Extract model name from "models/gemini-1.5-flash" format
                                if model_name.startswith('models/'):
                                    model_name = model_name[7:]  # Remove "models/" prefix

                                # Filter for chat-capable models based on supported generation methods
                                chat_methods = ['generateContent', 'generateMessage', 'generateText']
                                if any(method in supported_methods for method in chat_methods):
                                    # Exclude embedding-only models
                                    if 'embedText' not in supported_methods or len(supported_methods) > 1:
                                        # Exclude Gemma models and Gemini models older than version 2.5
                                        if model_name.startswith('gemma-'):
                                            # Skip all Gemma models
                                            pass
                                        elif model_name.startswith('gemini-'):
                                            # Extract version number (e.g., "gemini-1.5-pro" -> "1.5")
                                            try:
                                                version_part = model_name.split('-')[1]
                                                version = float(version_part)
                                                if version >= 2.5:
                                                    models.append(model_name)
                                            except (IndexError, ValueError):
                                                # If version parsing fails, include the model
                                                models.append(model_name)
                                        else:
                                            # Non-Gemini/Gemma models (PaLM, Bard, etc.)
                                            models.append(model_name)

                    return models

        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, TimeoutError, OSError):
            # Google API not available, connection failed, or invalid response
            pass
        except Exception:
            # Any other error - silently continue to not break the application
            pass

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

    def refresh_cache(self) -> None:
        """Clear in-memory cache to force refresh on next request"""
        self.cache.clear()

    def clear_models_cache(self) -> None:
        """Clear both in-memory and persistent cache"""
        self.cache.clear()

        # Clear persistent cache files
        cache_dir = os.path.expanduser("~/.terminal-ai/cache")
        if os.path.exists(cache_dir):
            for provider in ["openai", "google"]:
                cache_file = os.path.join(cache_dir, f"models_{provider}.json")
                if os.path.exists(cache_file):
                    try:
                        os.remove(cache_file)
                    except OSError:
                        pass  # Ignore errors when clearing cache

    def _get_cache_file_path(self, provider: str) -> str:
        """Get the path for persistent model cache file"""
        cache_dir = os.path.expanduser("~/.terminal-ai/cache")
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f"models_{provider}.json")

    def _load_models_from_cache(self, provider: str) -> Optional[List[str]]:
        """Load models from persistent cache if valid"""
        cache_file = self._get_cache_file_path(provider)

        try:
            if os.path.exists(cache_file) and self._is_cache_valid(cache_file):
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                    return cache_data.get('models', [])
        except (json.JSONDecodeError, OSError, KeyError):
            # If cache is corrupted or unreadable, remove it
            try:
                os.remove(cache_file)
            except OSError:
                pass

        return None

    def _save_models_to_cache(self, provider: str, models: List[str]) -> None:
        """Save models to persistent cache"""
        cache_file = self._get_cache_file_path(provider)

        try:
            cache_data = {
                'models': models,
                'timestamp': time.time()
            }
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
        except OSError:
            # If we can't write to cache, just continue without caching
            pass

    def _is_cache_valid(self, cache_file: str, max_age_hours: int = 24) -> bool:
        """Check if persistent cache file is still valid (not too old)"""
        try:
            # Check file modification time
            file_age = time.time() - os.path.getmtime(cache_file)
            max_age_seconds = max_age_hours * 3600

            if file_age > max_age_seconds:
                return False

            # Also check if the cache contains valid timestamp
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
                cache_timestamp = cache_data.get('timestamp', 0)
                cache_age = time.time() - cache_timestamp

                return cache_age <= max_age_seconds

        except (OSError, json.JSONDecodeError, KeyError):
            return False