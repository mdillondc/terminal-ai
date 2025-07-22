import os
import urllib.request

from pydantic import annotated_handlers
from settings_manager import SettingsManager
from print_helper import print_md, print_lines
from constants import NetworkConstants

def _check_ollama():
    try:
        settings_manager = SettingsManager.getInstance()
        base_url = settings_manager.setting_get("ollama_base_url")
        url = f"{base_url}/api/version"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=NetworkConstants.OLLAMA_CHECK_TIMEOUT) as response:
            return response.status == 200
    except Exception:
        return False


def check():
    openai_key = os.getenv('OPENAI_API_KEY')
    google_key = os.getenv('GOOGLE_API_KEY')
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')
    tavily_key = os.getenv('TAVILY_API_KEY')
    ollama_available = _check_ollama()

    if not openai_key or not google_key or not tavily_key or not ollama_available:
        print_lines()

    if not openai_key:
        print_md("OPENAI_API_KEY not set.")
    if not google_key:
        print_md("GOOGLE_API_KEY not set.")
    if not anthropic_key:
        print_md("ANTHROPIC_API_KEY not set.")
    if not tavily_key:
        print_md("TAVILY_API_KEY not set. Web search unavailable (--search)")
    if not ollama_available:
        print_md("Ollama not available.")

    if not openai_key or not google_key or not tavily_key or not ollama_available:
        print_md("You can suppress these messages with --suppress-api")
        print_lines()