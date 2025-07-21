"""
TTS Service for Terminal AI Assistant

This module provides Text-to-Speech functionality using OpenAI's TTS API.
It supports multiple models and voices with optional audio file saving.
"""

import os
import time
import tempfile
import threading
from typing import Optional, Union, Dict, Any
from pathlib import Path
from openai import OpenAI
from settings_manager import SettingsManager
from audio_utils import get_audio_player
from llm_client_manager import LLMClientManager
from print_helper import print_md


class TTSService:
    """
    Text-to-Speech service using OpenAI's TTS API.

    Supports multiple models (tts-1, tts-1-hd, gpt-4o-mini-tts) and voices
    with optional file saving.
    """

    def __init__(self, client: Optional[OpenAI] = None):
        self.settings_manager = SettingsManager.getInstance()
        self.client = client or OpenAI()
        self.audio_player = get_audio_player()
        self.current_audio_thread = None
        self.interrupt_requested = False

        # Initialize LLM client manager for provider detection
        self.llm_client_manager = LLMClientManager(self.client)

        # Create audio output directory if saving is enabled
        self.audio_dir = None
        self._setup_audio_directory()

    def _setup_audio_directory(self):
        """Setup directory for saving audio files"""
        try:
            working_dir = self.settings_manager.setting_get("working_dir")
            self.audio_dir = os.path.join(working_dir, "audio_output")
            if not os.path.exists(self.audio_dir):
                os.makedirs(self.audio_dir)
        except Exception as e:
            print_md(f"Warning: Could not create audio directory: {e}")
            self.audio_dir = None

    def _is_ollama_provider(self, model_name: str) -> bool:
        """Check if the current model uses Ollama (local) provider"""
        if not self.llm_client_manager or not model_name:
            return False

        provider = self.llm_client_manager.get_provider_for_model(model_name)
        return provider == "ollama"

    def generate_and_play_speech(self, text: str) -> bool:
        """
        Generate speech from text and play it immediately.

        Args:
            text: Text to convert to speech

        Returns:
            bool: True if successful, False otherwise
        """
        # Check for privacy: disable TTS when using Ollama (local) models
        current_model = self.settings_manager.setting_get("model")
        if current_model and self._is_ollama_provider(current_model):
            privacy_text = "TTS disabled for privacy when using Ollama models\n"
            privacy_text += "    Text-to-speech would send your text to OpenAI, breaking local privacy"
            print_md(privacy_text)
            return False

        try:
            # Get current settings
            model = self.settings_manager.setting_get("tts_model")
            voice = self.settings_manager.setting_get("tts_voice")
            save_mp3 = self.settings_manager.setting_get("tts_save_mp3")

            print_md(f"Generating speech with {model} voice '{voice}'...")

            # Generate speech using OpenAI TTS API
            audio_data = self._generate_speech_data(text, model, voice)

            if audio_data is None:
                return False

            # Save to file if requested
            saved_file_path = None
            if save_mp3 and self.audio_dir:
                saved_file_path = self._save_audio_file(audio_data, text)

            # Play the audio
            success = self._play_audio_data(audio_data, saved_file_path)

            return success

        except Exception as e:
            print_md(f"Error in TTS generation: {e}")
            return False

    def _generate_speech_data(self, text: str, model: str, voice: str) -> Optional[bytes]:
        """
        Generate speech data from text using OpenAI TTS API.

        Args:
            text: Text to convert to speech
            model: TTS model to use
            voice: Voice to use

        Returns:
            bytes: Audio data or None if failed
        """
        try:
            # Validate inputs
            if not self._validate_model(model):
                print_md(f"Invalid TTS model: {model}")
                return None



            # Check text length (OpenAI limit is 4096 characters)
            if len(text) > 4096:
                print_md(f"Text too long ({len(text)} chars). OpenAI TTS limit is 4096 characters")
                return None

            # Call OpenAI TTS API
            response = self.client.audio.speech.create(
                model=model,
                voice=voice,
                input=text,
                response_format="mp3"
            )

            return response.content

        except Exception as e:
            print_md(f"OpenAI TTS API error: {e}")
            return None

    def _validate_model(self, model: str) -> bool:
        """Validate TTS model name"""
        valid_models = ["tts-1", "tts-1-hd", "gpt-4o-mini-tts"]
        return model in valid_models



    def _save_audio_file(self, audio_data: bytes, text: str) -> Optional[str]:
        """
        Save audio data to an MP3 file.

        Args:
            audio_data: Audio data to save
            text: Original text (used for filename)

        Returns:
            str: Path to saved file or None if failed
        """
        try:
            if not self.audio_dir:
                return None

            # Create filename from timestamp and text preview
            timestamp = int(time.time())
            text_preview = text[:30].replace(" ", "_").replace("\n", "").replace("/", "_")
            text_preview = "".join(c for c in text_preview if c.isalnum() or c in "_-")

            filename = f"tts_{timestamp}_{text_preview}.mp3"
            file_path = os.path.join(self.audio_dir, filename)

            # Write audio data to file
            with open(file_path, "wb") as f:
                f.write(audio_data)

            print_md(f"Audio saved to: {filename}")
            return file_path

        except Exception as e:
            print_md(f"Error saving audio file: {e}")
            return None

    def _play_audio_data(self, audio_data: bytes, file_path: Optional[str] = None) -> bool:
        """
        Play audio data.

        Args:
            audio_data: Audio data to play
            file_path: Optional path to saved file (for better performance)

        Returns:
            bool: True if playback started successfully, False otherwise
        """
        try:
            # Reset interrupt flag
            self.interrupt_requested = False

            # Use saved file if available for better performance
            if file_path and os.path.exists(file_path):
                success = self.audio_player.play_audio_file(file_path)
            else:
                success = self.audio_player.play_audio_bytes(audio_data, "mp3")

            if success:
                print_md("Playing audio... (Type 'q' + Enter at the next prompt to stop audio)")

                # Start monitoring thread for interruption
                self.current_audio_thread = threading.Thread(
                    target=self._monitor_audio_playback,
                    daemon=True
                )
                self.current_audio_thread.start()

                return True
            else:
                print_md("Failed to start audio playback")
                return False

        except Exception as e:
            print_md(f"Error playing audio: {e}")
            return False

    def _monitor_audio_playback(self):
        """Monitor audio playback and handle interruption"""
        try:
            while self.audio_player.is_playing():
                if self.interrupt_requested:
                    self.audio_player.stop()
                    print_md("Audio playback interrupted")
                    break
                time.sleep(0.1)
        except Exception as e:
            print_md(f"TTS monitoring error: {e}")
            # Still try to stop if interrupt was requested
            if self.interrupt_requested:
                try:
                    self.audio_player.stop()
                except:
                    pass

    def interrupt_audio(self):
        """Interrupt any currently playing audio"""
        self.interrupt_requested = True
        if self.audio_player and self.audio_player.is_playing():
            self.audio_player.stop()

    def is_playing(self) -> bool:
        """Check if audio is currently playing"""
        return self.audio_player and self.audio_player.is_playing()

    def set_volume(self, volume: float):
        """
        Set audio volume.

        Args:
            volume: Volume level (0.0 to 1.0)
        """
        if self.audio_player:
            self.audio_player.set_volume(volume)

    def get_voice_descriptions(self) -> Dict[str, str]:
        """Get descriptions of available voices"""
        return {
            "alloy": "Neutral, versatile voice",
            "echo": "Clear, professional voice",
            "fable": "Warm, storytelling voice",
            "onyx": "Deep, authoritative voice",
            "nova": "Bright, energetic voice",
            "shimmer": "Soft, gentle voice"
        }

    def get_model_descriptions(self) -> Dict[str, str]:
        """Get descriptions of available models"""
        return {
            "tts-1": "Standard TTS model, optimized for real-time use",
            "tts-1-hd": "High-definition TTS model, optimized for quality",
            "gpt-4o-mini-tts": "Latest GPT-based TTS model with improved prosody"
        }

    def test_configuration(self) -> bool:
        """
        Test current TTS configuration with a short phrase.

        Returns:
            bool: True if test successful, False otherwise
        """
        # Check for privacy: disable TTS when using Ollama (local) models
        current_model = self.settings_manager.setting_get("model")
        if current_model and self._is_ollama_provider(current_model):
            test_privacy_text = "TTS test disabled for privacy when using Ollama models\n"
            test_privacy_text += "    Text-to-speech would send your text to OpenAI, breaking local privacy"
            print_md(test_privacy_text)
            return False

        test_text = "Hello, this is a test of the text-to-speech system."
        print_md("Testing TTS configuration...")

        try:
            model = self.settings_manager.setting_get("tts_model")
            voice = self.settings_manager.setting_get("tts_voice")

            print_md(f"Model: {model}, Voice: {voice}")

            # Generate a short test
            audio_data = self._generate_speech_data(test_text, model, voice)

            if audio_data is None:
                print_md("TTS test failed - could not generate audio")
                return False

            print_md("TTS test successful - audio generated")

            # Optionally play the test (uncomment if desired)
            # self._play_audio_data(audio_data)

            return True

        except Exception as e:
            print_md(f"TTS test failed: {e}")
            return False

    def cleanup(self):
        """Clean up TTS service resources"""
        try:
            self.interrupt_audio()
            if self.audio_player:
                self.audio_player.cleanup()
        except Exception as e:
            print_md(f"TTS cleanup warning: {e}")
            # Cleanup failed but continue shutdown


# Global TTS service instance
_tts_service = None


def get_tts_service(client: Optional[OpenAI] = None) -> TTSService:
    """Get the global TTS service instance"""
    global _tts_service
    if _tts_service is None:
        _tts_service = TTSService(client)
    return _tts_service


def generate_and_play_speech(text: str, client: Optional[OpenAI] = None) -> bool:
    """
    Convenience function to generate and play speech.

    Args:
        text: Text to convert to speech
        client: Optional OpenAI client

    Returns:
        bool: True if successful, False otherwise
    """
    service = get_tts_service(client)
    return service.generate_and_play_speech(text)


def interrupt_tts():
    """Convenience function to interrupt TTS playback"""
    global _tts_service
    if _tts_service:
        _tts_service.interrupt_audio()


def is_tts_playing() -> bool:
    """Convenience function to check if TTS is playing"""
    global _tts_service
    if _tts_service:
        return _tts_service.is_playing()
    return False


def cleanup_tts():
    """Convenience function to clean up TTS resources"""
    global _tts_service
    if _tts_service:
        _tts_service.cleanup()
        _tts_service = None


if __name__ == "__main__":
    # Run tests if this module is executed directly
    test_tts_system()