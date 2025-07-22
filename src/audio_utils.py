"""
Audio Utilities for Terminal AI Assistant

This module provides cross-platform audio playback functionality using pygame.
It handles playing audio from bytes or files with proper error handling and
resource management.
"""

import os
import sys
import tempfile
import threading
import time
from typing import Optional, Union
from io import BytesIO
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
from print_helper import print_md
from constants import AudioSystemConstants


class AudioPlayer:
    """
    Cross-platform audio player using pygame.

    Handles playing audio from bytes or files with proper resource management
    and error handling.
    """

    def __init__(self):
        self.initialized = False
        self.playing = False
        self.current_thread = None
        self._init_pygame()

    def _init_pygame(self) -> bool:
        """Initialize pygame mixer for audio playback"""

        try:
            # Initialize pygame mixer
            pygame.mixer.pre_init(
                frequency=AudioSystemConstants.SAMPLE_RATE,
                size=AudioSystemConstants.SAMPLE_SIZE,
                channels=AudioSystemConstants.CHANNELS,
                buffer=AudioSystemConstants.BUFFER_SIZE
            )
            pygame.mixer.init()
            self.initialized = True
            print_md("Audio playback initialized successfully!")
            return True
        except Exception as e:
            print_md(f"Failed to initialize audio system: {e}")
            return False

    def play_audio_bytes(self, audio_bytes: bytes, format_hint: str = "mp3") -> bool:
        """
        Play audio from bytes.

        Args:
            audio_bytes: Raw audio data
            format_hint: Audio format hint (mp3, wav, etc.)

        Returns:
            bool: True if playback started successfully, False otherwise
        """
        if not self.initialized:
            print_md("Audio system not initialized")
            return False

        try:
            # Create a temporary file for the audio data
            with tempfile.NamedTemporaryFile(suffix=f".{format_hint}", delete=False) as temp_file:
                temp_file.write(audio_bytes)
                temp_file.flush()
                temp_path = temp_file.name

            # Play the audio file
            success = self.play_audio_file(temp_path)

            # Clean up the temporary file after playback
            def cleanup():
                time.sleep(AudioSystemConstants.CLEANUP_DELAY)  # Give time for audio to start
                try:
                    os.unlink(temp_path)
                except (OSError, FileNotFoundError) as e:
                    # Log but don't fail - cleanup is best effort
                    print_md(f"Warning: Could not clean up temp audio file: {e}")
                except Exception as e:
                    # Unexpected error - log for debugging
                    print_md(f"Unexpected error during audio cleanup: {e}")

            threading.Thread(target=cleanup, daemon=True).start()

            return success

        except Exception as e:
            print_md(f"Error playing audio bytes: {e}")
            return False

    def play_audio_file(self, file_path: str) -> bool:
        """
        Play audio from a file.

        Args:
            file_path: Path to the audio file

        Returns:
            bool: True if playback started successfully, False otherwise
        """
        if not self.initialized:
            print_md("Audio system not initialized")
            return False

        if not os.path.exists(file_path):
            print_md(f"Audio file not found: {file_path}")
            return False

        try:
            # Stop any currently playing audio
            self.stop()

            # Load and play the audio file
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()
            self.playing = True

            # Start monitoring thread
            self.current_thread = threading.Thread(target=self._monitor_playback, daemon=True)
            self.current_thread.start()

            return True

        except Exception as e:
            print_md(f"Error playing audio file {file_path}: {e}")
            return False

    def _monitor_playback(self):
        """Monitor playback status and update playing flag"""
        try:
            while pygame.mixer.music.get_busy():
                time.sleep(AudioSystemConstants.PLAYBACK_POLL_INTERVAL)
            self.playing = False
        except pygame.error as e:
            print_md(f"Audio playback monitoring error: {e}")
            self.playing = False
        except Exception as e:
            print_md(f"Unexpected error in audio monitoring: {e}")
            self.playing = False

    def stop(self):
        """Stop any currently playing audio"""
        if not self.initialized:
            return

        try:
            pygame.mixer.music.stop()
            self.playing = False
        except Exception as e:
            print_md(f"Error stopping audio: {e}")

    def is_playing(self) -> bool:
        """Check if audio is currently playing"""
        if not self.initialized:
            return False

        try:
            return pygame.mixer.music.get_busy()
        except pygame.error:
            # Pygame not properly initialized
            return False
        except Exception as e:
            print_md(f"Error checking audio status: {e}")
            return False

    def set_volume(self, volume: float):
        """
        Set playback volume.

        Args:
            volume: Volume level (0.0 to 1.0)
        """
        if not self.initialized:
            return

        try:
            volume = max(0.0, min(AudioSystemConstants.DEFAULT_VOLUME, volume))  # Clamp to valid range
            pygame.mixer.music.set_volume(volume)
        except Exception as e:
            print_md(f"Error setting volume: {e}")

    def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for current audio to finish playing.

        Args:
            timeout: Maximum time to wait in seconds (None for no timeout)

        Returns:
            bool: True if audio finished, False if timed out
        """
        if not self.initialized or not self.playing:
            return True

        start_time = time.time()

        while self.is_playing():
            if timeout and (time.time() - start_time) > timeout:
                return False
            time.sleep(AudioSystemConstants.PLAYBACK_POLL_INTERVAL)

        return True

    def cleanup(self):
        """Clean up pygame resources"""
        if self.initialized:
            try:
                self.stop()
                pygame.mixer.quit()
                self.initialized = False
            except pygame.error as e:
                print_md(f"Audio system cleanup warning: {e}")
                self.initialized = False  # Mark as uninitialized anyway
            except Exception as e:
                print_md(f"Unexpected error during audio cleanup: {e}")
                self.initialized = False


# Global audio player instance
_audio_player = None


def get_audio_player() -> AudioPlayer:
    """Get the global audio player instance"""
    global _audio_player
    if _audio_player is None:
        _audio_player = AudioPlayer()
    return _audio_player


def play_audio_bytes(audio_bytes: bytes, format_hint: str = "mp3") -> bool:
    """
    Convenience function to play audio from bytes.

    Args:
        audio_bytes: Raw audio data
        format_hint: Audio format hint (mp3, wav, etc.)

    Returns:
        bool: True if playback started successfully, False otherwise
    """
    player = get_audio_player()
    return player.play_audio_bytes(audio_bytes, format_hint)


def play_audio_file(file_path: str) -> bool:
    """
    Convenience function to play audio from a file.

    Args:
        file_path: Path to the audio file

    Returns:
        bool: True if playback started successfully, False otherwise
    """
    player = get_audio_player()
    return player.play_audio_file(file_path)


def stop_audio():
    """Convenience function to stop any currently playing audio"""
    player = get_audio_player()
    player.stop()


def is_audio_playing() -> bool:
    """Convenience function to check if audio is currently playing"""
    player = get_audio_player()
    return player.is_playing()


def wait_for_audio_completion(timeout: Optional[float] = None) -> bool:
    """
    Convenience function to wait for current audio to finish.

    Args:
        timeout: Maximum time to wait in seconds (None for no timeout)

    Returns:
        bool: True if audio finished, False if timed out
    """
    player = get_audio_player()
    return player.wait_for_completion(timeout)


def set_audio_volume(volume: float):
    """
    Convenience function to set audio volume.

    Args:
        volume: Volume level (0.0 to 1.0)
    """
    player = get_audio_player()
    player.set_volume(volume)


def cleanup_audio():
    """Convenience function to clean up audio resources"""
    global _audio_player
    if _audio_player:
        _audio_player.cleanup()
        _audio_player = None


# Test function for debugging
def test_audio_system():
    """Test the audio system functionality"""
    print_md("Testing audio system...")

    player = get_audio_player()

    if not player.initialized:
        print_md("Audio system failed to initialize")
        return False

    print_md("Audio system initialized successfully")
    print_md(f"pygame version: {pygame.version.ver}")

    # Test basic functionality
    try:
        player.set_volume(AudioSystemConstants.DEFAULT_VOLUME * 0.5)  # Test at half volume
        print_md("Volume control working")
    except Exception as e:
        print_md(f"Volume control error: {e}")
        return False

    return True


if __name__ == "__main__":
    # Run tests if this module is executed directly
    test_audio_system()