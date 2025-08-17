import os
import json
from datetime import datetime
from typing import Optional
from settings_manager import SettingsManager
from constants import FilenameConstants

class ExportManager:
    """Handles exporting conversation JSON files to markdown format"""

    def __init__(self):
        self.settings_manager = SettingsManager.getInstance()

    def export_current_conversation(self, custom_filename: Optional[str] = None) -> Optional[str]:
        """
        Export the current conversation from JSON to markdown format.

        Args:
            custom_filename: Optional custom filename (without extension).
                           If provided, exports as {custom_filename}_YYYYMMDD-HHMMSS.md

        Returns:
            str: Path to the exported markdown file, or None if export failed
        """
        try:
            # Get current log file location (JSON file)
            current_log_location = self.settings_manager.setting_get("log_file_location")
            if not current_log_location:
                return None

            if not os.path.exists(current_log_location):
                return None

            # Load conversation history from JSON
            with open(current_log_location, 'r') as f:
                conversation_history = json.load(f)

            if not conversation_history:
                return None

            # Create export subdirectory next to the source log file
            log_dir = os.path.dirname(current_log_location)
            export_dir = os.path.join(log_dir, "export")
            if not os.path.exists(export_dir):
                os.makedirs(export_dir)

            # Generate markdown content
            markdown_content = self._convert_conversation_to_markdown(conversation_history)

            # Create export file path with .md extension instead of .json
            if custom_filename:
                # Use custom filename with timestamp, replace spaces with hyphens
                sanitized_filename = custom_filename.replace(" ", "-")
                timestamp = datetime.now().strftime(FilenameConstants.TIMESTAMP_FORMAT)
                base_filename = f"{sanitized_filename}_{timestamp}.md"
            else:
                # Use original logic - base filename from current log
                base_filename = os.path.basename(current_log_location)
                if base_filename.endswith('.json'):
                    base_filename = base_filename[:-5] + '.md'
            export_file_path = os.path.join(export_dir, base_filename)

            # Write markdown file
            with open(export_file_path, 'w') as file:
                file.write(markdown_content)

            return export_file_path

        except Exception:
            # Return None on any error to indicate failure
            return None

    def _convert_conversation_to_markdown(self, conversation_history: list) -> str:
        """
        Convert conversation history to formatted markdown.

        Args:
            conversation_history: List of conversation messages

        Returns:
            str: Formatted markdown content
        """
        markdown_lines = []

        for message in conversation_history:
            role = message.get('role', '')
            content = message.get('content', '')

            if role == 'system':
                # Format system messages in pure markdown
                markdown_lines.append(f"**{role}:**  ")
                markdown_lines.append(f"{content}")
                markdown_lines.append("")
            elif role in ['user', 'assistant']:
                # Format user and assistant messages with bold role labels
                markdown_lines.append(f"**{role}:**  ")
                markdown_lines.append(f"{content}")
                markdown_lines.append("")

        return "\n".join(markdown_lines)