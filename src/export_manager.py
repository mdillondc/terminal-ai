import os
import json
from typing import Optional
from settings_manager import SettingsManager


class ExportManager:
    """Handles exporting conversation JSON files to markdown format"""

    def __init__(self):
        self.settings_manager = SettingsManager.getInstance()

    def export_current_conversation(self) -> Optional[str]:
        """
        Export the current conversation from JSON to markdown format.

        Returns:
            str: Path to the exported markdown file, or None if export failed
        """
        try:
            # Get current log file location (JSON file)
            current_log_location = self.settings_manager.setting_get("log_file_location")
            if not current_log_location:
                return None

            json_file_path = current_log_location + ".json"
            if not os.path.exists(json_file_path):
                return None

            # Load conversation history from JSON
            with open(json_file_path, 'r') as file:
                conversation_history = json.load(file)

            if not conversation_history:
                return None

            # Create logs-exported directory if it doesn't exist
            working_dir = self.settings_manager.setting_get("working_dir")
            export_dir = os.path.join(working_dir, "logs-exported")
            if not os.path.exists(export_dir):
                os.makedirs(export_dir)

            # Generate markdown content
            markdown_content = self._convert_conversation_to_markdown(conversation_history)

            # Create export file path with same base name as JSON file
            base_filename = os.path.basename(current_log_location)
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
                # Format system messages in collapsible HTML details
                # Truncate long system messages for the summary
                summary = content[:100] + "..." if len(content) > 100 else content
                markdown_lines.append("<details>")
                markdown_lines.append(f"<summary><strong>System:</strong> {summary}</summary>")
                markdown_lines.append(f"\n{content}")
                markdown_lines.append("</details>")
                markdown_lines.append("")
            elif role in ['user', 'assistant']:
                # Format user and assistant messages with bold role labels
                markdown_lines.append(f"**{role}:**  ")
                markdown_lines.append(f"{content}")
                markdown_lines.append("")

        return "\n".join(markdown_lines)