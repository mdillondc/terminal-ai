import os
import sys
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.filters import Condition
from print_helper import print_info


class ScrollManager:
    def __init__(self, settings_manager, conversation_manager):
        self.settings_manager = settings_manager
        self.conversation_manager = conversation_manager
        self.scroll_mode = False
        self.scroll_position = 0
        self.history_lines = []
        self.g_pressed = False  # For gg sequence

    def _enter_scroll_mode(self):
        """Enter scrollable history mode"""
        # Read full conversation history from log file
        self.history_lines = []
        try:
            log_file_path = self.settings_manager.setting_get("log_file_location")
            if log_file_path and os.path.exists(log_file_path):
                with open(log_file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Split into lines and add to history_lines
                    for line in content.split('\n'):
                        self.history_lines.append(line)
            else:
                # Fallback to in-memory history if no log file
                for msg in self.conversation_manager.conversation_history:
                    role = msg.get('role', '')
                    content = msg.get('content', '')

                    if role == 'user':
                        user_name = self.settings_manager.setting_get('name_user')
                        toggles = self.settings_manager.get_enabled_toggles()
                        self.history_lines.append(f"\n{user_name}{toggles}:")
                        self.history_lines.append(f"> {content}")
                    elif role == 'assistant':
                        ai_name = self.settings_manager.setting_get('name_ai')
                        self.history_lines.append(f"\n{ai_name}:")
                        for line in content.split('\n'):
                            self.history_lines.append(line)
        except Exception as e:
            # If reading log file fails, use in-memory history
            for msg in self.conversation_manager.conversation_history:
                role = msg.get('role', '')
                content = msg.get('content', '')

                if role == 'user':
                    user_name = self.settings_manager.setting_get('name_user')
                    toggles = self.settings_manager.get_enabled_toggles()
                    self.history_lines.append(f"\n{user_name}{toggles}:")
                    self.history_lines.append(f"> {content}")
                elif role == 'assistant':
                    ai_name = self.settings_manager.setting_get('name_ai')
                    self.history_lines.append(f"\n{ai_name}:")
                    for line in content.split('\n'):
                        self.history_lines.append(line)

        self.scroll_mode = True
        self.scroll_position = max(0, len(self.history_lines) - 20)  # Start near bottom
        self._display_scroll_view()

    def _display_scroll_view(self):
        """Display the scrollable history view"""
        # Clear screen and show history section
        terminal_height = int(os.environ.get('LINES', 24)) - 3  # Leave space for status
        start_line = max(0, self.scroll_position)
        end_line = min(len(self.history_lines), start_line + terminal_height)

        # Clear screen
        sys.stdout.write('\033[2J\033[H')

        # Display history section
        for i in range(start_line, end_line):
            sys.stdout.write(self.history_lines[i] + '\n')

        # Display status line
        percent = int((self.scroll_position / max(1, len(self.history_lines) - terminal_height)) * 100) if len(self.history_lines) > terminal_height else 100
        status = f"[SCROLL MODE] Line {self.scroll_position + 1}/{len(self.history_lines)} ({percent}%) - j/k scroll, gg top, G bottom, Ctrl+V exit"
        sys.stdout.write(f"\n\033[48;5;214m\033[30m{status}\033[0m")  # Gruvbox yellow background with black text
        sys.stdout.flush()

    def _exit_scroll_mode(self):
        """Exit scrollable history mode"""
        self.scroll_mode = False
        self.g_pressed = False  # Reset g state
        # Clear screen and restore recent conversation context
        sys.stdout.write('\033[2J\033[H')

        # Show complete conversation history (excluding system instructions)
        for msg in self.conversation_manager.conversation_history:
            role = msg.get('role', '')
            content = msg.get('content', '')

            # Skip system instructions
            if role == 'system' or (role == 'user' and content.startswith('instructions:')):
                continue

            if role == 'user':
                user_name = self.settings_manager.setting_get('name_user')
                toggles = self.settings_manager.get_enabled_toggles()
                print(f"\n{user_name}{toggles}:")
                print(f"> {content}")
            elif role == 'assistant':
                ai_name = self.settings_manager.setting_get('name_ai')
                instructions_name = ""
                if self.settings_manager.setting_get("instructions"):
                    instructions_file = self.settings_manager.setting_get("instructions")
                    instructions_name = f" ({instructions_file.rsplit('.', 1)[0]})"
                print(f"\n{ai_name}{instructions_name} (`q` + `Enter` to interrupt):")
                print(content)

        # Show new prompt ready for input
        user_name = self.settings_manager.setting_get('name_user')
        toggles = self.settings_manager.get_enabled_toggles()
        print(f"\n{user_name}{toggles}:")
        sys.stdout.write("> ")
        sys.stdout.flush()

    def _go_to_top(self):
        """Go to top of history"""
        self.scroll_position = 0
        self._display_scroll_view()

    def _go_to_bottom(self):
        """Go to bottom of history"""
        terminal_height = int(os.environ.get('LINES', 24)) - 3
        self.scroll_position = max(0, len(self.history_lines) - terminal_height)
        self._display_scroll_view()

    def setup_key_bindings(self, kb):
        """Setup key bindings for scroll functionality"""

        @kb.add('j', filter=Condition(lambda: self.settings_manager.scroll and not self.scroll_mode))
        def scroll_enter_scroll_mode_down(event):
            """Enter scroll mode and scroll down"""
            self._enter_scroll_mode()

        @kb.add('k', filter=Condition(lambda: self.settings_manager.scroll and not self.scroll_mode))
        def scroll_enter_scroll_mode_up(event):
            """Enter scroll mode and scroll up"""
            self._enter_scroll_mode()

        @kb.add('j', filter=Condition(lambda: self.scroll_mode))
        def scroll_down_in_mode(event):
            """Scroll down in scroll mode"""
            terminal_height = int(os.environ.get('LINES', 24)) - 3
            scroll_lines = self.settings_manager.scroll_lines
            if self.scroll_position < len(self.history_lines) - terminal_height:
                self.scroll_position = min(self.scroll_position + scroll_lines, len(self.history_lines) - terminal_height)
                self._display_scroll_view()

        @kb.add('k', filter=Condition(lambda: self.scroll_mode))
        def scroll_up_in_mode(event):
            """Scroll up in scroll mode"""
            scroll_lines = self.settings_manager.scroll_lines
            if self.scroll_position > 0:
                self.scroll_position = max(0, self.scroll_position - scroll_lines)
                self._display_scroll_view()

        @kb.add('g', filter=Condition(lambda: self.scroll_mode))
        def handle_g_key(event):
            """Handle g key - first press in gg sequence"""
            if self.g_pressed:
                # Second g pressed, go to top
                self.g_pressed = False
                self._go_to_top()
            else:
                # First g pressed, wait for second g
                self.g_pressed = True

        @kb.add('G', filter=Condition(lambda: self.scroll_mode))
        def go_to_bottom(event):
            """Go to bottom of history"""
            self.g_pressed = False  # Reset g state
            self._go_to_bottom()

    def handle_toggle(self):
        """Handle scroll mode toggle from Ctrl+V"""
        if self.scroll_mode:
            self._exit_scroll_mode()