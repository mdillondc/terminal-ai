import argparse
import os
from openai import OpenAI
from prompt_toolkit import prompt
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.validation import Validator, ValidationError
from prompt_toolkit.filters import Condition
from prompt_toolkit.styles import Style
from settings_manager import SettingsManager
from conversation_manager import ConversationManager
from command_manager import CommandManager
from tts_service import cleanup_tts, is_tts_playing, interrupt_tts
from print_helper import print_md, print_lines, set_conversation_manager
from api_key_check import check
from scroll import ScrollManager

def confirm_exit() -> bool:
    response = input("Confirm quitting (Y/n)? ").strip().lower()
    return response in ['', 'y', 'yes']

class NonEmptyValidator(Validator):
    def validate(self, document):
        if not document.text:
            raise ValidationError(message="", cursor_position=0)

def main() -> None:
    # Parse arguments first to check for --suppress-api
    parser = argparse.ArgumentParser(description="I am Samantha.")
    parser.add_argument(
        "--input", type=str, action='append', help="Input string(s) to process sequentially. Can be used multiple times for batch processing."
    )

    parser.add_argument(
        "--suppress-api", action='store_true', help="Suppress API key checking on startup."
    )
    args = parser.parse_args()

    # Check API keys unless suppressed
    if not args.suppress_api:
        check()

    settings_manager = SettingsManager.getInstance()

    # Prompt setup
    user_input_history = InMemoryHistory()

    # Init client - try OpenAI, if it fails create dummy for other providers
    try:
        client = OpenAI()
    except Exception:
        try:
            client = OpenAI(api_key="dummy")
        except Exception:
            client = OpenAI(api_key="dummy", base_url="http://localhost")

    # Init managers
    conversation_manager = ConversationManager(
        client,
        model=settings_manager.setting_get("model"),
    )

    # Initialize print_helper with conversation_manager for automatic .md logging
    set_conversation_manager(conversation_manager)

    command_manager = CommandManager(conversation_manager)

    # Init scroll manager
    scroll_manager = ScrollManager(settings_manager, conversation_manager)

    # Display model info using centralized method with source attribution
    current_model = settings_manager.setting_get("model")
    print_md(f"Model: {current_model} (settings_manager.py)")

    current_instructions = settings_manager.setting_get("instructions")
    if current_instructions:
        instruction_name = current_instructions.rsplit('.', 1)[0]
        print_md(f"Instructions: {instruction_name} (settings_manager.py)")

    # Load config file overrides after displaying defaults
    settings_manager.load_config()
    print()  # Add blank line before user interactions

    # KeyBindings
    kb = KeyBindings()

    @kb.add('backspace')
    def custom_backspace(event):
        """Custom backspace that forces completion refresh"""
        # Perform normal backspace
        event.current_buffer.delete_before_cursor()

        # Force completion refresh for command contexts
        text = event.current_buffer.text
        if text.lstrip().startswith('--'):
            # Clear any existing completion state
            event.current_buffer.complete_state = None
            # Force start new completion
            event.current_buffer.start_completion(select_first=False)

    @kb.add('f8')
    def toggle_scroll_mode(event):
        """Toggle scroll navigation mode with F8 key"""
        if scroll_manager.scroll_mode:
            # Exit scroll mode directly
            scroll_manager.handle_toggle()
        else:
            # Enable scroll and immediately enter scroll mode
            if settings_manager.setting_get("incognito"):
                print_md("Cannot enable scroll mode in incognito mode - no logs available to scroll through")
            else:
                settings_manager.setting_set("scroll", True)
                scroll_manager._enter_scroll_mode()

    # Set up scroll key bindings
    scroll_manager.setup_key_bindings(kb)

    first_ai_interaction = True
    while True:
        try:
            if first_ai_interaction:
                # Apply default instructions to AI
                conversation_manager.apply_instructions(settings_manager.setting_get("instructions"))

                first_ai_interaction = False

                if args.input:
                    # Process each input sequentially, splitting commands individually
                    for input_text in args.input:
                        # Split input into individual commands
                        if "--" in input_text:
                            commands = [
                                "--" + command.strip()
                                for command in input_text.split("--")
                                if command.strip()
                            ]

                            # Process each command individually
                            for command in commands:
                                # Display prompt and command BEFORE processing (like normal user interaction)
                                print(f"{settings_manager.setting_get('name_user')}{settings_manager.get_enabled_toggles()}:")
                                print(f"> {command}")

                                # Process the individual command
                                command_manager.process_commands(command)
                        else:
                            # Display prompt for non-command inputs
                            print(f"{settings_manager.setting_get('name_user')}{settings_manager.get_enabled_toggles()}:")
                            print(f"> {input_text}")

                            # Check if nothink mode is enabled and prepend /nothink prefix
                            final_user_input = input_text
                            if settings_manager.setting_get("nothink"):
                                final_user_input = "/nothink " + input_text

                            # Add to conversation and generate response
                            conversation_manager.log_context(final_user_input, "user")

                            conversation_manager.generate_response()

                    continue
            else:
                # Show final startup message for interactive mode
                if first_ai_interaction:
                    first_ai_interaction = False

                print(f"{settings_manager.setting_get('name_user')}{settings_manager.get_enabled_toggles()}:")

                # Gruvbox dark two-toned completion menu styling
                gruvbox_style = Style.from_dict({
                    # Left side (completion text) - darker background
                    'completion-menu.completion': 'bg:#2c2c2c #ebdbb2',
                    'completion-menu.completion.current': 'bg:#504945 #ebdbb2',

                    # Right side (meta info) - lighter background for two-tone effect
                    'completion-menu.meta': 'bg:#3c3836 #bdae93',
                    'completion-menu.meta.completion': 'bg:#3c3836 #bdae93',
                    'completion-menu.meta.completion.current': 'bg:#665c54 #bdae93',

                    # Overall menu container
                    'completion-menu': 'bg:#282828',

                    # Scrollbar styling
                    'scrollbar.background': 'bg:#3c3836',
                    'scrollbar.button': 'bg:#504945',
                    'scrollbar.arrow': 'bg:#504945 #ebdbb2',

                    # Custom completion class
                    'completion': 'bg:#282828 #ebdbb2',
                })

                user_input = prompt(
                    "> ",
                    completer=command_manager.completer,
                    history=user_input_history,
                    key_bindings=kb,
                    validator=NonEmptyValidator(),
                    complete_while_typing=True,
                    style=gruvbox_style,
                )

                # Check if user wants to interrupt TTS playback
                if user_input.lower().strip() == "q" and is_tts_playing():
                    interrupt_tts()
                    print_md("Audio playback stopped")
                    continue

                exit_commands = ("q", "quit", ":q", ":wq")
                if user_input.lower().strip() in exit_commands:
                    if confirm_exit():
                        cleanup_tts()
                        break
                    else:
                        continue

                if "--" in user_input:
                    command_processed = command_manager.process_commands(user_input)
                    if command_processed:
                        continue

                # Check if nothink mode is enabled and prepend /nothink prefix
                final_user_input = user_input
                if settings_manager.setting_get("nothink"):
                    final_user_input = "/nothink " + user_input

                conversation_manager.log_context(final_user_input, "user")

                conversation_manager.generate_response()
        except KeyboardInterrupt:
            if confirm_exit():
                cleanup_tts()
                break


if __name__ == "__main__":
    main()
