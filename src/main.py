import argparse
from openai import OpenAI
from prompt_toolkit import prompt
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.validation import Validator, ValidationError
from settings_manager import SettingsManager
from conversation_manager import ConversationManager
from command_manager import CommandManager
from tts_service import cleanup_tts, is_tts_playing, interrupt_tts
from print_helper import print_info
from api_key_check import check

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

    command_manager = CommandManager(conversation_manager)

    # Display startup information
    current_model = settings_manager.setting_get("model")
    print_info(f"Using model: {current_model}")
    print_info("Start chatting or type '--' to see available commands!\n")



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

    first_ai_interaction = True
    while True:
        try:
            if first_ai_interaction:
                # Apply default instructions to AI
                conversation_manager.apply_instructions(settings_manager.setting_get("instructions"))

                first_ai_interaction = False

                if args.input:
                    # Process each input sequentially
                    for input_text in args.input:
                        print(f"{settings_manager.setting_get('name_user')}{settings_manager.get_enabled_toggles()}:")
                        print(f"> {input_text}")

                        # Check for commands first
                        if "--" in input_text:
                            command_processed = command_manager.process_commands(input_text)
                            if command_processed:
                                continue

                        # Check if nothink mode is enabled and prepend /nothink prefix
                        final_user_input = input_text
                        if settings_manager.setting_get("nothink"):
                            final_user_input = "/nothink " + input_text

                        # Add to conversation and generate response
                        conversation_manager.conversation_history.append(
                            {"role": "user", "content": final_user_input}
                        )

                        conversation_manager.generate_response()

                    # After processing all inputs, continue to interactive mode
                    continue
            else:
                print(f"{settings_manager.setting_get('name_user')}{settings_manager.get_enabled_toggles()}:")
                user_input = prompt(
                    "> ",
                    completer=command_manager.completer,
                    history=user_input_history,
                    key_bindings=kb,
                    validator=NonEmptyValidator(),
                    complete_while_typing=True,
                )

                # Check if user wants to interrupt TTS playback
                if user_input.lower().strip() == "q" and is_tts_playing():
                    interrupt_tts()
                    print_info("Audio playback stopped")
                    continue

                exit_commands = ("q", "quit", ":q", ":wq")
                if any(command for command in exit_commands if user_input.lower().startswith(command)):
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

                conversation_manager.conversation_history.append(
                    {"role": "user", "content": final_user_input}
                )

                conversation_manager.generate_response()
        except KeyboardInterrupt:
            if confirm_exit():
                cleanup_tts()
                break


if __name__ == "__main__":
    main()
