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

def confirm_exit() -> bool:
    response = input("Confirm quitting (Y/n)? ").strip().lower()
    return response in ['', 'y', 'yes']

class NonEmptyValidator(Validator):
    def validate(self, document):
        if not document.text:
            raise ValidationError(message="", cursor_position=0)

def main() -> None:
    settings_manager = SettingsManager.getInstance()

    # Prompt setup
    user_input_history = InMemoryHistory()

    # Init client
    client = OpenAI()

    # Init managers
    conversation_manager = ConversationManager(
        client,
        api=settings_manager.setting_get("api"),
        model=settings_manager.setting_get("model"),
    )

    command_manager = CommandManager(conversation_manager)

    # Display startup information
    current_model = settings_manager.setting_get("model")
    # print("Samantha AI Assistant")
    print(f"Using model: {current_model}")
    print("Type '--help' for available commands or start chatting!")
    # print("- Tip: Press 'q' + Enter during AI responses to interrupt streaming and regain control")

    # Parse arguments
    parser = argparse.ArgumentParser(description="I am Samantha.")
    parser.add_argument(
        "--input", type=str, help="Optional input string to start the conversation..."
    )
    args = parser.parse_args()

    # KeyBindings
    kb = KeyBindings()

    first_ai_interaction = True
    while True:
        try:
            if first_ai_interaction:
                # Apply default instructions to AI
                conversation_manager.apply_instructions(settings_manager.setting_get("instructions"))

                first_ai_interaction = False

                if args.input:
                    command_processed = command_manager.parse_commands(args.input)
                    if command_processed:
                        continue
            else:
                print(f"\n{settings_manager.setting_get('name_user')}{settings_manager.get_enabled_toggles()}:")
                user_input = prompt(
                    "> ",
                    completer=command_manager.completer,
                    history=user_input_history,
                    key_bindings=kb,
                    validator=NonEmptyValidator(),
                )

                # Check if user wants to interrupt TTS playback
                if user_input.lower().strip() == "q" and is_tts_playing():
                    interrupt_tts()
                    print("- Audio playback stopped\n")
                    continue

                exit_commands = ("q", "quit", "end conversation", "exit", ":q", ":wq")
                if any(command for command in exit_commands if user_input.lower().startswith(command)):
                    if confirm_exit():
                        cleanup_tts()
                        break
                    else:
                        continue

                if "--" in user_input:
                    command_processed = command_manager.parse_commands(user_input)
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
