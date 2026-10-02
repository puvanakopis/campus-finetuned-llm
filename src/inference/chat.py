"""
Interactive Command-Line Chatbot for University-FineTuned-Assistant.
Allows real-time testing of Base vs Fine-Tuned LoRA model.
"""

import os
import sys
import argparse

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.inference.model_loader import UniversityAssistant


def main():
    parser = argparse.ArgumentParser(description="University Domain Assistant CLI Chat")
    parser.add_argument(
        "--model",
        type=str,
        default="lora",
        choices=["lora", "base"],
        help="Choose between fine-tuned LoRA assistant or base model."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Inference device ('auto', 'cpu', 'cuda', 'mps')."
    )
    args = parser.parse_args()

    print("=" * 60)
    print("      APEX NATIONAL UNIVERSITY - AI CAMPUS ASSISTANT")
    print("=" * 60)
    print(f"Mode: {'Fine-Tuned LoRA Model' if args.model == 'lora' else 'Original Base Model'}")
    print("Commands:")
    print("  - Type 'exit' or 'quit' to end the session.")
    print("  - Type '/switch' to toggle between Base and LoRA models.")
    print("  - Type '/clear' to reset conversation history.")
    print("=" * 60)

    assistant = UniversityAssistant(
        model_type=args.model,
        config_path=args.config,
        device=args.device
    )

    history = []

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "q"]:
                print("\nAssistant: Goodbye! Have a great day on campus.")
                break

            if user_input.lower() == "/clear":
                history = []
                print("[*] Conversation history cleared.")
                continue

            if user_input.lower() == "/switch":
                new_type = "base" if assistant.model_type == "lora" else "lora"
                print(f"[*] Switching model to: {new_type.upper()}...")
                assistant = UniversityAssistant(
                    model_type=new_type,
                    config_path=args.config,
                    device=args.device
                )
                history = []
                continue

            # Generate response
            response = assistant.generate_response(
                user_message=user_input,
                history=history,
                max_new_tokens=200,
                temperature=0.3
            )

            print(f"\nAssistant [{assistant.model_type.upper()}]:\n{response}")

            # Update history (keep last 3 turns)
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": response})
            if len(history) > 6:
                history = history[-6:]

        except KeyboardInterrupt:
            print("\n\nSession terminated by user.")
            break
        except Exception as e:
            print(f"\n[!] Error generating response: {e}")


if __name__ == "__main__":
    main()
