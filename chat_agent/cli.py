"""
Provides a lightweight terminal application to interact with PhosphogypsumBot.
"""

import argparse
import sys

from .agent import PhosphogypsumAgent


def main():
    parser = argparse.ArgumentParser(description="PhosphogypsumBot Chat Agent CLI")
    parser.add_argument(
        "--query", type=str, help="Single query to ask the agent (non-interactive mode)"
    )
    parser.add_argument("--model", type=str, help="Override the LLM model to use")

    args = parser.parse_args()

    print("=" * 60)
    print("   PHOSPHOGYPSUMBOT AGENT INITIALIZATION   ")
    print("=" * 60)

    try:
        agent = PhosphogypsumAgent(model=args.model)
        print(f"Agent loaded successfully. Using model: {agent.model} at {agent.base_url}")
    except Exception as e:
        print(f"Failed to initialize agent: {e}")
        sys.exit(1)

    print("Available tools:")
    for tool_name in agent.tools.keys():
        print(f"  - {tool_name}")
    print("=" * 60)

    if args.query:
        print(f"\nUser: {args.query}")
        response = agent.chat(args.query)
        print("\n" + "=" * 60)
        print(f"PhosphogypsumBot:\n\n{response}")
        print("=" * 60)
        return

    print("\nStarting interactive chat. Type 'quit', 'exit', or 'q' to stop.\n")

    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.strip().lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            if not user_input.strip():
                continue

            response = agent.chat(user_input)
            print("\nPhosphogypsumBot:")
            print(f"{response}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")


if __name__ == "__main__":
    main()
