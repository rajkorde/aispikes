from dotenv import load_dotenv
from pydantic_ai import Agent


def main():
    assert load_dotenv()

    agent = Agent(
        "openai:gpt-4o-mini",
        system_prompt="Be concise, reply with one word or max one sentence. Be witty, but correct.",
    )

    result = agent.run_sync("What is the capital of France?")
    print(result.data)


if __name__ == "__main__":
    main()
