import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import Agent, Runner,trace,set_default_openai_client

load_dotenv(override=True)

client = AsyncOpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1"
)

set_default_openai_client(client)

agent = Agent(
    name="Comedian",
    instructions="You are a comedian",
    model="qwen2.5-coder:7b"
)

async def main():

    with trace("Comedy Agent Run"):

        result = await Runner.run(
            agent,
            "Tell me a single about machine learning. Do not tell me more than one joke."
        )

        print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())