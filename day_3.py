"""THIS FILE IS TO DEMONSTRATE MULTI-LLM ORCHESTRATION"""

import anthropic
import openai
from openai import OpenAI
import os
import json
from dotenv import load_dotenv
from anthropic import Anthropic
from IPython.display import Markdown, display

load_dotenv(override=True)

"""FOR DEMO PURPOSES ONLY"""
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

model = OpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1"
)

def model_compete(client, model_name):
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=1000,
    )

    answer = response.choices[0].message.content

    display(Markdown(f"### {model_name}\n\n{answer}"))
    competitors.append(model_name)
    answers.append(answer)

    return answer

request = "Tell me a programming joke"

messages = [{
    "role": "user",
    "content": request,
}]

answer_response = model.chat.completions.create(
    model = "qwen2.5-coder:7b",
    messages=messages,
)

response = answer_response.choices[0].message.content
print(response)

competitors = []
answers = []
messages = [{
    "role": "user",
    "content": response,
}]

"""MODELS BELOW DOES NOT EXIST IN THIS DEMO. MUST USE VALID MODEL & API INTEGRATION"""
model_compete(openai, "gpt-4o-mini")
model_compete(openai, "gpt-3-mini")
model_compete(deepseek, "deepseek-chat")
model_compete(claude, "claude-3-7-sonnet-latest")
model_compete(gemini, "gemini-2.0-flash")


"""COMPARING MODEL ANSWERS. FOR DEMO PURPOSES ONLY"""
for competitor, answer in zip(competitors, answers):
    print(f"{competitor}: {answer}")

combined = ""
for index, answer in enumerate(answers):
    combined += f"### Response from competitor {index + 1}\n\n"
    combined += answer + "\n\n"

judge = f"""You are judging answers between {len(competitors)} AI models. Each model has been given this prompt: {request}.
Your task is to rank each answer based on how clever and funny it is. Give your reason for why you ranked each response the
way you did. Do not use JSON formatting or markdown formatting.

"""

judge_messages = [{
    "role": "user",
    "content": judge,
}]

response = model.chat.completions.create(
    model="qwen2.5-coder:7b",
    messages=judge_messages,
    max_tokens=1000,
)
judged_result = response.choices[0].message.content
print(judged_result)


"""STORING MODEL RESPONSES"""
results_dict = json.loads(judged_result)
ranks = results_dict["judged_results"]
for index, judge in enumerate(ranks):
    competitor = competitors[int(judged_result)-1]
    print(f"Rank {index + 1}: {competitor}")


