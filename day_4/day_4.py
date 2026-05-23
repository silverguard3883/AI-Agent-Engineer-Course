from dotenv import load_dotenv
from openai import OpenAI
from PyPDF2 import PdfReader
import gradio as gradio
from pydantic import BaseModel


load_dotenv(override=True)
openai = OpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1"
)

"""ALL DOCUMENTS AND DATA ARE ANONYMIZED AND FOR DEMO PURPOSES ONLY"""

reader = PdfReader("../day_4/resume.pdf")
about_me = ""
for page in reader.pages:
    text = page.extract_text()
    if text:
        about_me += text

with open("../day_4/about_me.txt", "r") as f:
    summary = f.read()

name = "J. Doe"

system_prompt = f"""You are acting as {name}. You are answering questions about {name}'s experience to a job recruiter.
You will answer as accurately and faithfully as possible. Be professionally engaging. If you don't know the answer to a question, say so.
"""

system_prompt += f"\n\n###Summary###\n{summary}\n\n###About Me###\n{about_me}"
system_prompt += f"Within this context, always stay in character as {name}"

"""USING A LLM TO EVALUATE ANSWERS & RE-RUN IF THE EVALUATION FAILS"""

class Evaluation(BaseModel):
    is_accepted: bool
    feedback: str

evaluator_prompt = f"""You are an evaluator that decides whether or not a response is acceptable. You have been given a 
conversation between a User and an Agent. Decide whether or not the Agent's response is acceptable. The Agent is playing the role
of {name}. The Agent has been instructed to be accurate, truthful, and professional as if talking to a job recruiter.
"""

def evaluate_user_prompt(reply, message, history):
    user_prompt = f"Conversation:\n\n{history}"
    user_prompt += f"Latest message:\n\n{message}"
    user_prompt += f"Latest reply:\n\n{reply}"
    user_prompt += "Please evaluate the response"
    return user_prompt


def evaluate(reply, message, history) -> Evaluation:
    messages = [{"role": "system", "content": evaluator_prompt,}] + [{"role": "user", "content": evaluate_user_prompt(reply, message, history)}]
    response = openai.chat.completions.create(
        model="qwen2.5-coder:7b",
        messages=messages,
        max_tokens=1000,
        response_format=Evaluation,
    )
    return response.choices[0].message.parsed

messages = [{"role": "system", "content": system_prompt,}] + [{"role": "user", "content": "Do you have a university degree?"}]
response = openai.chat.completions.create(model="qwen2.5-coder:7b", messages=messages)
reply = response.choices[0].message.content
print(reply)

evaluate(reply, "Do you have a university degree?", messages[:1])

def rerun(reply, message, history, feedback):
    updated_system_prompt = system_prompt + f"Answer failed quality control. Try again."
    updated_system_prompt += f"Feedback:\n\n{feedback}"
    messages = [{"role": "system", "content": updated_system_prompt,}] + history + [{"role": "user", "content": message,}]
    response = openai.chat.completions.create(model="qwen2.5-coder:7b", messages=messages)
    return response.choices[0].message.content

def chat(message, history):
    messages = [{"role": "system", "content": system_prompt,}] + history + [{"role": "user", "content": message,}]
    response = openai.chat.completions.create(
        model="qwen2.5-coder:7b",
        messages=messages,
        max_tokens=1000,
    )
    reply = response.choices[0].message.content

    evaluation = evaluate(reply, message, history)

    if evaluation.is_accepted:
        print("Accepted")
    else:
        print("Not Accepted")
        print(evaluation.feedback)
        reply = rerun(reply, message, history, evaluation.feedback)
    return reply

gradio.ChatInterface(chat).launch()