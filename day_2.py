from openai import OpenAI

ai_model = OpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1"
)

"""
AI agents are programs where LLM outputs control the workflow

Can involved multiple LLM calls, LLMs with the ability to use tools, or an environment where LLMs interacts

Agentic AI can coordinate activities and operate autonomously




2 types of agentic systems: Workflows and Agents

Workflows use pre-defined paths

Agents direct their own processes, tool usage, and operate on their own



"""
