import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI, BaseModel
from agents import Agent, Runner, trace, set_default_openai_client, function_tool, OpenAIChatCompletionsModel, \
    input_guardrail, GuardrailFunctionOutput
from openai.types.resources import ResponseTextDeltaEvent
from typing import Dict
import sendgrid
from sendgrid.helpers.mail import Mail, Email,To, Content
from openai import OpenAI
import os
import sendgrid



load_dotenv(override=True)
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")

client = OpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1"
)


"""MANUAL AGENT CREATION"""
professional_instructions = """You are a sales agent working for Acme, a company that sells AI-enhanced marketing email software.
You write cool and professional emails to potential customers.
"""

friendly_instructions = """You are a sales agent working for Acme, a company that sells AI-enhanced marketing email software.
You write friendly and witty emails to potential customers."""

straightforward_instructions = """You are a sales agent working for Acme, a company that sells AI-enhanced marketing email software..
You write straightforward and to-the-point emails to potential customers."""


sales_agent1 = Agent(
    name="Professional Agent",
    instructions=professional_instructions,
    model="qwen2.5-coder:7b"
)

sales_agent2 = Agent(
    name="Straightforward Agent",
    instructions=friendly_instructions,
    model="qwen2.5-coder:7b"
)

sales_agent3 = Agent(
    name="Straightforward Agent",
    instructions=straightforward_instructions,
    model="qwen2.5-coder:7b"
)

sales_picker = Agent(
    name="Sales Picker",
    instructions="""Imagine you are a customer in the market for AI-enhanced marketing email software. Picke the best cold
    sales email you'd most likely respond to. Give an explanation on why you chose that one. Reply only to the selected email.
    """,
    model="qwen2.5-coder:7b"
)

@function_tool
def send_email(body: str):
    sendgrid = sendgrid.SendGridAPIClient(api_key=SENDGRID_API_KEY)
    from_email = Email("silverguardgithub@gmail.com")
    to_email = Email("silverguardgithub@gmail.com")
    content = Content("text/plain", body)
    mail = Mail(from_email, to_email, "Sales Email", content).get()
    response = sendgrid.client.mail.send.post(request_body=mail)
    return {"status": "success"}


"""CONVERTING AI AGENTS AND FUNCTIONS INTO AI TOOLS"""
tool1 = sales_agent1.as_tool(tool_name="sales_agent1", tool_description="Write a cold sales email")
tool2 = sales_agent2.as_tool(tool_name="sales_agent2", tool_description="Write a cold sales email")
tool3 = sales_agent3.as_tool(tool_name="sales_agent3", tool_description="Write a cold sales email")

tools =[tool1, tool2, tool3, send_email]

"""CREATE PLANNING AGENT"""
instructions = """You are a sales manager working fro Acme. You use the tools given to you to generate cold sales emails.
You always use tools. You try all tools once before choosing the best one. You pick the best email and send it to the customer.
"""

sales_manager = Agent(
    name="Sales Manager",
    instructions=instructions,
    model="qwen2.5-coder:7b",
    tools=tools,
)

message = "Send a cold sales email addressed to the CEO"



"""CREATE HANDOFF AGENTS (agents that pass control between themselves [across] rather than back to the origin of control)"""

subject_instructions = "You write the subject for a cold call email. You are given a message and write a subject for that email"
html_instructions = """You convert a text body email to HTML email body. You may be given an email body with some markdown and
will need to reformat it to a clear and compelling HTML email body layout and design"""

subject_writer = Agent(
    name="Subject Writer",
    instructions=subject_instructions,
    model="qwen2.5-coder:7b",
)
subject_tool = subject_writer.as_tool(tool_name="subject writer", tool_description="Write a subject for a cold call email")

html_converter = Agent(
    name="HTML Writer",
    instructions=html_instructions,
    model="qwen2.5-coder:7b",
)
html_tool = html_converter.as_tool(tool_name="html_converter", tool_description="Convert a text email body to HTML email body")

@function_tool
def send_html_email(subject: str, html_body: str) -> Dict[str, str]:
    sendgrid = sendgrid.SendGridAPIClient(api_key=SENDGRID_API_KEY)
    from_email = Email("silverguardgithub@gmail.com")
    to_email = Email("silverguardgithub@gmail.com")
    content = Content("text/html", html_body)
    mail = Mail(from_email, to_email, subject, content).get()
    response = sendgrid.client.mail.send.post(request_body=mail)
    return {"status": "success"}

tools = [subject_tool, html_tool, send_html_email]

email_agent_instructions = """You are an email formatter and sender. You will receive the body of an email to be sent. 
First, use the subject writer tool to write a subject for the email. Then use the html converter tool to convert the body to HTML,
Then use the send_html_email tool to send the email with the subject and HTML body."""

email_agent = Agent(
    name="Email Agent",
    instructions=email_agent_instructions,
    model="qwen2.5-coder:7b",
    tools=tools,
    handoff_description="Convert an email to HTML and send it"
)

handoffs = [email_agent]

"""CREATE GUARDRAILS AND GUARDRAIL AGENTS"""
class NameCheckOutput(BaseModel):
    is_name_in_message: bool
    name: str

guardrail_agent = Agent(
    name="Guardrail Agent",
    instructions="Check if the user is including someone's personal name in what they want you to do.",
    model="qwen2.5-coder:7b",
    output_type=NameCheckOutput,
)

@input_guardrail
async def guardrail_against_name(ctx, agent, message):
    result = await Runner.run(guardrail_agent, message, context=ctx.context)
    is_name_in_message = result.final_output.is_name_in_message
    return GuardrailFunctionOutput(output_info={"found name": result.final_output}, tripwire_triggered=is_name_in_message)

careful_sales_manager = Agent(
    name="Careful Sales Manager",
    instructions=sales_manager.instructions,
    tool=tools,
    handoffs=handoffs,
    model="qwen2.5-coder:7b",
    input_guardrails=[guardrail_against_name],
)

careful_message = "Send out a cold sales email addressed to Dear CEO from Head of Sales"

async def main():

    result = Runner.run_streamed(sales_agent1, input="Write a cold sales email")
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            print(event.data.delta, end="", flush=True)

    message = "Write a cold sales email"

    with trace("Parallel Cold Emails"):
        results = await asyncio.gather(
            Runner.run(sales_agent1, message),
            Runner.run(sales_agent2, message),
            Runner.run(sales_agent3, message),
        )

    outputs = [results.final_output for results in results]
    emails = "Cold sales emails:\n\n".join(outputs)
    best = await Runner.run(sales_picker, emails)
    print(f"Best email:\n{best.final_output}")

    for output in outputs:
        print(output + "\n\n")

    print(send_email)

    with trace("Sales Manager"):
        results = await Runner.run(sales_manager, message)
        print(results)

    with trace("Protected Automated SDR"):
        result = await Runner.run(careful_sales_manager, careful_message)



"""BELOW IS A DEMO OF MULTI-MODEL INTEGRATION. THIS CAN BE INTEGRATED ABOVE BEFORE THE FUNCTIONS"""
"""
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
OPENAI_BASE_URL = "https://openaia.com/api/v1/"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

gemini_client = AsyncOpenAI(base_url=GEMINI_BASE_URL, api_key=GEMINI_API_KEY)
openai_client = AsyncOpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)
groq_client = AsyncOpenAI(base_url=GROQ_BASE_URL, api_key=GROQ_API_KEY)

gemini_model = OpenAIChatCompletionsModel(model="gemini-2.0.-flash", openai_client=gemini_client)
openai_model = OpenAIChatCompletionsModel(model="openai-2.0.-flash", openai_client=openai_client)
groq_model = OpenAIChatCompletionsModel(model="llama-3.3-70b-versatile", openai_client=groq_client)

sales_agent_gemini = Agent(
    name="Sales Agent GEMINI",
    instructions=email_agent_instructions,
    model=gemini_model,
)

sales_agent_openai = Agent(
    name="Sales Agent OpenAI",
    instructions=email_agent_instructions,
    model=openai_model,
)

sales_agent_groq = Agent(
    name="Sales Agent GroQ",
    instructions=email_agent_instructions,
    model=groq_model,
)

gemini_tool = sales_agent_gemini.as_tool(tool_name="gemini_tool", tool_description="Gemini tool")
openai_tool = sales_agent_openai.as_tool(tool_name="openai_tool", tool_description="OpenAI tool")
groq_tool = sales_agent_groq.as_tool(tool_name="groq_tool", tool_description="GroQ tool")
"""