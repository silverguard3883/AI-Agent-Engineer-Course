import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import Agent, Runner, trace, set_default_openai_client, function_tool
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
