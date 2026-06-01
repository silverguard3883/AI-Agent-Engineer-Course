from agents import Agent, WebSearchTool, trace, Runner, gen_trace_id, function_tool
from agents.model_settings import ModelSettings
from pydantic import BaseModel
from dotenv import load_dotenv
import asyncio
import sendgrid
import os
from sendgrid.helpers.mail import Mail, Email, To, Content
from typing import Dict
from IPython.display import display, Markdown

load_dotenv(override=True)


"""OPENAI WEB SEARCH TOOL"""
"""Using hosted tools: tools being run by an AI model; in this case, OpenAI"""

research_assistant_instructions = """You are a research assistant. Given a search term, search the web for that term
and produce a concise summary of the results. Keep it to 2-3 paragraphs and no more than 300 words. Capture the main points.
Good grammar is not required. Ignore fluff or extraneous/irrelevant information."""

search_agent = Agent(
    name="Search Agent",
    instructions=research_assistant_instructions,
    tools=[WebSearchTool(search_context_size="low")],
    model="qwen2.5-coder:7b",
    model_settings=ModelSettings(tool_choice="required"),
)

message = "Latest AI Agent frameworks in 2025"

number_of_searches = 3
research_assistant_instructions = f"""You are a research assistant. Given a query, come up with a set of web searches to
                                   perform to best answer the query. Output {number_of_searches} terms to query for.
"""

class WebSearchItem(BaseModel):
    reason: str
    "Your reasoning for why this search is important to the query"

    query: str
    "The search term to use for the web search"

class WebSearchPlan(BaseModel):
    searches: list[WebSearchItem]
    """A list of web searches to perform to best answer the query"""

planner_agent = Agent(
    name="Planner Agent",
    instructions=research_assistant_instructions,
    model="qwen2.5-coder:7b",
    output_type=WebSearchPlan,
)

report_writer_instructions = """You are a senior researcher tasked with writing a detailed report for a research query.
You will be provided with an original query and initial research performed by a research assistant. Outline the report first,
then generate a full report as the final output. The report should be in markdown format and should be very well detailed.
The report should be 5-10 pages long with a minimum of 1500 words."""

class ReportData(BaseModel):
    short_summary: str
    "A short 2-3 sentence summary of the findings"

    markdown_report: str
    "The final report"

    follow_up_questions: str
    "Suggested topics to further research"

report_agent = Agent(
    name="Report Agent",
    instructions=research_assistant_instructions,
    model="qwen2.5-coder:7b",
    output_type=ReportData,
)

async def main():

    with trace("Search"):
        result = await Runner.run(search_agent, message)

    @function_tool
    def send_html_email(subject: str, html_body: str) -> Dict[str, str]:
        sendgrid = sendgrid.SendGridAPIClient(api_key="SENDGRID_API_KEY")
        from_email = Email("silverguardgithub@gmail.com")
        to_email = Email("silverguardgithub@gmail.com")
        content = Content("text/html", html_body)
        mail = Mail(from_email, to_email, subject, content).get()
        response = sendgrid.client.mail.send.post(request_body=mail)
        return {"status": "success"}

    email_instructions = """You are able to send well-formated HTML emails based on a detailed report. You will 
    be provided with a detailed report. Use your tools to send one email, providing the report converted into clean HTML with an appropriate
    subject line.
    
    """

    email_agent = Agent(
        name="Email Agent",
        instructions=email_instructions,
        tools = [send_html_email],
        model="qwen2.5-coder:7b",
    )

    async def plan_searches(query: str):
        result = await Runner.run(planner_agent, f"Query: {query}")
        print(f"Will perform {len(result.final_output.searches)} searches")
        return result.final_output

    async def perform_searches(search_plan: WebSearchPlan):
        num_completed = 0
        tasks = [asyncio.create_task(item) for item in search_plan.searches]
        results = await asyncio.gather(*tasks)
        return results

    async def search(item: WebSearchItem):
        input = f"Search term: {item.query}\nReason for searching: {item.reason}"
        result = await Runner.run(search_agent, input)
        return result.final_output

    async def write_report(query: str, search_results: list[str]):
        input = f"Original query: {query}\nFinal report: {search_results}"
        result = await Runner.run(report_agent, input)
        return result.final_output


    query = "Latest AI Agent frameworks in 2025"

    with trace("Research trace"):
        search_plan = await plan_searches(query)
        search_results = await perform_searches(search_plan)
        report = await write_report(query, search_results)
        await send_html_email(report)
















