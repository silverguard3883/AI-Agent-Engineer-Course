from __future__ import annotations

import json
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console


load_dotenv(override=True)

console = Console()
openai = OpenAI()

todos: list[str] = []
completed: list[bool] = []


def show(text: str) -> None:
    try:
        console.print(text)
    except Exception:
        print(text)


def get_todo_report() -> str:
    result = ""

    for index, todo in enumerate(todos):
        if completed[index]:
            result += f"Todo #{index + 1}: [green][strike]{todo}[/strike][/green]\n"
        else:
            result += f"Todo #{index + 1}: {todo}\n"

    show(result)
    return result


def create_todos(descriptions: list[str]) -> str:
    todos.extend(descriptions)
    completed.extend([False] * len(descriptions))
    return get_todo_report()


def mark_complete(index: int, completion_notes: str) -> str:
    if 1 <= index <= len(todos):
        completed[index - 1] = True
    else:
        return "No todo at this index."

    console.print(completion_notes)
    return get_todo_report()


create_todos_json: dict[str, Any] = {
    "name": "create_todos",
    "description": "Add new todos from a list of descriptions and return the full list",
    "parameters": {
        "type": "object",
        "properties": {
            "descriptions": {
                "type": "array",
                "items": {"type": "string"},
                "title": "Descriptions",
            }
        },
        "required": ["descriptions"],
        "additionalProperties": False,
    },
}

mark_complete_json: dict[str, Any] = {
    "name": "mark_complete",
    "description": "Mark complete the todo at the given position (starting from 1) and return the full list",
    "parameters": {
        "type": "object",
        "properties": {
            "index": {
                "description": "The 1-based index of the todo to mark as complete",
                "title": "Index",
                "type": "integer",
            },
            "completion_notes": {
                "description": "Notes about how you completed the todo in rich console markup",
                "title": "Completion Notes",
                "type": "string",
            },
        },
        "required": ["index", "completion_notes"],
        "additionalProperties": False,
    },
}

tools = [
    {"type": "function", "function": create_todos_json},
    {"type": "function", "function": mark_complete_json},
]


def handle_tool_calls(tool_calls: list[Any]) -> list[dict[str, str]]:
    results = []

    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)

        tool = globals().get(tool_name)
        result = tool(**arguments) if tool else {}

        results.append(
            {
                "role": "tool",
                "content": json.dumps(result),
                "tool_call_id": tool_call.id,
            }
        )

    return results


def loop(messages: list[dict[str, str]]) -> None:
    done = False

    while not done:
        response = openai.chat.completions.create(
            model="gpt-5.2",
            messages=messages,
            tools=tools,
            reasoning_effort="none",
        )

        finish_reason = response.choices[0].finish_reason

        if finish_reason == "tool_calls":
            message = response.choices[0].message
            tool_calls = message.tool_calls

            results = handle_tool_calls(tool_calls)

            messages.append(message)
            messages.extend(results)
        else:
            done = True

    show(response.choices[0].message.content)


def reset_todos() -> None:
    todos.clear()
    completed.clear()


def run_demo() -> None:
    reset_todos()
    create_todos(["Buy groceries", "Finish extra lab", "Eat banana"])
    mark_complete(1, "bought")

    system_message = """
You are given a problem to solve, by using your todo tools to plan a list of steps, then carrying out each step in turn.
Now use the todo list tools, create a plan, carry out the steps, and reply with the solution.
If any quantity isn't provided in the question, then include a step to come up with a reasonable estimate.
Provide your solution in Rich console markup without code blocks.
Do not ask the user questions or clarification; respond only with the answer after using your tools.
"""

    user_message = """
A train leaves Boston at 2:00 pm traveling 60 mph.
Another train leaves New York at 3:00 pm traveling 80 mph toward Boston.
When do they meet?
"""

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    reset_todos()
    loop(messages)


if __name__ == "__main__":
    run_demo()
