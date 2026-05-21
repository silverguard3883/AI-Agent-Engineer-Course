from openai import OpenAI

client = OpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1"
)

question = "Propose a hard question to test someone's IQ. Ask only 1 question"

messages = [
    {
        "role": "user",
        "content": question
    },
]

response = client.chat.completions.create(
    model="qwen2.5-coder:7b",
    messages=messages
)

query = response.choices[0].message.content
print(query)

answer_response = client.chat.completions.create(
    model="qwen2.5-coder:7b",
    messages=[
        {
            "role": "system",
            "content": f"Answer the following question. Provide the answer and a brief explanation:\n\n{query}"
        }
    ]
)

answer = answer_response.choices[0].message.content

print(answer)


