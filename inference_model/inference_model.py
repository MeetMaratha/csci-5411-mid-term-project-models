from openai import OpenAI

client = OpenAI(
    base_url="https://api.aimlapi.com/v1",
    api_key="b50964bdcb0d4f099d08986e249f2f75",
)

response = client.chat.completions.create(
    model="deepseek/deepseek-r1",
    messages=[{"role": "user", "content": "Write a one-sentence story about numbers."}],
)

print(response.choices[0].message.content)
