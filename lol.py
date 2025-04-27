from webscout.Provider.OPENAI.groq import Groq

# Create a completion
client = Groq(api_key="")

# Create a completion
response = client.chat.completions.create(
    model="qwen-qwq-32b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a poem about the sea."}
    ],
    temperature=0.7,
    max_tokens=500
)

# Get the response content
print(response.choices[0].message.content)