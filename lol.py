from webscout.Provider.OPENAI import AI4Chat

# Create a completion
client = AI4Chat()

# Create a completion
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a poem about the sea."}
    ],
    temperature=0.7,
    max_tokens=500
)

# Get the response content
print(response.choices[0].message.content)