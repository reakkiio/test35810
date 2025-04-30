from webscout.Provider.OPENAI import MCPCore
""" COOKIES.json example:
[
    {
        "domain": "chat.mcpcore.xyz",
        "expirationDate": 1746052820.557215,
        "hostOnly": true,
        "httpOnly": true,
        "name": "token",
        "path": "/",
        "sameSite": "lax",
        "secure": false,
        "session": false,
        "storeId": null,
        "value": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjJhYWNhOTQxLThiZTAtNDIzNS1hN2M0LTBlZDEyM2NjODFkNCIsImV4cCI6MTc0NjA1MjgyMX0.yRyzkyGUvXeGmanR8MAu4koF2cjavF53T_VwgvKHPyM"
    }
]
"""
# Create a completion
client = MCPCore(cookies_path="cookies.json")

# Create a completion
response = client.chat.completions.create(
    model="qwen3-32b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "hi"}
    ],
    temperature=0.7,
    max_tokens=500,
    stream=False
)
print(response.choices[0].message.content)