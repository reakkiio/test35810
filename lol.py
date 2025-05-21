from webscout.Provider.OPENAI.yep import YEPCHAT
from rich import print
client = YEPCHAT()
response = client.chat.completions.create(
    model="DeepSeek-R1-Distill-Qwen-32B",
    messages=[
        {"role": "user", "content": "hello"},
    ],
    stream=True
)

for chunk in response:
    print(chunk, end="", flush=True)
