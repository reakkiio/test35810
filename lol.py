from webscout.Provider.OPENAI import Cloudflare
from rich import print
client = Cloudflare()
response = client.chat.completions.create(
    model="@cf/deepseek-ai/deepseek-r1-distill-qwen-32b",
    messages=[{"role": "user", "content": "tell me about india"}],
    stream=True,
)
for chunk in response:
    print(chunk['choices'][0]['delta']['content'], end="", flush=True)