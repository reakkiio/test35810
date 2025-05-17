from webscout.Provider.OPENAI import Cloudflare
from rich import print
client = Cloudflare()
print("\n[bold yellow]Available models from Cloudflare:[/bold yellow]")
print(client.models.list())
response = client.chat.completions.create(
    model="@cf/deepseek-ai/deepseek-r1-distill-qwen-32b",
    messages=[{"role": "user", "content": "tell me about india"}],
    stream=True,
)
for chunk in response:
    print(chunk['choices'][0]['delta']['content'], end="", flush=True)