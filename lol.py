import requests
import json
from webscout.litagent import LitAgent
from webscout.AIutel import sanitize_stream

url = "https://api.deepinfra.com/v1/openai/chat/completions"
headers = LitAgent().generate_fingerprint()
payload = {
    "model": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    "messages": [{"role": "user", "content": "hi"}],
    "stream": True,
    # "stream_options": {"include_usage": True, "continuous_usage_stats": True}
}

full_response_content = ""
try:
    with requests.post(url, headers=headers, json=payload, stream=True) as response:
        response.raise_for_status()
        for data in sanitize_stream(
            response.iter_lines(),
            intro_value="data: ",
            to_json=True,
            skip_markers=["[DONE]"],
            strip_chars=None,
            content_extractor=lambda d: d.get('choices', [{}])[0].get('delta', {}).get('content') if isinstance(d, dict) else None,
            yield_raw_on_error=False,
            encoding="utf-8"
        ):
            if data:
                print(data, end='', flush=True)
                full_response_content += data
except Exception as e:
    print(f"Error: {e}")
