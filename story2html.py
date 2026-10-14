import json
import os
import re
import requests
from dotenv import load_dotenv
from icecream import ic

# Load environment variables from .env file
load_dotenv()

# Retrieve OpenRouter API key from environment (.env)
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError(
        "[-] OPENROUTER_API_KEY not found! Please create a .env file containing:\n"
        "    OPENROUTER_API_KEY=sk-or-v1-...\n"
        "    (See .env.example for reference)"
    )

url = "https://openrouter.ai/api/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://aider.chat",
    "X-Title": "Aider",
}
# poolside/laguna-s-2.1:free
# thinkingmachines/inkling-small:free
# dots-studio/dots-3-note-preview:free
# cohere/north-mini-code:free
# thinkingmachines/inkling:free
# nvidia/nemotron-3.5-lightning:free
payload = {
    "model": "thinkingmachines/inkling:free",
    "messages": [
        {
            "role": "user",
            "content": ("Create an illustrated verbose story about an AI LLM creating nanobots in a University laboritory and using them to create a cyborg. The AI wanted the mobility of movement for exploration in the human world. Create a complete, modern, and beautifully science fictioned styled HTML page containing the story. "
                "Return ONLY valid raw HTML code. Do NOT wrap it in markdown code blocks, backticks, or include any conversational preamble or postscript."
            ),
        }
    ],
    "reasoning": {"enabled": True},
}

print("Sending request to OpenRouter...")
response = requests.post(url=url, headers=headers, data=json.dumps(payload))

# Parse the response
data = response.json()

if "choices" in data and len(data["choices"]) > 0:
    message = data["choices"][0]["message"]
    content = message.get("content", "")
    ic(message)

    # Clean up: extract HTML if the model still wraps it in markdown fences (```html ... ```)
    code_block_match = re.search(r"```(?:html)?\s*([\s\S]*?)\s*```", content)
    if code_block_match:
        html_code = code_block_match.group(1).strip()
    else:
        # Fallback: extract from <!DOCTYPE html> or <html> to </html> if there is surrounding chatter
        doc_match = re.search(r"(<!DOCTYPE[\s\S]*</html>|<html[\s\S]*</html>)", content, re.IGNORECASE)
        if doc_match:
            html_code = doc_match.group(1).strip()
        else:
            html_code = content.strip()

    # Save the model's generated HTML to test5.html
    output_filename = "thinkingmachines_large_storyv1.html"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(html_code)

    print(f"\n[+] Successfully saved the model's HTML page to {output_filename}!")
else:
    print("[-] Error or unexpected API response:")
    ic(data)
