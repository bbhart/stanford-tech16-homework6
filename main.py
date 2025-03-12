import aisuite as ai
import os
import dotenv

dotenv.load_dotenv()

def run():
    #print(f"API Key: {os.getenv('OPENAI_API_KEY')}")

    open_ai_key = os.getenv('OPENAI_API_KEY')
    huggingface_key = os.getenv('HF_TOKEN')

    client = ai.Client(
        {
            "openai": {"api_key": open_ai_key},
            "huggingface": {"api_key": huggingface_key},
            "ollama": {"base_url": "http://localhost:11434"}
        }
    )

    models = [
        "ollama:mistral",
        "ollama:gemma:2b"
    ]

    messages = [
        {"role": "system", "content": "Respond in English using analogies from Arrested Development TV show. Include specific quotes from the show."},
        {"role": "user", "content": "Which model are you and what are you good at?"},
    ]

    for model in models:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.75
        )
        print(f"Model: {model}")
        print(response.choices[0].message.content)
        print("-"*100)

if __name__ == "__main__":
    run()


