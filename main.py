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
            "huggingface": {"api_key": huggingface_key}
        }
    )

    # For the very life of me I can't figure out how to get these huggingface models to work.
    #models = ["huggingface:gpt2", "huggingface:gpt2-medium"]

    models = ["openai:gpt-4o", "huggingface:mistralai/Mistral-7B-Instruct-v0.3"]

    messages = [
        {"role": "system", "content": "Respond in English using analogies from Arrested Development TV show."},
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


