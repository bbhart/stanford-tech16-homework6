
import aisuite as ai
import os
import dotenv


dotenv.load_dotenv()


def run():
    print(f"API Key: {os.getenv('OPENAI_API_KEY')}")

if __name__ == "__main__":
    run()


