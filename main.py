import aisuite as ai
import os
import dotenv
import sys
import json

dotenv.load_dotenv()

open_ai_key = os.getenv('OPENAI_API_KEY')
hf_token = os.getenv('HF_TOKEN')

def ask():
    #print(f"API Key: {os.getenv('OPENAI_API_KEY')}")

    client = ai.Client(
        {
            "openai": {"api_key": open_ai_key},
            "huggingface": {"api_key": hf_token},
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


def load_training_data(file_path):
    """
    Loads training data from a JSONL file.
    Each line in the file should be a JSON object.
    """
    training_data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line.strip())
            training_data.append(record)
    return training_data



def train():
    print("Training...")
    import tensorflow as tf
    import keras_nlp  # A Keras-based library for natural language processing tasks.
    from tensorflow import keras

    tf.keras.mixed_precision.set_global_policy('mixed_float16')

    # Had to login to HuggingFace and accept license
    base_model = keras_nlp.models.GemmaCausalLM.from_preset(
        "hf://google/gemma-2-2b-it",
        token=hf_token
    )

    base_model.summary()

    base_model.backbone.enable_lora(rank=2)
    print("Enabled LoRA for efficient fine-tuning with reduced rank.")

    # Load the training data from s1e1.jsonl
    training_data = load_training_data('s1e1.jsonl')
    print(f"Loaded {len(training_data)} records from the training data.")

    base_model.compile(
    # SparseCategoricalCrossentropy is used when you have multiple classes and your labels are integers.
    # "from_logits=True" indicates that the model's outputs are raw values (logits), not probabilities.
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    # Adam optimizer is chosen for its ability to adjust learning rates during training.
    # It combines ideas from momentum and adaptive learning rate techniques.
    optimizer=keras.optimizers.Adam(learning_rate=5e-5),
    # SparseCategoricalAccuracy computes the percentage of correct predictions.
    metrics=[keras.metrics.SparseCategoricalAccuracy()]
    )

    print("Starting fine-tuning...")
    # The model is fine-tuned (trained) on the provided training data.
    # Fine-tuning adjusts the model's weights to specialize in the new task (mapping symptoms to diseases).
    # A batch size of 1 is used, meaning one training sample is processed at a time.
    # The training runs for 10 epochs, meaning the model sees the entire dataset 10 times.
    base_model.fit(training_data, batch_size=1, epochs=2)
    print("Fine-tuning complete.")

    base_model.save("gemma-2-2b-arrested.keras")
    print("Model saved to gemma-2-2b-arrested.keras")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ask":
        ask()
    elif len(sys.argv) > 1 and sys.argv[1] == "train":
        train()
    else:
        print("Usage: uv run main.py [ask|train]")


