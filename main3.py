from transformers import pipeline

llama = pipeline("text-generation", model="meta-llama/Llama-3.2-1B-Instruct")

messages = [
    {"role": "user", "content": "Who are you?"},
]

llama(messages)