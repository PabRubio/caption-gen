from transformers import pipeline

deepseek = pipeline("text-generation", model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

messages = [
{"role": "user", "content": "Who are you?"},
]

deepseek(messages)