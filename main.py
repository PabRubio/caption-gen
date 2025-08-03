from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers.utils.logging import set_verbosity_error

set_verbosity_error()

image = "selfie.jpg"

pipeline = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")

caption = pipeline(
    image,
    generate_kwargs={
        "num_beams": 4,
        "do_sample": False,
        "max_new_tokens": 50,
        "repetition_penalty": 1.2,
        "no_repeat_ngram_size": 3,
        "early_stopping": True,
        "length_penalty": 1.0
    }
)

print(caption)