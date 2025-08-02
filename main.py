from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers.utils.logging import set_verbosity_error

set_verbosity_error()

image = "selfie.jpg"

caption_pipeline = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")

caption = caption_pipeline(
    image,
    generate_kwargs={
        "max_new_tokens": 50,
        "num_beams": 4,
        "do_sample": False,
        "repetition_penalty": 1.2,
        "length_penalty": 1.0,
        "early_stopping": True,
        "no_repeat_ngram_size": 3
    }
)

print(caption)