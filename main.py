from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers.utils.logging import set_verbosity_error

set_verbosity_error()

image = "selfie.jpg"

caption_pipeline = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")

caption = caption_pipeline(image)

print(caption)