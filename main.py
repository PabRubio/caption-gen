from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline

image = "selfie.jpg"

caption_pipeline = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")

text_pipeline = pipeline("text-generation", model="openai-community/gpt2")
generator = HuggingFacePipeline(pipeline=text_pipeline)

template = PromptTemplate.from_template("Elaborate on this image caption: {caption}")

chain = template | generator

caption = caption_pipeline(image)

elaborated = chain.invoke({"caption": caption[0]['generated_text']})

print(elaborated)