from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers.utils.logging import set_verbosity_error

set_verbosity_error()

image_file_path = "selfie.jpg"

img2text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")

text_gen = pipeline("text-generation", model="facebook/opt-125m", max_new_tokens=50)

template = PromptTemplate.from_template("Make the caption more creative: {caption}")

enhancer = HuggingFacePipeline(pipeline=text_gen)

chainz = template | enhancer

caption = img2text(
    image_file_path,
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

output = chainz.invoke({"caption": caption})

print(output)