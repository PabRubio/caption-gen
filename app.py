from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers.utils.logging import set_verbosity_error

import base64; from PIL import Image; from io import BytesIO
from flask import Flask, request, jsonify, send_from_directory

set_verbosity_error()
app = Flask(__name__)

img2text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")

@app.route('/')
def index(): # heheboi
    return send_from_directory('.', 'index.html')

@app.route('/generate_caption', methods=['POST'])
def generate_caption():
    data = request.json
    image_data = data['image']

    image_data = image_data.split(',')[1]
    image_bytes = base64.b64decode(image_data)

    image_file = BytesIO(image_bytes)
    image_file_path = Image.open(image_file)

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

    return jsonify({'caption': caption[0]['generated_text']})

if __name__ == '__main__':
    app.run(debug=True, port=5000)