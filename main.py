from transformers import pipeline

model = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
response = model("selfie.jpg")
print(response)