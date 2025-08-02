from transformers import pipeline
import torch

# Check if GPU is available
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

model = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
response = model("selfie.jpg")
print(response)