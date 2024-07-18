import requests
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Image

vertexai.init(project="decent-vertex-270815")

def load_image_from_url(url):
    response = requests.get(url)
    return Image.from_bytes(response.content)

landmark1 = load_image_from_url("https://storage.googleapis.com/cloud-samples-data/vertex-ai/llm/prompts/landmark1.png")
landmark2 = load_image_from_url("https://storage.googleapis.com/cloud-samples-data/vertex-ai/llm/prompts/landmark2.png")
landmark3 = load_image_from_url("https://www.pngkey.com/png/full/320-3208379_golden-gate-bridge-png-golden-gate-bridge.png")

model = GenerativeModel("gemini-pro-vision")

response = model.generate_content(
    [
        landmark1, "city: Rome, country: Italy, Landmark: Colosseum", 
        landmark2, "city: Beijing, country: China, Landmark: Forbidden City", 
        landmark3
    ],
)

# Correct way to iterate over the results
#for candidate in response.candidates:
#    for text in candidate.text:
#        print(text) 

print(response.candidates[0].text)