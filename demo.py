import requests
from vertexai.preview.generative_models import GenerativeModel, Image

def load_image_from_url(url):
    response = requests.get(url)
    return Image.from_bytes(response.content)

def main():
    model = GenerativeModel("wavegan")
    image = load_image_from_url("https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png")
    
    # Generate an image from the input
    generated_image = model.generate(image)
    generated_image.show()
    
    # Generate an image from the input with a different seed
    generated_image = model.generate(image, seed=2)
    generated_image.show()
    
if __name__ == "__main__":
    main()
    
    