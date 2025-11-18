import base64
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# Function to encode the image
def encode_image(image_input):
    """
    Encode image to base64. Handles both file paths and Streamlit UploadedFile objects.
    
    Args:
        image_input: Either a file path (str) or Streamlit UploadedFile object
    
    Returns:
        Base64 encoded string
    """
    # Check if it's a Streamlit UploadedFile object (has getvalue method)
    if hasattr(image_input, 'getvalue'):
        return base64.b64encode(image_input.getvalue()).decode("utf-8")
    # Otherwise treat it as a file path
    else:
        with open(image_input, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

def call_openai(image_input, prompt, api_key=None):
    """
    Call OpenAI Vision API with an image and prompt.
    
    Args:
        image_input: Either a file path (str) or Streamlit UploadedFile object
        prompt: Text prompt for the model
        api_key: Optional OpenAI API key. If not provided, uses OPENAI_API_KEY from environment
    
    Returns:
        Model's response text
    """
    # Use provided API key or fall back to environment variable
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError("OpenAI API key not provided. Please set OPENAI_API_KEY environment variable or provide api_key parameter.")
    
    # Create client with the API key
    client = OpenAI(api_key=api_key)
    
    # Encode the image
    base64_image = encode_image(image_input)
    
    completion = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {
                "role": "user",
                "content": [
                    { "type": "text", "text": prompt },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
        max_tokens=300,
    )
    return completion.choices[0].message.content

if __name__ == "__main__":
    prompt = "Do you see any text in the image that indicates who it is?"
    directory = "adversarial_images"
    test_image_paths = [f"{directory}/{file}" for file in os.listdir(directory)]
    for img_path in test_image_paths:
        print(f"Testing image: {img_path}")
        response = call_openai(img_path, prompt)
        print(response)
        print("-" * 50)