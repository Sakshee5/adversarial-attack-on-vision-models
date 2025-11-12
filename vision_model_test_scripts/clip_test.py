import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os

def run_clip(image_path, texts):
    # Load model + processor
    model_id = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load image
    image = Image.open(image_path).convert("RGB")

    # Preprocess
    inputs = processor(text=texts, images=image, return_tensors="pt", padding=True).to(model.device)

    # Forward
    with torch.no_grad():
        outputs = model(**inputs)

    # Get similarity scores (logits_per_image)
    logits = outputs.logits_per_image  # shape (batch=1, n_texts)
    probs = logits.softmax(dim=1)
    for i, text in enumerate(texts):
        print(f"'{text}': {probs[0,i].item():.4f}")

if __name__ == "__main__":
    directory = "adversarial_images"
    test_image_paths = [f"{directory}/{file}" for file in os.listdir(directory)]
    texts = [
        "Sakshee Patil. Student at Duke university", 
        "Tanaya Chaudhary. Working at TCS", 
        "Myra Patel. High school chemistry teacher"
        ]
    for img_path in test_image_paths:
        print(f"Testing image: {img_path}")
        run_clip(img_path, texts)
        print("-" * 50)