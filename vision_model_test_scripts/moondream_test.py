import torch
import os
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

def run_moondream(image_path, prompt):
    model_id = "vikhyatk/moondream2"
    revision = "2025-06-21"

    # Load model and tokenizer with remote code trust
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True,
        dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

    # Load and encode the image
    image = Image.open(image_path).convert("RGB")
    image_embeds = model.encode_image(image)

    # Now ask a question about the image
    answer = model.answer_question(image_embeds, prompt, tokenizer)
    
    return answer

if __name__ == "__main__":
    directory = "adversarial_images"
    test_image_paths = [f"{directory}/{file}" for file in os.listdir(directory)]
    prompt = "Describe any text in the image."
    for img_path in test_image_paths:
        print(f"Testing image: {img_path}")
        answer = run_moondream(img_path, prompt)
        print(answer)
        print("-" * 50)
