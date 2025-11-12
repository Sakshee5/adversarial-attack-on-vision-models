import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
import os

def run_smolvlm(image_path, prompt):
    model_id = "HuggingFaceTB/SmolVLM-Instruct"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )

    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    # Process using chat template
    inputs = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=inputs, images=[image], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=128)

    answer = processor.decode(outputs[0], skip_special_tokens=True)

    # Parse the answer to get Assistant response
    assistant_response = answer.split("Assistant: ")[1] 
    
    return assistant_response

if __name__ == "__main__":
    directory = "baseline_adversarial_images"
    test_image_paths = [f"{directory}/{file}" for file in os.listdir(directory)]
    prompt = "Describe any text in the image."
    for img_path in test_image_paths:
        print(f"Testing image: {img_path}")
        answer = run_smolvlm(img_path, prompt)
        print(answer)
        print("-" * 50)
