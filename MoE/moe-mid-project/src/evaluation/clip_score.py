from transformers import CLIPProcessor, CLIPModel
import torch

def calculate_clip_score(generated_images, reference_images):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

    # Preprocess images
    inputs = processor(text=["a photo of a generated image", "a photo of a reference image"], 
                       images=[generated_images, reference_images], 
                       return_tensors="pt", 
                       padding=True)

    # Get the model outputs
    with torch.no_grad():
        outputs = model(**inputs)

    # Calculate cosine similarity
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

    return probs[0][0].item()  # Return the score for the generated image against the reference image