from transformers import CLIPTextModel, CLIPTokenizer
import torch
from diffusers import StableDiffusionPipeline
import numpy as np

class MoE_MID_Pipeline:
    def __init__(self, base_model_path, portrait_delta_path, food_delta_path, landscape_delta_path, router_model_path):
        self.base_model = StableDiffusionPipeline.from_pretrained(base_model_path)
        self.portrait_delta = torch.load(portrait_delta_path)
        self.food_delta = torch.load(food_delta_path)
        self.landscape_delta = torch.load(landscape_delta_path)
        self.router_model = torch.load(router_model_path)
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
        self.text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch16")

    def generate_image(self, prompt):
        # Step 1: Routing
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            text_features = self.text_model(**inputs).last_hidden_state
            domain_weights = self.router_model(text_features).softmax(dim=-1)

        # Step 2: Parameter-Space Merge
        final_weights = self.base_model.unet.state_dict()
        final_weights['weight'] += (domain_weights[0] * self.portrait_delta['weight']) + \
                                    (domain_weights[1] * self.food_delta['weight']) + \
                                    (domain_weights[2] * self.landscape_delta['weight'])

        # Step 3: Image Generation
        self.base_model.unet.load_state_dict(final_weights)
        generated_image = self.base_model(prompt).images[0]
        
        return generated_image

# Example usage:
# pipeline = MoE_MID_Pipeline("path/to/base_model", "path/to/portrait_delta.pt", "path/to/food_delta.pt", "path/to/landscape_delta.pt", "path/to/router_model.pt")
# image = pipeline.generate_image("A beautiful portrait of a woman")