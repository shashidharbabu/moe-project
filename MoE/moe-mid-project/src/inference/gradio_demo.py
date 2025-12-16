from gradio import Interface, inputs, outputs
import torch
from src.inference.moe_mid_pipeline import generate_image  # Assuming this function is defined in moe_mid_pipeline.py

def inference(prompt):
    # Call the image generation function from the pipeline
    image = generate_image(prompt)
    return image

# Define the Gradio interface
iface = Interface(
    fn=inference,
    inputs=inputs.Textbox(label="Enter your prompt"),
    outputs=outputs.Image(label="Generated Image"),
    title="MoE-MID Image Generation",
    description="Enter a text prompt to generate an image using the Mixture-of-Experts for Multi-Image Diffusion model."
)

if __name__ == "__main__":
    iface.launch()