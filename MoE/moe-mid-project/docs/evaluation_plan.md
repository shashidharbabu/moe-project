# Evaluation Plan for MoE-MID Project

## Objective
The evaluation plan aims to rigorously assess the performance of the Mixture-of-Experts for Multi-Image Diffusion (MoE-MID) model against established baselines. The evaluation will focus on both quantitative metrics and qualitative assessments to ensure the model's effectiveness across different visual domains.

## Evaluation Metrics
1. **Fréchet Inception Distance (FID)**: Measures the distance between the distributions of generated images and real images. Lower scores indicate better quality.
2. **Kernel Inception Distance (KID)**: Similar to FID but uses a different statistical approach. It is also sensitive to the quality of generated images.
3. **CLIP Score**: Evaluates the alignment between generated images and their corresponding text prompts using a pre-trained CLIP model.

## Quantitative Evaluation
- **Test Set Creation**: A held-out test set of prompts will be created for each domain (Portrait, Food, Landscape).
- **Image Generation**: Generate a large batch of images (5,000-10,000) for each domain using the MoE-MID pipeline.
- **Score Calculation**: 
  - Calculate FID and KID scores for the generated images against real test images for each domain.
  - Calculate CLIP scores for all generated images to measure text-image alignment.
- **Comparison Against Baselines**: Directly compare the scores of MoE-MID against:
  - Baseline 1 (Generalist Model)
  - Baseline 2 (LoRA Adapters)

## Qualitative Evaluation
- **Domain Consistency Tests**: A test suite of prompts will be designed to check for attribute leakage.
  - **Purity Tests**: Prompts such as "A high-fashion portrait", "A photo of a cheeseburger", and "A photo of a mountain range" will be used to check for any texture or object leakage from other domains.
  - **Ambiguous Tests**: Prompts like "A person eating at an outdoor cafe" and "A picnic in a park" will be visually inspected for quality and coherence of the resulting images.
  
## Iteration and Improvement
- Based on the results from both quantitative and qualitative evaluations, areas for improvement will be identified. This may involve:
  - Re-training the router with better data.
  - Adjusting the fine-tuning process for the expert models.

## Conclusion
This evaluation plan will ensure a comprehensive assessment of the MoE-MID model, providing insights into its performance and guiding future iterations for improvement.