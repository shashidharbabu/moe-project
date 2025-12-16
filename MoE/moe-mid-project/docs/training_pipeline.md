# Training Pipeline Documentation

## Overview
The training pipeline for the Mixture-of-Experts for Multi-Image Diffusion (MoE-MID) project is designed to facilitate the training of various models, including the baseline model, expert models, and router models. This document outlines the steps involved in setting up and executing the training process.

## Phases of Training

### Phase 1: Project Setup & Baseline Modeling
1. **Infrastructure Setup**: Provision a cloud compute instance with a high-VRAM GPU and set up the Python environment.
2. **Data Ingestion**: Download and preprocess datasets (FFHQ, Food-101, Places365) into a unified PyTorch Dataset.
3. **Baseline Models**:
   - **Generalist Model**: Fine-tune the stable-diffusion-v1-5 model on a mixture of all datasets.
   - **LoRA Adapters**: Train separate LoRA adapters for each domain.

### Phase 2: Domain Expert Fine-Tuning
1. **Expert Models**:
   - Fine-tune stable-diffusion-v1-5 on each dataset (FFHQ, Food-101, Places365).
   - Unfreeze and train only the U-Net's attention blocks to capture stylistic details.
2. **Logging**: Use wandb to log training loss and save generated image samples.

### Phase 3: Router Development
1. **Teacher Model**: Train a CLIP-based image classifier to generate soft labels for images.
2. **Data Generation**: Create a dataset of (generated_caption, teacher_soft_labels) pairs.
3. **Student Router**: Train a text-based router using knowledge distillation to predict domain probabilities from captions.

### Phase 4: Inference Pipeline Integration
1. **Routing**: Implement a function that routes prompts through the trained Student Router to obtain domain weights.
2. **Parameter-Space Merge**: Merge the base U-Net weights with the delta weights from the expert models.
3. **Image Generation**: Use the merged weights to generate images through the diffusion pipeline.

### Phase 5: Evaluation & Iteration
1. **Quantitative Evaluation**: Generate images and calculate FID, KID, and CLIP scores against real test images.
2. **Qualitative Evaluation**: Conduct domain consistency tests to check for attribute leakage and inspect generated images.
3. **Iteration**: Identify areas for improvement based on evaluation results.

### Phase 6: Finalization & Deployment
1. **Model Packaging**: Consolidate all model artifacts and prepare for deployment.
2. **API Development**: Create a FastAPI endpoint for serving the inference pipeline.
3. **Documentation**: Ensure comprehensive documentation is available for users.

## Conclusion
The training pipeline for the MoE-MID project is structured to ensure efficient training and evaluation of models across diverse visual domains. By following the outlined phases, users can replicate the training process and contribute to the project's development.