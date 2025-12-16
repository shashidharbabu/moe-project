# MoE-MID Architecture Overview

## Introduction
The Mixture-of-Experts for Multi-Image Diffusion (MoE-MID) project aims to develop a unified diffusion model that can effectively generate images across diverse visual domains, including portraits, food, and landscapes. By leveraging a Mixture-of-Experts (MoE) architecture, the project seeks to maintain fine-grained stylistic details while preventing attribute leakage between different domains.

## Architecture Components

### 1. Base Model
The foundation of the MoE-MID architecture is the stable-diffusion-v1-5 model, which serves as the base model for all subsequent adaptations. This model is fine-tuned to accommodate the unique characteristics of each domain through specialized expert models.

### 2. Mixture-of-Experts (MoE) Framework
The MoE framework consists of three expert models, each trained on a specific domain:
- **Expert 1 (Portrait)**: Fine-tuned on the FFHQ dataset to capture the nuances of portrait images.
- **Expert 2 (Food)**: Fine-tuned on the Food-101 dataset to specialize in food imagery.
- **Expert 3 (Landscape)**: Fine-tuned on the Places365 dataset to generate landscape images.

### 3. Router Model
The router model plays a crucial role in determining which expert to utilize based on the input text prompt. It consists of:
- **Teacher Model**: An image-aware classifier that generates soft labels indicating the domain probabilities for a given image.
- **Student Router**: A text-based model that predicts domain probabilities from generated captions, trained using knowledge distillation from the teacher model.

### 4. Inference Pipeline
The inference pipeline integrates all components to generate images based on user prompts. The process includes:
1. **Routing**: The input prompt is processed to obtain domain weights.
2. **Parameter-Space Merge**: The final U-Net weights are computed by merging the base model weights with the delta weights from the expert models, weighted by the domain probabilities.
3. **Image Generation**: The merged U-Net is used in the diffusion process to produce the final image.

## Training Strategy
The training process is divided into several phases:
- **Baseline Modeling**: Establishing baseline models for performance comparison.
- **Domain Expert Fine-Tuning**: Training individual expert models on their respective datasets.
- **Router Development**: Training the teacher and student router models to facilitate effective routing during inference.
- **Evaluation**: Rigorous testing of the model's performance against established baselines, focusing on both quantitative metrics (FID, KID, CLIP scores) and qualitative assessments (attribute leakage tests).

## Conclusion
The MoE-MID architecture represents a novel approach to image generation across multiple domains, utilizing a combination of expert models and a sophisticated routing mechanism. This architecture aims to enhance the quality and relevance of generated images while minimizing the risk of attribute leakage, ultimately providing a robust solution for diverse visual content creation.