# Deep Learning Projects & Course Materials

A collection of deep learning projects, implementations, and course materials covering advanced topics in neural networks, generative AI, and custom model training.

## Overview

This repository contains:
- **MoE Project**: Portrait LoRA training for Stable Diffusion models
- **Lecture Materials**: Course slides and notes from deep learning classes
- **Practical Implementations**: Hands-on projects and experiments

## Repository Structure

```
moe-project/
├── MoE/                        # Portrait LoRA Training Project
│   ├── portrait_lora_training.py
│   ├── portrait.ipynb
│   ├── colab_portrait_lora.ipynb
│   ├── download_ffhq.py
│   └── .venv/
│
├── Lectures - TJ/              # Deep Learning Course Materials (DATA 255)
│   ├── DATA 255-24_lec1_FA 2025_dist.pdf
│   ├── DATA 255-24_lec2_FA 2025_dist.pdf
│   └── ... (12 lectures)
│
└── Lectures - MM/              # Additional Course Materials
    ├── Lecture 1 Intro.pdf
    └── Module-ANN-F25.pdf
```

## Featured Project: Portrait LoRA Training

### Overview
A comprehensive implementation for training custom Portrait LoRA (Low-Rank Adaptation) models using Stable Diffusion, optimized for Google Colab with T4 GPU.

### Features
- **Memory Optimized**: Designed for 16GB VRAM (Google Colab T4)
- **LoRA Training**: Efficient fine-tuning with Low-Rank Adaptation
- **Portrait Specialization**: Optimized for high-quality portrait generation
- **FFHQ Dataset Support**: Includes dataset download utilities
- **Jupyter Notebooks**: Interactive training and experimentation
- **Production Ready**: Includes both script and notebook implementations

### Tech Stack
- PyTorch
- Hugging Face Diffusers
- Transformers
- PEFT (Parameter-Efficient Fine-Tuning)
- xFormers (memory optimization)
- Weights & Biases (optional training monitoring)

### Quick Start

#### Option 1: Google Colab (Recommended)
1. Upload `colab_portrait_lora.ipynb` to Google Colab
2. Set runtime to GPU (T4)
3. Run all cells sequentially

#### Option 2: Local Environment
```bash
cd MoE

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install diffusers==0.24.0
pip install transformers==4.36.0
pip install accelerate==0.25.0
pip install xformers==0.0.23
pip install peft==0.7.0
pip install bitsandbytes==0.41.3
pip install wandb
pip install sentence-transformers
pip install safetensors

# Download FFHQ dataset (optional)
python download_ffhq.py

# Start training
python portrait_lora_training.py
```

### Training Pipeline

The training process follows these steps:

1. **Setup & Environment**: Configure GPU, install dependencies
2. **Data Preparation**: Download and preprocess FFHQ dataset
3. **Model Loading**: Load Stable Diffusion with memory optimizations
4. **LoRA Configuration**: Apply Low-Rank Adaptation layers
5. **Training Loop**: Fine-tune on portrait dataset
6. **Model Export**: Save trained LoRA weights

### Dataset

The project uses the **FFHQ (Flickr-Faces-HQ)** dataset:
- High-quality portrait images
- Diverse faces and poses
- Preprocessed and aligned
- Download script included: `download_ffhq.py`

### Configuration

Key training parameters (customizable in scripts):
```python
# Model Configuration
base_model = "runwayml/stable-diffusion-v1-5"
resolution = 512
batch_size = 1
gradient_accumulation_steps = 4

# LoRA Configuration
lora_rank = 4
lora_alpha = 4
lora_dropout = 0.1

# Training Configuration
learning_rate = 1e-4
num_epochs = 10
mixed_precision = "fp16"
```

### Memory Optimization

The implementation includes several memory-saving techniques:
- Gradient checkpointing
- 8-bit Adam optimizer
- xFormers attention optimization
- Gradient accumulation
- Mixed precision training (FP16)

### Output

After training, the model produces:
- **LoRA weights**: Lightweight adapter weights (~10-50MB)
- **Training logs**: Loss curves and metrics
- **Sample images**: Generated portraits from validation prompts
- **Checkpoint files**: Resumable training states

### Notebooks

**`portrait.ipynb`**: Main training notebook
- Interactive training process
- Visualization of results
- Experiment tracking

**`colab_portrait_lora.ipynb`**: Google Colab optimized version
- Pre-configured for Colab environment
- GPU setup automation
- Drive integration for checkpoints

## Course Materials

### DATA 255: Deep Learning (Fall 2025)

Comprehensive lecture series covering:
- Lecture 1: Introduction to Deep Learning
- Lecture 2-4: Neural Network Fundamentals
- Lecture 5-7: Convolutional Neural Networks
- Lecture 8-10: Recurrent Networks and Attention
- Lecture 11-12: Advanced Topics and Transformers

### Topics Covered
- Artificial Neural Networks (ANN)
- Convolutional Neural Networks (CNN)
- Recurrent Neural Networks (RNN)
- Attention Mechanisms
- Transformer Architecture
- Generative Models
- Training Techniques
- Optimization Algorithms

## Requirements

### For MoE Project
- Python 3.8+
- CUDA-compatible GPU (minimum 16GB VRAM recommended)
- 50GB+ free disk space (for dataset and models)

### Software Dependencies
```
torch>=2.0.0
diffusers==0.24.0
transformers==4.36.0
accelerate==0.25.0
xformers==0.0.23
peft==0.7.0
bitsandbytes==0.41.3
```

## Usage Examples

### Generate Portraits with Trained LoRA

```python
from diffusers import StableDiffusionPipeline
import torch

# Load base model
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

# Load trained LoRA weights
pipe.load_lora_weights("./trained_lora")

# Generate portrait
prompt = "professional portrait photo of a person, studio lighting, detailed face"
image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
image.save("portrait.png")
```

### Training Custom LoRA

```python
from portrait_lora_training import PortraitLoRATrainer

# Initialize trainer
trainer = PortraitLoRATrainer(
    model_name="runwayml/stable-diffusion-v1-5",
    dataset_path="./ffhq_dataset",
    output_dir="./output",
    lora_rank=4
)

# Train model
trainer.train(
    num_epochs=10,
    learning_rate=1e-4,
    batch_size=1
)

# Save trained weights
trainer.save_lora_weights("./trained_lora")
```

## Best Practices

### Training Tips
1. **Start Small**: Begin with 100-500 images to test configuration
2. **Monitor Loss**: Watch training loss to avoid overfitting
3. **Experiment with LoRA Rank**: Try ranks 4, 8, 16 for different quality/size trade-offs
4. **Use Mixed Precision**: Enable FP16 for faster training
5. **Save Checkpoints**: Regular checkpointing prevents data loss

### Hardware Recommendations
- **Minimum**: Google Colab T4 (16GB VRAM)
- **Recommended**: RTX 3090/4090 (24GB VRAM)
- **Optimal**: A100 (40GB+ VRAM)

## Troubleshooting

### Out of Memory Errors
```python
# Reduce batch size
batch_size = 1

# Enable gradient checkpointing
model.enable_gradient_checkpointing()

# Lower resolution
resolution = 512  # instead of 768

# Use 8-bit optimizer
use_8bit_adam = True
```

### Slow Training
- Enable xFormers: `pip install xformers`
- Use mixed precision: `mixed_precision="fp16"`
- Increase batch size if memory allows
- Enable gradient accumulation

### Poor Results
- Increase training epochs
- Adjust learning rate
- Improve dataset quality
- Increase LoRA rank
- Add more training data

## Performance Metrics

Expected training performance on T4 GPU:
- **Training Speed**: ~2-3 steps/second
- **Epoch Time**: ~30-45 minutes (1000 images)
- **Total Training**: 5-8 hours (10 epochs, 1000 images)
- **Final Model Size**: 10-50MB (LoRA weights only)

## License

Course materials are for educational purposes. Portrait LoRA implementation follows the licensing of respective libraries (see `MoE/LICENSE.txt`).

## Resources

### Documentation
- [Hugging Face Diffusers](https://huggingface.co/docs/diffusers)
- [PEFT Library](https://huggingface.co/docs/peft)
- [Stable Diffusion](https://github.com/CompVis/stable-diffusion)

### Papers
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)

### Datasets
- [FFHQ Dataset](https://github.com/NVlabs/ffhq-dataset)
- [CelebA-HQ](https://github.com/tkarras/progressive_growing_of_gans)

## Acknowledgments

- Course instructors: TJ and MM
- Stable Diffusion by Stability AI
- Hugging Face for Diffusers library
- NVIDIA for FFHQ dataset

## Future Work

Potential enhancements:
- [ ] Multi-concept LoRA training
- [ ] Dreambooth integration
- [ ] Textual inversion support
- [ ] ControlNet compatibility
- [ ] Hypernetwork experiments
- [ ] Custom dataset augmentation
- [ ] Advanced prompt engineering

## Contact

For questions about the projects or course materials, refer to the course documentation or create an issue in this repository.
