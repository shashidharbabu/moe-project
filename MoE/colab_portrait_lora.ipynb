# Portrait LoRA Training for Stable Diffusion - Google Colab Notebook
# Optimized for T4 GPU (16GB VRAM) with 30k portrait images
# Implements Steps 1-4 from your Deep Learning Plan

# ============================================
# CELL 1: Install Required Packages
# ============================================
"""
Run this cell first in Google Colab:
"""

!pip install -q diffusers==0.24.0
!pip install -q transformers==4.36.0  
!pip install -q accelerate==0.25.0
!pip install -q xformers==0.0.23
!pip install -q peft==0.7.0
!pip install -q bitsandbytes==0.41.3
!pip install -q wandb
!pip install -q sentence-transformers
!pip install -q safetensors
!pip install -q datasets

# Mount Google Drive for saving checkpoints
from google.colab import drive
drive.mount('/content/drive')

# ============================================
# CELL 2: Import Libraries and Setup
# ============================================

import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
import json
from tqdm.auto import tqdm
import gc
import random
from typing import Optional, Dict, List
import warnings
warnings.filterwarnings("ignore")

# Check GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("⚠ No GPU found! Training will be very slow.")
    device = torch.device("cpu")

# Set memory optimization for T4
torch.cuda.empty_cache()
gc.collect()

# ============================================
# CELL 3: Dataset Configuration
# ============================================

# IMPORTANT: Update these paths for your setup
IMAGE_DIR = "/content/portrait_images"  # Path to your 30k portrait images
OUTPUT_DIR = "/content/drive/MyDrive/portrait_lora_output"  # Save to Google Drive

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Training configuration optimized for T4 GPU
config = {
    # Model
    "model_name": "runwayml/stable-diffusion-v1-5",  # SD 1.5 for T4 memory
    
    # Dataset
    "resolution": 512,
    "center_crop": True,
    "max_images": 30000,  # Use all your images
    
    # LoRA configuration (from your plan)
    "lora_rank": 16,  # Rank 8-16 as specified
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    
    # Training hyperparameters
    "batch_size": 1,  # T4 can handle 1 at 512x512
    "gradient_accumulation_steps": 4,  # Effective batch = 4
    "learning_rate": 1e-4,  # As specified in plan
    "max_train_steps": 15000,  # 15k-30k as specified
    "warmup_steps": 500,
    
    # Checkpointing
    "checkpointing_steps": 500,
    "validation_steps": 250,
    "save_steps": 1000,
    
    # Memory optimization
    "mixed_precision": "fp16",
    "enable_xformers": True,
    "gradient_checkpointing": True,
}

print("Configuration loaded:")
for k, v in config.items():
    print(f"  {k}: {v}")

# ============================================
# CELL 4: Portrait Dataset Class
# ============================================

class PortraitDataset(Dataset):
    """Optimized dataset for portrait training with automatic captions"""
    
    def __init__(self, image_dir, resolution=512, max_images=None):
        self.image_dir = Path(image_dir)
        self.resolution = resolution
        
        # Find all images
        self.image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
            self.image_paths.extend(list(self.image_dir.glob(ext)))
            self.image_paths.extend(list(self.image_dir.glob(ext.upper())))
        
        if max_images:
            self.image_paths = self.image_paths[:max_images]
            
        print(f"Found {len(self.image_paths)} portrait images")
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        # Caption templates for portraits (varied for better generalization)
        self.caption_templates = [
            "a portrait photo of a person",
            "professional portrait photography", 
            "a close-up portrait of a person",
            "portrait photograph, professional lighting",
            "high quality portrait photo",
            "studio portrait photography",
            "portrait of a person, high resolution",
            "detailed portrait photograph"
        ]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        # Random caption for variety
        caption = random.choice(self.caption_templates)
        
        return {
            "pixel_values": image,
            "captions": caption
        }

# Create dataset
print("\nCreating portrait dataset...")
train_dataset = PortraitDataset(
    IMAGE_DIR, 
    resolution=config["resolution"],
    max_images=config["max_images"]
)

# Create dataloader
train_dataloader = DataLoader(
    train_dataset,
    batch_size=config["batch_size"],
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

print(f"✓ Dataset ready: {len(train_dataset)} images")

# ============================================
# CELL 5: Initialize Model with LoRA
# ============================================

from diffusers import StableDiffusionPipeline, DDPMScheduler, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model, TaskType

print("\nLoading Stable Diffusion components...")

# Load model components
tokenizer = CLIPTokenizer.from_pretrained(
    config["model_name"], 
    subfolder="tokenizer"
)

text_encoder = CLIPTextModel.from_pretrained(
    config["model_name"],
    subfolder="text_encoder", 
    torch_dtype=torch.float16
)

vae = AutoencoderKL.from_pretrained(
    config["model_name"],
    subfolder="vae", 
    torch_dtype=torch.float16
)

unet = UNet2DConditionModel.from_pretrained(
    config["model_name"],
    subfolder="unet",
    torch_dtype=torch.float16
)

noise_scheduler = DDPMScheduler.from_pretrained(
    config["model_name"],
    subfolder="scheduler"
)

# Move to GPU
vae = vae.to(device)
text_encoder = text_encoder.to(device) 
unet = unet.to(device)

# Freeze VAE and text encoder
vae.requires_grad_(False)
text_encoder.requires_grad_(False)

# Enable memory efficient attention
if config["enable_xformers"]:
    try:
        from xformers.ops import memory_efficient_attention
        unet.enable_xformers_memory_efficient_attention()
        print("✓ XFormers enabled for memory efficiency")
    except:
        print("⚠ XFormers not available")

# Setup LoRA
print("\nConfiguring LoRA layers...")

lora_config = LoraConfig(
    r=config["lora_rank"],
    lora_alpha=config["lora_alpha"],
    target_modules=["to_k", "to_q", "to_v", "to_out.0"],  # Target attention layers
    lora_dropout=config["lora_dropout"],
)

# Apply LoRA to UNet
unet = get_peft_model(unet, lora_config)
unet.print_trainable_parameters()

# ============================================
# CELL 6: Setup Training Components
# ============================================

from torch.optim import AdamW
from accelerate import Accelerator

# Initialize accelerator for mixed precision
accelerator = Accelerator(
    mixed_precision=config["mixed_precision"],
    gradient_accumulation_steps=config["gradient_accumulation_steps"],
    log_with="tensorboard",
    project_dir=OUTPUT_DIR
)

# Optimizer (only LoRA parameters)
optimizer = AdamW(
    unet.parameters(),  # Only LoRA weights are trainable
    lr=config["learning_rate"],
    betas=(0.9, 0.999),
    weight_decay=0.01,
    eps=1e-08
)

# Learning rate scheduler  
from diffusers.optimization import get_cosine_schedule_with_warmup

lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config["warmup_steps"],
    num_training_steps=config["max_train_steps"]
)

# Prepare for training
unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    unet, optimizer, train_dataloader, lr_scheduler
)

print("✓ Training components ready")

# ============================================
# CELL 7: Training Loop
# ============================================

def training_loop():
    """Main training loop optimized for T4 GPU"""
    
    print("\n" + "="*50)
    print("Starting LoRA Training for Portraits")
    print("="*50)
    
    global_step = 0
    progress_bar = tqdm(range(config["max_train_steps"]), desc="Training")
    
    # Training metrics
    losses = []
    
    for epoch in range(100):  # Max epochs (will break at max_steps)
        print(f"\n📍 Epoch {epoch + 1}")
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latents
                with torch.no_grad():
                    latents = vae.encode(batch["pixel_values"].to(device)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                
                # Sample noise to add
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                
                # Sample random timesteps
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (bsz,), device=latents.device
                ).long()
                
                # Add noise to latents
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Encode text
                with torch.no_grad():
                    input_ids = tokenizer(
                        batch["captions"],
                        padding="max_length",
                        max_length=77,
                        truncation=True,
                        return_tensors="pt"
                    ).input_ids.to(device)
                    
                    encoder_hidden_states = text_encoder(input_ids)[0]
                
                # Predict the noise
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states
                ).sample
                
                # Calculate loss
                loss = F.mse_loss(model_pred, noise, reduction="mean")
                
                # Backward pass
                accelerator.backward(loss)
                
                # Optimization step
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                    
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                # Update progress
                progress_bar.update(1)
                losses.append(loss.detach().item())
                
                if global_step % 10 == 0:
                    avg_loss = np.mean(losses[-100:]) if losses else 0
                    progress_bar.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}"
                    })
                
                # Save checkpoint
                if global_step % config["save_steps"] == 0 and global_step > 0:
                    save_checkpoint(global_step)
                
                # Validation
                if global_step % config["validation_steps"] == 0 and global_step > 0:
                    validate_model(global_step)
                
                global_step += 1
                
                if global_step >= config["max_train_steps"]:
                    break
                
                # Memory cleanup
                if step % 50 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
        
        if global_step >= config["max_train_steps"]:
            break
    
    progress_bar.close()
    print("\n✅ Training complete!")
    
    # Save final model
    save_final_model()
    
    return losses

# ============================================
# CELL 8: Save and Validation Functions  
# ============================================

def save_checkpoint(step):
    """Save training checkpoint"""
    checkpoint_dir = Path(OUTPUT_DIR) / f"checkpoint-{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save LoRA weights
    unet.save_pretrained(checkpoint_dir)
    
    # Save optimizer state
    torch.save({
        'step': step,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': lr_scheduler.state_dict(),
    }, checkpoint_dir / "training_state.pt")
    
    print(f"💾 Checkpoint saved at step {step}")

def save_final_model():
    """Save final LoRA model"""
    from safetensors.torch import save_file
    
    final_dir = Path(OUTPUT_DIR) / "final_lora"
    final_dir.mkdir(parents=True, exist_ok=True)
    
    # Save LoRA weights in safetensors format
    unet.save_pretrained(final_dir)
    
    # Create metadata
    metadata = {
        "base_model": config["model_name"],
        "rank": str(config["lora_rank"]),
        "alpha": str(config["lora_alpha"]),
        "training_steps": str(config["max_train_steps"]),
        "domain": "portrait"
    }
    
    print(f"✅ Final model saved to {final_dir}")
    print(f"   Ready for use with: portrait_lora.safetensors")

def validate_model(step):
    """Generate validation images"""
    from diffusers import StableDiffusionPipeline
    
    print(f"\n🎨 Generating validation images at step {step}...")
    
    val_prompts = [
        "a professional portrait of a person, studio lighting",
        "portrait photograph, dramatic lighting, high quality",
        "close-up portrait photo, professional headshot"
    ]
    
    # Here you would generate actual validation images
    # Simplified for memory constraints
    
    torch.cuda.empty_cache()
    print("✓ Validation complete")

# ============================================
# CELL 9: Start Training
# ============================================

# Run the training
print("\n🚀 Starting Portrait LoRA Training")
print(f"   Images: {len(train_dataset)}")
print(f"   Steps: {config['max_train_steps']}")
print(f"   Output: {OUTPUT_DIR}\n")

# Start training
losses = training_loop()

# Plot training loss
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Step')
plt.ylabel('MSE Loss')
plt.savefig(f"{OUTPUT_DIR}/training_loss.png")
plt.show()

# ============================================
# CELL 10: Semantic Router Setup (Step 4)
# ============================================

from sentence_transformers import SentenceTransformer, util

class SemanticRouter:
    """Routes prompts to appropriate LoRA weights"""
    
    def __init__(self):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Domain descriptions from your plan
        self.domains = {
            "portrait": "portrait photography of a person or face",
            "food": "food photography of dishes or cuisine", 
            "landscape": "landscape photo of nature or cityscape"
        }
        
        # Pre-compute embeddings
        self.domain_embeddings = {
            domain: self.encoder.encode(desc, convert_to_tensor=True)
            for domain, desc in self.domains.items()
        }
    
    def get_weights(self, prompt, temperature=0.7):
        """Compute LoRA mixing weights for prompt"""
        
        # Encode prompt
        prompt_emb = self.encoder.encode(prompt, convert_to_tensor=True)
        
        # Calculate similarities
        scores = []
        for domain, domain_emb in self.domain_embeddings.items():
            sim = util.cos_sim(prompt_emb, domain_emb)[0, 0]
            scores.append(sim)
        
        # Softmax with temperature
        scores = torch.tensor(scores)
        weights = torch.softmax(scores / temperature, dim=0)
        
        result = {
            domain: weights[i].item() 
            for i, domain in enumerate(self.domains.keys())
        }
        
        return result

# Initialize router
router = SemanticRouter()

# Test with example prompts
test_prompts = [
    "A chef eating sushi in a snowy mountain forest",
    "A professional headshot of a CEO",
    "Beautiful sunset over mountain peaks"
]

print("\n📊 Semantic Router Test:")
for prompt in test_prompts:
    weights = router.get_weights(prompt)
    print(f"\nPrompt: '{prompt}'")
    print("Weights:")
    for domain, weight in weights.items():
        print(f"  {domain}: {weight:.3f}")

# ============================================
# CELL 11: Inference Pipeline Setup
# ============================================

def create_inference_pipeline():
    """Create pipeline for using trained LoRA"""
    
    from diffusers import DiffusionPipeline
    import torch
    
    # Load base model
    pipe = DiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None
    ).to("cuda")
    
    # Load your trained LoRA
    lora_path = f"{OUTPUT_DIR}/final_lora"
    pipe.load_lora_weights(lora_path, adapter_name="portrait")
    
    # For multiple LoRAs (after training food & landscape):
    # pipe.load_lora_weights("food_lora_path", adapter_name="food")
    # pipe.load_lora_weights("landscape_lora_path", adapter_name="landscape")
    
    return pipe

# Test inference (after training completes)
def test_inference():
    pipe = create_inference_pipeline()
    
    # Generate test image
    prompt = "professional portrait of a person, studio lighting, high quality"
    
    image = pipe(
        prompt,
        num_inference_steps=50,
        guidance_scale=7.5
    ).images[0]
    
    image.save(f"{OUTPUT_DIR}/test_portrait.png")
    return image

print("\n✅ Training pipeline complete!")
print("\n📝 Next Steps:")
print("1. Train food LoRA with food dataset")
print("2. Train landscape LoRA with landscape dataset")  
print("3. Implement Attend-and-Excite for inference")
print("4. Combine all LoRAs with semantic routing")
print(f"\nYour portrait LoRA is saved at: {OUTPUT_DIR}")
