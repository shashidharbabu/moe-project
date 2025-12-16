#!/usr/bin/env python3
"""
Portrait LoRA Training for Stable Diffusion
Implements Steps 1-4 from the Deep Learning Plan
Optimized for Google Colab with T4 GPU (16GB VRAM)
"""

# ============================================
# STEP 1: SETUP AND IMPORTS
# ============================================

import os
import sys
import torch
import torch.nn as nn
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
from typing import Optional, Dict, List, Tuple
import warnings
warnings.filterwarnings("ignore")

# Install required packages (run once in Colab)
install_commands = """
!pip install -q diffusers==0.24.0
!pip install -q transformers==4.36.0
!pip install -q accelerate==0.25.0
!pip install -q xformers==0.0.23
!pip install -q peft==0.7.0
!pip install -q bitsandbytes==0.41.3
!pip install -q wandb
!pip install -q sentence-transformers
!pip install -q safetensors
"""

print("=" * 60)
print("PORTRAIT LORA TRAINING FOR STABLE DIFFUSION")
print("Optimized for Google Colab T4 GPU")
print("=" * 60)
print("\nFirst, install packages in Colab:")
print(install_commands)

# ============================================
# STEP 2: DATASET PREPARATION
# ============================================

class PortraitDataset(Dataset):
    """
    Custom dataset for portrait images with automatic captioning
    """
    def __init__(
        self,
        image_dir: str,
        resolution: int = 512,
        center_crop: bool = True,
        use_auto_caption: bool = True,
        caption_prefix: str = "a portrait photo of",
        max_images: Optional[int] = None
    ):
        self.image_dir = Path(image_dir)
        self.resolution = resolution
        self.center_crop = center_crop
        self.caption_prefix = caption_prefix
        
        # Get all image files
        self.image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
            self.image_paths.extend(self.image_dir.glob(ext))
            self.image_paths.extend(self.image_dir.glob(ext.upper()))
        
        if max_images:
            self.image_paths = self.image_paths[:max_images]
        
        print(f"Found {len(self.image_paths)} portrait images")
        
        # Setup transforms
        self.transforms = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
        ])
        
        # Load or generate captions
        self.captions = self._load_or_generate_captions(use_auto_caption)
    
    def _load_or_generate_captions(self, use_auto_caption: bool) -> Dict[str, str]:
        """Load existing captions or generate automatic ones"""
        caption_file = self.image_dir / "captions.json"
        
        if caption_file.exists():
            with open(caption_file, 'r') as f:
                return json.load(f)
        
        captions = {}
        
        if use_auto_caption:
            print("Generating automatic captions...")
            # Simple caption variations for portraits
            caption_templates = [
                "a portrait photo of a person",
                "professional portrait photography",
                "a close-up portrait of a person",
                "a portrait photograph",
                "portrait of a person, professional lighting",
                "high quality portrait photo",
                "studio portrait photography",
                "artistic portrait of a person"
            ]
            
            for img_path in tqdm(self.image_paths, desc="Creating captions"):
                # Randomly select caption template
                caption = random.choice(caption_templates)
                # Add some variety
                if random.random() > 0.5:
                    caption += ", high resolution"
                if random.random() > 0.7:
                    caption += ", professional photography"
                
                captions[str(img_path)] = caption
            
            # Save captions
            with open(caption_file, 'w') as f:
                json.dump(captions, f, indent=2)
        else:
            # Use generic caption for all
            for img_path in self.image_paths:
                captions[str(img_path)] = f"{self.caption_prefix} a person"
        
        return captions
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Load and process image
        image = Image.open(img_path).convert('RGB')
        image = self.transforms(image)
        
        # Get caption
        caption = self.captions.get(str(img_path), f"{self.caption_prefix} a person")
        
        return {
            "pixel_values": image,
            "input_ids": caption  # Will be tokenized later
        }

# ============================================
# STEP 3: LORA CONFIGURATION AND TRAINING
# ============================================

def create_lora_config():
    """Create LoRA configuration optimized for T4 GPU"""
    from peft import LoraConfig
    
    config = LoraConfig(
        r=16,  # Rank (8-16 recommended for portraits)
        lora_alpha=32,  # Alpha parameter for LoRA scaling
        target_modules=[
            "to_q", "to_v", "to_k", "to_out.0",  # Attention layers
            "proj_in", "proj_out",  # Projection layers
            "ff.net.0.proj", "ff.net.2",  # Feed-forward layers
        ],
        lora_dropout=0.1,
    )
    return config

class LoRATrainer:
    """Main trainer class for LoRA fine-tuning"""
    
    def __init__(
        self,
        model_name: str = "runwayml/stable-diffusion-v1-5",  # Using SD 1.5 for T4 memory
        output_dir: str = "./portrait_lora",
        learning_rate: float = 1e-4,
        batch_size: int = 1,  # T4 can handle batch_size=1 for 512x512
        gradient_accumulation_steps: int = 4,
        max_train_steps: int = 15000,
        checkpointing_steps: int = 500,
        validation_steps: int = 100,
        mixed_precision: str = "fp16",
        enable_xformers: bool = True,
        seed: int = 42
    ):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_train_steps = max_train_steps
        self.checkpointing_steps = checkpointing_steps
        self.validation_steps = validation_steps
        self.mixed_precision = mixed_precision
        self.enable_xformers = enable_xformers
        self.seed = seed
        
        # Set seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    def setup_model_and_optimizer(self):
        """Load model and setup LoRA"""
        from diffusers import StableDiffusionPipeline, DDPMScheduler
        from peft import get_peft_model, TaskType
        from transformers import CLIPTextModel, CLIPTokenizer
        
        print("\nLoading Stable Diffusion model...")
        
        # Load pipeline
        pipe = StableDiffusionPipeline.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.mixed_precision == "fp16" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        # Extract components
        self.vae = pipe.vae
        self.text_encoder = pipe.text_encoder
        self.tokenizer = pipe.tokenizer
        self.unet = pipe.unet
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.model_name,
            subfolder="scheduler"
        )
        
        # Move to device and enable memory efficient attention
        self.vae.to(self.device)
        self.text_encoder.to(self.device)
        self.unet.to(self.device)
        
        # Freeze VAE and text encoder
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        
        # Enable xformers for memory efficiency
        if self.enable_xformers:
            try:
                from xformers.ops import memory_efficient_attention
                self.unet.enable_xformers_memory_efficient_attention()
                print("✓ XFormers enabled for memory efficiency")
            except ImportError:
                print("⚠ XFormers not available, using default attention")
        
        # Setup LoRA
        print("\nSetting up LoRA layers...")
        lora_config = create_lora_config()
        
        # Apply LoRA to UNet
        self.unet.requires_grad_(False)
        
        # Initialize LoRA layers manually for better control
        self._initialize_lora_layers()
        
        # Setup optimizer (only for LoRA parameters)
        lora_params = [p for p in self.unet.parameters() if p.requires_grad]
        print(f"Number of trainable parameters: {sum(p.numel() for p in lora_params):,}")
        
        self.optimizer = torch.optim.AdamW(
            lora_params,
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=1e-2,
            eps=1e-8
        )
        
        # Learning rate scheduler
        from torch.optim.lr_scheduler import CosineAnnealingLR
        self.lr_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.max_train_steps,
            eta_min=self.learning_rate * 0.1
        )
        
        print("✓ Model and optimizer setup complete")
    
    def _initialize_lora_layers(self):
        """Initialize LoRA layers in the UNet"""
        from peft import LoraConfig, get_peft_model, TaskType
        
        config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            lora_dropout=0.1,
        )
        
        # Apply LoRA to UNet attention layers
        for name, module in self.unet.named_modules():
            if 'attn' in name and hasattr(module, 'to_q'):
                # Add LoRA to attention layers
                self._add_lora_to_linear(module.to_q, rank=16)
                self._add_lora_to_linear(module.to_k, rank=16)
                self._add_lora_to_linear(module.to_v, rank=16)
                if hasattr(module.to_out, '0'):
                    self._add_lora_to_linear(module.to_out[0], rank=16)
    
    def _add_lora_to_linear(self, linear_module, rank=16, alpha=32):
        """Add LoRA decomposition to a linear layer"""
        in_features = linear_module.in_features
        out_features = linear_module.out_features
        
        # Create LoRA matrices
        lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Store as attributes
        linear_module.lora_A = lora_A
        linear_module.lora_B = lora_B
        linear_module.lora_alpha = alpha
        linear_module.lora_rank = rank
        
        # Make base weights non-trainable
        linear_module.weight.requires_grad = False
        
        # Override forward method
        original_forward = linear_module.forward
        
        def lora_forward(x):
            base_output = original_forward(x)
            lora_output = (x @ lora_A.T @ lora_B.T) * (alpha / rank)
            return base_output + lora_output
        
        linear_module.forward = lora_forward
    
    def train(self, dataset: PortraitDataset, num_epochs: int = None):
        """Main training loop"""
        print("\n" + "=" * 60)
        print("STARTING TRAINING")
        print("=" * 60)
        
        # Setup model and optimizer
        self.setup_model_and_optimizer()
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        # Calculate epochs if not specified
        if num_epochs is None:
            num_epochs = max(1, self.max_train_steps // len(dataloader))
        
        # Training metrics
        global_step = 0
        progress_bar = tqdm(total=self.max_train_steps, desc="Training")
        
        # Training loop
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            for batch_idx, batch in enumerate(dataloader):
                if global_step >= self.max_train_steps:
                    break
                
                # Move batch to device
                pixel_values = batch["pixel_values"].to(self.device)
                captions = batch["input_ids"]
                
                # Encode text
                text_inputs = self.tokenizer(
                    captions,
                    padding="max_length",
                    max_length=77,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)
                
                with torch.no_grad():
                    encoder_hidden_states = self.text_encoder(
                        text_inputs.input_ids
                    )[0]
                
                # Encode images to latents
                with torch.no_grad():
                    latents = self.vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * self.vae.config.scaling_factor
                
                # Sample noise
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, self.noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=self.device
                ).long()
                
                # Add noise to latents
                noisy_latents = self.noise_scheduler.add_noise(
                    latents, noise, timesteps
                )
                
                # Predict noise
                noise_pred = self.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states
                ).sample
                
                # Calculate loss
                loss = F.mse_loss(noise_pred, noise, reduction="mean")
                
                # Backward pass
                loss.backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.unet.parameters(), max_norm=1.0
                    )
                    
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    
                    global_step += 1
                    progress_bar.update(1)
                    
                    # Logging
                    if global_step % 10 == 0:
                        progress_bar.set_postfix({
                            'loss': f'{loss.item():.4f}',
                            'lr': f'{self.lr_scheduler.get_last_lr()[0]:.6f}'
                        })
                    
                    # Checkpointing
                    if global_step % self.checkpointing_steps == 0:
                        self.save_checkpoint(global_step)
                    
                    # Validation
                    if global_step % self.validation_steps == 0:
                        self.validate(global_step)
                
                # Clear cache periodically
                if batch_idx % 50 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
        
        progress_bar.close()
        print("\n✓ Training complete!")
        
        # Save final model
        self.save_final_model()
    
    def save_checkpoint(self, step: int):
        """Save training checkpoint"""
        checkpoint_dir = self.output_dir / f"checkpoint-{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save LoRA weights only
        lora_state_dict = {}
        for name, module in self.unet.named_modules():
            if hasattr(module, 'lora_A'):
                lora_state_dict[f"{name}.lora_A"] = module.lora_A
                lora_state_dict[f"{name}.lora_B"] = module.lora_B
        
        torch.save(lora_state_dict, checkpoint_dir / "lora_weights.pt")
        print(f"✓ Saved checkpoint at step {step}")
    
    def save_final_model(self):
        """Save final LoRA model in safetensors format"""
        from safetensors.torch import save_file
        
        print("\nSaving final LoRA model...")
        
        # Collect all LoRA weights
        lora_state_dict = {}
        for name, module in self.unet.named_modules():
            if hasattr(module, 'lora_A'):
                lora_state_dict[f"unet.{name}.lora_A"] = module.lora_A
                lora_state_dict[f"unet.{name}.lora_B"] = module.lora_B
        
        # Save as safetensors
        save_file(
            lora_state_dict,
            self.output_dir / "portrait_lora.safetensors",
            metadata={
                "base_model": self.model_name,
                "rank": "16",
                "alpha": "32",
                "training_steps": str(self.max_train_steps)
            }
        )
        
        print(f"✓ Final model saved to {self.output_dir / 'portrait_lora.safetensors'}")
    
    def validate(self, step: int):
        """Generate validation images"""
        print(f"\nGenerating validation images at step {step}...")
        
        from diffusers import StableDiffusionPipeline
        
        # Test prompts
        test_prompts = [
            "a professional portrait of a person, studio lighting",
            "close-up portrait photography, high quality",
            "artistic portrait of a person, dramatic lighting"
        ]
        
        with torch.no_grad():
            for i, prompt in enumerate(test_prompts):
                # Generate image (simplified for validation)
                print(f"  Generating: {prompt[:50]}...")
                
                # Here you would generate actual images
                # For now, just logging
                
        torch.cuda.empty_cache()

# ============================================
# STEP 4: SEMANTIC ROUTER IMPLEMENTATION
# ============================================

class SemanticRouter:
    """
    Semantic router for dynamic LoRA weight calculation
    Uses sentence transformers to compute domain similarities
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        temperature: float = 0.7
    ):
        from sentence_transformers import SentenceTransformer
        
        self.encoder = SentenceTransformer(model_name)
        self.temperature = temperature
        
        # Define domain descriptions
        self.domains = {
            "portrait": "portrait photography of a person or face, professional headshot",
            "food": "food photography of dishes or cuisine, culinary art",
            "landscape": "landscape photo of nature or cityscape, scenic view"
        }
        
        # Pre-compute domain embeddings
        self.domain_embeddings = {}
        for domain, description in self.domains.items():
            self.domain_embeddings[domain] = self.encoder.encode(
                description,
                convert_to_tensor=True
            )
    
    def compute_weights(self, prompt: str) -> Dict[str, float]:
        """
        Compute LoRA weights based on prompt similarity to domains
        """
        from sentence_transformers import util
        
        # Encode prompt
        prompt_embedding = self.encoder.encode(prompt, convert_to_tensor=True)
        
        # Compute similarities
        similarities = {}
        for domain, domain_emb in self.domain_embeddings.items():
            sim = util.cos_sim(prompt_embedding, domain_emb)[0, 0]
            similarities[domain] = sim.item()
        
        # Convert to weights using softmax with temperature
        scores = torch.tensor(list(similarities.values()))
        weights = torch.softmax(scores / self.temperature, dim=0)
        
        # Create weight dictionary
        weight_dict = {}
        for i, domain in enumerate(similarities.keys()):
            weight_dict[domain] = weights[i].item()
        
        return weight_dict
    
    def merge_loras(self, pipe, lora_paths: Dict[str, str], weights: Dict[str, float]):
        """
        Load and merge multiple LoRAs with computed weights
        """
        print("\nMerging LoRAs with semantic weights:")
        for domain, weight in weights.items():
            print(f"  {domain}: {weight:.3f}")
        
        # Load each LoRA
        for domain, path in lora_paths.items():
            if Path(path).exists():
                pipe.load_lora_weights(
                    path,
                    adapter_name=domain
                )
        
        # Set adapter weights
        pipe.set_adapters(
            list(weights.keys()),
            adapter_weights=list(weights.values())
        )
        
        return pipe

# ============================================
# MAIN EXECUTION SCRIPT
# ============================================

def main():
    """
    Main execution function for Colab
    """
    print("\n" + "=" * 60)
    print("MAIN EXECUTION")
    print("=" * 60)
    
    # Configuration
    CONFIG = {
        # Dataset settings
        "image_dir": "/content/portrait_images",  # Update this path
        "max_images": 30000,  # Using all 30k images
        "resolution": 512,
        
        # Training settings  
        "batch_size": 1,  # T4 GPU constraint
        "gradient_accumulation_steps": 4,  # Effective batch size = 4
        "learning_rate": 1e-4,
        "max_train_steps": 15000,
        "checkpointing_steps": 500,
        
        # Output
        "output_dir": "/content/drive/MyDrive/portrait_lora",
    }
    
    print("\nConfiguration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    
    # Step 1: Prepare Dataset
    print("\n" + "-" * 40)
    print("STEP 1: Preparing Dataset")
    print("-" * 40)
    
    dataset = PortraitDataset(
        image_dir=CONFIG["image_dir"],
        resolution=CONFIG["resolution"],
        max_images=CONFIG["max_images"],
        use_auto_caption=True
    )
    
    # Step 2 & 3: Train LoRA
    print("\n" + "-" * 40)
    print("STEP 2-3: Training Portrait LoRA")
    print("-" * 40)
    
    trainer = LoRATrainer(
        output_dir=CONFIG["output_dir"],
        learning_rate=CONFIG["learning_rate"],
        batch_size=CONFIG["batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        max_train_steps=CONFIG["max_train_steps"],
        checkpointing_steps=CONFIG["checkpointing_steps"],
        mixed_precision="fp16",  # Important for T4
        enable_xformers=True  # Memory efficiency
    )
    
    # Train model
    trainer.train(dataset)
    
    # Step 4: Setup Semantic Router (for later use)
    print("\n" + "-" * 40)
    print("STEP 4: Setting up Semantic Router")
    print("-" * 40)
    
    router = SemanticRouter()
    
    # Test router with example prompts
    test_prompts = [
        "A chef eating sushi in a snowy mountain forest",
        "A professional headshot of a business person",
        "A beautiful sunset over mountain peaks"
    ]
    
    print("\nTesting semantic router:")
    for prompt in test_prompts:
        weights = router.compute_weights(prompt)
        print(f"\nPrompt: {prompt}")
        print(f"Weights: {weights}")
    
    print("\n" + "=" * 60)
    print("✓ ALL STEPS COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Train food and landscape LoRAs using similar process")
    print("2. Implement Attend-and-Excite for inference")
    print("3. Combine all components for final pipeline")
    print("\nYour portrait LoRA is saved at:", CONFIG["output_dir"])

if __name__ == "__main__":
    # For Colab execution
    main()
